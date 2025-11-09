import torch
import torch.nn
import torch.nn.functional as F
from .sh import eval_sh_bases
import numpy as np
import time

def positional_encoding(positions, freqs):
        freq_bands = (2**torch.arange(freqs, device=positions.device, dtype=positions.dtype))  # (F,)
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma*dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:,-1:]


def SHRender(xyz_sampled, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(xyz_sampled, viewdirs, features):
    rgb = features
    return rgb


class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb=aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0 / self.aabbSize * 2

        # alpha_volume might be BoolTensor([D,H,W]); cast to float
        av = alpha_volume.to(self.device).float()
        # reshape into [N,C,D,H,W] for grid_sample
        self.alpha_volume = av.view(1, 1, av.shape[-3], av.shape[-2], av.shape[-1])
      
        self.gridSize = torch.LongTensor([av.shape[-1], av.shape[-2], av.shape[-3]]).to(self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_norm = (xyz_sampled - self.aabb[0]) / self.aabbSize * 2 - 1   # [N,3]
        grid = xyz_norm.view(1, -1, 1, 1, 3)  # [1,N,1,1,3]
        grid = grid.to(dtype=self.alpha_volume.dtype)
        alpha_vals = F.grid_sample(self.alpha_volume, grid, align_corners=True).view(-1)
        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        span = (self.aabb[1] - self.aabb[0])
        span = torch.where(span == 0, torch.ones_like(span), span)

        norm01 = (xyz_sampled - self.aabb[0]) / span
        return norm01 * 2 - 1

class MLPRender_FeaNG(torch.nn.Module):
    """
    Safer MLP_Fea:
      - LayerNorm / normalize features
      - Gate warmup PE(features) 
      - Normalize view directions
      - (optional) PE after clamping
    """
    def __init__(self, inChanel, viewpe=2, feape=0, featureC=128,
                 use_layernorm=True, fea_gate_init=0.2, fea_gate_max=1.0,
                 pe_clamp=3.0):
        super().__init__()
        self.viewpe = int(viewpe)
        self.feape  = int(feape)
        self.inCh   = int(inChanel)

        # feature norm: LayerNorm by default (without affine to avoid learning statistics poorly)
        self.use_layernorm = bool(use_layernorm)
        self.ln = torch.nn.LayerNorm(self.inCh, elementwise_affine=False) if self.use_layernorm else None

        # gate: control strength of applying PE(features); recommend to modify by training warmup
        self.register_buffer('fea_gate', torch.tensor(float(fea_gate_init)))
        self.fea_gate_max = float(fea_gate_max)
        self.pe_clamp = float(pe_clamp)

        in_mlpC = self.inCh + 3
        if self.viewpe > 0:
            in_mlpC += 2 * self.viewpe * 3
        if self.feape > 0:
            in_mlpC += 2 * self.feape * self.inCh

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_mlpC, featureC), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, featureC), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, 3)
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0.0)

    @torch.no_grad()
    def set_fea_gate(self, value: float):
        # gate: 0~1
        v = float(value)
        v = 0.0 if v < 0.0 else v
        v = self.fea_gate_max if v > self.fea_gate_max else v
        self.fea_gate.fill_(v)

    def _norm_features(self, f: torch.Tensor) -> torch.Tensor:
        if self.ln is not None:
            return self.ln(f)
        mean = f.mean(dim=-1, keepdim=True)
        std  = f.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (f - mean) / std

    def forward(self, pts, viewdirs, features):
        v = viewdirs / (viewdirs.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        f = self._norm_features(features)
        parts = [f, v]

        # PE(viewdirs)
        if self.viewpe > 0:
            parts += [positional_encoding(v, self.viewpe)]

        # PE(features) + gate
        if self.feape > 0 and self.fea_gate.item() > 0.0:
            f_clamped = f.clamp(-self.pe_clamp, self.pe_clamp)
            pe_f = positional_encoding(f_clamped, self.feape)
            parts += [pe_f * self.fea_gate]

        mlp_in = torch.cat(parts, dim=-1)
        rgb = torch.sigmoid(self.mlp(mlp_in))
        return rgb

class MLPRender_Fea(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea, self).__init__()

        self.in_mlpC = 2*viewpe*3 + 2*feape*inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLPRender_PE(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, pospe=6, featureC=128):
        super(MLPRender_PE, self).__init__()

        self.in_mlpC = (3+2*viewpe*3)+ (3+2*pospe*3)  + inChanel #
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLPRender(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, featureC=128):
        super(MLPRender, self).__init__()

        self.in_mlpC = (3+2*viewpe*3) + inChanel
        self.viewpe = viewpe
        
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class TensorBase(torch.nn.Module):
    # Note that any method only including "pass" indicates it's implemented by subclasses.
    def __init__(self, aabb, gridSize, device, density_n_comp = 8, app_n_comp = 24, app_dim = 27,
                    shadingMode = 'MLP_PE', alphaMask = None, near_far=[2.0,6.0],
                    density_shift = -10.0, alphaMask_thres=0.001, distance_scale=25, rayMarch_weight_thres=0.0001,
                    pos_pe = 6, view_pe = 6, fea_pe = 6, featureC=128, step_ratio=2.0,
                    fea2denseAct = 'softplus'):
        super(TensorBase, self).__init__()
        self.device=device

        self.density_n_comp = density_n_comp
        self.app_n_comp = app_n_comp
        self.app_dim = app_dim
        self.aabb = aabb
        self.alphaMask = alphaMask

        self.density_shift = float(density_shift)
        self.alphaMask_thres = float(alphaMask_thres)
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio

        self.default_nSamples = 256  # or 512
        self.update_stepSize(gridSize)

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]

        self.shadingMode, self.pos_pe, self.view_pe, self.fea_pe, self.featureC = shadingMode, pos_pe, view_pe, fea_pe, featureC
        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, device)

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device):
        if shadingMode == 'MLP_PE':
            self.renderModule = MLPRender_PE(self.app_dim, view_pe, pos_pe, featureC).to(device)
        elif shadingMode == 'MLP_Fea':
            self.renderModule = MLPRender_Fea(self.app_dim, view_pe, fea_pe, featureC).to(device)
        elif shadingMode == 'MLP_FeaNG': 
            self.renderModule = MLPRender_FeaNG(self.app_dim, view_pe, fea_pe, featureC).to(device)
        elif shadingMode == 'MLP':
            self.renderModule = MLPRender(self.app_dim, view_pe, featureC).to(device)
        elif shadingMode == 'SH':
            self.renderModule = SHRender
        elif shadingMode == 'RGB':
            assert self.app_dim == 3
            self.renderModule = RGBRender
        else:
            print("Unrecognized shading module")
            exit()
        print("pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe)
        print(self.renderModule)

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        self.gridSize= torch.LongTensor(gridSize).to(self.device)
        self.units=self.aabbSize / (self.gridSize-1)
        self.stepSize=torch.mean(self.units)*self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        print("sampling step size: ", self.stepSize)

    def init_svd_volume(self, res, device):
        pass
    
    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial = 0.02, lr_init_network = 0.001):
        pass

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'density_n_comp': self.density_n_comp,
            'appe_n_comp': self.app_n_comp,
            'app_dim': self.app_dim,

            'density_shift': self.density_shift,
            'alphaMask_thres': self.alphaMask_thres,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'fea2denseAct': self.fea2denseAct,

            'near_far': self.near_far,
            'step_ratio': self.step_ratio,

            'shadingMode': self.shadingMode,
            'pos_pe': self.pos_pe,
            'view_pe': self.view_pe,
            'fea_pe': self.fea_pe,
            'featureC': self.featureC
        }

    # Note: Legacy ray sampling function for datasets with ndc_ray=True (e.g., LLFF).
    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        """
        Patch-adaptive ray sampling using the AABB ENTRY point to determine step size.

        Returns:
            rays_pts:    [N_rays, N_samples, 3]
            z_vals:      [N_rays, N_samples]
            mask_inbbox: [N_rays, N_samples] boolean, True means the sample is inside AABB
        """
        device = rays_o.device
        dtype  = rays_o.dtype
        aabb_min, aabb_max = self.aabb[0], self.aabb[1]
        near, far = self.near_far
        N_rays = rays_o.shape[0]
        S = self.default_nSamples if N_samples <= 0 else N_samples

        if N_rays == 0:
            empty_pts = torch.empty((0, S, 3), device=device, dtype=dtype)
            empty_z   = torch.empty((0, S),    device=device, dtype=dtype)
            empty_m   = torch.empty((0, S),    device=device, dtype=torch.bool)
            return empty_pts, empty_z, empty_m

        # ---- print control (rate-limited) ----
        gs = int(getattr(self, "global_step", getattr(self, "step", getattr(self, "iter", 0))))
        warn_interval = int(getattr(self, "warn_interval", 200))   # print every K steps when training
        warn_min_rays = int(getattr(self, "warn_min_rays", 512))   # only if batch is large enough
        debug_map_stats = bool(getattr(self, "debug_map_stats", False))
        should_print = (is_train and (gs % warn_interval == 0) and (N_rays >= warn_min_rays)) or debug_map_stats

        # avoid zero-division
        rays_d = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)

        # --- robust ray-AABB slab intersection ---
        t1 = (aabb_min - rays_o) / rays_d
        t2 = (aabb_max - rays_o) / rays_d
        tmin = torch.minimum(t1, t2).amax(dim=-1)  # entry across axes
        tmax = torch.maximum(t1, t2).amin(dim=-1)  # exit  across axes

        # clamp to near/far
        t_enter = torch.clamp(tmin, min=float(near))
        t_exit  = torch.clamp(tmax, max=float(far))
        hits = (t_exit > t_enter)  # [N_rays], whether the ray intersects the box

        # --- use ENTRY points to decide per-ray patch step size ---
        entry_pts = rays_o + t_enter.unsqueeze(-1) * rays_d  # [N_rays,3]
        ray_patch_coord, exists = self._map_coords_to_patch(entry_pts)  # coords: [N_rays,3], exists: [N_rays]

        if should_print:
            miss_ratio = 1.0 - exists.float().mean().item()
            if (miss_ratio > 0.3) and is_train:
                print(f"[WARNING] {miss_ratio:.0%} of ray entries map to UNALLOCATED patches (fallback risk)!")
            origin_in = ((rays_o >= aabb_min) & (rays_o <= aabb_max)).all(dim=-1).float().mean().item()
            hit_frac  = hits.float().mean().item()
            print(f"[dbg] step={gs} | N_rays={N_rays} | origin_inAABB={origin_in:.1%} | hits={hit_frac:.1%} | entries_in_map={(1.0 - miss_ratio):.1%}")

        aabb_len = (aabb_max - aabb_min).to(dtype=dtype)
        base_res = torch.as_tensor(self.gridSize, device=device, dtype=dtype)

        step_list = []
        for coord, hit in zip(ray_patch_coord, hits):
            if hit:
                key = tuple(coord.tolist())
                patch = self.patch_map.get(key, None)
                if patch is not None:
                    res = torch.as_tensor(patch['res'], device=device, dtype=dtype)
                else:
                    res = base_res
            else:
                res = base_res
            unit_size = aabb_len / res
            step_list.append(unit_size.mean() * self.step_ratio)
        patch_sizes = torch.stack(step_list)  # [N_rays]

        # --- build samples ---
        rng = torch.arange(S, device=device, dtype=dtype).unsqueeze(0).repeat(N_rays, 1)
        if is_train:
            rng = rng + torch.rand((N_rays, 1), device=device, dtype=dtype)

        z_vals = t_enter.unsqueeze(1) + rng * patch_sizes.unsqueeze(1)   # [N_rays, S]
        z_vals = torch.minimum(z_vals, t_exit.unsqueeze(1))              # stay within [enter, exit]

        rays_pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # [N_rays, S, 3]

        # strictly inside AABB and along valid segment (no forced True)
        in_box = ((rays_pts >= aabb_min) & (rays_pts <= aabb_max)).all(dim=-1)      # [N_rays, S]
        mask_in = in_box & hits.unsqueeze(1)                                        # [N_rays, S]

        if should_print:
            print(f"[dbg] step={gs} | mean_step={patch_sizes.mean().item():.6f} | min_max_step=({patch_sizes.min().item():.6f},{patch_sizes.max().item():.6f})")

        return rays_pts, z_vals, mask_in

    def shrink(self, new_aabb, voxel_size):
        pass

    @torch.no_grad()
    def getDenseAlpha(self,gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = torch.stack(torch.meshgrid(torch.linspace(0, 1, gridSize[0]),
                                             torch.linspace(0, 1, gridSize[1]),
                                             torch.linspace(0, 1, gridSize[2]), 
                                             indexing='ij'), -1).to(self.device)
        dense_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples

        # dense_xyz = dense_xyz
        # print(self.stepSize, self.distance_scale*self.aabbDiag)
        alpha = torch.zeros_like(dense_xyz[...,0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1,3), self.stepSize).view((gridSize[1], gridSize[2]))
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200,200,200)):
        """
        Recompute alpha mask and shrink AABB to non-empty region.
        If no voxels survive the threshold, we simply leave the AABB unchanged.
        """
        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = dense_xyz.transpose(0,2).contiguous()          # [X,Y,Z,3] -> [Z,Y,X,3] then flatten
        alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None,None]  
        total_voxels = gridSize[0]*gridSize[1]*gridSize[2]

        ks = 3
        mask = F.max_pool3d(alpha, kernel_size=ks, padding=ks//2, stride=1).view(gridSize[::-1])
        mask = mask >= self.alphaMask_thres

        self.alphaMask = AlphaGridMask(self.device, self.aabb, mask)

        # find surviving points
        valid_flat = mask.view(-1)
        if valid_flat.sum() == 0:
            print("[updateAlphaMask] Warning: no voxels exceed threshold; keeping old AABB.")
            return self.aabb  # nothing to shrink to

        valid_xyz = dense_xyz[mask]
        xyz_min = valid_xyz.amin(dim=0)
        xyz_max = valid_xyz.amax(dim=0)
        new_aabb = torch.stack((xyz_min, xyz_max), dim=0)

        total = valid_flat.sum().item()
        print(f"bbox: min={xyz_min}, max={xyz_max}")
        print(f"{total} / {total_voxels} voxels remain ({100.0*total/total_voxels:.2f}%)")

        return new_aabb

    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=None, bbox_only=False):
        """
        Filter rays outside bounding box or with no alpha contribution.

        Speedups:
        - Auto-chunk when chunk is None or <=0 (keeps ~6–9M samples per chunk). 
        - Two-stage alpha check with early-exit (stage1 = N_samples//4, stage2 refine only near-boundary rays).
        - Always do a fast bbox prefilter before alphaMask sampling.
        """
        print('========> filtering rays (fast) ...')
        tt = time.time()

        device = self.device
        rays_flat = all_rays.view(-1, all_rays.shape[-1])  # [N,6]
        rgbs_flat = all_rgbs.view(-1, all_rgbs.shape[-1])  # [N,3]
        N = rays_flat.shape[0]

        # auto-chunk
        if chunk is None or int(chunk) <= 0:
            target_samples = 8_000_000
            chunk_alpha = max(8192, min(131072, int(target_samples // max(1, N_samples))))
        else:
            chunk_alpha = int(chunk)

        chunk_bbox = max(chunk_alpha * 4, 200_000)

        thr = float(self.alphaMask_thres)  
        has_mask = hasattr(self, "alphaMask") and (self.alphaMask is not None)

        # bbox quick reject
        idx_keep_bbox = []
        aabb_min = self.aabb[0].to(device)
        aabb_max = self.aabb[1].to(device)
        near, far = float(self.near_far[0]), float(self.near_far[1])

        for idx_chunk in torch.split(torch.arange(N), chunk_bbox):
            rays_chunk = rays_flat[idx_chunk].to(device, non_blocking=True)
            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]

            vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
            rate_a = (aabb_max - rays_o) / vec
            rate_b = (aabb_min - rays_o) / vec
            t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
            t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
            mask_inbbox = t_max > t_min

            if mask_inbbox.any():
                idx_keep_bbox.append(idx_chunk[mask_inbbox].cpu())

        if len(idx_keep_bbox) == 0:
            print(f'Ray filtering done! takes {time.time()-tt:.2f}s. ray mask ratio: 0.00%')
            return all_rays[:0], all_rgbs[:0]

        idx_keep_bbox = torch.cat(idx_keep_bbox) 

        if bbox_only or (not has_mask):
            mask = torch.zeros(N, dtype=torch.bool)
            mask[idx_keep_bbox] = True
            mask = mask.view(all_rgbs.shape[:-1])
            print(f'Ray filtering (bbox-only) done! takes {time.time()-tt:.2f}s. keep ratio: {mask.sum()/mask.numel():.2%}')
            return all_rays[mask], all_rgbs[mask]

        n1 = max(32, N_samples // 4)  
        n2 = max(0, N_samples - n1)   # refine (only near-boundary)
        idx_final_keep = []

        for idx_chunk in torch.split(idx_keep_bbox, chunk_alpha):
            rays_chunk = rays_flat[idx_chunk].to(device, non_blocking=True)
            rays_o, rays_d = rays_chunk[:, :3], rays_chunk[:, 3:6]

            xyz_s1, _, valid1 = self.sample_ray(rays_o, rays_d, N_samples=n1, is_train=False)
            if valid1.any():
                alpha1 = self.alphaMask.sample_alpha(xyz_s1).view(xyz_s1.shape[:-1])
            else:
                alpha1 = torch.zeros((rays_chunk.shape[0], n1), device=device, dtype=rays_o.dtype)

            hit1 = (alpha1 > thr).any(dim=-1)  
            keep_idx1 = idx_chunk[hit1]
            if keep_idx1.numel() > 0:
                idx_final_keep.append(keep_idx1.cpu())

            if n2 > 0:
                near_mask = (~hit1) & ((alpha1.max(dim=-1).values) > (0.5 * thr))
                if near_mask.any():
                    rays_refine = rays_chunk[near_mask]
                    ro2, rd2 = rays_refine[:, :3], rays_refine[:, 3:6]
                    xyz_s2, _, valid2 = self.sample_ray(ro2, rd2, N_samples=n2, is_train=False)
                    if valid2.any():
                        alpha2 = self.alphaMask.sample_alpha(xyz_s2).view(xyz_s2.shape[:-1])
                        hit2 = (alpha2 > thr).any(dim=-1)
                        base = idx_chunk[near_mask.nonzero(as_tuple=False).squeeze(-1)]
                        keep_idx2 = base[hit2]
                        if keep_idx2.numel() > 0:
                            idx_final_keep.append(keep_idx2.cpu())

        if len(idx_final_keep) == 0:
            print(f'Ray filtering done! takes {time.time()-tt:.2f}s. ray mask ratio: 0.00%')
            return all_rays[:0], all_rgbs[:0]

        idx_final_keep = torch.cat(idx_final_keep).unique(sorted=True)

        mask = torch.zeros(N, dtype=torch.bool)
        mask[idx_final_keep] = True
        mask = mask.view(all_rgbs.shape[:-1])

        print(f'Ray filtering done! takes {time.time()-tt:.2f}s. ray mask ratio: {mask.sum()/mask.numel():.2%} '
            f'(chunk_alpha={chunk_alpha}, n1={n1}, n2={n2})')
        return all_rays[mask], all_rgbs[mask]


    def feature2density(self, density_features):
        density_shift = 0.0 if self.density_shift is None else self.density_shift
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features + density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)

    @torch.no_grad()
    def _gather_patch_res(self, patch_coords: torch.Tensor) -> torch.Tensor:
        """
        Map each sample's integer patch coords [N,3] to that patch's resolution (Rx,Ry,Rz).
        If a patch is missing or has no 'res' field, fall back to the global gridSize.
        Returns:
            Tensor of shape [N,3], dtype=float32, device aligned with input.
        """
        pc = patch_coords.long().view(-1, 3)
        dev = pc.device
        out = torch.empty((pc.shape[0], 3), device=dev, dtype=torch.float32)

        if isinstance(self.gridSize, torch.Tensor):
            gs_list = self.gridSize.tolist()
        else:
            gs_list = list(self.gridSize)
        gs = torch.as_tensor(gs_list, device=dev, dtype=torch.float32)

        if not hasattr(self, "patch_map") or len(self.patch_map) == 0:
            out[:] = gs
            return out

        # group by unique keys to avoid per-point Python dict lookup in a tight loop
        uniq, inv = torch.unique(pc, dim=0, return_inverse=True)
        for u in range(uniq.shape[0]):
            mask = (inv == u)
            key = tuple(int(v) for v in uniq[u].tolist())
            patch = self.patch_map.get(key, None)
            if patch is None or ('res' not in patch):
                r = gs
            else:
                r = torch.as_tensor(patch['res'], device=dev, dtype=torch.float32)
            out[mask] = r
        return out

    @torch.no_grad()
    def compute_alpha(self, xyz_locs, length=1.0):
        """
        Compute per-point alpha consistent with training:
            alpha = 1 - exp(-sigma * step_len * distance_scale * alpha_gate_scale)
        Notes:
        - step_len is per-sample. By default we use the global marching step,
            but for points whose entry maps to a valid patch we derive a per-patch step
            via that patch's resolution (mean voxel size * step_ratio).
        """
        dev = xyz_locs.device
        self.ensure_default_patch()

        # determine which points are inside the (pooled) alpha mask
        if self.alphaMask is not None:
            sampled = self.alphaMask.sample_alpha(xyz_locs)
            inside = (sampled > 0)
        else:
            inside = torch.ones(xyz_locs.shape[0], dtype=torch.bool, device=dev)

        sigma = torch.zeros(xyz_locs.shape[0], device=dev)

        if inside.any():
            idxs = inside.nonzero(as_tuple=False).squeeze(-1)
            pts_world = xyz_locs[idxs]  # [M,3] 
            pts_norm  = self.normalize_coord(pts_world)

            # default step lengths: global marching step
            lengths = torch.full((pts_world.shape[0],), float(self.stepSize), dtype=sigma.dtype, device=dev)

            # for points that map to allocated patches, use per-patch step derived from patch reso
            patch_coords, exists = self._map_coords_to_patch(pts_world)
            if exists.any():
                ok = exists.nonzero(as_tuple=False).squeeze(-1)
                aabb_span = (self.aabb[1] - self.aabb[0])
                R = self._gather_patch_res(patch_coords[ok])         # [M_ok,3]
                units = aabb_span / (R - 1).clamp(min=1)
                steps = units.mean(dim=-1) * float(self.step_ratio)  # [M_ok]
                lengths[ok] = steps

            dens_feat  = self.compute_density_patchwise_fast(pts_norm, patch_coords)
            sigma_vals = self.feature2density(dens_feat)
            sigma_vals = torch.nan_to_num(sigma_vals, nan=0.0, posinf=1e6, neginf=0.0)
            sigma = sigma.masked_scatter(inside, sigma_vals)

        scale = float(getattr(self, "alpha_gate_scale", 1.0))
        global_len = 1.0 if (length is None) else float(length)

        if inside.any():
            alpha = torch.zeros_like(sigma)
            alpha[idxs] = 1.0 - torch.exp(-sigma[idxs] * lengths * global_len * float(self.distance_scale) * scale)
        else:
            alpha = 1.0 - torch.exp(-sigma * float(self.stepSize) * global_len * float(self.distance_scale) * scale)

        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
        return alpha

    def forward(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):
        pass    

