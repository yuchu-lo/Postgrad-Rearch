import torch
import torch.nn.functional as F
import numpy as np
import os, sys, imageio
from tqdm.auto import tqdm
from utils import visualize_depth_numpy, rgb_ssim, rgb_lpips
from dataLoader.ray_utils import get_rays, ndc_rays_blender
import imageio

class PatchTrainStep:
    def __init__(self, tensorf, render_step_size=1.0, white_bg=True, rm_weight_mask_thre=1e-4):
        self.tensorf = tensorf
        self.render_step_size = render_step_size
        self.white_bg = white_bg
        self.rm_weight_mask_thre = float(rm_weight_mask_thre)

    def __call__(self, rays_chunk, target_rgb):
        rgb_map, alphas_map, depth_map, weights, uncertainty = self.forward_render(rays_chunk, is_train=True)
        acc_map = weights.sum(dim=-1)  # [N_rays]

        # loss masking by accumulated weight (detach to avoid weird gradients)
        thr = self.rm_weight_mask_thre
        if thr > 0.0:
            m = (acc_map.detach() > thr).to(rgb_map.dtype).unsqueeze(-1)  # [N,1]
            # weighted/normalized MSE over valid pixels
            diff2 = (rgb_map - target_rgb) ** 2
            num = (diff2 * m).sum()
            den = (m.sum() * 3.0).clamp_min(1e-8)
            loss_rgb = num / den
        else:
            loss_rgb = F.mse_loss(rgb_map, target_rgb)

        # not add any reg terms here (assigned outside if needed)
        total_loss = loss_rgb
        return total_loss, loss_rgb

    def forward_render(self, rays_chunk, is_train=False):
        outputs = self.tensorf(rays_chunk, white_bg=self.white_bg, is_train=is_train)
        if isinstance(outputs, tuple) and len(outputs) == 2:
            # compatible to some simplified model (only obtain rgb_map & depth_map)
            rgb_map, depth_map = outputs
            alphas_map = depth_map.new_zeros(depth_map.shape)
            weights = depth_map.new_zeros(depth_map.shape)
            uncertainty = None
        else:  
            rgb_map, alphas_map, depth_map, weights, uncertainty = outputs

        return rgb_map, alphas_map, depth_map, weights, uncertainty

    @torch.no_grad()
    def render(self, rays_chunk, is_train=False):
        return self.forward_render(rays_chunk, is_train=is_train)

    @torch.no_grad()
    def render_path(self, rays, chunk=2048):
        # given the entire ray path (e.g., for render path), perform chunk-wise rendering and merge the results
        results = []
        for i in range(0, rays.shape[0], chunk):
            rgb, *_ = self.render(rays[i:i+chunk])
            results.append(rgb.cpu())
        return torch.cat(results, dim=0)

    @torch.no_grad()
    def render_patch_mask(self, rays, patch_mask):
        """
        Only rendering to rays that pass through patches in patch_mask.
        patch_mask: set of (i,j,k) tuples, or a bool tensor matching patch_map shape
        """
        rays = rays.to(self.tensorf.aabb.device)
        rays_o, rays_d = rays[:, :3], rays[:, 3:6]

        mid, hit = self.tensorf._ray_aabb_midpoint(rays_o, rays_d)
        if not hit.any():
            return rays.new_zeros((0, 3))

        coords_hit, exists = self.tensorf._map_coords_to_patch(mid[hit])  # [Nh,3]
        miss_ratio = 1.0 - exists.float().mean().item()
        if miss_ratio > 0.3:
            print(f"[WARNING] {miss_ratio:.0%} of rendering patch samples map to MISSING patches (fallback risk)!")
        in_mask_hit = torch.tensor(
            [tuple(c.tolist()) in patch_mask for c in coords_hit],
            device=rays.device, dtype=torch.bool
        )

        valid = torch.zeros(rays.shape[0], dtype=torch.bool, device=rays.device)
        idx_hit = torch.nonzero(hit, as_tuple=False).squeeze(-1)
        valid[idx_hit] = in_mask_hit

        if not valid.any():
            return rays.new_zeros((0, 3))

        rays_in_mask = rays[valid]
        rgb_map, *_ = self.render(rays_in_mask, is_train=False)
        return rgb_map

@torch.no_grad()
def evaluation(test_dataset, tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    """
    Patch-compatible evaluation function for testing UVG (uneven voxel grid) rendering.
    Supports PSNR/SSIM/LPIPS metric computation and optional video export.
    """
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims, l_alex, l_vgg = [], [], []

    os.makedirs(savePath, exist_ok=True)
    os.makedirs(os.path.join(savePath, "rgbd"), exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis, 1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))

    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):
        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])

        # chunked rendering to avoid OOM
        chunk = 2048
        rgb_chunks, depth_chunks = [], []
        for i in range(0, rays.shape[0], chunk):
            rgb_c, _, depth_c, _, _ = renderer(rays[i:i+chunk], is_train=False)
            rgb_chunks.append(rgb_c.cpu())
            depth_chunks.append(depth_c.cpu())

        rgb_map = torch.cat(rgb_chunks, dim=0).clamp(0.0, 1.0)
        depth_map = torch.cat(depth_chunks, dim=0)

        print(f"[DEBUG eval] frame={idx}  rgb_map.min={rgb_map.min():.4f}, max={rgb_map.max():.4f}, mean={rgb_map.mean():.4f}")
        if rgb_map.max() == 0:
            print(f"[WARNING] rgb_map is all zeros at frame {idx}!")

        rgb_map, depth_map = rgb_map.reshape(H, W, 3), depth_map.reshape(H, W)
        depth_map_np, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssims.append(rgb_ssim(rgb_map, gt_rgb, 1))
                l_alex.append(rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device))
                l_vgg.append(rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device))

        rgb_map_np = (rgb_map.numpy() * 255).astype('uint8')
        rgb_maps.append(rgb_map_np)
        depth_maps.append(depth_map_np)

        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map_np)
            rgb_combined = np.concatenate((rgb_map_np, depth_map_np), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_combined)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    if PSNRs:
        avg_psnr = float(np.mean(PSNRs))
        if compute_extra_metrics:
            avg_ssim = float(np.mean(ssims))
            avg_alex = float(np.mean(l_alex))
            avg_vgg = float(np.mean(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([avg_psnr, avg_ssim, avg_alex, avg_vgg]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([avg_psnr]))

    return PSNRs

@torch.no_grad()
def evaluation_path(test_dataset, tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    """
    Patch-compatible path rendering for novel view synthesis video.
    """
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims, l_alex, l_vgg = [], [], []

    os.makedirs(savePath, exist_ok=True)
    os.makedirs(os.path.join(savePath, "rgbd"), exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    W, H = test_dataset.img_wh

    for idx, c2w in tqdm(enumerate(c2ws), total=len(c2ws), desc="Rendering path views"):
        c2w = torch.FloatTensor(c2w).to(device)
        rays_o, rays_d = get_rays(test_dataset.directions.to(device), c2w)

        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)

        rays = torch.cat([rays_o, rays_d], dim=1)

        chunk = 2048
        rgb_chunks, depth_chunks = [], []
        for i in range(0, rays.shape[0], chunk):
            rgb_c, _, depth_c, _, _ = renderer(rays[i:i+chunk], is_train=False)
            rgb_chunks.append(rgb_c.cpu())
            depth_chunks.append(depth_c.cpu())

        rgb_map = torch.cat(rgb_chunks, dim=0).clamp(0.0, 1.0)
        depth_map = torch.cat(depth_chunks, dim=0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3), depth_map.reshape(H, W)
        depth_map_np, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

        rgb_map_np = (rgb_map.numpy() * 255).astype('uint8')
        rgb_maps.append(rgb_map_np)
        depth_maps.append(depth_map_np)

        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map_np)
            rgb_combined = np.concatenate((rgb_map_np, depth_map_np), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_combined)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))

    return PSNRs
