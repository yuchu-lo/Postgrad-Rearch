import os, sys, time
import math
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from models.vm_patch_tensorf import TensorVMSplitPatch
from opt import config_parser
from utils import *
from dataLoader import dataset_dict
from renderer_patch import PatchTrainStep, evaluation, evaluation_path
from uneven_criterion_patch import uneven_critrn
from patch_visualization_utils import export_patch_viz_bundle
import json, shlex, shutil, subprocess
from collections import defaultdict
import copy
from copy import deepcopy
from itertools import chain
from types import SimpleNamespace
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExperimentLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.metrics = defaultdict(list)
        
    def log_iteration(self, iteration, psnr, memory_mb, n_patches, total_voxels):
        """記錄每次迭代的關鍵指標"""
        self.metrics['iteration'].append(iteration)
        self.metrics['psnr'].append(psnr)
        self.metrics['memory_mb'].append(memory_mb)
        self.metrics['n_patches'].append(n_patches)
        self.metrics['total_voxels'].append(total_voxels)
        
        # 計算效率指標
        if memory_mb > 0:
            self.metrics['psnr_per_mb'].append(psnr / memory_mb)
        
    def generate_plots(self):
        """自動生成論文用圖表"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # PSNR vs Iteration
        axes[0, 0].plot(self.metrics['iteration'], self.metrics['psnr'])
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('PSNR (dB)')
        axes[0, 0].grid(True)
        
        # Memory vs Iteration
        axes[0, 1].plot(self.metrics['iteration'], self.metrics['memory_mb'])
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].grid(True)
        
        # PSNR vs Memory (最重要的圖)
        axes[1, 0].scatter(self.metrics['memory_mb'], self.metrics['psnr'])
        axes[1, 0].set_xlabel('Memory (MB)')
        axes[1, 0].set_ylabel('PSNR (dB)')
        axes[1, 0].grid(True)
        
        # Efficiency over time
        axes[1, 1].plot(self.metrics['iteration'], self.metrics['psnr_per_mb'])
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('PSNR per MB')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.log_dir}/experiment_results.png', dpi=300)
        
    def generate_latex_table(self):
        """生成 LaTeX 表格供論文使用"""
        final_idx = -1
        latex = f"""
        \\begin{{table}}[h]
        \\centering
        \\begin{{tabular}}{{|l|c|c|c|}}
        \\hline
        Metric & Initial & Final & Improvement \\\\
        \\hline
        PSNR (dB) & {self.metrics['psnr'][0]:.2f} & {self.metrics['psnr'][final_idx]:.2f} & 
            {self.metrics['psnr'][final_idx] - self.metrics['psnr'][0]:.2f} \\\\
        Memory (MB) & {self.metrics['memory_mb'][0]:.1f} & {self.metrics['memory_mb'][final_idx]:.1f} & 
            {(1 - self.metrics['memory_mb'][final_idx]/self.metrics['memory_mb'][0])*100:.1f}\\% \\\\
        Patches & {self.metrics['n_patches'][0]} & {self.metrics['n_patches'][final_idx]} & 
            {self.metrics['n_patches'][final_idx] - self.metrics['n_patches'][0]} \\\\
        \\hline
        \\end{{tabular}}
        \\caption{{Performance comparison of our uneven patch method}}
        \\end{{table}}
        """
        with open(f'{self.log_dir}/results_table.tex', 'w') as f:
            f.write(latex)

class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr + self.batch]

class MixedSampler:
    """
    Used for refilter (annealed mixing) after alphaMask update.
    Sample from a concatenated pool [old ; new] with a time-varying ratio.
    new_ratio linearly increases from start_ratio to 1.0 over anneal_iters.
    """
    def __init__(self, N_old, N_new, batch, start_iter, anneal_iters=1500, start_ratio=0.8, device="cpu"):
        self.N_old = int(N_old)
        self.N_new = int(N_new)
        self.N_tot = self.N_old + self.N_new
        self.batch = int(batch)
        self.start_iter = int(start_iter)
        self.anneal_iters = int(anneal_iters)
        self.start_ratio = float(start_ratio)
        self.cur_iter = int(start_iter)
        self.device = device

    def set_iter(self, it):
        self.cur_iter = int(it)

    def _cur_new_ratio(self):
        if self.anneal_iters <= 0:
            return 1.0
        t = max(0, self.cur_iter - self.start_iter)
        s = min(1.0, t / max(1, self.anneal_iters))
        return min(1.0, self.start_ratio + (1.0 - self.start_ratio) * s)

    def nextids(self):
        r = self._cur_new_ratio()
        b_new = int(math.ceil(self.batch * r))
        b_old = self.batch - b_new
        # sample without replacement within each subrange; wrap-around by random perm if needed
        idx_old = torch.randint(low=0, high=self.N_old, size=(b_old,), device=self.device)
        idx_new = torch.randint(low=0, high=self.N_new, size=(b_new,), device=self.device) + self.N_old
        return torch.cat([idx_old, idx_new], dim=0)

def _as_minmax(aabb: torch.Tensor):
    """
    Accepts aabb as shape (6,) [xmin,ymin,zmin, xmax,ymax,zmax] OR (2,3) [[min],[max]].
    Returns (aabb_min[3], aabb_max[3]) on current device.
    """
    aabb = aabb.detach()
    if aabb.numel() == 6 and aabb.dim() == 1:
        aabb_min = aabb[:3]
        aabb_max = aabb[3:6]
    elif aabb.shape == (2, 3):
        aabb_min, aabb_max = aabb[0], aabb[1]
    else:
        raise ValueError(f"Unexpected AABB shape {tuple(aabb.shape)}; expected (6,) or (2,3).")
    return aabb_min, aabb_max

def _infer_fov_wh(train_dataset, fallback_fov_deg=50.0):
    """
    Try to infer (FOVx, FOVy, W, H) from dataset. 
    Fallbacks: square pixels, FOVx=FOVy=fallback_fov_deg, W/H=800 if missing.
    """
    W = getattr(train_dataset, 'W', None)
    H = getattr(train_dataset, 'H', None)
    # per-frame list/tensor → take median
    def _scalar(x, default):
        if x is None:
            return default
        if isinstance(x, (list, tuple)):
            return float(np.median(x))
        if torch.is_tensor(x):
            return float(x.float().median().item())
        try:
            return float(x)
        except Exception:
            return default

    W = int(round(_scalar(W, 800)))
    H = int(round(_scalar(H, 800)))

    fx = getattr(train_dataset, 'focal_x', None)
    fy = getattr(train_dataset, 'focal_y', None)
    if fx is None and hasattr(train_dataset, 'focal'):
        fx = getattr(train_dataset, 'focal')
        fy = fx

    fx = _scalar(fx, None)
    fy = _scalar(fy, None)

    # if focal is known: FOV (rad) = 2*atan(size/(2*focal))
    if fx is not None and fx > 0:
        FOVx = 2.0 * math.atan(W / (2.0 * fx))
    else:
        FOVx = math.radians(fallback_fov_deg)

    if fy is not None and fy > 0:
        FOVy = 2.0 * math.atan(H / (2.0 * fy))
    else:
        FOVy = math.radians(fallback_fov_deg)

    return FOVx, FOVy, W, H

def _estimate_zmin(train_dataset, aabb_min, aabb_max, near_far=None):
    """
    Prefer near plane if provided; otherwise use 5% of AABB diagonal length as a conservative 'visible' near depth.
    """
    if near_far is not None and len(near_far) >= 1:
        zmin = float(near_far[0])
        if zmin > 0:
            return zmin
    diag = torch.norm(aabb_max - aabb_min).item()
    return max(1e-3, 0.05 * diag)  # very conservative fallback

def _round_pow2(n):
    """Round to nearest power of two."""
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p if (p - n) <= (n - (p >> 1)) else (p >> 1)

def pick_bootstrap_resolutions(train_dataset, aabb, near_far=None,
                               R_min=8, R_max=64, kappa=1.0,
                               boot_G=(2, 2, 2)):
    """
    Pixel footprint + Nyquist: choose per-patch VM res (boot_R) so that voxel spacing ~= pixel world width at z_min.
        - aabb: (6,) or (2,3) tensor on any device
        - near_far: (near, far) if available
        - R_min/R_max: clamp range (power of 2 recommended)
        - kappa: safety factor on footprint (1.0~2.0 typical; bigger → finer VM grid)
        - boot_G: initial patch-grid resolution (number of patches per axis)

    Returns: (boot_G, boot_R) as tuples of ints
    """
    aabb_min, aabb_max = _as_minmax(aabb)
    L = (aabb_max - aabb_min)  # [Lx, Ly, Lz]
    Lx, Ly, Lz = [float(v) for v in L.tolist()]

    FOVx, FOVy, W, H = _infer_fov_wh(train_dataset)  # radians + ints
    zmin = _estimate_zmin(train_dataset, aabb_min, aabb_max, near_far)

    # pixel world footprint at zmin (pinhole): Δx(z)=2*z*tan(FOV/2)/W
    dx_min = 2.0 * zmin * math.tan(FOVx * 0.5) / max(1, W)
    dy_min = 2.0 * zmin * math.tan(FOVy * 0.5) / max(1, H)
    h0 = kappa * min(dx_min, dy_min)  # target voxel spacing (per-axis)

    # prevent degenerate (e.g., extremely small FOV / small z)
    h0 = max(h0, 1e-6)

    # per-patch VM res per axis (rounded to pow2, then clamped)
    Rx = max(R_min, min(R_max, _round_pow2(math.ceil(Lx / h0))))
    Ry = max(R_min, min(R_max, _round_pow2(math.ceil(Ly / h0))))
    Rz = max(R_min, min(R_max, _round_pow2(math.ceil(Lz / h0))))

    boot_R = (int(Rx), int(Ry), int(Rz))
    return tuple(boot_G), boot_R

def adjust_batch_size(reso_cur, args):
    # rays_chunk is selected based on the given batch size.
    batch_default = args.batch_size

    if max(reso_cur) >= 256:
        args.batch_size = min(1024, batch_default)
    elif max(reso_cur) >= 128:
        args.batch_size = min(2048, batch_default)
    else:
        args.batch_size = min(4096, batch_default)

    print(f"[INFO] Adjusted batch size: {args.batch_size}")

def build_optimizer_with_scale(tensorf, args, lr_scale):
    grad_vars = tensorf.get_optparam_groups(args.lr_init * lr_scale, args.lr_basis * lr_scale)
    return torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

def make_rank_policy(init_sigma, init_app, *, ref_res=64, gamma=0.6, cmin=4, cmax=64, round_to=4):
    """
    依據初始 per-axis ranks（可異向）、參考解析度 ref_res 與縮放指數 gamma
    回傳一個 policy(reso_hint) -> (sigma_new, app_new)。

    - 維持各軸比例（沿用 init_* 的三軸比例）
    - 以 r_min/res_ref 的 gamma 次方作溫和放大
    - 夾在 [cmin, cmax]，並以 round_to 對齊
    """
    init_sigma = np.array(list(map(int, init_sigma)), dtype=np.float64)  
    init_app   = np.array(list(map(int, init_app)),   dtype=np.float64)  

    def _round_and_clip(v):
        v = np.maximum(v, cmin)
        v = np.minimum(v, cmax)
        if round_to and round_to > 1:
            v = (np.round(v / round_to) * round_to)
        return v.astype(int).tolist()

    def policy(reso_hint):
        # reso_hint 接 (Rx, Ry, Rz)；用 r_min 做尺度（保守、穩定）
        r = np.array(list(map(float, reso_hint)))
        rmin = float(np.min(r))
        scale = (rmin / float(ref_res)) ** float(gamma)
        sigma_new = _round_and_clip(np.ceil(init_sigma * scale))
        app_new   = _round_and_clip(np.ceil(init_app   * scale))
        return sigma_new, app_new

    return policy

def current_reso_hint(tensorf, *, mode="max"):
    """
    回傳 (Rx,Ry,Rz) 作為 rank_policy 的輸入：
    - 若有 patch_map：取所有 patch 的 res 做 per-axis max/min
    - 否則退回 tensorf.gridSize 或 (8,8,8)
    """
    try:
        if getattr(tensorf, "patch_map", None):
            res_list = [p.get("res", (1,1,1)) for p in tensorf.patch_map.values()]
            if not res_list:
                raise ValueError
            arr = np.array(res_list, dtype=int)
            vec = arr.max(axis=0) if mode == "max" else arr.min(axis=0)
            return tuple(map(int, vec.tolist()))
    except Exception:
        pass

    if hasattr(tensorf, "gridSize"):
        return tuple(int(x) for x in tensorf.gridSize)
    return (8, 8, 8)

def snapshot_model(tensorf, device='cuda'):
    torch_state = {
        k: t.detach().cpu().clone()
        for k, t in tensorf.state_dict().items()
        if torch.is_tensor(t)
    }
    patch_state = {
        k: {
            'res': list(p['res']),
            'density_plane': [x.detach().cpu().clone() for x in p['density_plane']],
            'density_line':  [x.detach().cpu().clone() for x in p['density_line']],
            'app_plane':     [x.detach().cpu().clone() for x in p['app_plane']],
            'app_line':      [x.detach().cpu().clone() for x in p['app_line']],
        }
        for k, p in tensorf.patch_map.items()
    }
    meta = {
        'patch_grid_reso': tensorf.patch_grid_reso,
        'aabb': tensorf.aabb.detach().cpu().clone(),
    }
    return {'torch': torch_state, 'patch': patch_state, 'meta': meta}

def restore_model(tensorf, snap, device):
    """
    snap: torch.save 出來的 dict，至少包含 snap['torch'] 與其他訓練中需要的 meta。
    """
    st = snap.get('torch', None)
    if st is None:
        raise RuntimeError("ckpt missing key 'torch'")

    tensorf.to(device)
    tensorf.load_state_dict(st, strict=False)

    if 'global_step' in snap:
        tensorf.global_step = int(snap['global_step'])
    return tensorf

def get_val_dataset(args, train_dataset, test_dataset):
    """Prefer test set if built; otherwise fall back to train for quick-val."""
    return test_dataset if (test_dataset is not None) else train_dataset

@torch.no_grad()
def quick_val_psnr(test_dataset, tensorf, renderer, device, n_views=2, rays_per_view=2048):
    total_mse, total_cnt = 0.0, 0
    step = max(len(test_dataset.all_rgbs) // max(1, n_views), 1)
    for i in range(0, len(test_dataset.all_rgbs), step):
        rays = test_dataset.all_rays[i].to(device)[:rays_per_view]
        gt   = test_dataset.all_rgbs[i].view(-1,3).to(device)[:rays_per_view]
        if rays.numel() == 0: continue
        rgb, *_ = renderer(rays, is_train=False)
        mse = F.mse_loss(rgb, gt, reduction='mean').item()
        total_mse += mse * rays.shape[0]
        total_cnt += rays.shape[0]
        if total_cnt >= n_views * rays_per_view: break
    if total_cnt == 0: 
        return None
    mean_mse = total_mse / total_cnt
    return -10.0 * math.log10(max(1e-10, mean_mse))

@torch.no_grad()
def quick_val_psnr_safe(dataset, render, device,
                        target_views=3, target_rpv=4096,
                        min_views=1, min_rpv=1024,
                        chunk=1024):
    """
    Dataset-agnostic quick PSNR for stacked or flat loaders (NSVF/Blender/LLFF/Tanks).
    - Normalize all_rays/all_rgbs to [V, N, C] (or add a fake V=1 if flat).
    - Render in chunks (do NOT pass 'chunk' into render()).
    - Auto downscale on OOM: reduce rays_per_view first, then views.
    - Compatible with render outputs: tuple/list (rgb first) or dict (rgb_marched/rgb).
    """
    torch.cuda.empty_cache()

    def _to_VN(x, last_dim=3):
        # Accepts shapes: [N, C], [V, N, C], [V, H, W, C] -> returns [V, N, C]
        if x.dim() == 2:         # [N, C]
            return x.unsqueeze(0)
        if x.dim() == 3:         # [V, N, C]
            return x
        if x.dim() == 4:         # [V, H, W, C]
            V, H, W, C = x.shape
            return x.view(V, H * W, C)
        raise RuntimeError(f"Unsupported tensor dim: {x.shape}")

    rays_all = dataset.all_rays  # possible shapes: [N,6], [V,N,6], [V,H,W,6]
    rgbs_all = dataset.all_rgbs  # possible shapes: [N,3], [V,N,3], [V,H,W,3]

    # 一些載入器會把 rays 存成 [V,N,6]，rgb 存成 [V,H,W,3]；先攤平到 [V,N,C]
    if rays_all.dim() == 4:  # [V,H,W,6]
        V, H, W, C = rays_all.shape
        raysV = rays_all.view(V, H * W, C)
    elif rays_all.dim() == 3:  # [V,N,6]
        raysV = rays_all
    elif rays_all.dim() == 2:  # [N,6]
        raysV = rays_all.unsqueeze(0)
    else:
        raise RuntimeError(f"Unsupported rays shape: {rays_all.shape}")

    if rgbs_all.dim() == 4:  # [V,H,W,3]
        Vg, H, W, C = rgbs_all.shape
        rgbsV = rgbs_all.view(Vg, H * W, C)
    elif rgbs_all.dim() == 3 and rgbs_all.shape[-1] == 3:
        # could be [V,N,3] or [N,3]
        if rgbs_all.shape[0] == raysV.shape[0]:
            rgbsV = rgbs_all
        else:
            rgbsV = rgbs_all.unsqueeze(0)
    elif rgbs_all.dim() == 2:  # [N,3]
        rgbsV = rgbs_all.unsqueeze(0)
    else:
        raise RuntimeError(f"Unsupported rgbs shape: {rgbs_all.shape}")

    V = int(raysV.shape[0])
    N = int(raysV.shape[1])
    nv = min(int(target_views), V)
    idx_views = torch.linspace(0, V - 1, steps=nv).round().long().tolist()

    rpv = int(target_rpv)
    chunk = int(chunk)

    def _pred_rgb(rr):
        out = render(rr, is_train=False)  # do NOT pass 'chunk' into render()
        if isinstance(out, dict):
            return out.get("rgb_marched", out.get("rgb", None))
        if isinstance(out, (tuple, list)):  # assume first is rgb
            return out[0]
        return out

    while True:
        try:
            psnrs = []
            for v in idx_views:
                if N >= rpv:
                    idx = torch.randint(0, N, (rpv,))
                else:
                    idx = torch.randint(0, N, (N,))

                rays = raysV[v, idx].to(device, non_blocking=True)
                gt   = rgbsV[v, idx].to(device, non_blocking=True)

                mse_acc, cnt = 0.0, 0
                for s in range(0, rays.shape[0], chunk):
                    rr = rays[s:s+chunk]
                    gg = gt[s:s+chunk]
                    pred = _pred_rgb(rr)
                    if pred is None:
                        raise RuntimeError("render() output neither dict nor tuple with RGB.")
                    mse_acc += torch.mean((pred - gg) ** 2).item() * rr.shape[0]
                    cnt     += rr.shape[0]

                psnrs.append(-10.0 * math.log10(max(mse_acc / max(cnt, 1), 1e-8)))

            return float(sum(psnrs) / max(len(psnrs), 1))

        except RuntimeError as e:
            msg = str(e).lower()
            # OOM：先降 rpv，再降 views
            if ("out of memory" in msg) or ("cuda" in msg and "memory" in msg):
                if rpv > min_rpv:
                    rpv = max(min_rpv, rpv // 2)
                elif nv > min_views:
                    nv = max(min_views, nv - 1)
                    idx_views = torch.linspace(0, V - 1, steps=nv).round().long().tolist()
                else:
                    raise
                try: torch.cuda.empty_cache()
                except: pass
                continue
            else:
                raise

@torch.no_grad()
def export_mesh_patch(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt.get("state_dict", {})
    print('has patch_map:', 'patch_map' in sd)

    Model = eval(args.model_name)
    kwargs = dict(ckpt.get("kwargs", {}))
    kwargs["device"] = device
    tensorf = Model(**kwargs).to(device)

    tensorf.load_state_dict(sd, strict=False)

    if ('alphaMask.shape' in ckpt) and ('alphaMask.mask' in ckpt):
        from models.tensorBase import AlphaGridMask
        shape = tuple(ckpt['alphaMask.shape'])
        bits  = ckpt['alphaMask.mask']
        aabb  = ckpt.get('alphaMask.aabb', tensorf.aabb).to(device)
        flat  = np.unpackbits(bits)[:np.prod(shape)]
        mask  = torch.from_numpy(flat.reshape(shape)).bool().to(device)
        tensorf.alphaMask = AlphaGridMask(device, aabb, mask)

    if hasattr(tensorf, "assert_zero_origin_and_contiguous"):
        try: tensorf.assert_zero_origin_and_contiguous()
        except Exception as e: print(f"[WARN] assert_zero_origin_and_contiguous failed: {e}")

    alpha_volume = tensorf.get_dense_alpha_from_patch().cpu()
    convert_sdf_samples_to_ply(
        alpha_volume,
        save_path=f'{args.ckpt[:-3]}_patch.ply',
        bbox=tensorf.aabb.cpu(),
        level=0.005
    )
    print(f"[export_mesh_patch] Saved mesh to {args.ckpt[:-3]}_patch.ply")

def reconstruction(args):
    def _git_sha():
        try:
            return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        except Exception:
            return None

    def dump_run_config(logfolder, args):
        cfg = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "git": _git_sha(),
            "argv": " ".join(shlex.quote(x) for x in sys.argv),
            "args": vars(args),  
        }
        with open(os.path.join(logfolder, "args.json"), "w") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)
        with open(os.path.join(logfolder, "args_cli.txt"), "w") as f:
            f.write(cfg["argv"] + "\n")
       
        cfg_path = getattr(args, "config", None)
        if cfg_path:
            try:
                shutil.copy(cfg_path, os.path.join(logfolder, "config_used.txt"))
            except Exception as e:
                with open(os.path.join(logfolder, "config_used.copy_error.txt"), "w") as ferr:
                    ferr.write(repr(e) + "\n")

    def log_event(logfolder, tag, iteration, **kw):
        line = f"[{tag}] iter={iteration} " + " ".join([f"{k}={v}" for k,v in kw.items()])
        with open(os.path.join(logfolder, "events_log.txt"), "a") as f:
            f.write(line + "\n")
        print(line)

    def log_hparams_boundary(args, summary_writer, logfolder, iteration=0):
        """
        Log boundary/residual/seam/PUF-related hyperparameters to TensorBoard, events_log, and a JSON file.
        """
        def _s(tag, val):
            try:
                summary_writer.add_scalar(tag, float(val), int(iteration))
            except Exception:
                pass

        _s('hparams/puf_alpha_boundary',        getattr(args, 'puf_alpha_boundary', 0.3))
        _s('hparams/boundary_smooth_strength',  getattr(args, 'boundary_smooth_strength', 1.0))
        _s('hparams/enable_child_residual',     1.0 if getattr(args, 'enable_child_residual', True) else 0.0)
        _s('hparams/residual_gate_tau',         getattr(args, 'residual_gate_tau', 0.10))
        _s('hparams/enable_seam_blend',         1.0 if getattr(args, 'enable_seam_blend', True) else 0.0)
        _s('hparams/seam_band_width',           getattr(args, 'seam_band_width', 0.05))

        try:
            summary_writer.add_text('hparams/boundary_cost_mode',
                                    str(getattr(args, 'boundary_cost_mode', 'dof')),
                                    global_step=int(iteration))
        except Exception:
            pass

        # mirror to events_log.txt
        try:
            log_event(logfolder, "hparams", int(iteration),
                    puf_alpha_boundary=getattr(args, 'puf_alpha_boundary', 0.3),
                    boundary_cost_mode=getattr(args, 'boundary_cost_mode', 'dof'),
                    boundary_smooth_strength=getattr(args, 'boundary_smooth_strength', 1.0),
                    enable_child_residual=int(bool(getattr(args, 'enable_child_residual', True))),
                    residual_gate_tau=getattr(args, 'residual_gate_tau', 0.10),
                    enable_seam_blend=int(bool(getattr(args, 'enable_seam_blend', True))),
                    seam_band_width=getattr(args, 'seam_band_width', 0.05))
        except Exception:
            pass

        # persist to JSON in logfolder
        try:
            rec = dict(
                puf_alpha_boundary=float(getattr(args, 'puf_alpha_boundary', 0.3)),
                boundary_cost_mode=str(getattr(args, 'boundary_cost_mode', 'dof')),
                boundary_smooth_strength=float(getattr(args, 'boundary_smooth_strength', 1.0)),
                enable_child_residual=bool(getattr(args, 'enable_child_residual', True)),
                residual_gate_tau=float(getattr(args, 'residual_gate_tau', 0.10)),
                enable_seam_blend=bool(getattr(args, 'enable_seam_blend', True)),
                seam_band_width=float(getattr(args, 'seam_band_width', 0.05)),
            )
            with open(os.path.join(logfolder, "hparams_boundary.json"), "w") as f:
                json.dump(rec, f, indent=2, sort_keys=True)
        except Exception as e:
            print(f"[WARN] failed to write hparams_boundary.json: {repr(e)}")

    def _tensor_nbytes(t):
        try:
            return t.numel() * t.element_size()
        except Exception:
            return 0

    def _named_param_bytes(model):
        total = 0
        for n, p in model.named_parameters(recurse=True):
            if p is None:
                continue
            total += _tensor_nbytes(p)
        return total

    def _named_buffer_bytes(model):
        total = 0
        for n, b in model.named_buffers(recurse=True):
            if b is None:
                continue
            total += _tensor_nbytes(b)
        return total

    def _grad_bytes(model):
        total = 0
        for p in model.parameters():
            if p.grad is not None:
                total += _tensor_nbytes(p.grad)
        return total

    def _optimizer_state_bytes(optim):
        total = 0
        if optim is None:
            return 0
        for group in optim.param_groups:
            for p in group.get("params", []):
                st = optim.state.get(p, {})
                for v in st.values():
                    if torch.is_tensor(v):
                        total += _tensor_nbytes(v)
                    elif isinstance(v, (list, tuple)):
                        for x in v:
                            if torch.is_tensor(x):
                                total += _tensor_nbytes(x)
        return total

    def memory_breakdown_bytes(model, optim=None):
        """
        Returns a dict of memory components in bytes.
        Does NOT include activations; CUDA stats are sampled separately.
        """
        return dict(
            model_param=_named_param_bytes(model),
            model_buffers=_named_buffer_bytes(model),
            grads=_grad_bytes(model),
            optim_states=_optimizer_state_bytes(optim),
        )

    def cuda_memory_stats_bytes():
        """
        Returns allocated/reserved CUDA memory in bytes (per PyTorch caching allocator).
        """
        try:
            alloc = torch.cuda.memory_allocated()
            reserv = torch.cuda.memory_reserved()
        except Exception:
            alloc = reserv = 0
        return dict(cuda_allocated=alloc, cuda_reserved=reserv)

    def log_memory_snapshot(writer, logfolder, step, model, optim=None, tag_prefix="mem",
                            *, rotate_json=False, write_latest=True, tb_enable=True, events_enable=True):
        """
        Minimal-overhead memory logger.
        - By default: writes only a single 'mem_latest.json' (overwrite) + TB/events if enabled.
        - If rotate_json=True, it will also write 'memory_step_XXXXXX.json' (heavier I/O).
        """
        snap = memory_breakdown_bytes(model, optim)
        cuda = cuda_memory_stats_bytes()
        allm = {**snap, **cuda}

        if tb_enable and writer is not None:
            for k, v in allm.items():
                try:
                    writer.add_scalar(f"{tag_prefix}/{k}_MB", float(v) / (1024**2), step)
                except Exception:
                    pass

        if events_enable:
            try:
                rec = {k: round(float(v) / (1024**2), 3) for k, v in allm.items()}
                log_event(logfolder, "memory", int(step), **rec)
            except Exception:
                pass

        if write_latest:
            try:
                path = os.path.join(logfolder, "mem_latest.json")
                with open(path, "w") as f:
                    json.dump({k: int(v) for k, v in allm.items()}, f, indent=2, sort_keys=True)
            except Exception:
                pass

        if rotate_json:  # (expensive way)
            try:
                outp = os.path.join(logfolder, f"memory_step_{int(step):07d}.json")
                with open(outp, "w") as f:
                    json.dump({k: int(v) for k, v in allm.items()}, f, indent=2, sort_keys=True)
            except Exception:
                pass

    def reset_cuda_peak():
        """
        Reset CUDA peak memory stats so that subsequent calls to
        torch.cuda.max_memory_allocated()/reserved() report peaks
        since this reset point.
        """
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            print(f"[WARN] reset_peak_memory_stats failed: {repr(e)}")

    def update_peak_vram_txt(logfolder, best_holder):
        """
        Keep a single-line 'peak_vram.txt' with the highest observed CUDA allocated peak (MB).
        best_holder: a 1-element list used as mutable holder, e.g., [0].
        """
        if not torch.cuda.is_available():
            return
        try:
            cur_peak = torch.cuda.max_memory_allocated()
            if cur_peak > int(best_holder[0]):
                best_holder[0] = int(cur_peak)
                mb = best_holder[0] // (1024 * 1024)
                with open(os.path.join(logfolder, "peak_vram.txt"), "w") as f:
                    f.write(str(mb))
                # Optional: mirror to events for grep-ability
                try:
                    log_event(logfolder, "mem", int(locals().get("iteration", -1)), peak_MB=int(mb))
                except Exception:
                    pass
        except Exception:
            pass
        
    def log_cuda_peak(summary_writer, logfolder, step, tag_prefix="mem"):
        """
        Log CUDA peak memory (since last reset) to TensorBoard and events log.
        Records both max_memory_allocated and max_memory_reserved in MB.
        """
        try:
            if torch.cuda.is_available():
                peak_alloc = torch.cuda.max_memory_allocated()
                peak_resv  = torch.cuda.max_memory_reserved()
            else:
                peak_alloc = 0
                peak_resv  = 0
        except Exception as e:
            print(f"[WARN] read peak memory failed: {repr(e)}")
            peak_alloc = 0
            peak_resv  = 0

        # TensorBoard scalars (MB)
        try:
            summary_writer.add_scalar(f"{tag_prefix}/cuda_peak_allocated_MB",
                                    float(peak_alloc) / (1024.0**2), int(step))
            summary_writer.add_scalar(f"{tag_prefix}/cuda_peak_reserved_MB",
                                    float(peak_resv) / (1024.0**2), int(step))
        except Exception:
            pass

        # Events log (single-line JSON-like entry)
        try:
            alloc_mb = round(float(peak_alloc) / (1024.0**2), 3)
            resv_mb  = round(float(peak_resv)  / (1024.0**2), 3)
            log_event(logfolder, "cuda_peak", int(step),
                      cuda_peak_allocated_MB=alloc_mb,
                      cuda_peak_reserved_MB=resv_mb)
        except Exception:
            pass
    
    def log_mem_breakdown(tensorf, iteration, logfolder):
        if not bool(getattr(args, "enable_mem_breakdown_jsonl", False)):
            return

        bytes_vox = 0
        bytes_basis = 0

        for p in tensorf.patch_map.values():
            # ParameterList cannot be added directly
            planes = list(chain(
                list(p['density_plane']),
                list(p['app_plane']),
                list(p['density_line']),
                list(p['app_line']),
            ))
            for t in planes:
                bytes_vox += t.numel() * t.element_size()

        basis_seen = set()
        for p in tensorf.patch_map.values():
            lin = p.get('basis_mat', None)
            if lin is None:
                continue
            lid = id(lin)
            if lid in basis_seen:
                continue
            basis_seen.add(lid)
            bytes_basis += lin.weight.numel() * lin.weight.element_size()
            if lin.bias is not None:
                bytes_basis += lin.bias.numel() * lin.bias.element_size()

        bytes_mlp = 0
        for _, m in getattr(tensorf, 'renderModule', torch.nn.Module()).named_parameters(recurse=True):
            bytes_mlp += m.numel() * m.element_size()

        try:
            cuda_alloc = torch.cuda.memory_allocated()
            cuda_resvd = torch.cuda.memory_reserved()
        except Exception:
            cuda_alloc = cuda_resvd = -1

        rec = dict(
            iter=iteration,
            n_patches=len(tensorf.patch_map),
            n_basis=len(basis_seen),
            voxels=int(tensorf.get_total_voxels()),
            vox_MB=bytes_vox/1024**2,
            basis_MB=bytes_basis/1024**2,
            mlp_MB=bytes_mlp/1024**2,
            cuda_alloc_MB=cuda_alloc/1024**2 if cuda_alloc>=0 else None,
            cuda_reserved_MB=cuda_resvd/1024**2 if cuda_resvd>=0 else None
        )
        with open(os.path.join(logfolder, "mem_breakdown.jsonl"), "a") as f:
            f.write(json.dumps(rec)+"\n")

    def field_kd_loss(tensorf, step, device, *, max_pts=4096, sigma_w=1.0, app_w=1.0, max_buffers=10): 
        """
        Compute light-weight field knowledge distillation (KD) loss on post-split children.
        Pulls (xyz, sigma_tgt, app_tgt) from tensorf._kd_buffers (CPU), evaluates current model,
        returns a single scalar loss. Expired buffers are auto-removed.

        This uses patchwise fast eval: map world xyz -> (norm coords, patch coords),
        then call compute_*_patchwise_fast once per batch.
        """
        if not hasattr(tensorf, "_kd_buffers") or not tensorf._kd_buffers:
            return None

        # drop expired AND limit total buffers
        alive = []
        for rec in tensorf._kd_buffers:
            if int(step) <= int(rec.get("expires_at", -1)):
                alive.append(rec)
        
        # Keep only the most recent buffers if exceeding limit
        if len(alive) > max_buffers:
            alive = sorted(alive, key=lambda x: x.get("expires_at", 0), reverse=True)[:max_buffers]
        
        tensorf._kd_buffers = alive
        if not alive:
            return None

        # gather up to max_pts across buffers
        picks_xyz, picks_sigma, picks_app = [], [], []
        remain = int(max_pts)
        for rec in alive:
            if remain <= 0:
                break
            xyz_cpu  = rec["world_xyz"]
            sig_cpu  = rec["sigma_tgt"]
            app_cpu  = rec["app_tgt"]
            n = int(min(remain, xyz_cpu.shape[0]))
            if n <= 0:
                continue
            idx = torch.randint(0, xyz_cpu.shape[0], (n,))
            picks_xyz.append(xyz_cpu[idx])
            picks_sigma.append(sig_cpu[idx])
            picks_app.append(app_cpu[idx])
            remain -= n

        if not picks_xyz:
            return None

        xyz_w   = torch.cat(picks_xyz, dim=0).to(device)    # [M,3]
        sigma_t = torch.cat(picks_sigma, dim=0).to(device)  # [M,1]
        app_t   = torch.cat(picks_app, dim=0).to(device)    # [M,Capp]

        # map to normalized coords + patch indices for current model
        pts_norm = tensorf.normalize_coord(xyz_w)                       
        coords, exists = tensorf._map_coords_to_patch(xyz_w)
        # filter to valid (should be almost all, given ones sampled from valid children)
        if exists.dim() > 1:  # if needed
            exists = exists.reshape(-1)
        if exists.sum() == 0:
            return None
        pts_norm = pts_norm[exists]
        coords   = coords[exists]
        sigma_t  = sigma_t[exists]
        app_t    = app_t[exists]

        # current predictions (student)
        sig_feat = tensorf.compute_density_patchwise_fast(pts_norm, coords)  # [m, C_sigma]
        sigma_s  = tensorf.feature2density(sig_feat).view(-1, 1)             # [m,1]
        app_s    = tensorf.compute_app_patchwise_fast(pts_norm, coords)      # [m, C_app]

        # MSE in field space
        ls = torch.mean((sigma_s - sigma_t)**2) if sigma_w > 0 else 0.0
        la = torch.mean((app_s   - app_t  )**2) if app_w   > 0 else 0.0

        total = float(sigma_w) * ls + float(app_w) * la
        return total

    @torch.no_grad()
    def _adjacent_patch_pairs(tensorf):
        """
        建出 (patch_i, patch_j, axis, normal) 列表。若 model 已有鄰接表，優先用它；
        否則從 grid 形狀推。axis in {0:x,1:y,2:z}, normal=+1/-1 表示 i->j 的法向。
        """
        pairs = []
        if hasattr(tensorf, 'patch_grid_reso'):
            gx, gy, gz = map(int, tensorf.patch_grid_reso)
            # 假設 patch id = i + j*gx + k*gx*gy
            def pid(i,j,k): return i + j*gx + k*gx*gy
            for k in range(gz):
                for j in range(gy):
                    for i in range(gx):
                        a = pid(i,j,k)
                        if i+1<gx: pairs.append((a, pid(i+1,j,k), 0, +1))
                        if j+1<gy: pairs.append((a, pid(i,j+1,k), 1, +1))
                        if k+1<gz: pairs.append((a, pid(i,j,k+1), 2, +1))
        elif hasattr(tensorf, 'get_adjacent_patches'):
            pairs = tensorf.get_adjacent_patches()  # 自備函式就用它
        return pairs

    def seam_consistency_loss(tensorf, device, samples_per_face=512, eps=1e-3,
                              sigma_w=1.0, app_w=1.0):
        """
        在相鄰 patch 的共享面上取樣，沿法向兩側 +/-eps 評估 field，做 MSE。
        回傳 torch scalar；若偵測不到鄰接，回傳 None。
        """
        pairs = _adjacent_patch_pairs(tensorf)
        if not pairs:
            return None

        # 取相鄰面上均勻樣本 (局部 uv -> world)，依你模型具體 API 可能要從 AABB 推；這裡採簡化：
        # 用 normalize coord 空間 [-1,1] 的面上隨機取樣，再 map 回世界座標。
        # 若你有 per-patch AABB：請直接用 AABB 生成面點。
        losses = []
        for (pa, pb, axis, normal) in pairs:
            # 在該面建立樣本：三維[-1,1]，其中 axis 對應座標固定在 +/-1
            u = torch.rand(samples_per_face, device=device)*2-1
            v = torch.rand(samples_per_face, device=device)*2-1
            xyz = torch.zeros(samples_per_face, 3, device=device)
            free = [0,1,2]; free.remove(axis)
            xyz[:, free[0]] = u
            xyz[:, free[1]] = v
            xyz[:, axis] = 1.0 * (1 if normal>0 else -1)  # 面在 +1 或 -1

            # 稍微往兩側偏移
            delta = torch.zeros_like(xyz); delta[:, axis] = eps * (1 if normal>0 else -1)
            xyz_a = xyz - delta
            xyz_b = xyz + delta

            # 映射到 patch／前向（用你現有的 patchwise fast 路徑）
            # 這裡用 normalize_coord + _map_coords_to_patch 來讓 student 走到對的 patch。
            # 注意：這段採 field 空間 supervising（跟 KD 一致）。
            with torch.enable_grad():
                pa_norm = tensorf.normalize_coord(xyz_a)
                pb_norm = tensorf.normalize_coord(xyz_b)
                # map 會回傳 (patch_index, valid_mask)；若你 API 不同，請依實作改名
                pa_idx, _ = tensorf._map_coords_to_patch(xyz_a)
                pb_idx, _ = tensorf._map_coords_to_patch(xyz_b)

                sig_a_feat = tensorf.compute_density_patchwise_fast(pa_norm, pa_idx)
                sig_b_feat = tensorf.compute_density_patchwise_fast(pb_norm, pb_idx)
                sigma_a = tensorf.feature2density(sig_a_feat)
                sigma_b = tensorf.feature2density(sig_b_feat)

                app_a = tensorf.compute_app_patchwise_fast(pa_norm, pa_idx)
                app_b = tensorf.compute_app_patchwise_fast(pb_norm, pb_idx)

            l_sigma = torch.mean((sigma_a - sigma_b)**2)
            l_app   = torch.mean((app_a   - app_b)**2)
            losses.append(sigma_w*l_sigma + app_w*l_app)

        if not losses:
            return None
        return torch.mean(torch.stack(losses))


    def _param_set(tensorf):
        """Return a set of parameter object ids for quick diff."""
        return set(id(p) for p in tensorf.parameters())

    def _collect_new_params(tensorf, prev_idset):
        """Collect newly created parameters (by comparing object id)."""
        cur = list(tensorf.parameters())
        news = [p for p in cur if id(p) not in prev_idset]
        return news

    def _split_param_groups_by_params(orig_groups, special_params_set, base_scale=1.0, special_mult=1.0):
        """
        Split existing param groups into two buckets:
        - child group: contains any param in special_params_set (lr scaled by base_scale*special_mult)
        - others: the rest (lr scaled by base_scale)
        Return a list of new groups.
        """
        new_groups = []
        for g in orig_groups:
            common_kwargs = {k:v for k,v in g.items() if k != 'params' and k != 'lr'}

            child_params = [p for p in g['params'] if id(p) in special_params_set]
            other_params = [p for p in g['params'] if id(p) not in special_params_set]

            if child_params:
                new_groups.append({
                    'params': child_params,
                    'lr': g['lr'] * float(base_scale) * float(special_mult),
                    **common_kwargs
                })
            if other_params:
                new_groups.append({
                    'params': other_params,
                    'lr': g['lr'] * float(base_scale),
                    **common_kwargs
                })
        return new_groups

    def _build_ema_from(tensorf):
        teacher = copy.deepcopy(tensorf).eval()
        for p in teacher.parameters(): p.requires_grad_(False)
        return teacher

    def _ema_update(teacher, student, decay=0.999):
        with torch.no_grad():
            t_params = dict(teacher.named_parameters())
            for name, s in student.named_parameters():
                if name in t_params:
                    t = t_params[name]
                    t.data.mul_(decay).add_(s.data, alpha=(1.0 - decay))

    def rebuild_optimizer_with_child_boost(tensorf, args, base_scale, child_params, child_mult):
        """
        Rebuild Adam optimizer so that:
        - params in `child_params` get (lr * base_scale * child_mult)
        - others get (lr * base_scale)
        We DO NOT attempt to carry old optimizer states for simplicity & stability
        across structural changes. (Your code already rebuilds on structure events.)
        """
        special = set(id(p) for p in child_params)
        # tensorf.get_optparam_groups returns canonical groups w/ .lr_init & .lr_basis applied
        base_groups = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
        new_groups = _split_param_groups_by_params(base_groups, special, base_scale=base_scale, special_mult=child_mult)
        new_opt = torch.optim.Adam(new_groups, betas=(0.9, 0.99))
        return new_opt

    def micro_upsample_children(tensorf, child_keys, max_res=64, scale=2):
        """
        Per-patch micro-upsample for selected children. Returns actually-promoted keys.
        It tries a signature (scale,max_res); if model expects target_res instead, fallback.
        """
        promoted = []
        for key in child_keys:
            if key not in tensorf.patch_map:
                continue
            cur_res = tuple(int(x) for x in tensorf.patch_map[key]['res'])
            tgt_res = tuple(min(int(x*scale), int(max_res)) for x in cur_res)
            if tgt_res == cur_res:
                continue
            try:
                # common signature: (key, scale=?, max_res=?)
                tensorf.upsample_VM(key, scale=scale, max_res=max_res)
            except TypeError:
                # fallback: (key, target_res=?)
                tensorf.upsample_VM(key, target_res=tgt_res)
            promoted.append(key)
        return promoted

    exp_logger = ExperimentLogger(logfolder)

    # TensoRF uses "divisor" for downsample; values < 1.0 upsample and will blow memory!
    if float(getattr(args, "downsample_train", 1.0)) < 1.0:
        print("[WARNING] downsample_train < 1.0 means UPSAMPLING; forcing to 1.0 to avoid OOM/Killed.")
        args.downsample_train = 1.0

    if float(getattr(args, "downsample_test", 1.0)) < 1.0:
        print("[WARNING] downsample_test < 1.0 means UPSAMPLING; forcing to 1.0 to avoid OOM/Killed.")
        args.downsample_test = 1.0

    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False)
    
    # not always build even if runs function validation / quick checking exp 
    need_test = bool(
        getattr(args, "render_test", 0) or
        getattr(args, "render_path", 0) or
        ((getattr(args, "N_vis", 0) > 0) and (getattr(args, "vis_every", 0) > 0))
    )
    test_dataset = None
    if need_test:  # default: build for complete model learning
        test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_test, is_stack=True)
    else:
        print("[loader] render_test=0 & no periodic vis -> skip loading test set entirely")
    
    white_bg = True if getattr(args, "white_bkgd", False) else train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    rgbs = train_dataset.all_rgbs
    print("[rgb] dtype:", rgbs.dtype, "min:", float(rgbs.min()), "max:", float(rgbs.max()))
    assert rgbs.dtype in (torch.float32, torch.float16), "RGB dtype should be float"
    assert float(rgbs.min()) >= -1e-3 and float(rgbs.max()) <= 1.0+1e-3, "RGB not in [0,1]"

    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    dump_run_config(logfolder, args)
    log_hparams_boundary(args, summary_writer, logfolder, iteration=0)

    now = datetime.now()
    with open(os.path.join(logfolder, 'time_stamp.txt'), 'a') as f:
        f.write(f'Begin: {now}\n')

    aabb = train_dataset.scene_bbox.to(device)
    if aabb.dim() == 2 and aabb.shape == (3, 2):
        aabb = aabb.t().contiguous()
    assert (aabb.numel() == 6 and aabb.dim() == 1) or (tuple(aabb.shape) == (2, 3)), \
        f"Unexpected scene_bbox shape {tuple(aabb.shape)}; expected (6,) or (2,3)"

    # # decide res by pixel footprint
    # boot_G, boot_R = pick_bootstrap_resolutions(
    #     train_dataset, aabb, near_far=train_dataset.near_far,
    #     R_min=getattr(args, "boot_R_min", 8),
    #     R_max=getattr(args, "boot_R_max", 64),    # 8~64 (tunable)
    #     kappa=getattr(args, "boot_kappa", 1.0),   # ~0.75 to safer; 1.5~2.0 to aggressive
    #     boot_G=getattr(args, "boot_G", (2,2,2)),
    # )
    boot_G = tuple(args.init_grid_res)
    boot_R = tuple(args.init_vm_res)

    try:
        path = os.path.join(logfolder, "args.json")
        obj = json.load(open(path))
        obj["derived"] = {"boot_G": list(boot_G), "boot_R": list(boot_R)}
        json.dump(obj, open(path, "w"), indent=2, sort_keys=True)
    except Exception:
        pass

    print(f"[boot] G={boot_G}, R={boot_R} | AABB size={tuple((aabb[1]-aabb[0]).tolist() if aabb.ndim==2 else (aabb[3:6]-aabb[:3]).tolist())}")
    reso_cur = list(boot_R)

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        sd = ckpt['state_dict']
        print('has patch_map:', 'patch_map' in sd)
        tensorf = eval(args.model_name)(**{**ckpt['kwargs'], 'device': device})
        tensorf.load(ckpt)
    else:  
        def _triplet_from_arg(x, default_scalar):
            """
            Accept arg that may be None / int / str / [..] and return [s, s, s].
            default_scalar will be used if x is None or unparsable.
            """
            if x is None:
                s = int(default_scalar)
                return [s, s, s]
            # list/tuple of numbers -> take max then broadcast
            if isinstance(x, (list, tuple)):
                vals = []
                for v in x:
                    if isinstance(v, (int, float)):
                        vals.append(int(v))
                    elif isinstance(v, str) and v.strip():
                        for p in v.replace(",", " ").split():
                            try: vals.append(int(p))
                            except: pass
                if vals:
                    s = int(max(vals))
                    return [s, s, s]
                else:
                    s = int(default_scalar)
                    return [s, s, s]
            # str "16" / "16,16,16" -> take max then broadcast
            if isinstance(x, str) and x.strip():
                parts = x.replace(",", " ").split()
                try:
                    nums = [int(p) for p in parts]
                    s = int(max(nums))
                except Exception:
                    s = int(default_scalar)
                return [s, s, s]
            # scalar number
            if isinstance(x, (int, float)):
                s = int(x)
                return [s, s, s]
            # fallback
            s = int(default_scalar)
            return [s, s, s]

        sigma_trip = _triplet_from_arg(getattr(args, "n_factors_sigma", None), 16)
        app_trip   = _triplet_from_arg(getattr(args, "n_factors_app",   None), 48)
        
        # default: TensorVMSplitPatch
        tensorf = eval(args.model_name)(
            aabb, list(boot_R), device,
            density_n_comp=sigma_trip,
            app_n_comp=app_trip,
            app_dim=args.data_dim_color,
            near_far=train_dataset.near_far,
            shadingMode=args.shadingMode,
            alphaMask_thres=args.alpha_mask_thre,
            density_shift=args.density_shift,
            distance_scale=args.distance_scale,
            pos_pe=args.pos_pe,
            view_pe=args.view_pe,
            fea_pe=args.fea_pe,
            featureC=args.featureC,
            step_ratio=args.step_ratio,
            fea2denseAct=args.fea2denseAct,
            rank_cap_sigma=args.rank_cap_sigma,
            rank_cap_app=args.rank_cap_app,
            rank_base_floor_sig=args.rank_base_floor_sig,
            rank_base_floor_app=args.rank_base_floor_app,
            global_basis_enable=bool(args.global_basis_enable),
            global_basis_k_sigma=args.global_basis_k_sigma,
            global_basis_k_app=args.global_basis_k_app,
            min_rank=args.min_rank,
            max_rank=args.max_rank,
            repair_enable=args.repair_enable,
            repair_tau=args.repair_tau,
            repair_adjacent_only=args.repair_adjacent_only,
            repair_grad_scale_sigma=args.repair_grad_scale_sigma,
            repair_grad_scale_app=args.repair_grad_scale_app,
            seam_lowrank_enable=args.seam_lowrank_enable,
            seam_lowrank_scope=args.seam_lowrank_scope,
            seam_rank_sigma=args.seam_rank_sigma,
            seam_rank_app=args.seam_rank_app,
        )
        
        if hasattr(tensorf, 'enable_seam_blend'):
            tensorf.enable_seam_blend = False
            print("[INFO] Seam blending DISABLED during training (will enable for eval)")

        # set residual/seam controls as attributes on the model (not part of TensorBase.__init__ signature)
        try:
            tensorf.enable_child_residual = bool(getattr(args, "enable_child_residual", True))
            tensorf.residual_gate_tau     = float(getattr(args, "residual_gate_tau", 0.10))
            tensorf.enable_seam_blend     = bool(getattr(args, "enable_seam_blend", True))
            tensorf.seam_band_width       = float(getattr(args, "seam_band_width", 0.05))
        except Exception as e:
            print(f"[WARN] set residual/seam flags failed: {e}")

        if getattr(args, "seam_lowrank_enable", False):
            with torch.no_grad():
                wired = tensorf.init_seam_lowrank(
                    rank_sigma=args.seam_rank_sigma,
                    rank_app=args.seam_rank_app,
                    scope=args.seam_lowrank_scope
                )
                if wired:
                    print(f"[seam-lr] initial seams wired: {wired}")

        if not isinstance(getattr(tensorf, "density_n_comp", None), (list, tuple)):
            tensorf.density_n_comp = sigma_trip
        if not isinstance(getattr(tensorf, "app_n_comp", None), (list, tuple)):
            tensorf.app_n_comp = app_trip

    
    def quick_miss_ratio(tensorf, n_samples: int = 8192):
        """
        Evenly sample within AABB, estimate number of samples not mapped to existing maps.
        """
        device = tensorf.aabb.device
        aabb = tensorf.aabb
        if aabb.numel() == 6:
            aabb = aabb.reshape(2, 3)
        a0, a1 = aabb[0], aabb[1]
        pts = a0 + (a1 - a0) * torch.rand(n_samples, 3, device=device)

        _, exists = tensorf._map_coords_to_patch(pts) 
        miss_ratio = float((~exists).float().mean().item())
        return miss_ratio

    def append_covlog(logfolder, iteration, phase, miss_before=None, miss_after=None,
                      added=None, n_patches=None, G=None, reso_cur=None):
        path = os.path.join(logfolder, "coverage_log.tsv")
        hdr  = "iter\tphase\tmiss_before\tmiss_after\tadded\tn_patches\tG\treso_cur\n"
        line = f"{iteration}\t{phase}\t{miss_before}\t{miss_after}\t{added}\t{n_patches}\t{G}\t{reso_cur}\n"
        if not os.path.exists(path):
            with open(path, "w") as f: f.write(hdr)
        with open(path, "a") as f: f.write(line)

    def heartbeat_guard_coverage(tensorf, *, target_miss=0.15, seed_cells=4, note="",
                                 iter_for_log=None, reso_for_log=None):
        if not getattr(tensorf, "patch_map", None):
            tensorf.ensure_default_patch()

        tensorf.patch_grid_reso = tuple(tensorf.infer_patch_grid_reso_from_keys(tensorf.patch_map))
        if hasattr(tensorf, "assert_zero_origin_and_contiguous"):
            try: tensorf.assert_zero_origin_and_contiguous()
            except Exception as e: print(f"[WARN] assert_zero_origin_and_contiguous failed: {e}")

        miss_b = quick_miss_ratio(tensorf)
        added = tensorf.ensure_min_coverage(target_miss=target_miss, seed_cells=seed_cells)
        if int(added or 0) > 0:
            try:
                tensorf.assert_zero_origin_and_contiguous()
            except Exception as e:
                print(f"[WARN] assert_zero_origin_and_contiguous failed (post-ensure): {e}")
            if hasattr(tensorf, "infer_patch_grid_reso_from_keys"):
                tensorf.patch_grid_reso = tuple(tensorf.infer_patch_grid_reso_from_keys(tensorf.patch_map))
            tensorf.current_patch_keys = sorted(list(tensorf.patch_map.keys()))
        miss_a = quick_miss_ratio(tensorf)

        append_covlog(logfolder, int(iter_for_log if iter_for_log is not None else -1),
                    f"heartbeat-{note}",
                    miss_before=f"{miss_b:.4f}", miss_after=f"{miss_a:.4f}",
                    added=int(added or 0), n_patches=len(tensorf.patch_map),
                    G=str(tuple(tensorf.patch_grid_reso)),
                    reso_cur=str(tuple(reso_for_log) if reso_for_log is not None else None))

        added_int = int(added or 0)
        if added_int > 0:
            print(f"[cover][{note}] +{added_int} cells; G={tensorf.patch_grid_reso}")
        return added_int

    def run_heartbeat(*, note="", iter_for_log=None, reso_for_log=None):
        return heartbeat_guard_coverage(
            tensorf,
            target_miss=getattr(args, "heartbeat_target_miss", 0.15),
            seed_cells=getattr(args, "heartbeat_seed_cells", 8),
            note=note, iter_for_log=iter_for_log, reso_for_log=reso_for_log
        )
    
    def maybe_guard_missing(iteration, *, tag="loop"):
        mr = quick_miss_ratio(tensorf, n_samples=int(getattr(args, "miss_ratio_samples", 8192)))
        summary_writer.add_scalar('coverage/miss_ratio', float(mr), iteration)

        th = float(getattr(args, "miss_guard_thres", 0.35)) 
        if mr > th:
            added = event_full_recover(tensorf, note=f"guard-{tag}", iteration=iteration, reso_cur=tuple(reso_cur))
            if added > 0:
                _normalize_patch_keys(tensorf) 

    def event_full_recover(tensorf, *, note, iteration, reso_cur):
        total, added = 0, 1
        for _ in range(3): 
            if not added:
                break
            added = heartbeat_guard_coverage(
                tensorf,
                target_miss=0.0,         
                seed_cells=10**6,        
                note=f"{note}-full",
                iter_for_log=iteration,
                reso_for_log=tuple(reso_cur),
            )
            _normalize_patch_keys(tensorf)
            total += int(added or 0)
        if total > 0:
            print(f"[event-full] +{total} cells; G={tensorf.patch_grid_reso}")
        return total

    def _normalize_patch_keys(tensorf):
        try:
            tensorf.assert_zero_origin_and_contiguous()
        except Exception as e:
            print(f"[WARN] assert_zero_origin_and_contiguous failed: {e}")

        if hasattr(tensorf, "infer_patch_grid_reso_from_keys"):
            tensorf.patch_grid_reso = tuple(tensorf.infer_patch_grid_reso_from_keys(tensorf.patch_map))
        tensorf.current_patch_keys = sorted(list(tensorf.patch_map.keys()))
        tensorf.base_patch_grid_reso = tensorf.patch_grid_reso

    init_sigma = [int(x) for x in getattr(tensorf, "density_n_comp")]
    init_app   = [int(x) for x in getattr(tensorf, "app_n_comp")]

    tensorf.rank_policy = make_rank_policy(init_sigma=init_sigma, init_app=init_app,
                                           ref_res=args.vm_reso_max,
                                           gamma=args.rank_autoscale_gamma,
                                           cmin=args.min_rank, cmax=args.max_rank,
                                           round_to=4)
        
    tensorf.init_uniform_patches(grid_reso=boot_G, vm_reso=boot_R)
    _ = heartbeat_guard_coverage(tensorf, target_miss=0.0, seed_cells=10**6,
                                 note="init-uniform", iter_for_log=0, reso_for_log=tuple(reso_cur))
    _normalize_patch_keys(tensorf)
    tensorf.patch_grid_reso = tuple(tensorf.infer_patch_grid_reso_from_keys(tensorf.patch_map))
    tensorf.base_patch_grid_reso = tensorf.patch_grid_reso 
    tensorf.ensure_default_patch()

    tensorf.split_child_res_policy = "half"   # "arg"| "half"| "scale"
    tensorf.split_child_min = 16  

    teacher = _build_ema_from(tensorf) if args.kd_ema_enable else None

    if bool(getattr(args, "dynamic_rank", True)) and hasattr(tensorf, "try_rank_upgrade"):
        boot_info = tensorf.try_rank_upgrade(args, reso_hint=tuple(boot_R), iteration=0)
        if int(boot_info.get("up", 0)) > 0:
            print(f"[rank-init-boot] upgraded at init: {boot_info}")

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if not args.ndc_ray:
        allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)

    allrays_old = allrays
    allrgbs_old = allrgbs
    allrays_new = None
    allrgbs_new = None

    bs_prob_floor = None   # keep a floor for args.batch_size during probation
    bs_prob_until = -1     # iteration <= this => enforce floor

    trainer = PatchTrainStep(tensorf, render_step_size=1.0, white_bg=white_bg, rm_weight_mask_thre=args.rm_weight_mask_thre)

    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    if args.lr_decay_iters <= 0:
        args.lr_decay_iters = args.n_iters
    lr_gamma = float(args.lr_decay_target_ratio) ** (1.0 / float(args.lr_decay_iters))

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)

    PSNRs,PSNRs_test = [],[0]
    psnr_full, psnr_fast = None, None
    rd_log_list = [] 

    tensorf.warn_interval   = 200   
    tensorf.warn_min_rays   = 512   
    tensorf.debug_map_stats = False

    warmup_until = -1
    maybe_warmup_hook = None
    pending_restruct_probation = None  
    first_ups_done_iter = getattr(tensorf, "_first_ups_done_iter", None)
    last_even_selective_iter = -10**9 
    last_structure_change_iter = -10**9 
    last_softprune_iter = -10**9
    last_heartbeat_iter = -10**9

    ALPHA_KICK_1 = 8000   
    ALPHA_KICK_2 = 14000  
    
    did_alpha_32 = False
    did_alpha_64 = False
    did_refilter = False

    Ortho_reg_weight  = 0.0
    L1_reg_weight     = 0.0
    TV_weight_density = 0.0
    TV_weight_app     = 0.0

    Orth_INIT   = float(getattr(args, "Ortho_weight",      0.0))
    L1_INIT     = float(getattr(args, "L1_weight_inital",  0.0))   
    L1_REST     = float(getattr(args, "L1_weight_rest",    0.0))   
    TV_D_BASE   = float(getattr(args, "TV_weight_density", 0.0)) 
    TV_A_BASE   = float(getattr(args, "TV_weight_app",     0.0))  
    TV_A_BUMP64 = float(getattr(args, "TV_app_bump_at64",  1.25)) 

    USE_L1_ONLY = ((L1_INIT > 0.0 or L1_REST > 0.0) and (TV_D_BASE == 0.0 and TV_A_BASE == 0.0))
    USE_TV_ONLY = ((TV_D_BASE > 0.0 or TV_A_BASE > 0.0) and (L1_INIT == 0.0 and L1_REST == 0.0))
    if USE_L1_ONLY and USE_TV_ONLY:
        print("[WARN] Both L1 and TV are non-zero in config. Falling back to L1-only.")
        USE_TV_ONLY = False
    USE_NO_REG = (not USE_L1_ONLY) and (not USE_TV_ONLY)

    reg32_start = None
    reg64_start = None
    tv32_start  = None
    tv64_start  = None
    L1_ref_vox = None
    TV_ref_h = None
    TV_ref_patches = None

    did_enable_reg_32 = False
    did_enable_reg_64 = False

    REG_BURNIN_32 = int(getattr(args, "REG_BURNIN_32", 2000))
    REG_BURNIN_64 = int(getattr(args, "REG_BURNIN_64", 6000))
    REG_WARMUP_ITERS = int(getattr(args, "reg_warmup_iters", 1500))
    REG_DECAY_ITERS  = int(getattr(args, "reg_decay_iters", 1500))

    print("initial Ortho_reg_weight", Orth_INIT)
    print(f"initial L1_reg_weight: {L1_INIT}; rest weight: {L1_REST}")
    print(f"initial TV_weight density: {TV_D_BASE} appearance: {TV_A_BASE}")
    print(f"[reg] gates: NO_REG={USE_NO_REG}  L1_ONLY={USE_L1_ONLY}  TV_ONLY={USE_TV_ONLY}")
    print(f"[reg] args TV base  D={TV_D_BASE}  A={TV_A_BASE}  bump64={TV_A_BUMP64}")
    
    ALPHA_FREEZE_ITERS = int(getattr(args, "alpha_freeze_after_event", 400))

    VOX_BUDGET = int(getattr(args, "voxel_budget", 30_000_000))  
    VRAM_BUDGET_MB = float(getattr(args, "vram_budget_MB", 9000.0))

    Orth_INIT = float(getattr(args, "Ortho_weight", 0.0))
    L1_INIT = float(getattr(args, "L1_weight_inital", 0.0))  # e.g., NSVF/Wineholder: 8e-5
    L1_REST = float(getattr(args, "L1_weight_rest",   0.0))  # e.g., NSVF/Wineholder: 4e-5
    TV_D_BASE = float(getattr(args, "TV_weight_density", 0.0))  # e.g., LLFF: 1.0 / Tanks: 0.1
    TV_A_BASE = float(getattr(args, "TV_weight_app",     0.0))  # e.g., LLFF: 1.0 / Tanks: 0.01

    tvreg = TVLoss()

    def _current_min_vm_res(tensorf):
        for name in ('vm_res', 'vmRes', 'cur_vm_res', 'H'):
            v = getattr(tensorf, name, None)
            if v is None: continue
            try:
                if isinstance(v, (tuple, list)): 
                    return int(min(v))
                if torch.is_tensor(v):
                    return int(v.min().item()) if v.numel() > 0 else -1
                return int(min(v))
            except Exception:
                pass
 
        for attr in ('density_line', 'app_line', 'sigma_line', 'sh_line'):
            obj = getattr(tensorf, attr, None)
            if obj is None: continue
            Hs = []
            seq = obj if isinstance(obj, (list, tuple)) else [obj]
            for t in seq:
                if hasattr(t, 'shape') and len(t.shape) > 0:
                    Hs.append(int(t.shape[-1]))
            if Hs: return min(Hs)
        return -1
        
    def _alpha_grid_from_aabb(args, aabb):
        # aabb: shape (6,) 或 (2,3)；若是 (3,2)（常見的轉置），先轉回 (2,3)
        if isinstance(aabb, torch.Tensor):
            if aabb.dim() == 2 and aabb.shape == (3, 2):
                aabb = aabb.t().contiguous()
        else:
            raise TypeError(f"AABB must be a torch.Tensor, got {type(aabb)}")

        # 與文件內既有的 _as_minmax 統一規則（只接受 (6,) 或 (2,3)）
        a0, a1 = _as_minmax(aabb)  

        ext = (a1 - a0).abs()  # [Lx, Ly, Lz]

        base = int(getattr(args, "alpha_mask_base_res", 192))
        rmin = int(getattr(args, "alpha_mask_min_res", 64))
        rmax = int(getattr(args, "alpha_mask_max_res", 512))
        aspect = str(getattr(args, "alpha_mask_aspect", 'short')).lower()

        if aspect == 'long':
            ref = float(ext.max().item())
        elif aspect == 'mean':
            ref = float(ext.mean().item())
        else: 
            ref = float(ext.min().item())

        ref = max(ref, 1e-6)
        s = max(1.0, float(base) / ref)

        res = torch.clamp((ext * s).round().long(), rmin, rmax)
        res = torch.clamp(res, min=4)

        return tuple(int(x) for x in res.tolist())

    def maybe_softprune(note="softprune"):
        nonlocal last_softprune_iter, last_structure_change_iter

        if iteration < getattr(args, "softprune_start_iter", 12000):
            return
        if (iteration - last_structure_change_iter) < getattr(args, "softprune_cooldown", 1500):
            return
        if not tensorf.alpha_has_signal(eps=1e-5):
            return

        n_removed = tensorf.soft_prune_empty_patches(
            alpha_quantile=getattr(args, "softprune_alpha_q", 0.90),
            min_alpha=getattr(args, "softprune_alpha_min", 5e-3),
            min_reso=getattr(args, "softprune_min_reso", 16),
            keep_topk=getattr(args, "softprune_keep_topk", 1)
        )
        if n_removed > 0:
            print(f"[soft-prune] removed {n_removed} patches ({note})")
            tensorf.patch_grid_reso = tuple(tensorf.infer_patch_grid_reso_from_keys(tensorf.patch_map))
            tensorf.base_patch_grid_reso = tensorf.patch_grid_reso 
            if hasattr(tensorf, "assert_zero_origin_and_contiguous"):
                try: tensorf.assert_zero_origin_and_contiguous()
                except Exception as e: print(f"[WARN] assert_zero_origin_and_contiguous failed: {e}")
            run_heartbeat(note="post-softprune")
            _normalize_patch_keys(tensorf)
      
            last_structure_change_iter = iteration
            last_softprune_iter = iteration

    def total_voxels_now():
        return max(1, int(tensorf.get_total_voxels()))

    def h_min_now():
        return 1.0 / max(1, int(min(reso_cur)))

    def safe_export_patch_viz(tensorf, save_prefix, tag=""):
        try:
            pmap = getattr(tensorf, "patch_map", None)
            if not isinstance(pmap, dict) or len(pmap) == 0:
                print(f"[viz-skip] empty patch_map ({tag})")
                return
            export_patch_viz_bundle(pmap, tensorf, save_prefix)
            print(f'[viz] saved "{save_prefix}_*.png" ({tag})')
        except Exception as e:
            print(f"[viz-error] {repr(e)}")

    def maybe_viz_patches(iteration, tag=""):
        ev_ok = bool(getattr(args, "viz_patch_on_events", True))
        cyc = int(getattr(args, "viz_patch_every", 0))
        do_cyc = (cyc > 0 and (iteration % cyc == 0))
        if ev_ok or do_cyc:
            save_prefix = f"{logfolder}/patch_distribution_{iteration:06d}"
            safe_export_patch_viz(tensorf, save_prefix, tag)

    def log_patch_status(tensorf, iteration=None, note=""):
        try:
            G = tuple(tensorf.patch_grid_reso)
        except Exception:
            G = tuple(tensorf.infer_patch_grid_reso_from_keys(tensorf.patch_map))

        # internal VM res (own res of each patch)
        res_set = sorted({tuple(p['res']) for p in tensorf.patch_map.values()})
        n_patches = len(tensorf.patch_map)
        vox = getattr(tensorf, "get_total_voxels", lambda: -1)()

        tag = f"Iter {iteration:06d} " if iteration is not None else ""
        extra = f" {note}" if note else ""
        print(f"[INFO] {tag}{extra} patches={n_patches}, G={G}, res_set={res_set}, vox={vox}")
    
    def log_rd(iteration, note="", psnr_train_val=None, psnr_fast_val=None, psnr_full_val=None):
        nonlocal psnr_full, psnr_fast  

        if psnr_train_val is not None:
            try:
                summary_writer.add_scalar('rd/psnr_train', float(psnr_train_val), int(iteration))
            except Exception:
                pass
        if psnr_fast_val is not None:
            psnr_fast = float(psnr_fast_val)
        if psnr_full_val is not None:
            psnr_full = float(psnr_full_val)

        n_vox = int(tensorf.get_total_voxels())
        mem_mb = float(tensorf.get_total_mem()) / (1024.0 ** 2)

        try:
            summary_writer.add_scalar('rd/voxels', n_vox, int(iteration))
            summary_writer.add_scalar('rd/memory_MB', mem_mb, int(iteration))
            if psnr_fast is not None and not (isinstance(psnr_fast, float) and math.isnan(psnr_fast)):
                summary_writer.add_scalar('rd/psnr_fast', float(psnr_fast), int(iteration))
            if psnr_full is not None and not (isinstance(psnr_full, float) and math.isnan(psnr_full)):
                summary_writer.add_scalar('rd/psnr_full', float(psnr_full), int(iteration))
        except Exception:
            pass

        rd_log_list.append({
            "iter": int(iteration),
            "note": str(note),
            "lambda": float(getattr(args, "critrn_lambda", 0.0)),
            "tau": float(getattr(args, "logmargin_tau", 0.0)),
            "accept_ratio": float(getattr(args, "critrn_accept_ratio", 0.0)),
            "refine_frac": float(getattr(args, "critrn_refine_frac", 0.0)),
            "voxels": n_vox,
            "memory_MB": round(mem_mb, 2),
            "PSNR_train": float(psnr_train_val) if psnr_train_val is not None else None,
            "PSNR_test_full": float(psnr_full) if psnr_full is not None else None,
            "PSNR_test_fast": float(psnr_fast) if psnr_fast is not None else None
        })


    if getattr(args, "peak_vram_on_start_reset", True):
        reset_cuda_peak()
    best_peak_bytes = [0]

    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout) 
    for iteration in pbar:
        if iteration % 100 == 0:  
            exp_logger.log_iteration(
                iteration=iteration,
                psnr=PSNR,
                memory_mb=tensorf.get_total_mem() / (1024**2),
                n_patches=len(tensorf.patch_map),
                total_voxels=tensorf.get_total_voxels()
            )

        # pre-step warm-up: apply updated lr
        if maybe_warmup_hook is not None:
            maybe_warmup_hook(iteration) 
            if warmup_until >= 0 and iteration >= warmup_until:
                maybe_warmup_hook = None
                warmup_until = -1

        if getattr(tensorf, "renderModule", None) and hasattr(tensorf.renderModule, "set_fea_gate"):
            start, end = 1000, 3000
            if iteration <= end:
                t = max(0.0, min(1.0, (iteration - start) / max(1, end - start)))
                gate = 0.2 + 0.8 * t
                tensorf.renderModule.set_fea_gate(gate)

        # periodical cleanup
        if iteration % 1000 == 0:
            if hasattr(tensorf, 'cleanup_seam_banks'):
                removed = tensorf.cleanup_seam_banks(keep_active_only=True)
                if removed > 0:
                    print(f"[Cleanup] Removed {removed} unused seam banks")
            
            if hasattr(tensorf, 'shared_basis_manager'):
                active_keys = list(tensorf.patch_map.keys())
                removed = tensorf.shared_basis_manager.cleanup_unused(active_keys)
                if removed > 0:
                    print(f"[Cleanup] Removed {removed} unused basis coefficients")

            if hasattr(tensorf, '_kd_buffers') and len(tensorf._kd_buffers) > 10:
                tensorf._kd_buffers = tensorf._kd_buffers[-10:]  # last 10 new 

        tensorf.global_step = iteration

        def _coalesce_base_floors(args, model):
            """
            Return (base_floor_sig:int, base_floor_app:int)
            """
            def _scalar_from_any(x, default):
                if x is None:
                    return int(default)
                if isinstance(x, (int, float)):
                    return int(x)
                if isinstance(x, (list, tuple)):
                    vals = []
                    for v in x:
                        if isinstance(v, (int, float)):
                            vals.append(int(v))
                        elif isinstance(v, str) and v.strip():
                            # e.g., "16" or "16,16,16" or "16 16 16"
                            parts = v.replace(",", " ").split()
                            vals.extend(int(p) for p in parts)
                    if vals:
                        return int(max(vals))
                    return int(default)
                # str -> e.g., "16" or "16,16,16" or "16 16 16"
                if isinstance(x, str) and x.strip():
                    parts = x.replace(",", " ").split()
                    try:
                        nums = [int(p) for p in parts]
                        return int(max(nums))
                    except Exception:
                        return int(default)
                return int(default)

            def _first_non_none(*names):
                for n in names:
                    v = getattr(args, n, None)
                    if v is not None:
                        return v
                return None

            sig_src = _first_non_none("rank_base_floor_sig", "n_factors_sigma", "n_lamb_sigma")
            app_src = _first_non_none("rank_base_floor_app", "n_factors_app",   "n_lamb_sh")

            if sig_src is None:
                sig_src = getattr(model, "density_n_comp_init",
                                getattr(model, "density_n_comp", [16, 16, 16]))
            if app_src is None:
                app_src = getattr(model, "app_n_comp_init",
                                getattr(model, "app_n_comp", [48, 48, 48]))

            base_floor_sig = _scalar_from_any(sig_src, 16)
            base_floor_app = _scalar_from_any(app_src, 48)
            return int(base_floor_sig), int(base_floor_app)

        def _prob_cfg(phase: str):
            def coalesce(val, default):
                return default if (val is None) else val

            IMMEDIATE = float(coalesce(
                getattr(args, f"{phase}_immediate_abort_dB", None),
                getattr(args, "restruct_immediate_abort_dB", 4.0),
            ))
            ALLOW = float(coalesce(
                getattr(args, f"{phase}_probation_allow_dB", None),
                getattr(args, "restruct_probation_allow_dB", 3.0),
            ))
            PROB_ITERS = int(coalesce(
                getattr(args, f"{phase}_prob_iters", None),
                getattr(args, "restruct_prob_iters", 800),
            ))
            FINAL_TOL = float(coalesce(
                getattr(args, f"{phase}_prob_final_tol_dB", None),
                getattr(args, "restruct_prob_final_tol_dB", 0.5),
            ))
            CHECK_EVERY = int(coalesce(
                getattr(args, f"{phase}_prob_check_every", None),
                getattr(args, "restruct_prob_check_every", 100),
            ))
            return IMMEDIATE, ALLOW, PROB_ITERS, FINAL_TOL, CHECK_EVERY

        # ======== Probation rollback check ========
        if pending_restruct_probation is not None:
            prob = pending_restruct_probation
            phase = prob["phase"]  # will be even-selective phase or vm-upsample phase
            _, _, PROB_ITERS, PROB_FINAL_TOL, PROB_CHECK_EVERY = _prob_cfg(phase)
            t0 = int(prob["iter_applied"])

            need_check = (iteration >= t0 + PROB_ITERS) or ((iteration - t0) % PROB_CHECK_EVERY == 0)
            if need_check:
                if hasattr(tensorf, "eval"): tensorf.eval()
                flag = tensorf.repair_enable
                tensorf.repair_enable = False
                with torch.no_grad():
                    val_dataset = get_val_dataset(args, train_dataset, test_dataset)
                    cur_psnr_fast = quick_val_psnr_safe(
                        val_dataset, trainer.render, device,
                        target_views=3, target_rpv=4096,
                        min_views=1, min_rpv=1024, chunk=1024
                    )
                tensorf.repair_enable = flag
                if hasattr(tensorf, "train"): tensorf.train()
                pre_psnr = float(prob["pre_psnr_fast"])
    
                if cur_psnr_fast >= (pre_psnr - PROB_FINAL_TOL):
                    log_event(logfolder, "probation-commit", iteration,
                              phase=phase, applied_at=t0,
                              pre_psnr=f"{pre_psnr:.4f}", cur_psnr=f"{cur_psnr_fast:.4f}",
                              tol_dB=PROB_FINAL_TOL)
                    last_structure_change_iter = t0
                    if phase == "even":
                        last_even_selective_iter = t0
                    pending_restruct_probation = None

                elif iteration >= (t0 + PROB_ITERS):
                    restore_model(tensorf, prob["pre_snap"], device)
                    _normalize_patch_keys(tensorf)

                    cur_scale = float(args.lr_decay_target_ratio ** (iteration / args.n_iters))
                    optimizer = build_optimizer_with_scale(tensorf, args, cur_scale)

                    if phase == "even":
                        last_even_selective_iter = int(prob.get("prev_last_even_iter", last_even_selective_iter))
                    elif phase == "ups":
                        reso_cur = list(prob.get("prev_reso_cur", reso_cur))

                    last_structure_change_iter = iteration
                    log_event(logfolder, "probation-rollback-delayed", iteration,
                              phase=phase, applied_at=t0,
                              pre_psnr=f"{pre_psnr:.4f}", cur_psnr=f"{cur_psnr_fast:.4f}",
                              delta=f"{cur_psnr_fast - pre_psnr:+.4f}dB", tol_dB=PROB_FINAL_TOL)
                    pending_restruct_probation = None
                    continue

        def _should_run_autoscale(iteration, args, *,
                                  warmup_until=None,
                                  last_structure_change_iter=None,
                                  pending_probation=None):
            if pending_probation is not None:
                return False
            if warmup_until is not None and iteration <= int(warmup_until):
                return False
            start = int(getattr(args, "rank_down_after_iter", 20000))
            if iteration < start:
                return False
            cool = int(getattr(args, "autoscale_cooldown_after_event", 400))
            if (last_structure_change_iter is not None) and (iteration - int(last_structure_change_iter) < cool):
                return False
            return True

        enable_now = (args.repair_mode in ['conservative','full']) and (tensorf.global_step >= args.repair_warmup)
        tensorf.repair_enable = bool(enable_now)
        did_cov_this_step = False

        if bs_prob_until is not None and iteration <= int(bs_prob_until):
            if (bs_prob_floor is not None) and (args.batch_size < int(bs_prob_floor)):
                args.batch_size = int(bs_prob_floor)
                try:
                    trainingSampler.batch = int(args.batch_size)
                except Exception:
                    pass

        if hasattr(trainingSampler, "set_iter"): 
            trainingSampler.set_iter(iteration)

        if isinstance(trainingSampler, MixedSampler) and (iteration % 200 == 0):
            try:
                ratio = trainingSampler._cur_new_ratio()
                log_event(logfolder, "refilter-ratio", iteration, new_ratio=f"{ratio:.3f}")
            except Exception:
                pass

        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)
        total_loss, loss_rgb = trainer(rays_train, rgb_train) 

        # --- Post-split field KD (teacher = pre-split parent snapshot) ---
        kd_enable  = bool(int(getattr(args, "post_event_kd", 1)) == 1)
        kd_weight  = float(getattr(args, "post_event_kd_w", 0.1))
        kd_max_pts = int(getattr(args, "post_event_kd_pts", 4096))
        kd_sig_w   = float(getattr(args, "kd_sigma_weight", 1.0))
        kd_app_w   = float(getattr(args, "kd_app_weight", 1.0))
        kd_every   = int(getattr(args, "post_event_kd_every", 1))  # default 1 = every iter

        kd_val = None 
        if kd_enable and (iteration % kd_every == 0) and kd_weight > 0.0:
            kd = field_kd_loss(tensorf, iteration, device,
                            max_pts=kd_max_pts, sigma_w=kd_sig_w, app_w=kd_app_w)
            if kd is not None:
                total_loss = total_loss + kd_weight * kd
                kd_val = float(kd.detach())
        
        seam_val = None
        if args.seam_tying_enable and args.seam_tying_mode == 'soft' and args.seam_loss_w > 0:
            seam = seam_consistency_loss(
                tensorf, device,
                samples_per_face=args.seam_sample_per_face,
                eps=args.seam_eps,
                sigma_w=args.kd_sigma_weight, app_w=args.kd_app_weight
            )
            if seam is not None:
                total_loss = total_loss + args.seam_loss_w * seam
                seam_val = float(seam.detach())
                
        # ========== Regularization enable / schedule ==========
        # L1-only (for NSVF / Blender) 
        if (not USE_NO_REG) and USE_L1_ONLY and (L1_INIT > 0.0):
            if 'reg32_start' not in locals():
                reg32_start = None
            if 'reg64_start' not in locals():
                reg64_start = None

            # @32 res stage: enable and linearly warmup 0 -> L1_INIT
            if (min(reso_cur) >= 32) and (iteration >= REG_BURNIN_32):
                if L1_ref_vox is None:
                    L1_ref_vox = total_voxels_now()
                if reg32_start is None:  
                    reg32_start = iteration
                    print(f"[REG-L1] enable @32^3: warmup to {L1_INIT:g}, ref_vox={L1_ref_vox}")
                r = min(1.0, max(0.0, (iteration - reg32_start) / float(REG_WARMUP_ITERS)))
                L1_raw = L1_INIT * r
                L1_reg_weight = L1_raw * min(1.0, L1_ref_vox / float(total_voxels_now()))
                
                if iteration % 50 == 0:
                    print(f"[INFO] min_res: {min(reso_cur)} | L1 weight: {L1_reg_weight} | Vox: {total_voxels_now()} | {L1_ref_vox / float(total_voxels_now())}")

            # @64 res stage: slowly decay from L1_INIT -> L1_REST
            if (min(reso_cur) >= 64) and (iteration >= REG_BURNIN_64):
                if reg64_start is None:
                    reg64_start = iteration
                    print(f"[REG-L1] decay @64^3: {L1_INIT:g} → {L1_REST:g} (scaled by vox)")
                r = min(1.0, max(0.0, (iteration - reg64_start) / float(REG_DECAY_ITERS)))
                L1_raw = L1_INIT * (1.0 - r) + L1_REST * r
                L1_reg_weight = L1_raw * min(1.0, L1_ref_vox / float(total_voxels_now()))
            
                if iteration % 50 == 0:
                    print(f"[INFO] min_res: {min(reso_cur)} | L1 weight: {L1_reg_weight} | Vox: {total_voxels_now()} | {L1_ref_vox / float(total_voxels_now())}")

        # TV-only (for LLFF / Tanks) 
        if (not USE_NO_REG) and USE_TV_ONLY and (TV_D_BASE > 0.0 or TV_A_BASE > 0.0):
            if TV_ref_h is None:
                TV_ref_h = h_min_now()
            if TV_ref_patches is None:
                TV_ref_patches = max(1, len(tensorf.patch_map))

            scale_res     = h_min_now() / TV_ref_h
            scale_patches = TV_ref_patches / max(1, len(tensorf.patch_map))

            # @32: warmup 0 -> BASE over REG_WARMUP_ITERS 
            if iteration >= REG_BURNIN_32:
                if tv32_start is None:
                    tv32_start = iteration
                    print(f"[REG-TV] enable @32^3: warmup to d/a=({TV_D_BASE:g},{TV_A_BASE:g}), "
                        f"h_ref={TV_ref_h:.4g}, patches_ref={TV_ref_patches}")
                r32 = min(1.0, max(0.0, (iteration - tv32_start) / float(REG_WARMUP_ITERS)))
                d_raw32 = TV_D_BASE * r32
                a_raw32 = TV_A_BASE * r32
                TV_weight_density = d_raw32 * scale_res * scale_patches
                TV_weight_app     = a_raw32 * scale_res * scale_patches

            # @64: density fixed at BASE; appearance linearly bump to bump*BASE over REG_DECAY_ITERS 
            bump_after = int(getattr(args, "tv_force_bump_after", -1))
            enable64   = (iteration >= REG_BURNIN_64) or (bump_after >= 0 and iteration >= bump_after)
            if enable64:
                if tv64_start is None:
                    tv64_start = iteration
                    print(f"[REG-TV] bump @64^3: app ×{TV_A_BUMP64:g} (scaled by res/patches)")
                r64 = min(1.0, max(0.0, (iteration - tv64_start) / float(REG_DECAY_ITERS)))
                d_raw64 = TV_D_BASE
                a_raw64 = TV_A_BASE * (1.0 + (TV_A_BUMP64 - 1.0) * r64)
                TV_weight_density = d_raw64 * scale_res * scale_patches
                TV_weight_app     = a_raw64 * scale_res * scale_patches

            if iteration % 50 == 0:
                print(f"[TV-info] iter={iteration} min_res={min(reso_cur)} "
                    f"d={TV_weight_density:g} a={TV_weight_app:g} "
                    f"(scale_res={scale_res:.3g}, scale_patches={scale_patches:.3g})")

        reg_terms = {}

        if Ortho_reg_weight > 0:
            loss_ortho = tensorf.vector_comp_diffs()
            total_loss = total_loss + Ortho_reg_weight * loss_ortho
            reg_terms["ortho"] = loss_ortho

        if L1_reg_weight > 0:
            loss_L1 = tensorf.density_L1()
            total_loss = total_loss + L1_reg_weight * loss_L1
            reg_terms["l1"] = loss_L1

        if TV_weight_density > 0:
            # TV_weight_density *= lr_factor  
            loss_tv_d = tensorf.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv_d
            reg_terms["tv_density"] = loss_tv_d

        if TV_weight_app > 0:
            # TV_weight_app *= lr_factor
            loss_tv_a = tensorf.TV_loss_app(tvreg) * TV_weight_app
            total_loss = total_loss + loss_tv_a
            reg_terms["tv_app"] = loss_tv_a

        if not torch.isfinite(total_loss):
            print("[WARN] total_loss is NaN/Inf. Skipping step.")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if getattr(args, "kd_ema_enable", False) and (teacher is not None):
            _ema_update(teacher, tensorf, args.kd_ema_decay)

        if args.seam_tying_enable and args.seam_tying_mode == 'hard':
            if hasattr(tensorf, 'apply_seam_tying'):
                tensorf.apply_seam_tying() 
    
        if maybe_warmup_hook is not None:
            maybe_warmup_hook(iteration)
        else:
            for pg in optimizer.param_groups:
                pg['lr'] *= lr_gamma

        loss_rgb_val = float(loss_rgb.detach())
        total_reg_val = sum(float(t.detach()) for t in reg_terms.values())
        total_loss_val = float(total_loss.detach())

        for k in ("L1", "TVd", "TVa", "Ortho"):
            if k not in reg_terms:
                reg_terms[k] = torch.tensor(0.0, device=device)

        PSNR = -10.0 * math.log10(max(loss_rgb_val, 1e-8))
        PSNRs.append(PSNR)

        summary_writer.add_scalar('train/PSNR', PSNR, iteration)
        summary_writer.add_scalar('train/mse',  loss_rgb_val, iteration)
        summary_writer.add_scalar('train/loss_rgb', loss_rgb_val, iteration)
        if kd_val is not None:  
            summary_writer.add_scalar('train/loss_kd', kd_val, iteration)
        if seam_val is not None:
            summary_writer.add_scalar('train/loss_seam', seam_val, iteration)
        for name, t in reg_terms.items(): 
            summary_writer.add_scalar(f'train/loss_reg_{name}', float(t.detach()), iteration)
        summary_writer.add_scalar('train/total_reg', total_reg_val, iteration)
        summary_writer.add_scalar('train/total_loss', total_loss_val, iteration)
        summary_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], iteration)

        n_vox = tensorf.get_total_voxels()
        mem_bytes = tensorf.get_total_mem()
        summary_writer.add_scalar('stats/total_voxels', n_vox, iteration)
        summary_writer.add_scalar('stats/total_memory_MB', mem_bytes / (1024 ** 2), iteration)
        summary_writer.add_scalar('stats/total_patches', len(tensorf.patch_map), iteration)

        rd_every = int(getattr(args, "rd_every", 50))
        if (iteration % rd_every) == 0:
            log_rd(iteration, note="train_psnr", psnr_train_val=PSNR)

        with open(os.path.join(logfolder, 'total_voxels.txt'), 'a') as n_vox_log:
            n_vox_log.write(f'{n_vox}\n')
        with open(os.path.join(logfolder, 'grid_memory.txt'), 'a') as grid_mem_log:
            grid_mem_log.write(f'{mem_bytes}\n')
        
        if iteration % (5*args.progress_refresh_rate) == 0:
            print(f">> patch grid reso = {tensorf.patch_grid_reso}, num patches = {len(tensorf.patch_map)}")
            log_mem_breakdown(tensorf, iteration, logfolder)

        if iteration % args.progress_refresh_rate == 0:
            psnr_train_avg = np.mean(PSNRs)
            psnr_test_avg = float(np.mean(PSNRs_test)) if ('PSNRs_test' in locals() and len(PSNRs_test) > 0) else float('nan')

            total_reg_val = float(sum(v.detach().item() for v in reg_terms.values())) if reg_terms else 0.0
            reg_str = " ".join(f"{k}:{float(v.detach().item()):.3g}" for k,v in reg_terms.items()) or "none"
            pbar.set_description(f"Iteration {iteration:05d} | train_psnr: {psnr_train_avg:.4f} | test_psnr: {psnr_test_avg:.4f} | "
                                 f"RGB: {loss_rgb_val:.3g} | Reg: {total_reg_val:.3g} ({reg_str}) | Vox: {n_vox} | Mem: {mem_bytes // 1024} KB")

            with open(os.path.join(logfolder, 'psnr_log.txt'), 'a') as psnr_log:
                psnr_log.write(f"Iteration {iteration:05d} | train_psnr: {psnr_train_avg:.4f} | test_psnr: {psnr_test_avg:.4f} | "
                               f"RGB: {loss_rgb_val:.3g} | Reg: {total_reg_val:.3g} ({reg_str}) | Vox: {n_vox} | Mem: {mem_bytes // 1024} KB\n")
            PSNRs = []  
        
        if iteration % 500 == 0:
            vm_min = _current_min_vm_res(tensorf)
            reg32_on = (vm_min >= 32) and (iteration >= REG_BURNIN_32)
            reg64_on = (vm_min >= 64) and (iteration >= REG_BURNIN_64)
            print(f"[reg] it={iteration} vm_min={vm_min} | L1_stage32_on={reg32_on} L1_stage64_on={reg64_on} | "
                  f"L1_init={args.L1_weight_inital} L1_rest={args.L1_weight_rest} TVd={args.TV_weight_density} TVa={args.TV_weight_app}")
        
        if getattr(tensorf, "debug_map_stats", False) and (iteration % getattr(tensorf, "warn_interval", 200) == 0):
            with torch.no_grad():
                r = rays_train
                if r.shape[0] > 4096:
                    r = r[torch.randperm(r.shape[0])[:4096]]
                tensorf.log_patch_distribution(r.to(tensorf.aabb.device), iteration)

        # render visualization img and compute test PSNR
        do_vis = (getattr(args, "N_vis", 0) > 0) and (getattr(args, "vis_every", 0) > 0) and (((iteration + 1) % args.vis_every) == 0)
        if do_vis and (test_dataset is not None):
            tensorf_eval_flag = tensorf.repair_enable
            tensorf.repair_enable = False
            PSNRs_test = evaluation(
                test_dataset, tensorf, args, trainer.render,
                f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                prtx=f'{iteration:06d}_', N_samples=-1,
                white_bg=white_bg, ndc_ray=ndc_ray,
                compute_extra_metrics=False
            )
            tensorf.repair_enable = tensorf_eval_flag
            psnr_full = float(np.mean(PSNRs_test))
            summary_writer.add_scalar('test/psnr', psnr_full, global_step=iteration)
            log_rd(iteration, note="vis_full", psnr_full_val=psnr_full)


        # ========== AlphaMask / shrink / prune (staged process) ==========
        if iteration == ALPHA_KICK_1:
            tensorf.alpha_gate_scale.fill_(1.6)
        if iteration == ALPHA_KICK_1 + 1500:
            tensorf.alpha_gate_scale.fill_(1.0)

        # @32 res stage
        if (not did_alpha_32) and (min(reso_cur) >= 32) and (iteration >= ALPHA_KICK_1):
            if tensorf.any_patch_confident(alpha_quantile=0.90, min_val=1e-3):
                if iteration - last_structure_change_iter >= ALPHA_FREEZE_ITERS:
                    adjust_batch_size(reso_cur, args)
                    prev_aabb = tensorf.aabb.detach().clone()
                    new_aabb = tensorf.updateAlphaMask(_alpha_grid_from_aabb(args, prev_aabb))
                    tensorf.alphaMask.alpha_volume.data = tensorf.alphaMask.alpha_volume.data.half()
                    tensorf.shrink(new_aabb)

                    if getattr(args, "patch_crop_inside", False):
                        ok_cooldown = (iteration - last_structure_change_iter >= args.split_cooldown)
                
                        old_min, old_max = _as_minmax(prev_aabb)  
                        new_min, new_max = _as_minmax(new_aabb)
                        old_vol = float(torch.prod(old_max - old_min).item())
                        new_vol = float(torch.prod(new_max - new_min).item())
                        ok_gain = (new_vol / max(old_vol, 1e-6)) < 0.90  

                        if ok_cooldown and ok_gain:
                            n_cropped = tensorf.shrink_inside_patches(new_aabb, pad=int(args.patch_crop_pad))
                            if n_cropped > 0:
                                print(f"[patch-crop] cropped {n_cropped} patches internally after shrink")

                    # never upgrade ranks but allow to autoscale
                    if args.dynamic_rank:
                        tensorf.update_alpha_mass_per_patch(n_per=getattr(args, "alpha_mass_n", 1024))
                        
                        sigma_trip, app_trip = tensorf.rank_policy(tuple(reso_cur))
                        base_floor_sig, base_floor_app = _coalesce_base_floors(args, tensorf)
                        c_sig_base = max(base_floor_sig, int(np.round(np.mean(sigma_trip))))
                        c_app_base = max(base_floor_app, int(np.round(np.mean(app_trip))))
  
                        changed = 0
                        ALLOW_RANK_DOWN = (iteration >= int(getattr(args, "rank_down_after_iter", 12000))) and (pending_restruct_probation is None)
                        ALLOW_RANK_UP = (pending_restruct_probation is not None)

                        ref_res = int(getattr(args, "vm_reso_max", 64))
                        max_rank = int(getattr(args, "max_rank", 64))
                        scale_gamma = float(getattr(args, "rank_autoscale_gamma", 0.6))
                        alpha_keep_q = float(getattr(args, "rank_autoscale_alpha_keep_q", 0.85))
                        print(f"[rank-auto] ref_res={ref_res} gamma={scale_gamma} keep_q={alpha_keep_q}")

                        changed = tensorf.selective_rank_autoscale(ref_res=ref_res, gamma=scale_gamma,
                                                                   c_sig_base=c_sig_base, c_app_base=c_app_base,  
                                                                   c_min=8, c_max=max_rank, round_to=4, 
                                                                   allow_down=ALLOW_RANK_DOWN, allow_up=ALLOW_RANK_UP,                       
                                                                   alpha_keep_q=alpha_keep_q, verbose=True)
                        if changed > 0:
                            print(f"[rank-auto] resized ranks: changed={changed}")
                            optimizer = build_optimizer_with_scale(tensorf, args, 1.0)
                            last_structure_change_iter = iteration
                        
                        log_event(logfolder, "rank-resize", iteration, phase="shrink@32",
                                  up=0, down=int(changed), reso_cur=str(reso_cur),
                                  base_floor_sig=int(base_floor_sig), base_floor_app=int(base_floor_app),
                                  policy_sig=str(list(map(int, sigma_trip))),  
                                  policy_app=str(list(map(int, app_trip))),
                                  c_sig_base=int(c_sig_base), c_app_base=int(c_app_base),
                                  allow_down=bool(ALLOW_RANK_DOWN),
                                  alpha_keep_q=float(alpha_keep_q))

                    tensorf.updateAlphaMask(_alpha_grid_from_aabb(args, new_aabb))
                    tensorf.alphaMask.alpha_volume.data = tensorf.alphaMask.alpha_volume.data.half()
                    print(f"[INFO] AlphaMask @32: shrunk AABB to {new_aabb.tolist()}")
                else:
                    print(f"[alpha] skip @32 shrink due to freeze ({iteration - last_structure_change_iter} < {ALPHA_FREEZE_ITERS})")

                # density-based patch pruning (coarse-cell safeguard) 
                _ = tensorf.prune_empty_patches(alpha_thres=None, min_reso=32, group_size=2)

                # sync grid reso + repair coverage
                tensorf.patch_grid_reso = tuple(tensorf.infer_patch_grid_reso_from_keys(tensorf.patch_map))
                if hasattr(tensorf, "assert_zero_origin_and_contiguous"):
                    try: tensorf.assert_zero_origin_and_contiguous()
                    except Exception as e: print(f"[WARN] assert_zero_origin_and_contiguous failed: {e}")
                
                added_full = heartbeat_guard_coverage(tensorf, target_miss=0.10, seed_cells=8, note="@32-post-shrink/prune", 
                                                      iter_for_log=iteration, reso_for_log=tuple(reso_cur))
                _normalize_patch_keys(tensorf)
                if getattr(args, "seam_lowrank_enable", False):
                    prev_ids = _param_set(tensorf)  
                    wired = tensorf.init_seam_lowrank(
                        rank_sigma=args.seam_rank_sigma,
                        rank_app=args.seam_rank_app,
                        scope=args.seam_lowrank_scope
                    )
                    if wired:
                        new_params = _collect_new_params(tensorf, prev_ids)
                        special = set(map(id, new_params))
                        base_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
                        child_mult = float(getattr(args, "child_lr_boost_mult", 1.5))
                        optimizer = rebuild_optimizer_with_child_boost(
                            tensorf, args, base_scale=base_scale,
                            child_params=new_params, child_mult=child_mult
                        )

                if added_full > 0:
                    if getattr(args, "seam_lowrank_enable", False):
                        prev_ids = _param_set(tensorf)  
                        wired = tensorf.init_seam_lowrank(
                            rank_sigma=args.seam_rank_sigma,
                            rank_app=args.seam_rank_app,
                            scope=args.seam_lowrank_scope
                        )
                        if wired:
                            new_params = _collect_new_params(tensorf, prev_ids)
                            special = set(map(id, new_params))
                            base_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
                            child_mult = float(getattr(args, "child_lr_boost_mult", 1.5))
                            optimizer = rebuild_optimizer_with_child_boost(
                                tensorf, args, base_scale=base_scale,
                                child_params=new_params, child_mult=child_mult
                            )
                    print(f"[cover] +{added_full} coarse patches (FULL coverage); G={tensorf.patch_grid_reso}")

                did_alpha_32 = True

        # @64 res stage
        if (not did_alpha_64) and (min(reso_cur) >= 64) and (iteration >= ALPHA_KICK_2):
            if tensorf.any_patch_confident(alpha_quantile=0.90, min_val=1e-3):
                if iteration - last_structure_change_iter >= ALPHA_FREEZE_ITERS:
                    adjust_batch_size(reso_cur, args)
                    prev_aabb = tensorf.aabb.detach().clone()
                    new_aabb = tensorf.updateAlphaMask(_alpha_grid_from_aabb(args, prev_aabb))
                    tensorf.alphaMask.alpha_volume.data = tensorf.alphaMask.alpha_volume.data.half()
                    tensorf.shrink(new_aabb)

                    if getattr(args, "patch_crop_inside", False):
                        ok_cooldown = (iteration - last_structure_change_iter >= args.split_cooldown)

                        old_min, old_max = _as_minmax(prev_aabb)   
                        new_min, new_max = _as_minmax(new_aabb)
                        old_vol = float(torch.prod(old_max - old_min).item())
                        new_vol = float(torch.prod(new_max - new_min).item())
                        ok_gain = (new_vol / max(old_vol, 1e-6)) < 0.90  

                        if ok_cooldown and ok_gain:
                            n_cropped = tensorf.shrink_inside_patches(new_aabb, pad=int(args.patch_crop_pad))
                            if n_cropped > 0:
                                print(f"[patch-crop] cropped {n_cropped} patches internally after shrink")

                    # never upgrade ranks but allow to autoscale
                    if args.dynamic_rank:
                        tensorf.update_alpha_mass_per_patch(n_per=getattr(args, "alpha_mass_n", 1024))
                        
                        sigma_trip, app_trip = tensorf.rank_policy(tuple(reso_cur))
                        base_floor_sig, base_floor_app = _coalesce_base_floors(args, tensorf)
                        c_sig_base = max(base_floor_sig, int(np.round(np.mean(sigma_trip))))
                        c_app_base = max(base_floor_app, int(np.round(np.mean(app_trip))))
                        
                        changed = 0
                        ALLOW_RANK_DOWN = (iteration >= int(getattr(args, "rank_down_after_iter", 12000))) and (pending_restruct_probation is None)
                        ALLOW_RANK_UP = (pending_restruct_probation is not None)

                        ref_res = int(getattr(args, "vm_reso_max", 64))
                        max_rank = int(getattr(args, "max_rank", 64))
                        scale_gamma = float(getattr(args, "rank_autoscale_gamma", 0.6))
                        alpha_keep_q = float(getattr(args, "rank_autoscale_alpha_keep_q", 0.85))
                        print(f"[rank-auto] ref_res={ref_res} gamma={scale_gamma} keep_q={alpha_keep_q}")

                        changed = tensorf.selective_rank_autoscale(ref_res=ref_res, gamma=scale_gamma,
                                                                   c_sig_base=c_sig_base, c_app_base=c_app_base,  
                                                                   c_min=8, c_max=max_rank, round_to=4, 
                                                                   allow_down=ALLOW_RANK_DOWN, allow_up=ALLOW_RANK_UP,                       
                                                                   alpha_keep_q=alpha_keep_q, verbose=True)
                        if changed > 0:
                            print(f"[rank-auto] resized ranks: changed={changed}")
                            optimizer = build_optimizer_with_scale(tensorf, args, 1.0)
                            last_structure_change_iter = iteration
                        
                        log_event(logfolder, "rank-resize", iteration, phase="shrink@64",
                                  up=0, down=int(changed), reso_cur=str(reso_cur),
                                  base_floor_sig=int(base_floor_sig), base_floor_app=int(base_floor_app),
                                  policy_sig=str(list(map(int, sigma_trip))),  
                                  policy_app=str(list(map(int, app_trip))),
                                  c_sig_base=int(c_sig_base), c_app_base=int(c_app_base),
                                  allow_down=bool(ALLOW_RANK_DOWN),
                                  alpha_keep_q=float(alpha_keep_q))

                    tensorf.updateAlphaMask(_alpha_grid_from_aabb(args, new_aabb))
                    tensorf.alphaMask.alpha_volume.data = tensorf.alphaMask.alpha_volume.data.half()
                    print(f"[INFO] AlphaMask @64: shrunk AABB to {new_aabb.tolist()}")
                else:
                    print(f"[alpha] skip @64 shrink due to freeze ({iteration - last_structure_change_iter} < {ALPHA_FREEZE_ITERS})")

                _ = tensorf.prune_empty_patches(alpha_thres=None, min_reso=64, group_size=2)

                tensorf.patch_grid_reso = tuple(tensorf.infer_patch_grid_reso_from_keys(tensorf.patch_map))
                if hasattr(tensorf, "assert_zero_origin_and_contiguous"):
                    try: tensorf.assert_zero_origin_and_contiguous()
                    except Exception as e: print(f"[WARN] assert_zero_origin_and_contiguous failed: {e}")
                
                added_full = heartbeat_guard_coverage(tensorf, target_miss=0.10, seed_cells=8, note="@64-post-shrink/prune", 
                                                      iter_for_log=iteration, reso_for_log=tuple(reso_cur))
                _normalize_patch_keys(tensorf)
                if getattr(args, "seam_lowrank_enable", False):
                    prev_ids = _param_set(tensorf)  
                    wired = tensorf.init_seam_lowrank(
                        rank_sigma=args.seam_rank_sigma,
                        rank_app=args.seam_rank_app,
                        scope=args.seam_lowrank_scope
                    )
                    if wired:
                        new_params = _collect_new_params(tensorf, prev_ids)
                        special = set(map(id, new_params))
                        base_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
                        child_mult = float(getattr(args, "child_lr_boost_mult", 1.5))
                        optimizer = rebuild_optimizer_with_child_boost(
                            tensorf, args, base_scale=base_scale,
                            child_params=new_params, child_mult=child_mult
                        )
                
                if added_full > 0:
                    if getattr(args, "seam_lowrank_enable", False):
                        prev_ids = _param_set(tensorf)  
                        wired = tensorf.init_seam_lowrank(
                            rank_sigma=args.seam_rank_sigma,
                            rank_app=args.seam_rank_app,
                            scope=args.seam_lowrank_scope
                        )
                        if wired:
                            new_params = _collect_new_params(tensorf, prev_ids)
                            special = set(map(id, new_params))
                            base_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
                            child_mult = float(getattr(args, "child_lr_boost_mult", 1.5))
                            optimizer = rebuild_optimizer_with_child_boost(
                                tensorf, args, base_scale=base_scale,
                                child_params=new_params, child_mult=child_mult
                            )
                    print(f"[cover] +{added_full} coarse patches (FULL coverage); G={tensorf.patch_grid_reso}")

                did_alpha_64 = True

        # ========== Strict-even split (run once) ==========
        STRICT_EVEN_KICK     = int(getattr(args, "strict_even_kick", 1500))
        STRICT_EVEN_TARGET_G = tuple(getattr(args, "strict_even_target_G", (4,4,4)))
        STRICT_WARMUP_ITERS  = int(getattr(args, "strict_even_warmup_iters", 300))
        SPLIT_LR_POW         = float(getattr(args, "split_lr_pow", 0.5))
        SPLIT_LR_MIN         = float(getattr(args, "split_lr_min", 0.6))

        if iteration == STRICT_EVEN_KICK:
            if iteration - last_structure_change_iter < args.split_cooldown:
                log_event(logfolder, "strict-even-skip", iteration, reason="cooldown")
                print(f"[strict-even] cooldown skip @ {iteration}")
            elif len(tensorf.patch_map) >= args.patch_cap:
                log_event(logfolder, "strict-even-skip", iteration, reason="PATCH_CAP")
                print(f"[strict-even] reached PATCH_CAP={args.patch_cap}, skip.")
            elif tensorf.get_total_voxels() > VOX_BUDGET:
                log_event(logfolder, "strict-even-skip", iteration, reason="voxels")
                print(f"[strict-even] voxel guard: ({tensorf.get_total_voxels()} > {VOX_BUDGET}), skip.")
            elif (tensorf.get_total_mem()/1024**2) > VRAM_BUDGET_MB:
                log_event(logfolder, "strict-even-skip", iteration, reason="VRAM")
                print(f"[strict-even] VRAM guard: ({tensorf.get_total_mem()/1024**2:.1f} MB > {VRAM_BUDGET_MB} MB), skip.")
            else:
                if tensorf.alpha_has_signal(eps=1e-5):
                    prev_aabb = tensorf.aabb.detach().clone()
                    new_aabb = tensorf.updateAlphaMask(_alpha_grid_from_aabb(args, prev_aabb))
                    tensorf.alphaMask.alpha_volume.data = tensorf.alphaMask.alpha_volume.data.half()
                    tensorf.shrink(new_aabb)

                    if getattr(args, "patch_crop_inside", False):
                        ok_cooldown = (iteration - last_structure_change_iter >= args.split_cooldown)
                       
                        old_min, old_max = _as_minmax(prev_aabb)   
                        new_min, new_max = _as_minmax(new_aabb)
                        old_vol = float(torch.prod(old_max - old_min).item())
                        new_vol = float(torch.prod(new_max - new_min).item())
                        ok_gain = (new_vol / max(old_vol, 1e-6)) < 0.90  

                        if ok_cooldown and ok_gain:
                            n_cropped = tensorf.shrink_inside_patches(new_aabb, pad=int(args.patch_crop_pad))
                            if n_cropped > 0:
                                print(f"[patch-crop] cropped {n_cropped} patches internally after shrink")
                    
                    tensorf.updateAlphaMask(_alpha_grid_from_aabb(args, new_aabb))
                    tensorf.alphaMask.alpha_volume.data = tensorf.alphaMask.alpha_volume.data.half()
                    run_heartbeat(note="post-shrink", iter_for_log=iteration, reso_for_log=tuple(reso_cur))
                    did_cov_this_step = True
                    _normalize_patch_keys(tensorf)
                
                prev_n = len(tensorf.patch_map)
                miss_b = quick_miss_ratio(tensorf)

                print(f"========> STRICT-EVEN to G={STRICT_EVEN_TARGET_G} (keep VM={tuple(reso_cur)})")
                tensorf.strict_evenize_once(STRICT_EVEN_TARGET_G, reso_cur)
                tensorf.assert_zero_origin_and_contiguous()
                tensorf.debug_dump_basis_stats()

                added_full = event_full_recover(tensorf, note="strict-even", iteration=iteration, reso_cur=reso_cur)      
                did_cov_this_step = True    
                tensorf.update_alpha_mass_per_patch(n_per=getattr(args, "alpha_mass_n", 1024))
                _normalize_patch_keys(tensorf)
                miss_a = quick_miss_ratio(tensorf)
                tensorf.patch_grid_reso = tuple(tensorf.infer_patch_grid_reso_from_keys(tensorf.patch_map))
                
                append_covlog(logfolder, iteration, "strict-even",
                              miss_before=f"{miss_b:.4f}", miss_after=f"{miss_a:.4f}",
                              added=added_full, n_patches=len(tensorf.patch_map),
                              G=str(tuple(tensorf.patch_grid_reso)), reso_cur=str(reso_cur))
                maybe_viz_patches(iteration, "strict-even")

                n_new = len(tensorf.patch_map)
                cur_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
                raw_split_scale = (prev_n / max(1, n_new)) ** SPLIT_LR_POW
                split_scale = max(SPLIT_LR_MIN, min(1.0, raw_split_scale))
                eff_scale = cur_scale * split_scale
                print(f"[strict-even lr] old={prev_n} new={n_new} | cur={cur_scale:.4g} * split={split_scale:.3f} -> eff={eff_scale:.4g}")

                if args.dynamic_rank and hasattr(tensorf, "rank_policy"):
                    tensorf.update_alpha_mass_per_patch(n_per=getattr(args, "alpha_mass_n", 1024))
                    
                    sigma_trip, app_trip = tensorf.rank_policy(tuple(reso_cur))
                    base_floor_sig, base_floor_app = _coalesce_base_floors(args, tensorf)
                    c_sig_base = max(base_floor_sig, int(np.round(np.mean(sigma_trip))))
                    c_app_base = max(base_floor_app, int(np.round(np.mean(app_trip))))

                    Rmin = int(min(reso_cur))
                    print(f"[rank-up precheck] Rmin={Rmin} rank_min_res={getattr(args,'rank_min_res',8)}")

                    tensorf._last_rank_resize_iter = None
                    upgrade_info = tensorf.try_rank_upgrade(args, reso_hint=tuple(reso_cur), iteration=iteration)
                    upgraded = int(upgrade_info.get("up", 0)) 

                    if upgraded == 0 and "skipped" in upgrade_info: 
                        reason = str(upgrade_info.get("skipped"))
                        pol_sig_mean = int(np.round(np.mean(sigma_trip)))
                        pol_app_mean = int(np.round(np.mean(app_trip)))
                        min_res_eff = int(min(getattr(args, "rank_min_res", 8), Rmin))
                        max_rank_eff = int(getattr(args, "max_rank", 96))

                        print(f"[rank-up][skip] reason={reason} info={ {k:v for k,v in upgrade_info.items() if k!='skipped'} }")
                        log_event(logfolder, "rank-up-skip", iteration, phase="strict-even", reason=reason,
                                  Rmin=int(Rmin), min_res=min_res_eff,                   
                                  warmup_until=upgrade_info.get("warmup_until"),
                                  cooldown_left=upgrade_info.get("cooldown_left"),
                                  last_resize_iter=upgrade_info.get("last_resize_iter"),
                                  floor_sig=int(base_floor_sig), floor_app=int(base_floor_app),
                                  policy_sig_mean=pol_sig_mean, policy_app_mean=pol_app_mean,
                                  floor_eff_sig=upgrade_info.get("floor_eff_sig"), floor_eff_app=upgrade_info.get("floor_eff_app"),
                                  policy_sigma_raw=upgrade_info.get("policy_sigma_raw"), policy_app_raw=upgrade_info.get("policy_app_raw"),
                                  min_res_eff=upgrade_info.get("min_res_eff"), max_rank=max_rank_eff)

                    if upgraded > 0:
                        print(f"[rank-up] resized → sigma={sigma_trip}, app={app_trip}")
                        
                    optimizer = build_optimizer_with_scale(tensorf, args, cur_scale)
                    last_structure_change_iter = iteration
                    log_event(logfolder, "rank-upgrade", iteration, phase="strict-even",
                              up=int(upgraded), down=0, reso_cur=str(reso_cur))
                
                base_lrs = [g['lr'] for g in optimizer.param_groups]
                warmup_floor = float(getattr(args, "postcrit_warmup_floor", 0.3))
                for g, lr0 in zip(optimizer.param_groups, base_lrs):
                    g['lr'] = lr0 * warmup_floor

                s_iter = iteration
                e_iter = iteration + STRICT_WARMUP_ITERS
                def _maybe_warmup(cur_iter, opt=optimizer, base=base_lrs, floor=warmup_floor, s=s_iter, e=e_iter):
                    if cur_iter <= e:
                        t = (cur_iter - s) / max(1, e - s)
                        t = max(0.0, min(1.0, t))
                        frac = floor + (1.0 - floor) * t
                        if hasattr(tensorf.renderModule, "set_fea_gate"):
                            tensorf.renderModule.set_fea_gate(max(0.9, float(frac)))
                        for pg, lr0 in zip(opt.param_groups, base):
                            pg['lr'] = lr0 * frac
                maybe_warmup_hook = _maybe_warmup
                warmup_until = e_iter

                log_event(logfolder, "strict-even", iteration,
                          prev_n=prev_n, new_n=n_new, 
                          G=str(tensorf.patch_grid_reso), reso_cur=str(reso_cur), 
                          miss_before=f"{miss_b:.2%}", miss_after=f"{miss_a:.2%}")
                last_structure_change_iter = iteration

        # ========== Re-filter rays (annealed mixing; no shrink/prune) ==========
        COOL_AFTER_EVENT = 300  # 300~800 recommended
        safe_to_refilter = True

        if pending_restruct_probation is not None:
            safe_to_refilter = False
        if (iteration - last_structure_change_iter) < COOL_AFTER_EVENT:
            safe_to_refilter = False

        if (not did_refilter) and safe_to_refilter and (getattr(args, "rebucket_at", -1) > 0) \
            and (getattr(tensorf, "alphaMask", None) is not None) and tensorf.alpha_has_signal(1e-5) \
            and (iteration == int(args.rebucket_at)):

                print("========> refiltering rays with current alphaMask (annealed mixing)")
                rays_new, rgbs_new = tensorf.filtering_rays(allrays_old, allrgbs_old,
                                                            N_samples=int(getattr(args, "refilter_samples", 256)),
                                                            bbox_only=False)
                allrays = torch.cat([allrays_old, rays_new], dim=0)
                allrgbs = torch.cat([allrgbs_old, rgbs_new], dim=0)

                ANNEAL_ITERS = int(getattr(args, "refilter_anneal_iters", 1500))
                START_RATIO  = float(getattr(args, "refilter_start_ratio", 0.80))
                trainingSampler = MixedSampler(N_old=allrays_old.shape[0], N_new=rays_new.shape[0], batch=int(args.batch_size),
                                               start_iter=iteration, anneal_iters=ANNEAL_ITERS, start_ratio=START_RATIO,
                                               device="cpu")
                log_event(logfolder, "refilter-start", iteration,
                          old=allrays_old.shape[0], new=rays_new.shape[0],
                          anneal_iters=ANNEAL_ITERS, start_ratio=START_RATIO)

                did_refilter = True

        # ========== Selective-even splits ==========
        SELECTIVE_EVEN_KICKS = list(getattr(args, "split_even_kicks", []))
        HEALTH_DROP_THRES    = float(getattr(args, "split_psnr_drop_thres", 0.3))
        
        if iteration in SELECTIVE_EVEN_KICKS:
            # slight structure rearrangement after a finer res mapping 
            if getattr(tensorf, "_first_ups_done_iter", None) is None:     
                print("[even-selective] gated: need a VM upsample first; skip")
                continue

            if iteration - last_structure_change_iter < args.split_cooldown:
                log_event(logfolder, "even-selective-skip", iteration, reason="cooldown")
                print(f"[even-selective] cooldown skip @ {iteration}")
            elif len(tensorf.patch_map) >= args.patch_cap:
                log_event(logfolder, "even-selective-skip", iteration, reason="PATCH_CAP")
                print(f"[even-selective] reached PATCH_CAP={args.patch_cap}, skip.")
            elif tensorf.get_total_voxels() > VOX_BUDGET:
                log_event(logfolder, "even-selective-skip", iteration, reason="voxels")
                print(f"[even-selective] voxel guard: ({tensorf.get_total_voxels()} > {VOX_BUDGET}), skip.")
            elif (tensorf.get_total_mem()/1024**2) > VRAM_BUDGET_MB:
                log_event(logfolder, "even-selective-skip", iteration, reason="VRAM")
                print(f"[even-selective] VRAM guard: ({tensorf.get_total_mem()/1024**2:.1f} MB > {VRAM_BUDGET_MB} MB), skip.")
            else:
                val = getattr(args, "even_gate_miss_q", None)
                if val is None:
                    val = getattr(args, "heartbeat_target_miss", 0.15)
                TARGET_MISS = float(val)
                cur_miss = getattr(tensorf, "recent_missing_ratio_ema", None)

                if cur_miss is None:
                    cur_miss = getattr(tensorf, "last_missing_ratio", None)
                    if cur_miss is None:
                        try:
                            cur_miss = float(quick_miss_ratio(tensorf))
                        except Exception:
                            cur_miss = 0.0
                try:
                    cur_miss = float(cur_miss)
                except Exception:
                    cur_miss = 0.0

                if cur_miss > TARGET_MISS:
                    log_event(logfolder, "even-selective-skip", iteration,
                              reason="GATE_MISS_Q",
                              miss_q=f"{cur_miss:.4f}", thresh=TARGET_MISS)
                    print(f"[even-selective] gated by coverage: {cur_miss:.2%} > {TARGET_MISS:.2%}")
                    continue

                print("========> Even-phase selective split")
                print(">> Before split patch distribution:")
                r = rays_train
                if r.shape[0] > 8192:
                    idx = torch.randperm(r.shape[0])[:8192]
                    r = r[idx]
                tensorf.log_patch_distribution(r.to(tensorf.aabb.device), iteration)

                pre_snap = None
                pre_psnr_fast = post_psnr_fast = None
                
                if HEALTH_DROP_THRES > 0:
                    pre_snap = {"torch": tensorf.state_dict()} 

                    if hasattr(tensorf, "eval"): tensorf.eval()
                    tensorf_eval_flag = tensorf.repair_enable
                    tensorf.repair_enable = False
                    with torch.no_grad():
                        val_dataset = get_val_dataset(args, train_dataset, test_dataset)
                        pre_psnr_fast = quick_val_psnr_safe(
                            val_dataset, trainer.render, device,
                            target_views=3, target_rpv=4096, 
                            min_views=1, min_rpv=1024, chunk=1024
                        )
                    tensorf.repair_enable = tensorf_eval_flag
                    if hasattr(tensorf, "train"): tensorf.train()

                prev_n = len(tensorf.patch_map)
                miss_b = quick_miss_ratio(tensorf)

                pre_event_batch = int(getattr(args, "batch_size", 2048))  # for probation batch-floor

                if (test_dataset is None) or (not hasattr(test_dataset, "all_rays")) or (not hasattr(test_dataset, "all_rgbs")):
                    _rays_stub = rays_train
                    _rgbs_stub = rgb_train

                    if _rays_stub.dim() > 2:
                        _rays_stub = _rays_stub.view(-1, _rays_stub.shape[-1])
                    if _rgbs_stub.dim() > 2:
                        _rgbs_stub = _rgbs_stub.view(-1, _rgbs_stub.shape[-1])

                    test_dataset_for_critrn = SimpleNamespace(
                        all_rays=[_rays_stub],   # list of [N,6]
                        all_rgbs=[_rgbs_stub],   # list of [N,3]
                    )
                else:
                    test_dataset_for_critrn = test_dataset
                
                prev_patch_keys = set(tensorf.patch_map.keys())
                prev_param_ids  = _param_set(tensorf)
                
                n_split = uneven_critrn(test_dataset_for_critrn, tensorf, reso_cur, args, trainer.render,
                                        step=iteration, device=device)
                tensorf.debug_dump_basis_stats()

                tensorf.patch_grid_reso = tuple(tensorf.infer_patch_grid_reso_from_keys(tensorf.patch_map))
                if hasattr(tensorf, "assert_zero_origin_and_contiguous"):
                    try: tensorf.assert_zero_origin_and_contiguous()
                    except Exception as e: print(f"[WARN] assert_zero_origin_and_contiguous failed: {e}")
                
                added = run_heartbeat(note="even-selective", iter_for_log=iteration, reso_for_log=tuple(reso_cur))
                did_cov_this_step = True
                tensorf.update_alpha_mass_per_patch(n_per=getattr(args, "alpha_mass_n", 1024))
                _normalize_patch_keys(tensorf)
                miss_a = quick_miss_ratio(tensorf)

                append_covlog(logfolder, iteration, "even-selective",
                              miss_before=f"{miss_b:.4f}", miss_after=f"{miss_a:.4f}",
                              added=added, n_patches=len(tensorf.patch_map),
                              G=str(tuple(tensorf.patch_grid_reso)), reso_cur=str(reso_cur))

                print(">> After split patch distribution:")
                r = rays_train
                if r.shape[0] > 8192:
                    idx = torch.randperm(r.shape[0])[:8192]
                    r = r[idx]
                tensorf.log_patch_distribution(r.to(tensorf.aabb.device), iteration)

                # Post-split quick health check → Immediate abort OR start probation
                IMMEDIATE, ALLOW, PROB_ITERS, PROB_FINAL_TOL, PROB_CHECK_EVERY = _prob_cfg("even")

                if (n_split and n_split > 0):
                    if hasattr(tensorf, "eval"): tensorf.eval()
                    tensorf_eval_flag = tensorf.repair_enable
                    tensorf.repair_enable = False
                    with torch.no_grad():
                        val_dataset = get_val_dataset(args, train_dataset, test_dataset)
                        post_psnr_fast = quick_val_psnr_safe(
                            val_dataset, trainer.render, device,
                            target_views=3, target_rpv=4096,
                            min_views=1, min_rpv=1024, chunk=1024
                        )
                    tensorf.repair_enable = tensorf_eval_flag
                    if hasattr(tensorf, "train"): tensorf.train()

                    dpsnr_now = (post_psnr_fast or 0.0) - (pre_psnr_fast or 0.0)

                    if dpsnr_now < -IMMEDIATE:  # immediate rollback
                        restore_model(tensorf, pre_snap, device)
                        _normalize_patch_keys(tensorf)
                        if hasattr(tensorf, "clean_caches"):
                            try: tensorf.clean_caches()
                            except Exception: pass

                        maybe_warmup_hook = None
                        warmup_until = -1

                        log_patch_status(tensorf, iteration, "restore/load")
                        if hasattr(tensorf, "train"): tensorf.train()
                        log_event(logfolder, "even-selective-rollback", iteration,
                                  prev_n=prev_n, new_n=len(tensorf.patch_map),
                                  miss_before=f"{miss_b:.2%}", miss_after=f"{miss_a:.2%}",
                                  dpsnr=f"{dpsnr_now:+.4f}dB", reason="immediate_abort")
                        print(f"[even-selective] immediate rollback @ {iteration}: ΔPSNR={dpsnr_now:+.4f} dB")
                        continue
                    
                    if args.dynamic_rank and hasattr(tensorf, "rank_policy"):
                        rebuild_opt = False
                        tensorf.update_alpha_mass_per_patch(n_per=getattr(args, "alpha_mass_n", 1024))
                        
                        sigma_trip, app_trip = tensorf.rank_policy(tuple(reso_cur))
                        base_floor_sig, base_floor_app = _coalesce_base_floors(args, tensorf)
                        c_sig_base = max(base_floor_sig, int(np.round(np.mean(sigma_trip))))
                        c_app_base = max(base_floor_app, int(np.round(np.mean(app_trip))))

                        Rmin = int(min(reso_cur))
                        print(f"[rank-up precheck] Rmin={Rmin} rank_min_res={getattr(args,'rank_min_res',8)}")

                        tensorf._last_rank_resize_iter = None
                        upgrade_info = tensorf.try_rank_upgrade(args, reso_hint=tuple(reso_cur), iteration=iteration)
                        upgraded = int(upgrade_info.get("up", 0)) 

                        if upgraded == 0 and "skipped" in upgrade_info: 
                            reason = str(upgrade_info.get("skipped"))
                            pol_sig_mean = int(np.round(np.mean(sigma_trip)))
                            pol_app_mean = int(np.round(np.mean(app_trip)))
                            min_res_eff = int(min(getattr(args, "rank_min_res", 8), Rmin))
                            max_rank_eff = int(getattr(args, "max_rank", 96))

                            print(f"[rank-up][skip] reason={reason} info={ {k:v for k,v in upgrade_info.items() if k!='skipped'} }")
                            log_event(logfolder, "rank-up-skip", iteration, phase="even-selective", reason=reason,
                                      Rmin=int(Rmin), min_res=min_res_eff,                   
                                      warmup_until=upgrade_info.get("warmup_until"),
                                      cooldown_left=upgrade_info.get("cooldown_left"),
                                      last_resize_iter=upgrade_info.get("last_resize_iter"),
                                      floor_sig=int(base_floor_sig), floor_app=int(base_floor_app),
                                      policy_sig_mean=pol_sig_mean, policy_app_mean=pol_app_mean,
                                      floor_eff_sig=upgrade_info.get("floor_eff_sig"), floor_eff_app=upgrade_info.get("floor_eff_app"),
                                      policy_sigma_raw=upgrade_info.get("policy_sigma_raw"), policy_app_raw=upgrade_info.get("policy_app_raw"),
                                      min_res_eff=upgrade_info.get("min_res_eff"), max_rank=max_rank_eff)
                        
                        if upgraded > 0:
                            print(f"[rank-up] resized → sigma={sigma_trip}, app={app_trip}")
                            rebuild_opt = True
                        
                        ALLOW_RANK_DOWN = (iteration >= int(getattr(args, "rank_down_after_iter", 12000))) and (pending_restruct_probation is None)
                        ALLOW_RANK_UP = (pending_restruct_probation is not None)
                        DO_AUTOSCALE = ALLOW_RANK_UP or _should_run_autoscale(iteration, args, warmup_until=warmup_until,
                                                                              last_structure_change_iter=last_structure_change_iter,
                                                                              pending_probation=pending_restruct_probation)
                        
                        ref_res = int(getattr(args, "vm_reso_max", 64))
                        max_rank = int(getattr(args, "max_rank", 64))
                        scale_gamma = float(getattr(args, "rank_autoscale_gamma", 0.6))
                        alpha_keep_q = float(getattr(args, "rank_autoscale_alpha_keep_q", 0.85))
                        
                        if DO_AUTOSCALE:
                            print(f"[rank-auto] ref_res={ref_res} gamma={scale_gamma} keep_q={alpha_keep_q}")

                            changed = tensorf.selective_rank_autoscale(ref_res=ref_res, gamma=scale_gamma,
                                                                       c_sig_base=c_sig_base, c_app_base=c_app_base,  
                                                                       c_min=8, c_max=max_rank, round_to=4, 
                                                                       allow_down=ALLOW_RANK_DOWN, allow_up=ALLOW_RANK_UP,                       
                                                                       alpha_keep_q=alpha_keep_q, verbose=True)
                            if changed > 0:
                                print(f"[rank-auto] resized ranks: changed={changed}")
                                rebuild_opt = True
                        else:
                            changed = 0
                            
                        if rebuild_opt:
                            optimizer = build_optimizer_with_scale(tensorf, args, cur_scale)
                            if getattr(args, "kd_ema_enable", False):
                                teacher = _build_ema_from(tensorf)
                            last_structure_change_iter = iteration
                        log_event(logfolder, "rank-resize", iteration, phase="even-selective",
                                  up=int(upgraded), down=int(changed), reso_cur=str(reso_cur),
                                  base_floor_sig=int(base_floor_sig), base_floor_app=int(base_floor_app),
                                  policy_sig=str(list(map(int, sigma_trip))),  
                                  policy_app=str(list(map(int, app_trip))),
                                  c_sig_base=int(c_sig_base), c_app_base=int(c_app_base),
                                  allow_down=bool(ALLOW_RANK_DOWN),
                                  alpha_keep_q=float(alpha_keep_q))

                    if dpsnr_now <= -ALLOW:
                        pending_restruct_probation = {"phase": "even",
                                                      "iter_applied": iteration,
                                                      "pre_psnr_fast": float(pre_psnr_fast or 0.0),
                                                      "pre_snap": pre_snap,
                                                      "prev_lsc_iter": int(last_structure_change_iter),
                                                      "prev_last_even_iter": int(last_even_selective_iter)}
                        
                        bs_prob_floor = max(int(pre_event_batch), int(getattr(args, "batch_size", pre_event_batch)))
                        bs_prob_until = int(iteration + PROB_ITERS)
                        print(f"[probation] batch floor set to {bs_prob_floor} until iter {bs_prob_until} (phase=even-selective)")
                        
                        log_event(logfolder, "probation-start", iteration,
                                  phase="even-selective", dpsnr=f"{dpsnr_now:+.4f}dB",
                                  allow_dB=ALLOW, abort_dB=IMMEDIATE,
                                  iters=PROB_ITERS, final_tol_dB=PROB_FINAL_TOL,
                                  check_every=PROB_CHECK_EVERY)

                    n_new = len(tensorf.patch_map)
                    cur_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
                    raw_split_scale = (prev_n / max(1, n_new)) ** SPLIT_LR_POW
                    split_scale = max(SPLIT_LR_MIN, min(1.0, raw_split_scale))
                    eff_scale = cur_scale * split_scale
                    print(f"[even-selective lr] old={prev_n} new={n_new} | cur={cur_scale:.4g} * split={split_scale:.3f} -> eff={eff_scale:.4g}")
                    
                    warmup_iters = int(getattr(args, "postcrit_warmup_iters", 200))
                    warmup_floor = float(getattr(args, "postcrit_warmup_floor", 0.5))
                    base_lrs = [g['lr'] for g in optimizer.param_groups]
                    for g, lr0 in zip(optimizer.param_groups, base_lrs):
                        g['lr'] = lr0 * warmup_floor
                    s_iter, e_iter = iteration, iteration + warmup_iters
                    def _maybe_warmup(cur_iter, opt=optimizer, base=base_lrs, floor=warmup_floor, s=s_iter, e=e_iter):
                        if cur_iter <= e:
                            t = (cur_iter - s) / max(1, e - s)
                            t = max(0.0, min(1.0, t))
                            frac = floor + (1.0 - floor) * t
                            if hasattr(tensorf.renderModule, "set_fea_gate"):
                                tensorf.renderModule.set_fea_gate(max(0.9, float(frac)))
                            for pg, lr0 in zip(opt.param_groups, base):
                                pg['lr'] = lr0 * frac
                    maybe_warmup_hook = _maybe_warmup
                    warmup_until = e_iter

                    log_event(logfolder, "even-selective", iteration,
                              prev_n=prev_n, new_n=n_new, n_split=n_split,
                              G=str(tensorf.patch_grid_reso), reso_cur=str(reso_cur),
                              miss_before=f"{miss_b:.2%}", miss_after=f"{miss_a:.2%}",
                              dpsnr=f"{(post_psnr_fast or 0)-(pre_psnr_fast or 0):+.4f}dB")

                    if not (pending_restruct_probation is not None
                            and pending_restruct_probation.get("phase") == "even"
                            and int(pending_restruct_probation.get("iter_applied", -1)) == iteration):
                        last_structure_change_iter = iteration
                        last_even_selective_iter = iteration

                    run_heartbeat(note="post-even-split", iter_for_log=iteration, reso_for_log=tuple(reso_cur))
                    did_cov_this_step = True
                    _normalize_patch_keys(tensorf)
                    maybe_softprune(note="post-even-split")

                maybe_viz_patches(iteration, "even-selective") 

                do_full = (getattr(args, "N_vis", 0) > 0) and (getattr(args, "vis_every", 0) > 0) and (((iteration + 1) % args.vis_every) == 0)
                if do_full and (test_dataset is not None):
                    if hasattr(tensorf, "eval"): tensorf.eval()
                    tensorf_eval_flag = tensorf.repair_enable
                    tensorf.repair_enable = False
                    with torch.no_grad():
                        PSNRs_test = evaluation(
                            test_dataset, tensorf, args, trainer.render, f'{logfolder}/imgs_vis/',
                            N_vis=args.N_vis, prtx=f'{iteration:06d}_', N_samples=-1,
                            white_bg=white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False
                        )
                    tensorf.repair_enable = tensorf_eval_flag
                    if hasattr(tensorf, "train"): tensorf.train()
                    psnr_full = float(np.mean(PSNRs_test))
                    summary_writer.add_scalar('test/psnr', psnr_full, iteration)
                    log_rd(iteration, note="even_selective_full", psnr_full_val=psnr_full)
                else:
                    if hasattr(tensorf, "eval"): tensorf.eval()
                    tensorf_eval_flag = tensorf.repair_enable
                    tensorf.repair_enable = False
                    with torch.no_grad():
                        val_dataset = get_val_dataset(args, train_dataset, test_dataset)
                        psnr_fast = quick_val_psnr_safe(
                            val_dataset, trainer.render, device,
                            target_views=3, target_rpv=4096,  
                            min_views=1, min_rpv=1024, chunk=1024
                        )
                    tensorf.repair_enable = tensorf_eval_flag
                    if hasattr(tensorf, "train"): tensorf.train()
                    summary_writer.add_scalar('test/psnr_fast', psnr_fast, iteration)
                    log_rd(iteration, note="even_selective_fast", psnr_fast_val=psnr_fast)

        # ========== Uneven phase: increase VM res ==========
        VM_MAX = int(getattr(args, "vm_reso_max", 256)) 

        if iteration in args.vm_upsamp_list:
            reso_target = [min(r * 2, VM_MAX) for r in reso_cur]

            if reso_target == reso_cur:
                log_event(logfolder, "vm-ups-skip", iteration, reason="VM_MAX", reso_cur=str(reso_cur), target=str(reso_target))
                print("[vm upsample] hit VM_MAX; skip.")
            elif len(tensorf.patch_map) >= args.patch_cap:
                log_event(logfolder, "vm-ups-skip", iteration, reason="PATCH_CAP")
                print(f"[vm upsample] reached PATCH_CAP={args.patch_cap}, skip.")
            elif tensorf.get_total_voxels() > VOX_BUDGET:
                log_event(logfolder, "vm-ups-skip", iteration, reason="voxels", vox_now=tensorf.get_total_voxels(), vox_budget=VOX_BUDGET)
                print(f"[vm upsample] voxel guard: ({tensorf.get_total_voxels()} > {VOX_BUDGET}), skip.")
            elif (tensorf.get_total_mem()/1024**2) > VRAM_BUDGET_MB:
                log_event(logfolder, "vm-ups-skip", iteration, reason="VRAM", mem_now_MB=f"{tensorf.get_total_mem()/1024**2:.1f}", mem_budget_MB=VRAM_BUDGET_MB)
                print(f"[vm upsample] VRAM guard: ({tensorf.get_total_mem()/1024**2:.1f} MB > {VRAM_BUDGET_MB} MB), skip.")
            else:
                SPLIT_TO_UPS_COOLDOWN = int(getattr(args, "ups_cooldown_after_split", 800))
                MISS_Q_THRESH = float(getattr(args, "ups_gate_miss_q", 0.08))
                MIN_PATCH_FACTOR = float(getattr(args, "ups_gate_min_patch_factor", 2.0))

                last_split_iter = last_even_selective_iter

                def _coverage_ok(model, q_thresh=MISS_Q_THRESH):
                    q = getattr(model, "recent_missing_ratio_ema", None)
                    if q is None:
                        q = float(getattr(model, "last_missing_ratio", 0.0))
                    try:
                        q = float(q)
                    except Exception:
                        q = 0.0
                    return q < q_thresh

                def _enough_patches(model, target_G, factor=MIN_PATCH_FACTOR):
                    cur_n = int(getattr(model, "num_patches", len(getattr(model, "patch_map", {}))))
                    need = int(target_G[0] * target_G[1] * target_G[2] * factor)
                    return cur_n >= need

                ALLOW_UPS = True
                # gate 1: split -> ups cooldown
                if last_split_iter is not None and (iteration - int(last_split_iter)) < SPLIT_TO_UPS_COOLDOWN:
                    ALLOW_UPS = False
                    log_event(logfolder, "vm-ups-skip", iteration, reason="GATE_COOLDOWN",
                            delta=int(iteration - int(last_split_iter)), need=SPLIT_TO_UPS_COOLDOWN)
                # gate 2: coverage
                if ALLOW_UPS and not _coverage_ok(tensorf):
                    ALLOW_UPS = False
                    log_event(logfolder, "vm-ups-skip", iteration, reason="GATE_MISS_Q",
                            miss_q=float(getattr(tensorf, "recent_missing_ratio_ema",
                                                getattr(tensorf, "last_missing_ratio", 0.0))),
                            thresh=MISS_Q_THRESH)
                # gate 3: enough patches
                target_G = tuple(getattr(args, "strict_even_target_G", [3, 3, 3]))
                _first = getattr(tensorf, "_first_ups_done_iter", None) is None
                factor_for_this = 1.0 if _first else MIN_PATCH_FACTOR

                if ALLOW_UPS and not _enough_patches(tensorf, target_G=target_G, factor=factor_for_this):
                    ALLOW_UPS = False
                    cur_n = int(getattr(tensorf, "num_patches", len(getattr(tensorf, "patch_map", {}))))
                    need_n = int(target_G[0]*target_G[1]*target_G[2]*factor_for_this)
                    log_event(logfolder, "vm-ups-skip", iteration, reason="GATE_PATCHES",
                              cur=cur_n, need=need_n, target_G=str(target_G))

                if not ALLOW_UPS:
                    print("[vm upsample] gated off; skip this round.")
                else:
                    _ = event_full_recover(tensorf, note="pre-VM-upsample", iteration=iteration, reso_cur=reso_cur)
                    did_cov_this_step = True
                    _normalize_patch_keys(tensorf)

                    tensorf.update_alpha_mass_per_patch(n_per=getattr(args, "alpha_mass_n", 1024))
                    min_k = int(getattr(args, "perpatch_ups_min_k", 8))
                    max_k = int(getattr(args, "perpatch_ups_max_k", 64))
                    ratio = float(getattr(args, "perpatch_ups_topk_ratio", 0.20))
                    topk_arg = getattr(args, "perpatch_ups_topk", None)  # never use fixed top-K by default
                    topk = int(topk_arg) if topk_arg is not None else None

                    before_res = sorted({tuple(p['res']) for p in tensorf.patch_map.values()})

                    # top-K selected by certain ratio 
                    sel_keys, eligible = tensorf.select_patches_by_alpha_mass(tuple(reso_target), ratio=ratio, topk=topk, min_k=min_k, max_k=max_k)
                    log_event(logfolder, "vm-ups-select", iteration,
                              topk=(int(topk_arg) if topk_arg is not None else -1),
                              ratio=ratio, eligible=eligible, picked=len(sel_keys),
                              res_before=str(before_res),
                              target=str(reso_target))

                    if len(sel_keys) == 0:
                        print("[per-patch upsample] no eligible patches this round; skip.")
                    else:
                        if hasattr(tensorf, "eval"): tensorf.eval()
                        tensorf_eval_flag = tensorf.repair_enable
                        tensorf.repair_enable = False
                        with torch.no_grad():
                            val_dataset = get_val_dataset(args, train_dataset, test_dataset)
                            psnr_fast = quick_val_psnr_safe(
                                val_dataset, trainer.render, device,
                                target_views=3, target_rpv=4096, 
                                min_views=1, min_rpv=1024, chunk=1024
                            )
                        tensorf.repair_enable = tensorf_eval_flag
                        if hasattr(tensorf, "train"): tensorf.train()
                        log_rd(iteration, note="pre_upsample", psnr_fast_val=psnr_fast)
                        
                        # snapshot for delayed rollback (VM upsample)
                        pre_snap       = {"torch": tensorf.state_dict()}
                        prev_lsc_iter  = int(last_structure_change_iter)
                        prev_reso_cur  = list(reso_cur)
                        pre_psnr_fast  = float(psnr_fast or 0.0)  # reuse pre-UPS quick PSNR

                        vox_before = tensorf.get_total_voxels()
                        mem_before = int(tensorf.get_total_mem()/1024**2)

                        pre_event_batch = int(getattr(args, "batch_size", 2048))

                        # actually upgrade res in patches: top-K alpha-mass
                        adjust_batch_size(reso_target, args)
                        n_promoted = tensorf.upsample_patches(sel_keys, tuple(reso_target), mode="bilinear", align_corners=False, verbose=True)

                        _ = event_full_recover(tensorf, note="post-VM-upsample", iteration=iteration, reso_cur=reso_cur)
                        did_cov_this_step = True
                        _normalize_patch_keys(tensorf)

                        after_res = sorted({tuple(p['res']) for p in tensorf.patch_map.values()})

                        log_event(logfolder, "vm-ups-apply", iteration, promoted=n_promoted,
                                  vox_before=vox_before, vox_after=tensorf.get_total_voxels(),
                                  mem_before_MB=mem_before, mem_after_MB=int(tensorf.get_total_mem()/1024**2),
                                  res_after=str(after_res))
                        
                        topk_disp = topk if topk is not None else f"ratio*{ratio:.2f}[{min_k},{max_k}]"
                        print(f"[sanity] vm_upsample@{iteration} res_before={before_res} res_after={after_res} "
                              f"selected={len(sel_keys)} upgraded={n_promoted} topk={topk_disp}")

                        if n_promoted > 0:
                            # Post-UPS quick health check → Immediate abort OR start probation
                            IMMEDIATE, ALLOW, PROB_ITERS, PROB_FINAL_TOL, PROB_CHECK_EVERY = _prob_cfg("ups")

                            if hasattr(tensorf, "eval"): tensorf.eval()
                            tensorf_eval_flag = tensorf.repair_enable
                            tensorf.repair_enable = False
                            with torch.no_grad():
                                val_dataset = get_val_dataset(args, train_dataset, test_dataset)
                                post_psnr_fast = quick_val_psnr_safe(
                                    val_dataset, trainer.render, device,
                                    target_views=3, target_rpv=4096,
                                    min_views=1, min_rpv=1024, chunk=1024
                                )
                            tensorf.repair_enable = tensorf_eval_flag
                            if hasattr(tensorf, "train"): tensorf.train()

                            dpsnr_now = (post_psnr_fast or 0.0) - (pre_psnr_fast or 0.0)

                            if dpsnr_now < -IMMEDIATE:  # immediate rollback
                                restore_model(tensorf, pre_snap, device)                     
                                _normalize_patch_keys(tensorf)
                                if hasattr(tensorf, "clean_caches"):
                                    try: tensorf.clean_caches()
                                    except Exception: pass

                                maybe_warmup_hook = None
                                warmup_until = -1

                                log_patch_status(tensorf, iteration, "restore/load")
                                if hasattr(tensorf, "train"): tensorf.train()
                                log_event(logfolder, "vm-upsample-rollback", iteration,
                                        prev_n=prev_n, new_n=len(tensorf.patch_map),
                                        miss_before=f"{miss_b:.2%}", miss_after=f"{miss_a:.2%}",
                                        dpsnr=f"{(post_psnr_fast or 0)-(pre_psnr_fast or 0):+.4f}dB", reason="immediate_abort")
                                print(f"[vm-upsample] immediate rollback @ {iteration}: ΔPSNR={(post_psnr_fast or 0)-(pre_psnr_fast or 0):+.4f} dB")
                                continue

                            if dpsnr_now <= -ALLOW:
                                pending_restruct_probation = {"phase": "ups",
                                                              "iter_applied": iteration,
                                                              "pre_psnr_fast": float(pre_psnr_fast or 0.0),
                                                              "pre_snap": pre_snap,
                                                              "prev_lsc_iter": prev_lsc_iter,
                                                              "prev_reso_cur": list(prev_reso_cur)}
                                
                                bs_prob_floor = max(int(pre_event_batch), int(getattr(args, "batch_size", pre_event_batch)))
                                bs_prob_until = int(iteration + PROB_ITERS)
                                print(f"[probation] batch floor set to {bs_prob_floor} until iter {bs_prob_until} (phase=vm-upsample)")
                                
                                log_event(logfolder, "probation-start", iteration,
                                          phase="vm-upsample", dpsnr=f"{dpsnr_now:+.4f}dB",
                                          allow_dB=ALLOW, abort_dB=IMMEDIATE,
                                          iters=PROB_ITERS, final_tol_dB=PROB_FINAL_TOL,
                                          check_every=PROB_CHECK_EVERY)

                            reso_cur = reso_target
                            summary_writer.add_scalar('events/vm_promoted', n_promoted, iteration)
                            log_event(logfolder, "perpatch-ups", iteration, promoted=len(sel_keys), target_res=tuple(reso_cur))
                            with open(os.path.join(logfolder, 'vm_upsampling_reso.txt'), 'a') as f:
                                f.write(f"Iter {iteration} per-patch upsample → {reso_cur}\n")
                            
                            cur_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)

                            if args.dynamic_rank and hasattr(tensorf, "rank_policy"):
                                rebuild_opt = False
                                tensorf.update_alpha_mass_per_patch(n_per=getattr(args, "alpha_mass_n", 1024))
                                
                                sigma_trip, app_trip = tensorf.rank_policy(tuple(reso_cur))
                                base_floor_sig, base_floor_app = _coalesce_base_floors(args, tensorf)
                                c_sig_base = max(base_floor_sig, int(np.round(np.mean(sigma_trip))))
                                c_app_base = max(base_floor_app, int(np.round(np.mean(app_trip))))

                                Rmin = int(min(reso_cur))
                                print(f"[rank-up precheck] Rmin={Rmin} rank_min_res={getattr(args,'rank_min_res',8)}")

                                # must promote ranks as VM res increases
                                tensorf._last_rank_resize_iter = None
                                upgrade_info = tensorf.try_rank_upgrade(args, reso_hint=tuple(reso_cur), iteration=iteration)
                                upgraded = int(upgrade_info.get("up", 0)) 

                                if upgraded == 0 and "skipped" in upgrade_info: 
                                    reason = str(upgrade_info.get("skipped"))
                                    pol_sig_mean = int(np.round(np.mean(sigma_trip)))
                                    pol_app_mean = int(np.round(np.mean(app_trip)))
                                    min_res_eff = int(min(getattr(args, "rank_min_res", 8), Rmin))
                                    max_rank_eff = int(getattr(args, "max_rank", 96))

                                    print(f"[rank-up][skip] reason={reason} info={ {k:v for k,v in upgrade_info.items() if k!='skipped'} }")
                                    log_event(logfolder, "rank-up-skip", iteration, phase="vm-upsample", reason=reason,
                                              Rmin=int(Rmin), min_res=min_res_eff,                   
                                              warmup_until=upgrade_info.get("warmup_until"),
                                              cooldown_left=upgrade_info.get("cooldown_left"),
                                              last_resize_iter=upgrade_info.get("last_resize_iter"),
                                              floor_sig=int(base_floor_sig), floor_app=int(base_floor_app),
                                              policy_sig_mean=pol_sig_mean, policy_app_mean=pol_app_mean,
                                              floor_eff_sig=upgrade_info.get("floor_eff_sig"), floor_eff_app=upgrade_info.get("floor_eff_app"),
                                              policy_sigma_raw=upgrade_info.get("policy_sigma_raw"), policy_app_raw=upgrade_info.get("policy_app_raw"),
                                              min_res_eff=upgrade_info.get("min_res_eff"), max_rank=max_rank_eff)

                                if upgraded > 0:
                                    print(f"[rank-up] resized → sigma={sigma_trip}, app={app_trip}")
                                    rebuild_opt = True
                                
                                ALLOW_RANK_DOWN = (iteration >= int(getattr(args, "rank_down_after_iter", 12000))) and (pending_restruct_probation is None)
                                ALLOW_RANK_UP = (pending_restruct_probation is not None)
                                DO_AUTOSCALE = ALLOW_RANK_UP or _should_run_autoscale(iteration, args, warmup_until=warmup_until,
                                                                                      last_structure_change_iter=last_structure_change_iter,
                                                                                      pending_probation=pending_restruct_probation)
                                
                                ref_res = int(getattr(args, "vm_reso_max", 64))
                                max_rank = int(getattr(args, "max_rank", 64))
                                scale_gamma = float(getattr(args, "rank_autoscale_gamma", 0.6))
                                alpha_keep_q = float(getattr(args, "rank_autoscale_alpha_keep_q", 0.85))
                                
                                if DO_AUTOSCALE:
                                    print(f"[rank-auto] ref_res={ref_res} gamma={scale_gamma} keep_q={alpha_keep_q}")

                                    changed = tensorf.selective_rank_autoscale(ref_res=ref_res, gamma=scale_gamma,
                                                                               c_sig_base=c_sig_base, c_app_base=c_app_base,  
                                                                               c_min=8, c_max=max_rank, round_to=4, 
                                                                               allow_down=ALLOW_RANK_DOWN, allow_up=False,                       
                                                                               alpha_keep_q=alpha_keep_q, verbose=True)
                                    if changed > 0:
                                        print(f"[rank-auto] resized ranks: changed={changed}")
                                        rebuild_opt = True
                                else:
                                    changed = 0
                                    
                                if rebuild_opt:
                                    optimizer = build_optimizer_with_scale(tensorf, args, cur_scale)
                                    if getattr(args, "kd_ema_enable", False):
                                        teacher = _build_ema_from(tensorf)
                                    last_structure_change_iter = iteration
                                log_event(logfolder, "rank-resize", iteration, phase="vm-upsample",
                                          up=int(upgraded), down=int(changed), reso_cur=str(reso_cur),
                                          base_floor_sig=int(base_floor_sig), base_floor_app=int(base_floor_app),
                                          policy_sig=str(list(map(int, sigma_trip))),  
                                          policy_app=str(list(map(int, app_trip))),
                                          c_sig_base=int(c_sig_base), c_app_base=int(c_app_base),
                                          allow_down=bool(ALLOW_RANK_DOWN),
                                          alpha_keep_q=float(alpha_keep_q))

                                if getattr(args, "seam_lowrank_enable", False):
                                    prev_ids = _param_set(tensorf)  
                                    wired = tensorf.init_seam_lowrank(
                                        rank_sigma=args.seam_rank_sigma,
                                        rank_app=args.seam_rank_app,
                                        scope=args.seam_lowrank_scope
                                    )
                                    if wired:
                                        new_params = _collect_new_params(tensorf, prev_ids)
                                        special = set(map(id, new_params))
                                        base_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
                                        child_mult = float(getattr(args, "child_lr_boost_mult", 1.5))
                                        optimizer = rebuild_optimizer_with_child_boost(
                                            tensorf, args, base_scale=base_scale,
                                            child_params=new_params, child_mult=child_mult
                                        )

                            tensorf._first_ups_done_iter = getattr(tensorf, "_first_ups_done_iter", None) or iteration

                            warmup_iters = int(getattr(args, "postcrit_warmup_iters", 300))
                            warmup_floor = float(getattr(args, "postcrit_warmup_floor", 0.3))
                            base_lrs = [g['lr'] for g in optimizer.param_groups]
                            for g, lr0 in zip(optimizer.param_groups, base_lrs):
                                g['lr'] = lr0 * warmup_floor

                            s_iter, e_iter = iteration, iteration + warmup_iters
                            def _maybe_warmup(cur_iter, opt=optimizer, base=base_lrs, floor=warmup_floor, s=s_iter, e=e_iter):
                                if cur_iter <= e:
                                    t = (cur_iter - s) / max(1, e - s)
                                    t = max(0.0, min(1.0, t))
                                    frac = floor + (1.0 - floor) * t
                                    if hasattr(tensorf.renderModule, "set_fea_gate"):
                                        tensorf.renderModule.set_fea_gate(max(0.9, float(frac)))
                                    for pg, lr0 in zip(opt.param_groups, base):
                                        pg['lr'] = lr0 * frac
                            maybe_warmup_hook = _maybe_warmup
                            warmup_until = e_iter

                            log_patch_status(tensorf, iteration, "upsample")
                            unique_res = {tuple(p['res']) for p in tensorf.patch_map.values()}
                            print(f"[INFO] Current per-patch VM resolutions: {unique_res}; total voxels: {tensorf.get_total_voxels()}")
                            maybe_viz_patches(iteration, "upsample")

                        else:
                            print("[per-patch upsample] selected but 0 promoted (unexpected); skip LR rebuild.")

        # ========== Post-uneven split ==========
        """
        Do not apply this by default. Structure refinement process is basically handled by last 2 phases.
        If REALLY needs, only slowly split and auto-downscale ranks to further optimize structure.
        """
        _offset = int(getattr(args, "split_after_upsamp_offset", 0))
        POST_VM_SPLIT_ITERS = set(int(u) + _offset for u in args.vm_upsamp_list)

        if args.postcrit_apply and (iteration in POST_VM_SPLIT_ITERS):
            if iteration - last_structure_change_iter < args.split_cooldown:
                print(f"[uneven split] cooldown skip @ {iteration}")
            elif len(tensorf.patch_map) >= args.patch_cap:
                print(f"[uneven split] reached PATCH_CAP={args.patch_cap}, skip.")
            elif tensorf.get_total_voxels() > VOX_BUDGET:
                print(f"[uneven split] voxel guard: ({tensorf.get_total_voxels()} > {VOX_BUDGET}), skip.")
            elif (tensorf.get_total_mem()/1024**2) > VRAM_BUDGET_MB:
                print(f"[uneven split] VRAM guard: ({tensorf.get_total_mem()/1024**2:.1f} MB > {VRAM_BUDGET_MB} MB), skip.")
            else:
                print("========> Post-VM split (health-check + warmup)")
                print(">> Before criterion patch distribution:")
                tensorf.log_patch_distribution(rays_train.to(tensorf.aabb.device), iteration)

                heartbeat_guard_coverage(tensorf, target_miss=0.10, seed_cells=8, note="pre-postVM-split")

                # late-stage convergence processing if needed
                MISS_Q = float(getattr(args, "postcrit_gate_miss_q", 0.05))  
                PLATEAU_STEPS = int(getattr(args, "postcrit_plateau_steps", 800)) 
                PLATEAU_EPS = float(getattr(args, "postcrit_plateau_eps", 0.05))   

                q = getattr(tensorf, "recent_missing_ratio_ema",
                            getattr(tensorf, "last_missing_ratio", 0.0))
                try:
                    q = float(q)
                except Exception:
                    q = 0.0

                plateau_ok = False
                if len(PSNRs) >= PLATEAU_STEPS:
                    delta = np.diff(PSNRs[-PLATEAU_STEPS:])
                    plateau_ok = (abs(delta.mean()) < PLATEAU_EPS)

                if (q > MISS_Q) or (not plateau_ok):
                    log_event(logfolder, "post-split-skip", iteration,
                              reason="gate",
                              miss_q=f"{q:.4f}", miss_q_th=MISS_Q,
                              plateau_ok=bool(plateau_ok),
                              plateau_steps=PLATEAU_STEPS, plateau_eps=PLATEAU_EPS)
                    print(f"[post-VM split] gated: miss_q={q:.2%} (th={MISS_Q:.2%}), plateau_ok={plateau_ok}")
                    continue
                
                pre_snap = {"torch": tensorf.state_dict()} 

                if (iteration % 1000) == 0:          
                    run_heartbeat(note="pre-eval", iter_for_log=iteration, reso_for_log=tuple(reso_cur))
                    did_cov_this_step = True
                    _normalize_patch_keys(tensorf)
                
                if hasattr(tensorf, "eval"): tensorf.eval()
                tensorf_eval_flag = tensorf.repair_enable
                tensorf.repair_enable = False
                with torch.no_grad():
                    val_dataset = get_val_dataset(args, train_dataset, test_dataset)
                    pre_psnr_fast = quick_val_psnr_safe(
                        val_dataset, trainer.render, device,
                        target_views=3, target_rpv=4096,  
                        min_views=1, min_rpv=1024, chunk=1024
                    )
                tensorf.repair_enable = tensorf_eval_flag
                if hasattr(tensorf, "train"): tensorf.train()

                miss_b = quick_miss_ratio(tensorf)
                prev_n = len(tensorf.patch_map)
                setattr(tensorf, "_n_patches_before_split", prev_n)

                if (test_dataset is None) or (not hasattr(test_dataset, "all_rays")) or (not hasattr(test_dataset, "all_rgbs")):
                    _rays_stub = rays_train
                    _rgbs_stub = rgb_train

                    if _rays_stub.dim() > 2:
                        _rays_stub = _rays_stub.view(-1, _rays_stub.shape[-1])
                    if _rgbs_stub.dim() > 2:
                        _rgbs_stub = _rgbs_stub.view(-1, _rgbs_stub.shape[-1])

                    test_dataset_for_critrn = SimpleNamespace(
                        all_rays=[_rays_stub],   # list of [N,6]
                        all_rgbs=[_rgbs_stub],   # list of [N,3]
                    )
                else:
                    test_dataset_for_critrn = test_dataset
                
                n_split = uneven_critrn(test_dataset_for_critrn, tensorf, reso_cur, args, trainer.render,
                                        step=iteration, device=device)
                tensorf.debug_dump_basis_stats()

                run_heartbeat(note="post-VM-split", iter_for_log=iteration, reso_for_log=tuple(reso_cur))
                did_cov_this_step = True
                did_cov_this_step = True
                tensorf.update_alpha_mass_per_patch(n_per=getattr(args, "alpha_mass_n", 1024))
                miss_a = quick_miss_ratio(tensorf)
                append_covlog(logfolder, iteration, f"postVM-split",
                                miss_before=f"{miss_b:.4f}", miss_after=f"{miss_a:.4f}",
                                added=n_split, n_patches=len(tensorf.patch_map),
                                G=str(tuple(tensorf.patch_grid_reso)), reso_cur=str(tuple(reso_cur)))

                run_heartbeat(note="post-VM-criterion", iter_for_log=iteration, reso_for_log=tuple(reso_cur))
                did_cov_this_step = True
                _normalize_patch_keys(tensorf)
                _ = heartbeat_guard_coverage(tensorf, target_miss=0.10, seed_cells=8, note="post-VM-criterion", 
                                                iter_for_log=iteration, reso_for_log=tuple(reso_cur))
                log_patch_status(tensorf, iteration, "post-VM-criterion")
                tensorf.update_alpha_mass_per_patch(n_per=getattr(args, "alpha_mass_n", 1024))

                print(">> After criterion patch distribution:")
                tensorf.log_patch_distribution(rays_train.to(tensorf.aabb.device), iteration)

                unhealthy, post_psnr_fast = False, None
                if n_split and n_split > 0:
                    if (iteration % 1000) == 0:          
                        run_heartbeat(note="pre-eval", iter_for_log=iteration, reso_for_log=tuple(reso_cur))
                        did_cov_this_step = True
                        _normalize_patch_keys(tensorf)

                    if hasattr(tensorf, "eval"): tensorf.eval()
                    tensorf_eval_flag = tensorf.repair_enable
                    tensorf.repair_enable = False
                    with torch.no_grad():
                        val_dataset = get_val_dataset(args, train_dataset, test_dataset)
                        post_psnr_fast = quick_val_psnr_safe(
                            val_dataset, trainer.render, device,
                            target_views=3, target_rpv=4096,  
                            min_views=1, min_rpv=1024, chunk=1024
                        )
                    tensorf.repair_enable = tensorf_eval_flag
                    if hasattr(tensorf, "train"): tensorf.train()
                    if (pre_psnr_fast is not None) and (post_psnr_fast is not None):
                        if (post_psnr_fast - pre_psnr_fast) < -0.3:
                            unhealthy = True

                if unhealthy:  # immediate rollback
                    restore_model(tensorf, pre_snap, device)
                    _normalize_patch_keys(tensorf)
                    if hasattr(tensorf, "clean_caches"):
                        try: tensorf.clean_caches()
                        except Exception: pass

                    maybe_warmup_hook = None
                    warmup_until = -1

                    log_patch_status(tensorf, iteration, "restore/load")
                    if hasattr(tensorf, "train"): tensorf.train()
                    log_event(logfolder, "uneven-split-rollback", iteration,
                                prev_n=prev_n, new_n=len(tensorf.patch_map),
                                miss_before=f"{miss_b:.2%}", miss_after=f"{miss_a:.2%}",
                                dpsnr=f"{dpsnr_now:+.4f}dB", reason="immediate_abort")
                    print(f"[uneven-split] immediate rollback @ {iteration}: ΔPSNR={dpsnr_now:+.4f} dB")
                    continue

                else:
                    # split-aware LR: lower a bit and warmup back later
                    n_old = getattr(tensorf, "_n_patches_before_split", prev_n)
                    n_new = len(tensorf.patch_map)

                    cur_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
                    split_lr_pow = getattr(args, "split_lr_pow", 0.5)
                    split_lr_min = getattr(args, "split_lr_min", 0.6)  # 0.5~0.8
                    raw_split_scale = (n_old / max(n_new, 1)) ** split_lr_pow
                    split_scale = max(split_lr_min, min(1.0, raw_split_scale))
                    eff_scale = cur_scale * split_scale
                    print(f"[uneven-split lr] old={n_old} new={n_new} | cur_scale={cur_scale:.4g} * split_scale={split_scale:.3f} -> eff_scale={eff_scale:.4g}")


                    cur_patch_keys = set(tensorf.patch_map.keys())
                    new_children = sorted(list(cur_patch_keys - prev_patch_keys))

                    # per-child micro upsample right after split (small, local)
                    if len(new_children) > 0 and bool(int(getattr(args, "child_micro_ups_enable", 1))):
                        micro_scale = int(getattr(args, "child_micro_ups_scale", 2))       # e.g. 2
                        vmax        = int(getattr(args, "vm_reso_max", 64))                # hard cap
                        promoted = micro_upsample_children(tensorf, new_children, max_res=vmax, scale=micro_scale)
                        if len(promoted) > 0:
                            optimizer = build_optimizer_with_scale(tensorf, args, eff_scale)
                            last_structure_change_iter = iteration
                            log_event(logfolder, "micro-ups-on-split", iteration,
                                      promoted=len(promoted), keys=str(promoted), scale=micro_scale, vmax=vmax)

                    # short-horizon "child LR boost": make new child parameters learn faster
                    # collect newly created parameters (includes micro-upsampled params) and rebuild optimizer
                    child_lr_mult   = float(getattr(args, "child_lr_boost_mult", 1.5))     # e.g. 1.3~1.8
                    child_lr_iters  = int(getattr(args,  "child_lr_boost_iters", 300))     # decay/finish horizon
                    enable_boost    = bool(int(getattr(args, "child_lr_boost_enable", 1)))

                    if enable_boost and (len(new_children) > 0) and (child_lr_mult > 1.0) and (child_lr_iters > 0):
                        new_params = _collect_new_params(tensorf, prev_param_ids)
                        if len(new_params) > 0:
                            optimizer = rebuild_optimizer_with_child_boost(
                                tensorf, args, base_scale=eff_scale,
                                child_params=new_params, child_mult=child_lr_mult
                            )
                            last_structure_change_iter = iteration
                            log_event(logfolder, "child-lr-boost", iteration,
                                    new_params=len(new_params), mult=child_lr_mult, horizon=child_lr_iters)

                            # schedule a soft decay back to normal (rebuild to base groups after horizon)
                            # reuse the existing warmup hook slot if it's free; otherwise we layer a tiny hook
                            child_boost_end = iteration + child_lr_iters
                            base_groups_snapshot = [pg['lr'] for pg in optimizer.param_groups]

                            def _maybe_decay_child_boost(cur_iter, opt=optimizer, base_lrs=base_groups_snapshot,
                                                        end_iter=child_boost_end, mult0=child_lr_mult):
                                if cur_iter <= end_iter:
                                    # linear decay: from mult0 -> 1.0
                                    t = (cur_iter - iteration) / max(1, end_iter - iteration)
                                    t = max(0.0, min(1.0, t))
                                    cur_mult = mult0 + (1.0 - mult0) * t
                                    # group-0 is not guaranteed to be child group; so recompute per step:
                                    # scale any group whose lr > its "base_lrs * eff_scale" a bit more
                                    for gi, pg in enumerate(opt.param_groups):
                                        # detect child groups by checking if their current lr is higher than base
                                        base_lr = base_lrs[gi]
                                        if pg['lr'] > base_lr + 1e-12:
                                            # child group -> follow decayed multiplier
                                            pg['lr'] = base_lr * cur_mult
                                        else:
                                            # others keep their base lr (already at eff_scale)
                                            pg['lr'] = base_lr

                            # chain with existing maybe_warmup if present.
                            prev_hook = maybe_warmup_hook
                            def _chained_hook(cur_iter, *a, **kw):
                                if prev_hook is not None:
                                    prev_hook(cur_iter)
                                _maybe_decay_child_boost(cur_iter)

                            maybe_warmup_hook = _chained_hook
                            warmup_until = max(int(warmup_until), int(child_boost_end))


                    # basically only downscaling is allowed for structure optimization
                    if args.dynamic_rank:
                        tensorf.update_alpha_mass_per_patch(n_per=getattr(args, "alpha_mass_n", 1024))

                        sigma_trip, app_trip = tensorf.rank_policy(tuple(reso_cur))
                        base_floor_sig, base_floor_app = _coalesce_base_floors(args, tensorf)
                        c_sig_base = max(base_floor_sig, int(np.round(np.mean(sigma_trip))))
                        c_app_base = max(base_floor_app, int(np.round(np.mean(app_trip))))
                        
                        changed = 0
                        ALLOW_RANK_UP = bool(getattr(args, "postcrit_allow_up", False)) and (pending_restruct_probation is not None)

                        ref_res = int(getattr(args, "vm_reso_max", 64))
                        max_rank = int(getattr(args, "max_rank", 64))
                        scale_gamma = float(getattr(args, "rank_autoscale_gamma", 0.6))
                        alpha_keep_q = float(getattr(args, "rank_autoscale_alpha_keep_q", 0.95))
                        print(f"[rank-auto] ref_res={ref_res} gamma={scale_gamma} keep_q={alpha_keep_q}")

                        changed = tensorf.selective_rank_autoscale(ref_res=ref_res, gamma=scale_gamma,
                                                                    c_sig_base=c_sig_base, c_app_base=c_app_base,  
                                                                    c_min=8, c_max=max_rank, round_to=4, 
                                                                    allow_down=True, allow_up=ALLOW_RANK_UP,                       
                                                                    alpha_keep_q=alpha_keep_q, verbose=True)
                        if changed > 0:
                            print(f"[rank-auto] resized ranks: changed={changed}")
                            optimizer = build_optimizer_with_scale(tensorf, args, eff_scale)
                            last_structure_change_iter = iteration
                        
                        log_event(logfolder, "rank-resize", iteration, phase="uneven-split",
                                  up=0, down=int(changed), reso_cur=str(reso_cur),
                                  base_floor_sig=int(base_floor_sig), base_floor_app=int(base_floor_app),
                                  policy_sig=str(list(map(int, sigma_trip))),  
                                  policy_app=str(list(map(int, app_trip))),
                                  c_sig_base=int(c_sig_base), c_app_base=int(c_app_base),
                                  allow_down=bool(ALLOW_RANK_DOWN),
                                  alpha_keep_q=float(alpha_keep_q))

                    warmup_iters = getattr(args, "postcrit_warmup_iters", 300)
                    warmup_floor = getattr(args, "postcrit_warmup_floor", 0.3)
                    base_lrs = [g['lr'] for g in optimizer.param_groups]
                    for g, lr0 in zip(optimizer.param_groups, base_lrs):
                        g['lr'] = lr0 * warmup_floor

                    s_iter, e_iter = iteration, iteration + warmup_iters
                    def _maybe_warmup(cur_iter, opt=optimizer, base=base_lrs, floor=warmup_floor,
                                    s=s_iter, e=e_iter):
                        if cur_iter <= e:
                            t = (cur_iter - s) / max(1, e - s)
                            t = max(0.0, min(1.0, t))
                            frac = floor + (1.0 - floor) * t
                            if hasattr(tensorf.renderModule, "set_fea_gate"):
                                tensorf.renderModule.set_fea_gate(max(0.9, float(frac)))
                            for pg, lr0 in zip(opt.param_groups, base):
                                pg['lr'] = lr0 * frac
                    maybe_warmup_hook = _maybe_warmup
                    warmup_until = e_iter

                    if hasattr(tensorf, "_n_patches_before_split"):
                        delattr(tensorf, "_n_patches_before_split")

                maybe_viz_patches(iteration, "post-VM-split") 

                do_full = (getattr(args, "N_vis", 0) > 0) and (getattr(args, "vis_every", 0) > 0) and (((iteration + 1) % args.vis_every) == 0)
                if do_full and (test_dataset is not None):
                    if (iteration % 1000) == 0:          
                        run_heartbeat(note="pre-eval", iter_for_log=iteration, reso_for_log=tuple(reso_cur))
                        did_cov_this_step = True
                        _normalize_patch_keys(tensorf)

                    tensorf_eval_flag = tensorf.repair_enable
                    tensorf.repair_enable = False
                    PSNRs_test = evaluation(
                        test_dataset, tensorf, args, trainer.render, f'{logfolder}/imgs_vis/',
                        N_vis=args.N_vis, prtx=f'{iteration:06d}_', N_samples=-1,
                        white_bg=white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False
                    )
                    tensorf.repair_enable = tensorf_eval_flag
                    psnr_full = float(np.mean(PSNRs_test))
                    summary_writer.add_scalar('test/psnr', psnr_full, iteration)
                    log_rd(iteration, note="post_VM_split_full", psnr_full_val=psnr_full)
                else:
                    if (iteration % 1000) == 0:          
                        run_heartbeat(note="pre-eval", iter_for_log=iteration, reso_for_log=tuple(reso_cur))
                        did_cov_this_step = True
                        _normalize_patch_keys(tensorf)
                    
                    if hasattr(tensorf, "eval"): tensorf.eval()
                    tensorf_eval_flag = tensorf.repair_enable
                    tensorf.repair_enable = False
                    with torch.no_grad():
                        val_dataset = get_val_dataset(args, train_dataset, test_dataset)
                        psnr_fast = quick_val_psnr_safe(
                            val_dataset, trainer.render, device,
                            target_views=3, target_rpv=4096,  
                            min_views=1, min_rpv=1024, chunk=1024
                        )
                    tensorf.repair_enable = tensorf_eval_flag
                    if hasattr(tensorf, "train"): tensorf.train()
                    summary_writer.add_scalar('test/psnr_fast', psnr_fast, iteration)
                    log_rd(iteration, note="post_VM_split_fast", psnr_fast_val=psnr_fast)

                last_structure_change_iter = iteration
                run_heartbeat(note="post-VM-split", iter_for_log=iteration, reso_for_log=tuple(reso_cur))  
                did_cov_this_step = True
                _normalize_patch_keys(tensorf) 
                maybe_softprune(note="post-VM-split")

        if isinstance(trainingSampler, MixedSampler):
            done = (iteration - trainingSampler.start_iter) >= trainingSampler.anneal_iters
            if done:
                N_old = allrays_old.shape[0]
                allrays, allrgbs = allrays[N_old:], allrgbs[N_old:]
                trainingSampler = SimpleSampler(allrays.shape[0], int(args.batch_size))
                log_event(logfolder, "refilter-anneal-done", iteration, keep=allrays.shape[0])

        if iteration % args.heartbeat_every == 0:
            update_peak_vram_txt(logfolder, best_peak_bytes)

            peak_every = getattr(args, "peak_vram_log_every", 0)
            should_log_peak = False
            if peak_every > 0:
                should_log_peak = (iteration % peak_every == 0)
            else:
                should_log_peak = (iteration % args.heartbeat_every == 0)

            if should_log_peak:
                log_cuda_peak(summary_writer, logfolder, iteration,
                              tag_prefix=getattr(args, "peak_vram_tag_prefix", "mem"))

        if not did_cov_this_step:
            if (iteration % 100) == 0:
                maybe_guard_missing(iteration, tag="periodic")  

            if (iteration - last_heartbeat_iter) >= args.heartbeat_every:
                run_heartbeat(note=f"heartbeat@{iteration}", iter_for_log=iteration, reso_for_log=tuple(reso_cur))
                _normalize_patch_keys(tensorf)
                last_heartbeat_iter = iteration
                did_cov_this_step = True

    log_rd(iteration, note="final")
    summary_writer.close()
    final_ckpt = os.path.join(logfolder, f"{args.expname}.th")
    tensorf.save(final_ckpt, extra_meta={'iter': iteration, 'args': vars(args)})
    print('has patch_map:', 'patch_map' in tensorf.state_dict())
    print("Training completed and final checkpoint is saved.")

    if iteration == args.n_iters - 1:
        exp_logger.generate_plots()
        exp_logger.generate_latex_table()
        print(f"[INFO] Experiment results saved to {logfolder}")

    if args.render_train or args.render_test or args.render_path:
        args.ckpt = final_ckpt
        render_test(args, tensorf)

    with open(os.path.join(logfolder, 'final_grid_stats.txt'), 'a') as final_train_log:
        final_train_log.write(f'Final voxel count: {tensorf.get_total_voxels()}\n')
        final_train_log.write(f'Final memory usage: {tensorf.get_total_mem() / 1024**2:.2f} MB\n')

    with open(os.path.join(logfolder, 'rd_curve_log.json'), 'w') as f:
        json.dump(rd_log_list, f, indent=2)
    print(f"[LOG] RD data saved to {logfolder}/rd_curve_log.json")


@torch.no_grad()
def render_test(args, tensorf=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = dataset_dict[args.dataset_name]

    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_test, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if tensorf is None:
        ckpt_path = getattr(args, "ckpt", None)
        if (not ckpt_path) or (not os.path.isfile(ckpt_path)):
            print(f"[WARN] render_test skipped: invalid --ckpt={ckpt_path}")
            return

        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get('state_dict', {})
        print('has patch_map:', 'patch_map' in sd)

        Model = eval(args.model_name)
        model_kwargs = dict(ckpt.get("kwargs", {}))
        model_kwargs["device"] = device
        tensorf = Model(**model_kwargs)
        tensorf.load_state_dict(sd, strict=False)
        tensorf.to(device)

        if ('alphaMask.shape' in ckpt) and ('alphaMask.mask' in ckpt):
            from models.tensorBase import AlphaGridMask
            shape = tuple(ckpt['alphaMask.shape'])
            bits  = ckpt['alphaMask.mask']
            aabb  = ckpt.get('alphaMask.aabb', tensorf.aabb).to(device)
            flat  = np.unpackbits(bits)[:np.prod(shape)]
            mask  = torch.from_numpy(flat.reshape(shape)).bool().to(device)
            tensorf.alphaMask = AlphaGridMask(device, aabb, mask)
        
        if not getattr(tensorf, "patch_map", None) or len(tensorf.patch_map) == 0:
            raise RuntimeError("[render_test] checkpoint has no patch_map — was it saved with the new saver?")

    print('num patches:', len(tensorf.patch_map))
    print('grid G:', getattr(tensorf, 'patch_grid_reso', None))
    any_patch = next(iter(tensorf.patch_map.values()))
    print('basis device:', any_patch['basis_mat'].weight.device)
    if hasattr(tensorf, "assert_zero_origin_and_contiguous"):
        try: tensorf.assert_zero_origin_and_contiguous()
        except Exception as e: print('[WARN]', e)

    tensorf.eval()
    trainer = PatchTrainStep(tensorf, render_step_size=1.0, white_bg=white_bg)

    logfolder = os.path.dirname(args.ckpt) if getattr(args, "ckpt", None) else os.path.join(args.basedir, args.expname)
    os.makedirs(logfolder, exist_ok=True)

    flag = getattr(tensorf, "repair_enable", False)
    tensorf.repair_enable = False
    try:
        if args.render_train:
            out_dir = os.path.join(logfolder, "imgs_train_all"); os.makedirs(out_dir, exist_ok=True)
            train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
            PSNRs_train = evaluation(train_dataset, tensorf, args, trainer.render, out_dir,
                                     N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
            print(f'======> {args.expname} train all psnr: {float(np.mean(PSNRs_train)):.4f} <======')

        if args.render_test:
            out_dir = os.path.join(logfolder, "imgs_test_all"); os.makedirs(out_dir, exist_ok=True)
            PSNRs_test = evaluation(test_dataset, tensorf, args, trainer.render, out_dir,
                                    N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
            PSNR_mean = float(np.mean(PSNRs_test))
            print(f'======> {args.expname} test all psnr: {PSNR_mean:.4f} <======')
            with open(os.path.join(logfolder, 'eval_test_all.txt'), 'a') as f:
                f.write(f'{PSNR_mean:.4f}\n')
            with open(os.path.join(logfolder, 'time_stamp.txt'), 'a') as f:
                f.write(f'Final render-test: {datetime.now()}\n')

        if args.render_path:
            out_dir = os.path.join(logfolder, "imgs_path_all"); os.makedirs(out_dir, exist_ok=True)
            c2ws = test_dataset.render_path
            evaluation_path(test_dataset, tensorf, c2ws, trainer.render, out_dir,
                            N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
    finally:
        tensorf.repair_enable = flag

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    torch.backends.cudnn.benchmark = True       
    torch.backends.cudnn.deterministic = False  

    args = config_parser()

    if args.logmargin_base is not None:
        args.logmargin_tau = math.log(float(args.logmargin_base))
    
    print(args)

    if args.export_mesh:
        export_mesh_patch(args)

    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args)
