import torch
import torch.nn
import torch.nn.functional as F
from .tensorBase import *
import numpy as np
import math
import os
import collections
from collections import Counter, defaultdict
from contextlib import contextmanager
from typing import Optional, Sequence, List, Tuple, Dict, Any

class SharedBasisManager(torch.nn.Module):
    """
    全域共享基底管理器
    每個 patch 只需學習係數，共享全域基底以節省記憶體
    """
    def __init__(self, n_basis=128, app_dim=27, device='cuda', dtype=torch.float32):
        super().__init__()
        self.n_basis = n_basis
        self.app_dim = app_dim
        self.device = device
        self.dtype = dtype
        
        # 全域共享基底
        self.global_basis = torch.nn.Linear(n_basis, app_dim, bias=False, device=device, dtype=dtype)
        torch.nn.init.orthogonal_(self.global_basis.weight)  # 正交初始化
        
        # 每個 patch 的係數矩陣
        self.patch_coeffs = torch.nn.ModuleDict()
        
    def get_or_create_coeffs(self, patch_key, in_dim):
        """獲取或創建 patch 的係數矩陣"""
        key_str = f"{patch_key[0]}_{patch_key[1]}_{patch_key[2]}"
        
        if key_str not in self.patch_coeffs:
            coeffs = torch.nn.Linear(in_dim, self.n_basis, bias=False, 
                                    device=self.device, dtype=self.dtype)
            torch.nn.init.xavier_uniform_(coeffs.weight, gain=0.1)
            self.patch_coeffs[key_str] = coeffs
            
        return self.patch_coeffs[key_str]
    
    def forward(self, patch_key, vm_features):
        """
        前向傳播：VM features -> coefficients -> global basis
        Args:
            patch_key: (i,j,k) tuple
            vm_features: [N, in_dim] VM 特徵
        Returns:
            [N, app_dim] 最終 appearance 特徵
        """
        in_dim = vm_features.shape[-1]
        coeffs_layer = self.get_or_create_coeffs(patch_key, in_dim)
        
        # 投影到基底空間
        coeffs = coeffs_layer(vm_features)  # [N, n_basis]
        
        # 應用全域基底
        output = self.global_basis(coeffs)  # [N, app_dim]
        
        return output
    
    def cleanup_unused(self, active_keys):
        """清理未使用的係數矩陣"""
        active_set = {f"{k[0]}_{k[1]}_{k[2]}" for k in active_keys}
        to_remove = []
        for key_str in self.patch_coeffs.keys():
            if key_str not in active_set:
                to_remove.append(key_str)
        
        for key_str in to_remove:
            del self.patch_coeffs[key_str]
        
        return len(to_remove)

class _SeamLR2D(torch.nn.Module):
    def __init__(self, C, H, W, k, device=None, dtype=None):
        super().__init__()
        self.C, self.H, self.W, self.k = C, H, W, k
        self.U  = torch.nn.Parameter(torch.empty(C, k, device=device, dtype=dtype).normal_(0, 0.02))
        self.Ba = torch.nn.Parameter(torch.zeros(k, H, W, device=device, dtype=dtype))
        self.Bb = torch.nn.Parameter(torch.zeros(k, H, W, device=device, dtype=dtype))
    def reconstruct(self, side: int):
        B = self.Ba if side == 0 else self.Bb
        X = torch.matmul(self.U, B.view(self.k, self.H*self.W)).view(1, self.C, self.H, self.W)
        return X.contiguous()

class _SeamLR1D(torch.nn.Module):
    def __init__(self, C, L, k, device=None, dtype=None):
        super().__init__()
        self.C, self.L, self.k = C, L, k
        self.U  = torch.nn.Parameter(torch.empty(C, k, device=device, dtype=dtype).normal_(0, 0.02))
        self.Ba = torch.nn.Parameter(torch.zeros(k, L, device=device, dtype=dtype))
        self.Bb = torch.nn.Parameter(torch.zeros(k, L, device=device, dtype=dtype))
    def reconstruct(self, side: int):
        B = self.Ba if side == 0 else self.Bb
        X = torch.matmul(self.U, B).view(1, self.C, self.L, 1)
        return X.contiguous()

class TensorVMSplitPatch(TensorBase):
    def __init__(self, aabb, gridSize, device, patch_grid_reso=8, **kargs):
        self.basis_lowrank_enable = bool(kargs.pop("basis_lowrank_enable", False))
        self.basis_rank           = int(kargs.pop("basis_rank", 16))
        self.rank_cap_sigma       = kargs.pop("rank_cap_sigma", None)
        self.rank_cap_app         = kargs.pop("rank_cap_app", None)
        self.rank_base_floor_sig  = kargs.pop("rank_base_floor_sig", None)
        self.rank_base_floor_app  = kargs.pop("rank_base_floor_app", None)
        self.min_rank             = kargs.pop("min_rank", None)
        self.max_rank             = kargs.pop("max_rank", None)
        self.seam_lowrank_enable  = bool(kargs.pop("seam_lowrank_enable", False))
        self.seam_lowrank_scope   = str(kargs.pop("seam_lowrank_scope", "both"))
        self.seam_rank_sigma      = int(kargs.pop("seam_rank_sigma", 8))
        self.seam_rank_app        = int(kargs.pop("seam_rank_app", 8))

        self.repair_enable            = bool(kargs.pop("repair_enable", True))
        self.repair_tau               = float(kargs.pop("repair_tau", 1.0))
        self.repair_adjacent_only     = bool(kargs.pop("repair_adjacent_only", True))
        self.repair_grad_scale_sigma  = float(kargs.pop("repair_grad_scale_sigma", 0.0))
        self.repair_grad_scale_app    = float(kargs.pop("repair_grad_scale_app", 0.3))

        self.split_child_res_policy = str(kargs.pop("split_child_res_policy", "arg"))
        self.split_child_scale      = float(kargs.pop("split_child_scale", 1.0))
        self.split_child_min        = int(kargs.pop("split_child_min", 16))
        
        self.global_basis_enable = bool(kargs.pop("global_basis_enable", False))
        self.global_basis_k_sigma = int(kargs.pop("global_basis_k_sigma", 64))
        self.global_basis_k_app   = int(kargs.pop("global_basis_k_app", 96))
        
        self.patch_map = {}
        self.patch_grid_reso = patch_grid_reso
        self._basis_registry = {}
        self._last_rank_resize_iter = None
        
        super().__init__(aabb, gridSize, device, **kargs)
        
        self._basis_bank = torch.nn.ModuleDict()
        self._seam_banks = torch.nn.ModuleDict()
        self.basis_dtype = getattr(self, "basis_dtype", torch.float32)
        
        if self.global_basis_enable:
            self.shared_basis_manager = SharedBasisManager(
                n_basis=self.global_basis_k_app,
                app_dim=self.app_dim,  
                device=device,
                dtype=self.basis_dtype
            )
            print(f"[INFO] Using SharedBasisManager with {self.global_basis_k_app} basis vectors")
        
        gate_dtype = self.aabb.dtype  
        self.register_buffer("alpha_gate_scale", torch.ones((), dtype=gate_dtype))
        
        for pid, p in self.patch_map.items():
            self._validate_one_patch_shapes(f"{pid[0]}_{pid[1]}_{pid[2]}", p)
    
    @staticmethod
    def _grad_scale(x, g):
        # g could be float or tensor; only grad changes and forward value does not
        if not torch.is_tensor(g):
            g = torch.tensor(g, dtype=x.dtype, device=x.device)
        return x * g + x.detach() * (1.0 - g)

    def _cheby_nearest_keys(self, keys_tensor, q_tensor):
        # keys: [K,3] long, q: [M,3] long. Return index [M]
        # Chebyshev/L∞ distance
        dif = (keys_tensor[None, ...].float() - q_tensor[:, None, :].float()).abs()  # [M,K,3]
        dist = dif.amax(dim=-1)                                                      # [M,K]
        return dist.argmin(dim=1)                                                    # [M]

    @torch.no_grad()
    def _get_patch_grid_reso(self):
        """
        Normalize self.patch_grid_reso to a 3-tuple (Gx,Gy,Gz).
        Accepts int / 1-elem list/tuple/tensor (→ (g,g,g)) or 3-elem (→ tuple).
        """
        pgr = getattr(self, "patch_grid_reso", None)

        if pgr is None:
            return (1, 1, 1)

        if isinstance(pgr, int):
            g = int(pgr)
            return (g, g, g)

        if isinstance(pgr, (list, tuple)):
            if len(pgr) == 3:
                return (int(pgr[0]), int(pgr[1]), int(pgr[2]))
            elif len(pgr) == 1:
                g = int(pgr[0])
                return (g, g, g)

        if torch.is_tensor(pgr):
            if pgr.numel() == 1:
                g = int(pgr.item())
                return (g, g, g)
            elif pgr.numel() == 3:
                a = pgr.flatten().tolist()
                return (int(a[0]), int(a[1]), int(a[2]))

        raise ValueError(f"Invalid patch_grid_reso: {pgr!r}")

    @torch.no_grad()
    def build_patch_lookup_table(self):
        """
        建立高效的 patch 查詢表，避免 Python dict 查詢
        """
        if not hasattr(self, 'patch_grid_reso'):
            return
            
        Gx, Gy, Gz = self._get_patch_grid_reso()
        
        # 建立查詢表：-1 表示沒有 patch
        self.patch_lookup = torch.full((Gx, Gy, Gz), -1, 
                                    dtype=torch.long, device=self.device)
        
        # 建立索引到 key 的映射
        self.patch_idx_to_key = {}
        
        for idx, key in enumerate(self.patch_map.keys()):
            i, j, k = key
            if 0 <= i < Gx and 0 <= j < Gy and 0 <= k < Gz:
                self.patch_lookup[i, j, k] = idx
                self.patch_idx_to_key[idx] = key
        
        self._lookup_table_valid = True

    @torch.no_grad()
    def _map_coords_to_patch_fast(self, xyz_coords, snap_missing=False, snap_tau=0.5,
                                adjacent_only=True, return_snapped=False):
        """
        高效版本的 patch 映射，使用查詢表而非 dict
        """
        # 確保查詢表是最新的
        if not hasattr(self, '_lookup_table_valid') or not self._lookup_table_valid:
            self.build_patch_lookup_table()
        
        device = xyz_coords.device
        
        # AABB 正規化
        aabb = self.aabb
        if aabb.numel() == 6:
            aabb = aabb.reshape(2, 3)
        a0, a1 = aabb[0], aabb[1]
        extent = (a1 - a0).clamp_min(1e-8)
        
        # 空 map 處理
        if not self.patch_map or not hasattr(self, 'patch_lookup'):
            coords = torch.zeros((*xyz_coords.shape[:-1], 3), dtype=torch.long, device=device)
            exists = torch.zeros((*xyz_coords.shape[:-1],), dtype=torch.bool, device=device)
            if return_snapped:
                snapped = torch.zeros_like(exists)
                return coords, exists, snapped
            return coords, exists
        
        Gx, Gy, Gz = self._get_patch_grid_reso()
        G_t = torch.tensor([Gx, Gy, Gz], device=device, dtype=torch.float32)
        
        # 正規化到 [0,1)
        eps = 1e-6
        p = ((xyz_coords - a0) / extent).clamp(0.0, 1.0 - eps)
        idx = torch.floor(p * G_t).long()
        
        # Clamp 到有效範圍
        idx[..., 0] = torch.clamp(idx[..., 0], 0, Gx - 1)
        idx[..., 1] = torch.clamp(idx[..., 1], 0, Gy - 1)
        idx[..., 2] = torch.clamp(idx[..., 2], 0, Gz - 1)
        
        # 使用查詢表快速檢查存在性
        flat_shape = idx.shape[:-1]
        idx_flat = idx.view(-1, 3)
        
        # 批量查詢
        patch_indices = self.patch_lookup[idx_flat[:, 0], idx_flat[:, 1], idx_flat[:, 2]]
        exists_flat = (patch_indices >= 0)
        
        exists = exists_flat.view(*flat_shape)
        coords_out = idx
        
        # Snap missing（如果需要）
        snapped = torch.zeros_like(exists, dtype=torch.bool)
        if snap_missing:
            # ... snap 邏輯保持不變 ...
            pass
        
        if return_snapped:
            return coords_out, exists, snapped
        return coords_out, exists

    @torch.no_grad()
    def _map_coords_to_patch(self, xyz_coords, snap_missing: bool = False, snap_tau: float = 0.5,
                             adjacent_only: bool = True, return_snapped: bool = False):
        device = xyz_coords.device

        # --- AABB to (2,3) ---
        aabb = self.aabb
        if aabb.numel() == 6:
            aabb = aabb.reshape(2, 3)
        a0, a1 = aabb[0], aabb[1]
        extent = (a1 - a0).clamp_min(1e-8)

        # empty map
        if not self.patch_map:
            coords = torch.zeros((*xyz_coords.shape[:-1], 3), dtype=torch.long, device=device)
            exists = torch.zeros((*xyz_coords.shape[:-1],), dtype=torch.bool, device=device)
            if return_snapped:
                snapped = torch.zeros_like(exists)
                return coords, exists, snapped
            return coords, exists

        # ---- 推回當前 G ----
        keys_t = torch.as_tensor(list(self.patch_map.keys()), device=device, dtype=torch.long)  # [K,3]
        G_t = keys_t.max(dim=0).values + 1            # tensor [3]，用來做計算/廣播
        G = tuple(int(v) for v in G_t.tolist())       # ints，用來建 tensor shape / 索引

        # 佔據表（用 ints 建 shape）
        occ = torch.zeros(G, dtype=torch.bool, device=device)
        occ[keys_t[:, 0], keys_t[:, 1], keys_t[:, 2]] = True

        # ---- 正規化 -> 量化到 cell ----
        eps = 1e-6
        p = ((xyz_coords - a0) / extent).clamp(0.0, 1.0 - eps)          # [...,3] in [0,1)
        idx = torch.floor(p * G_t.to(p.dtype)).long()                   # [...,3]
        hi = (G_t - 1).view(*([1] * (idx.dim() - 1)), 3)                # [...,3]
        lo = torch.zeros_like(hi)
        idx = torch.minimum(torch.maximum(idx, lo), hi)                 # clamp 到 [0, G-1]

        exists = occ[idx[..., 0], idx[..., 1], idx[..., 2]]
        coords_out = idx.clone()
        snapped = torch.zeros_like(exists, dtype=torch.bool)

        # ---- 鄰域 snap（半徑=1；若想支援非 adjacent-only，可把半徑調 2）----
        if snap_missing:
            # 攤平成 1D 處理（避免多維 nonzero/squeeze 陷阱）
            sh = idx.shape[:-1]
            idx_flat = idx.view(-1, 3)
            p_flat = p.view(-1, 3)
            exists_flat = exists.view(-1)
            coords_flat = coords_out.view(-1, 3)
            snapped_flat = snapped.view(-1)

            miss_mask = ~exists_flat
            if miss_mask.any():
                miss = miss_mask.nonzero(as_tuple=False).squeeze(1)       # [M]

                # 距離門檻（在 [0,1) 的座標系）
                cell = (1.0 / G_t.to(p.dtype))                            # [3]
                base_ctr = (idx_flat[miss].to(p.dtype) + 0.5) * cell      # [M,3]
                dist_cell = (p_flat[miss] - base_ctr).abs().amax(dim=-1)  # [M]
                ok_tau = (dist_cell <= float(snap_tau))

                # 鄰域
                rad = 1 if adjacent_only else 2
                rng = torch.arange(-rad, rad + 1, device=device, dtype=torch.long)
                dx, dy, dz = torch.meshgrid(rng, rng, rng, indexing='ij')
                nbr = torch.stack([dx.reshape(-1), dy.reshape(-1), dz.reshape(-1)], dim=1)  # [K,3], K=(2r+1)^3

                nei = idx_flat[miss].unsqueeze(1) + nbr.unsqueeze(0)      # [M,K,3]
                upper = (G_t - 1).view(1, 1, 3)                           # broadcastable
                lower = torch.zeros_like(nei)
                nei = torch.minimum(torch.maximum(nei, lower), upper)     # clamp

                occ_nei = occ[nei[..., 0], nei[..., 1], nei[..., 2]]      # [M,K] (bool)
                hit_any = occ_nei.any(dim=1) & ok_tau

                if hit_any.any():
                    # 先用「第一個命中」；你要最近中心也行
                    first = occ_nei[hit_any].float().argmax(dim=1)         # [H]
                    chosen = nei[hit_any, first, :]                        # [H,3]
                    coords_flat[miss[hit_any]] = chosen
                    exists_flat[miss[hit_any]] = True
                    snapped_flat[miss[hit_any]] = True

            # 還原形狀
            coords_out = coords_flat.view(*sh, 3)
            exists = exists_flat.view(*sh)
            snapped = snapped_flat.view(*sh)

        return (coords_out, exists, snapped) if return_snapped else (coords_out, exists)

    def _plane_param_for(self, patch, which: str, axis: int):
        info = patch.get("_seam_lr", {}).get((which, 'plane', axis))
        if info is None:
            return patch[f'{which}_plane'][axis]
        bank = self._seam_banks[info["id"]]
        return bank.reconstruct(info["side"])

    def _line_param_for(self, patch, which: str, axis: int):
        info = patch.get("_seam_lr", {}).get((which, 'line', axis))
        if info is None:
            return patch[f'{which}_line'][axis]
        bank = self._seam_banks[info["id"]]
        return bank.reconstruct(info["side"])

    @torch.no_grad()
    def _mk_seam_id(self, a, b, axis, which, kind):
        # a,b 是 patch key tuple (i,j,k)；規範化方向避免重複
        a_, b_ = (a, b) if a < b else (b, a)
        return f"seam:{which}:{kind}:axis{axis}:{a_}->{b_}"

    def _map_which_key(self, which: str) -> str:
        # 'sigma' | 'density' → 'density'；'app' → 'app'
        return "density" if which in ("sigma", "density") else "app"

    @torch.no_grad()
    def init_seam_lowrank(self, rank_sigma=None, rank_app=None, scope="both"):
        if not bool(getattr(self, "seam_lowrank_enable", False)):
            return 0

        k_sig = int(rank_sigma if rank_sigma is not None else getattr(self, "seam_rank_sigma", 8))
        k_app = int(rank_app   if rank_app   is not None else getattr(self, "seam_rank_app",   8))
        do_plane = scope in ("plane", "both")
        do_line  = scope in ("line",  "both")

        wired = 0
        for (ix, iy, iz), pa in list(self.patch_map.items()):
            for axis, neigh in [(0, (ix+1, iy, iz)), (1, (ix, iy+1, iz)), (2, (ix, iy, iz+1))]:
                if neigh not in self.patch_map:
                    continue
                pb = self.patch_map[neigh]

                for which, k in (('sigma', k_sig), ('app', k_app)):
                    wkey = self._map_which_key(which)

                    # ---- plane seam ----
                    if do_plane and (f'{wkey}_plane' in pa) and (f'{wkey}_plane' in pb):
                        Ca = pa[f'{wkey}_plane'][axis].shape[1]
                        Cb = pb[f'{wkey}_plane'][axis].shape[1]
                        Ha = pa[f'{wkey}_plane'][axis].shape[2]; Wa = pa[f'{wkey}_plane'][axis].shape[3]
                        Hb = pb[f'{wkey}_plane'][axis].shape[2]; Wb = pb[f'{wkey}_plane'][axis].shape[3]
                        if (Ca == Cb) and (Ha == Hb) and (Wa == Wb):
                            C, H, W = int(Ca), int(Ha), int(Wa)
                            sid = self._mk_seam_id((ix, iy, iz), neigh, axis, which, "plane")
                            if sid not in self._seam_banks:
                                bank = _SeamLR2D(C, H, W, min(k, C),
                                                 device=self.aabb.device,
                                                 dtype=pa[f'{wkey}_plane'][axis].dtype)
                                self._seam_banks[sid] = bank
                                # LS 初始化：U 隨機、Ba/Bb = pinv(U) @ A/B
                                U  = bank.U
                                Ua = torch.linalg.pinv(U)
                                A  = pa[f'{wkey}_plane'][axis].detach().squeeze(0).view(C, H*W)
                                B  = pb[f'{wkey}_plane'][axis].detach().squeeze(0).view(C, H*W)
                                bank.Ba.copy_((Ua @ A).view(-1, H, W))
                                bank.Bb.copy_((Ua @ B).view(-1, H, W))
                            pa.setdefault("_seam_lr", {})[(which, 'plane', axis)] = {"id": sid, "side": 0}
                            pb.setdefault("_seam_lr", {})[(which, 'plane', axis)] = {"id": sid, "side": 1}
                            wired += 1

                    # ---- line seam ----
                    if do_line and (f'{wkey}_line' in pa) and (f'{wkey}_line' in pb):
                        Ca = pa[f'{wkey}_line'][axis].shape[1]
                        Cb = pb[f'{wkey}_line'][axis].shape[1]
                        La = pa[f'{wkey}_line'][axis].shape[2]; Lb = pb[f'{wkey}_line'][axis].shape[2]
                        if (Ca == Cb) and (La == Lb):
                            C, L = int(Ca), int(La)
                            sid = self._mk_seam_id((ix, iy, iz), neigh, axis, which, "line")
                            if sid not in self._seam_banks:
                                bank = _SeamLR1D(C, L, min(k, C),
                                                 device=self.aabb.device,
                                                 dtype=pa[f'{wkey}_line'][axis].dtype)
                                self._seam_banks[sid] = bank
                                U  = bank.U
                                Ua = torch.linalg.pinv(U)
                                A  = pa[f'{wkey}_line'][axis].detach().squeeze(0).squeeze(-1)  # [C,L]
                                B  = pb[f'{wkey}_line'][axis].detach().squeeze(0).squeeze(-1)  # [C,L]
                                bank.Ba.copy_(Ua @ A)
                                bank.Bb.copy_(Ua @ B)
                            pa.setdefault("_seam_lr", {})[(which, 'line', axis)] = {"id": sid, "side": 0}
                            pb.setdefault("_seam_lr", {})[(which, 'line', axis)] = {"id": sid, "side": 1}
                            wired += 1

        return wired

    @torch.no_grad()
    def cleanup_seam_banks(self, keep_active_only=True):
        if not hasattr(self, '_seam_banks'):
            return 0
        
        if keep_active_only:
            active_seams = set()
            for patch in self.patch_map.values():
                if '_seam_lr' in patch:
                    for info in patch['_seam_lr'].values():
                        active_seams.add(info['id'])
            
            to_remove = []
            for sid in self._seam_banks.keys():
                if sid not in active_seams:
                    to_remove.append(sid)
            
            for sid in to_remove:
                del self._seam_banks[sid]
            
            return len(to_remove)
        return 0

    @torch.no_grad()
    def assert_zero_origin_and_contiguous(self, patch_map=None):
        """
        將 patch keys 平移到 (0,0,0) 起點，並確保形成稠密連續的格點。
        即便原本就 OK，也刷新 self.patch_grid_reso / self.current_patch_keys。
        """
        pm = self.patch_map if patch_map is None else patch_map
        if not pm:
            print("[WARN] assert_zero_origin_and_contiguous: empty patch_map")
            return

        ks = torch.as_tensor(list(pm.keys()), dtype=torch.long)
        kmin = ks.min(0).values
        kmax = ks.max(0).values
        size = (kmax - kmin + 1)
        expected = int(size[0] * size[1] * size[2])

        need_shift = bool((kmin != 0).any().item())
        not_contig = (len(pm) != expected)

        new_map = pm
        shift_tuple = (int(kmin[0].item()), int(kmin[1].item()), int(kmin[2].item()))

        if need_shift or not_contig:
            # 平移 keys 讓最小值對齊 0
            tmp = {}
            for (i, j, k), patch in pm.items():
                ni = int(i - kmin[0])
                nj = int(j - kmin[1])
                nk = int(k - kmin[2])
                tmp[(ni, nj, nk)] = patch
            new_map = tmp
            print(f"[fix] patch keys shifted by {shift_tuple} to zero-origin; G={(int(size[0]), int(size[1]), int(size[2]))}")

        self.patch_map = new_map
        self.patch_grid_reso = (int(size[0]), int(size[1]), int(size[2]))
        self.base_patch_grid_reso = self.patch_grid_reso
        self.current_patch_keys = sorted(list(new_map.keys()))

    def ensure_default_patch(self):
        """
        Build a complete fallback patch if patch_map is empty.
            - key = (0,0,0): ensure that each ray can be mapped into it.
            - fallback_patch: must include all keys that will be visited afterward.
        """
        if not self.patch_map:
            print("[WARNING] patch_map is empty! Creating default fallback patch.")
            device = self.aabb.device 
            fallback_key = (0, 0, 0)

            fallback_patch = self._create_patch(self.gridSize.tolist(), device)
            fallback_patch.setdefault('res', self.gridSize.tolist())
            fallback_patch['depth'] = 0  

            # set the only patch_map as fallback
            self.patch_map = {fallback_key: fallback_patch}

            if getattr(self, "base_patch_grid_reso", None) is None:
                self.base_patch_grid_reso = tuple(self._get_patch_grid_reso())
            self.patch_grid_reso = (1, 1, 1) 
    
    def _ensure_patch_device(self, patch, device):
        def _to_paramlist(pl):
            return torch.nn.ParameterList([
                torch.nn.Parameter(p.detach().to(device), requires_grad=True) for p in pl
            ])
        patch['density_plane'] = _to_paramlist(patch['density_plane'])
        patch['density_line']  = _to_paramlist(patch['density_line'])
        patch['app_plane']     = _to_paramlist(patch['app_plane'])
        patch['app_line']      = _to_paramlist(patch['app_line'])

        if 'basis_mat' in patch and isinstance(patch['basis_mat'], torch.nn.Module):
            patch['basis_mat'] = patch['basis_mat'].to(device)
        if 'basis_B' in patch and isinstance(patch['basis_B'], torch.nn.Module):
            patch['basis_B'] = patch['basis_B'].to(device)
        if 'mix_W' in patch and isinstance(patch['mix_W'], torch.nn.Parameter):
            patch['mix_W'] = torch.nn.Parameter(patch['mix_W'].detach().to(device), requires_grad=True)

        return patch


    def _init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]), device=device)))
            line_coef.append(torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1), device=device)))
        return torch.nn.ParameterList(plane_coef), torch.nn.ParameterList(line_coef)

    def _init_basis_mat(self):
        return torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False)
    
    def _create_patch(self, res, device):
        gridSize = list(res)
        
        if not self.global_basis_enable:
            # Fallback to original implementation (for A/B testing)
            density_plane, density_line = self._init_one_svd(self.density_n_comp, gridSize, 0.1, device)
            app_plane,     app_line     = self._init_one_svd(self.app_n_comp,     gridSize, 0.1, device)
            
            patch = {
                'res': gridSize.copy(),
                'density_plane': density_plane,
                'density_line':  density_line,
                'app_plane':     app_plane,
                'app_line':      app_line,
            }
        else:
            # NEW: Use shared basis + per-patch coefficients
            K_sigma = self.global_basis_k_sigma
            K_app   = self.global_basis_k_app
            
            # Create coefficient matrices (much smaller than full tensors!)
            density_plane = torch.nn.ParameterList([
                torch.nn.Parameter(0.01 * torch.randn(self.density_n_comp[i], K_sigma, device=device))
                for i in range(3)
            ])
            density_line = torch.nn.ParameterList([
                torch.nn.Parameter(0.01 * torch.randn(self.density_n_comp[i], K_sigma, device=device))
                for i in range(3)
            ])
            
            app_plane = torch.nn.ParameterList([
                torch.nn.Parameter(0.01 * torch.randn(self.app_n_comp[i], K_app, device=device))
                for i in range(3)
            ])
            app_line = torch.nn.ParameterList([
                torch.nn.Parameter(0.01 * torch.randn(self.app_n_comp[i], K_app, device=device))
                for i in range(3)
            ])
            
            patch = {
                'res': gridSize.copy(),
                'density_plane': density_plane,
                'density_line':  density_line,
                'app_plane':     app_plane,
                'app_line':      app_line,
            }
        
        # Appearance basis (unchanged)
        in_dim = self._app_in_dim_from_vm_or_coef(patch)
        
        if getattr(self, "basis_lowrank_enable", False):
            r = int(getattr(self, "basis_rank", 16))
            mix_W = torch.nn.Parameter(0.01 * torch.randn(r, in_dim, device=device), requires_grad=True)
            basis_B = self.get_shared_basis(r, self.app_dim)
            patch['mix_W']  = mix_W
            patch['basis_B'] = basis_B
        else:
            basis = self.get_shared_basis(in_dim, self.app_dim)
            patch['basis_mat'] = basis

        # Residuals (if enabled)
        if bool(getattr(self, "enable_child_residual", True)):
            if not self.global_basis_enable:
                def _zeros_like_pl(pl):
                    return torch.nn.ParameterList([
                        torch.nn.Parameter(torch.zeros_like(p), requires_grad=True) for p in pl
                    ])
                patch['density_plane_res'] = _zeros_like_pl(patch['density_plane'])
                patch['density_line_res']  = _zeros_like_pl(patch['density_line'])
                patch['app_plane_res']     = _zeros_like_pl(patch['app_plane'])
                patch['app_line_res']      = _zeros_like_pl(patch['app_line'])
            else:
                # Residuals also use coefficients
                patch['density_plane_res'] = torch.nn.ParameterList([
                    torch.nn.Parameter(torch.zeros(self.density_n_comp[i], K_sigma, device=device))
                    for i in range(3)
                ])
                patch['density_line_res'] = torch.nn.ParameterList([
                    torch.nn.Parameter(torch.zeros(self.density_n_comp[i], K_sigma, device=device))
                    for i in range(3)
                ])
                patch['app_plane_res'] = torch.nn.ParameterList([
                    torch.nn.Parameter(torch.zeros(self.app_n_comp[i], K_app, device=device))
                    for i in range(3)
                ])
                patch['app_line_res'] = torch.nn.ParameterList([
                    torch.nn.Parameter(torch.zeros(self.app_n_comp[i], K_app, device=device))
                    for i in range(3)
                ])

        return patch

    def _app_in_dim_from_vm_or_coef(self, patch):
        """
        Get appearance input dimension from either full VM tensors or coefficients.
        Compatible with both old and new patch structures.
        """
        if 'app_plane' in patch:
            return int(sum(p.shape[1] for p in patch['app_plane']) + 
                      sum(l.shape[1] for l in patch['app_line']))
        elif 'app_plane' in patch:
            return int(sum(c.shape[0] for c in patch['app_plane']) + 
                      sum(c.shape[0] for c in patch['app_line']))
        else:
            raise KeyError("Patch has neither app_plane nor app_plane")
    
    def _get_or_create_global_basis(self, res, basis_type, axis):
        """
        Lazy initialization of global shared basis for a given (res, type, axis).
        
        Args:
            res: (Rx, Ry, Rz) tuple - resolution of this patch
            basis_type: "density" or "app"
            axis: 0, 1, 2 (corresponding to XY, YZ, XZ planes)
        
        Returns:
            (plane_basis, line_basis): each a nn.Parameter
        """
        key = (tuple(res), basis_type, axis)
        
        if key in self._global_basis_cache:
            return self._global_basis_cache[key]
        
        # Determine spatial dimensions for this axis
        mat_id_0, mat_id_1 = self.matMode[axis]
        vec_id = self.vecMode[axis]
        H, W = res[mat_id_1], res[mat_id_0]
        L = res[vec_id]
        
        K = self.global_basis_k_sigma if basis_type == "density" else self.global_basis_k_app
        device = self.aabb.device
        
        # Initialize with small random values
        plane_basis = torch.nn.Parameter(
            0.02 * torch.randn(1, K, H, W, device=device),
            requires_grad=True
        )
        line_basis = torch.nn.Parameter(
            0.02 * torch.randn(1, K, L, 1, device=device),
            requires_grad=True
        )
        
        # Register as model parameters (important for optimizer!)
        param_name_plane = f"_gb_{basis_type}_ax{axis}_r{res[0]}x{res[1]}x{res[2]}_plane"
        param_name_line  = f"_gb_{basis_type}_ax{axis}_r{res[0]}x{res[1]}x{res[2]}_line"
        self.register_parameter(param_name_plane, plane_basis)
        self.register_parameter(param_name_line, line_basis)
        
        self._global_basis_cache[key] = (plane_basis, line_basis)
        
        return plane_basis, line_basis
    
    def _reconstruct_from_coef(self, coef_plane, coef_line, global_plane, global_line):
        """
        Reconstruct full VM tensor from coefficients and global basis.
        
        Args:
            coef_plane: [rank, K]
            coef_line: [rank, K]
            global_plane: [1, K, H, W]
            global_line: [1, K, L, 1]
        
        Returns:
            (reconstructed_plane [1, rank, H, W], reconstructed_line [1, rank, L, 1])
        """
        # Einstein summation: coef[rank, K] @ basis[K, spatial] -> [rank, spatial]
        plane_recon = torch.einsum('rk,bkhw->brhw', coef_plane, global_plane)  # [1, rank, H, W]
        line_recon  = torch.einsum('rk,bklw->brlw', coef_line,  global_line)   # [1, rank, L, 1]
        
        return plane_recon, line_recon


    def set_patch(self, patch_map=None, patch_key=None, d_plane=None, d_line=None, a_plane=None, a_line=None):
        """
        Flexible patch setter:
            - If only patch_map is given, replace self.patch_map entirely.
            - If patch_key and VM tensors are given, update VM parameters of the specified patch.
        Ensures all tensors are registered as trainable nn.Parameter.
        """
        if (patch_map is not None) and not isinstance(patch_map, dict):
            raise TypeError("set_patch(patch_map=...) expects a dict; did you mean patch_key=?")

        dev = self.aabb.device

        def to_param_list(tensors):
            return torch.nn.ParameterList([
                (t if isinstance(t, torch.nn.Parameter) else torch.nn.Parameter(t))
                .detach().to(dev).requires_grad_()
                for t in tensors
            ])

        self.ensure_default_patch()

        if patch_map is not None:
            new_map = {}
            for k, p in patch_map.items():
                p = dict(p)  # shallow copy
                p.setdefault('res', self.gridSize.tolist())
                p['density_plane'] = to_param_list(p['density_plane'])
                p['density_line']  = to_param_list(p['density_line'])
                p['app_plane']     = to_param_list(p['app_plane'])
                p['app_line']      = to_param_list(p['app_line'])

                # basis family
                if 'basis_mat' in p and isinstance(p['basis_mat'], torch.nn.Module):
                    p['basis_mat'] = p['basis_mat'].to(dev)
                if 'basis_B' in p and isinstance(p['basis_B'], torch.nn.Module):
                    p['basis_B'] = p['basis_B'].to(dev)
                if 'mix_W' in p and isinstance(p['mix_W'], torch.nn.Parameter):
                    p['mix_W'] = torch.nn.Parameter(p['mix_W'].detach().to(dev), requires_grad=True)

                new_map[k] = p
            self.patch_map = new_map

        elif patch_key is not None:
            if patch_key not in self.patch_map:
                raise ValueError(f"[set_patch] patch {patch_key} not found.")
            P = self.patch_map[patch_key]
            if d_plane is not None:
                P['density_plane'] = to_param_list(d_plane)
            if d_line  is not None:
                P['density_line']  = to_param_list(d_line)
            if a_plane is not None:
                P['app_plane']     = to_param_list(a_plane)
            if a_line  is not None:
                P['app_line']      = to_param_list(a_line)

        else:
            raise ValueError("Either patch_map or patch_key must be provided.")

        self.current_patch_keys = list(self.patch_map.keys())
    
    @contextmanager
    def fast_patch_vm(self, patch_key, d_plane, d_line, a_plane, a_line):
        """
        Zero-copy swap of VM ParameterLists for a single patch (PUF/eval-only).

        Usage:
            with self.fast_patch_vm(key, dp, dl, ap, al):
                rgb = renderer(rays)[0]
        """
        if patch_key not in self.patch_map:
            raise KeyError(f"patch {patch_key} not found")
        P = self.patch_map[patch_key]
        old = (P['density_plane'], P['density_line'], P['app_plane'], P['app_line'])
        try:
            P['density_plane'], P['density_line'], P['app_plane'], P['app_line'] = d_plane, d_line, a_plane, a_line
            yield
        finally:
            P['density_plane'], P['density_line'], P['app_plane'], P['app_line'] = old


    @torch.no_grad()
    def init_uniform_patches(self, grid_reso=(2,2,2), vm_reso=(8,8,8)):
        """
        Create a uniform patch grid of shape `grid_reso` (Gx,Gy,Gz).
        Each patch holds its own VM tensors at per-patch resolution `vm_reso`.
        """
        print(f"[basis] bank size = {len(self._basis_bank)} | keys = {list(self._basis_bank.keys())[:4]}...")
        Gx, Gy, Gz = map(int, grid_reso)
        Rx, Ry, Rz = map(int, vm_reso)
        dev = self.aabb.device

        new_map = {}
        template = self._create_patch([Rx, Ry, Rz], dev)  # 讓 template 依照當前 flags 建好 basis_mat 或 basis_B+mix_W

        def _clone_pl(pl):
            return torch.nn.ParameterList([
                torch.nn.Parameter(p.detach().clone().to(dev), requires_grad=True) for p in pl
            ])

        for i in range(Gx):
            for j in range(Gy):
                for k in range(Gz):
                    p = {
                        'res': [Rx, Ry, Rz],
                        'density_plane': _clone_pl(template['density_plane']),
                        'density_line':  _clone_pl(template['density_line']),
                        'app_plane':     _clone_pl(template['app_plane']),
                        'app_line':      _clone_pl(template['app_line']),
                    }

                    # 基底：對齊 template 的路徑
                    if 'basis_mat' in template:
                        in_dim = self._app_in_dim_from_vm(p['app_plane'], p['app_line'])
                        p['basis_mat'] = self.get_shared_basis(in_dim, self.app_dim)
                    else:
                        # 低秩共享：新建 per-patch mix_W，basis_B 使用共享 bank
                        r = int(getattr(self, "basis_rank", 16))
                        in_dim = self._app_in_dim_from_vm(p['app_plane'], p['app_line'])
                        p['mix_W']  = torch.nn.Parameter(0.01 * torch.randn(r, in_dim, device=dev), requires_grad=True)
                        p['basis_B'] = self.get_shared_basis(r, self.app_dim)

                    new_map[(i, j, k)] = p

        self.patch_map = new_map
        self.current_patch_keys = list(self.patch_map.keys())
        self.gridSize = torch.LongTensor([Rx, Ry, Rz]).to(dev)
        self.update_stepSize([Rx, Ry, Rz])
        self.patch_grid_reso = (Gx, Gy, Gz)
        self.assert_zero_origin_and_contiguous()

    def _as_NC(self, t: torch.Tensor, N_expected: int) -> torch.Tensor:
        """
        Convert input to [N, C].
        Allowed shapes:
        - [1, C, N, 1]  (typical for grid_sample)
        - [C, N]
        - [N, C]
        """
        if t.dim() == 4:  # [1, C, N, 1]
            t = t.squeeze(0).squeeze(-1).transpose(0, 1)  # -> [N, C]
            return t
        if t.dim() == 2:
            if t.shape[0] == N_expected:
                return t  # [N, C]
            if t.shape[1] == N_expected:
                return t.transpose(0, 1)  # [C, N] -> [N, C]
        raise ValueError(f"_as_NC: unexpected shape {tuple(t.shape)} (expect N={N_expected})")

    def _as_CN(self, t: torch.Tensor, N_expected: int) -> torch.Tensor:
        """
        Convert input to [C, N].
        Allowed shapes:
        - [1, C, N, 1]  (typical for grid_sample)
        - [C, N]
        - [N, C]
        """
        if t.dim() == 4:  # [1, C, N, 1]
            t = t.squeeze(0).squeeze(-1)  # -> [C, N]
            return t
        if t.dim() == 2:
            if t.shape[1] == N_expected:
                return t  # [C, N]
            if t.shape[0] == N_expected:
                return t.transpose(0, 1)  # [N, C] -> [C, N]
        raise ValueError(f"_as_CN: unexpected shape {tuple(t.shape)} (expect N={N_expected})")

    def _smoothstep(self, x, edge0, edge1):
        t = (x - edge0) / (edge1 - edge0 + 1e-8)
        t = t.clamp(0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)  # values in [0,1]

    def _interior_gate(self, xyz_01):
        """
        Interior gate for residuals.
        xyz_01: [N,3] local coords in [0,1].
        Returns g: [N,1] where g≈0 at faces and g≈1 in the interior.
        """
        tau = float(getattr(self, "residual_gate_tau", 0.10))  # interior width
        s = self._smoothstep(xyz_01, tau, 1.0 - tau)           # [N,3]
        g = (s[:, 0] * s[:, 1] * s[:, 2]).unsqueeze(-1)        # [N,1]
        return g

    def _blend_with_neighbors_if_needed(self, patch, xyz_local_01, base_val, kind="density"):
        """
        Lightweight seam blending near faces (inference-time smoothing).
        - patch: current patch dict; expects patch['_key'] = (ix,iy,iz) to be set by the dispatcher.
        - xyz_local_01: [N,3] local coords in [0,1] for THIS patch.
        - base_val: density -> [N], app -> [N, D]
        - kind: "density" or "app"
        Returns blended values (same shape as base_val). No extra params or loss.
        """
        if not bool(getattr(self, "enable_seam_blend", True)):
            return base_val

        if patch.get("_no_seam", False):
            return base_val  # prevent recursion during neighbor calls

        key = patch.get("_key", None)
        if key is None or not hasattr(self, "patch_grid_reso"):
            return base_val

        Gx, Gy, Gz = map(int, self.patch_grid_reso)
        band = float(getattr(self, "seam_band_width", 0.05))  # width in [0,1] from each face

        x, y, z = xyz_local_01[:, 0], xyz_local_01[:, 1], xyz_local_01[:, 2]
        out = base_val.clone()

        tasks = []
        def add(mask, nk, coords):
            if mask.any() and (nk in self.patch_map):
                tasks.append((mask, nk, coords))

        # X- faces
        mask_l = (x < band)
        mask_r = (x > (1.0 - band))
        if key[0] - 1 >= 0:
            coords = xyz_local_01.clone(); coords[:, 0] = coords[:, 0] + 1.0  # map to left neighbor
            add(mask_l, (key[0]-1, key[1], key[2]), coords)
        if key[0] + 1 < Gx:
            coords = xyz_local_01.clone(); coords[:, 0] = coords[:, 0] - 1.0  # map to right neighbor
            add(mask_r, (key[0]+1, key[1], key[2]), coords)

        # Y- faces
        mask_d = (y < band)
        mask_u = (y > (1.0 - band))
        if key[1] - 1 >= 0:
            coords = xyz_local_01.clone(); coords[:, 1] = coords[:, 1] + 1.0
            add(mask_d, (key[0], key[1]-1, key[2]), coords)
        if key[1] + 1 < Gy:
            coords = xyz_local_01.clone(); coords[:, 1] = coords[:, 1] - 1.0
            add(mask_u, (key[0], key[1]+1, key[2]), coords)

        # Z- faces
        mask_b = (z < band)
        mask_f = (z > (1.0 - band))
        if key[2] - 1 >= 0:
            coords = xyz_local_01.clone(); coords[:, 2] = coords[:, 2] + 1.0
            add(mask_b, (key[0], key[1], key[2]-1), coords)
        if key[2] + 1 < Gz:
            coords = xyz_local_01.clone(); coords[:, 2] = coords[:, 2] - 1.0
            add(mask_f, (key[0], key[1], key[2]+1), coords)

        if not tasks:
            return out

        for mask, nkey, ncoords in tasks:
            neighbor = self.patch_map[nkey]
            neighbor["_no_seam"] = True
            neighbor["_key"] = nkey
            if kind == "density":
                sub = self.compute_density_patch(neighbor, ncoords[mask])  # [M]
            else:
                sub = self.compute_app_patch(neighbor, ncoords[mask])      # [M,D]
            neighbor.pop("_no_seam", None)
            neighbor.pop("_key", None)

            out_mask = out[mask]
            out[mask] = 0.5 * out_mask + 0.5 * sub  # simple 50/50; can be distance-weighted
        return out
    
    @contextmanager
    def no_seam_blend(self):
        """
        Temporarily disable seam blending (PUF/eval only).
        Usage:
            with tensorf.no_seam_blend():
                # do renderer(...) calls
        """
        old = bool(getattr(self, "enable_seam_blend", True))
        self.enable_seam_blend = False
        try:
            yield
        finally:
            self.enable_seam_blend = old
    
    def apply_seam_tying(self):
        """
        Hard seam tying 的預設入口。現在先做 no-op（不改參數），
        之後要做「邊界 slice 互拷」或「參數共享」就從這裡接。
        """
        return

    def compute_density_patch(self, patch, xyz_sampled):
        """
        Sample density features for one patch (before feature2density).
        Now supports both old (full tensors) and new (coefficients + shared basis) formats.
        """
        if patch.get('dead', False):
            return torch.zeros(xyz_sampled.shape[0], device=xyz_sampled.device)

        coord_plane, coord_line = self._get_patch_coords(xyz_sampled)
        N = xyz_sampled.shape[0]
        device = xyz_sampled.device
        
        sigma_acc = torch.zeros(N, device=device)
        res = tuple(patch['res'])
        
        # Check if using shared basis (new) or full tensors (old)
        use_shared_basis = ('density_plane' in patch) and self.global_basis_enable
        
        for i in range(3):
            if use_shared_basis:
                # NEW PATH: Reconstruct from coefficients + global basis
                global_plane, global_line = self._get_or_create_global_basis(res, "density", i)
                coef_p = patch['density_plane'][i]  # [rank, K]
                coef_l = patch['density_line'][i]   # [rank, K]
                
                plane_recon, line_recon = self._reconstruct_from_coef(
                    coef_p, coef_l, global_plane, global_line
                )
                
                pfeat = self._gs2d_1CHW(plane_recon, coord_plane[i])  # [N, rank]
                lfeat = self._gs1d_1CL1(line_recon,  coord_line[i])   # [N, rank]
            else:
                # OLD PATH: Direct sampling from full tensors
                plane = self._plane_param_for(patch, 'density', i)
                line  = self._line_param_for(patch, 'density', i)
                
                pfeat = self._gs2d_1CHW(plane, coord_plane[i])
                lfeat = self._gs1d_1CL1(line,  coord_line[i])
            
            sigma_acc += (pfeat * lfeat).sum(dim=1)

        # Residual (only active in interior for boundary continuity)
        if bool(getattr(self, "enable_child_residual", True)):
            if use_shared_basis:
                rpl = patch.get('density_plane_res', None)
                rln = patch.get('density_line_res',  None)
            else:
                rpl = patch.get('density_plane_res', None)
                rln = patch.get('density_line_res',  None)
            
            if (rpl is not None) and (rln is not None):
                g = self._interior_gate(xyz_sampled).squeeze(-1)  # [N]
                r_acc = torch.zeros_like(sigma_acc)
                
                for i in range(3):
                    if use_shared_basis:
                        global_plane, global_line = self._get_or_create_global_basis(res, "density", i)
                        rp_recon, rl_recon = self._reconstruct_from_coef(
                            rpl[i], rln[i], global_plane, global_line
                        )
                        rp = self._gs2d_1CHW(rp_recon, coord_plane[i])
                        rl = self._gs1d_1CL1(rl_recon, coord_line[i])
                    else:
                        rplane, rline = rpl[i], rln[i]
                        rp = self._gs2d_1CHW(rplane, coord_plane[i])
                        rl = self._gs1d_1CL1(rline,  coord_line[i])
                    
                    r_acc += (rp * rl).sum(dim=1)
                
                sigma_acc = sigma_acc + g * r_acc

        # Seam blending (inference-time smoothing)
        sigma_acc = self._blend_with_neighbors_if_needed(patch, xyz_sampled, sigma_acc, kind="density")
        return sigma_acc
    
    def compute_app_patch(self, patch, xyz):
        """
        Build appearance features and project via basis matrix.
        Now supports both old and new formats.
        """
        coord_plane, coord_line = self._get_patch_coords(xyz)
        N = xyz.shape[0]
        device = xyz.device
        res = tuple(patch['res'])
        
        use_shared_basis = ('app_plane' in patch) and self.global_basis_enable
        
        # Base features
        comps = []
        for i in range(3):
            if use_shared_basis:
                global_plane, global_line = self._get_or_create_global_basis(res, "app", i)
                coef_p = patch['app_plane'][i]
                coef_l = patch['app_line'][i]
                
                plane_recon, line_recon = self._reconstruct_from_coef(
                    coef_p, coef_l, global_plane, global_line
                )
                
                fx = self._gs2d_1CHW(plane_recon, coord_plane[i])
                gx = self._gs1d_1CL1(line_recon,  coord_line[i])
            else:
                px = self._plane_param_for(patch, 'app', i)
                lx = self._line_param_for(patch, 'app', i)
                
                fx = self._gs2d_1CHW(px, coord_plane[i])
                gx = self._gs1d_1CL1(lx, coord_line[i])
            
            comps.extend([fx, gx])
        
        feat = torch.cat(comps, dim=1)  # [N, C_total]

        # Interior-gated residuals
        if bool(getattr(self, "enable_child_residual", True)):
            if use_shared_basis:
                rpl = patch.get('app_plane_res', None)
                rln = patch.get('app_line_res',  None)
            else:
                rpl = patch.get('app_plane_res', None)
                rln = patch.get('app_line_res',  None)
            
            if (rpl is not None) and (rln is not None):
                rfeat_comps = []
                for i in range(3):
                    if use_shared_basis:
                        global_plane, global_line = self._get_or_create_global_basis(res, "app", i)
                        rp_recon, rl_recon = self._reconstruct_from_coef(
                            rpl[i], rln[i], global_plane, global_line
                        )
                        rp = self._gs2d_1CHW(rp_recon, coord_plane[i])
                        rl = self._gs1d_1CL1(rl_recon, coord_line[i])
                    else:
                        rp = self._gs2d_1CHW(rpl[i], coord_plane[i])
                        rl = self._gs1d_1CL1(rln[i], coord_line[i])
                    
                    rfeat_comps.extend([rp, rl])
                
                rfeat = torch.cat(rfeat_comps, dim=1)
                g = self._interior_gate(xyz)  # [N,1]
                feat = feat + g * rfeat

        self._ensure_basis_for_feat(patch, feat.shape[1])

        patch_key = patch.get("_key", (0, 0, 0))

        if self.use_shared_basis and hasattr(self, 'shared_basis_manager'):
            # 獲取 patch key
            patch_key = patch.get("_key", None)
            if patch_key is None:
                # 嘗試從 patch 反推 key
                for k, v in self.patch_map.items():
                    if v is patch:
                        patch_key = k
                        break
                if patch_key is None:
                    patch_key = (0, 0, 0)  # fallback
            
            out = self.shared_basis_manager(patch_key, feat)
        elif 'mix_W' in patch and 'basis_B' in patch:
            # 低秩分解路徑
            mid = torch.matmul(feat, patch['mix_W'].T)
            out = patch['basis_B'](mid)
        else:
            # 確保 basis_mat 存在
            self._ensure_basis_for_feat(patch, feat.shape[1])
            out = patch['basis_mat'](feat)

        out = self._blend_with_neighbors_if_needed(patch, xyz, out, kind="app")
        return out

    def compute_density_patchwise_fast(self, xyz, patch_coords):
        """
        Vectorized per-point density dispatch.
        Args:
            xyz:          [N,3]    (normalized coord)
            patch_coords: [N,3]    (int patch index)  
        Returns:
            out: [N]
        """
        patch_coords = patch_coords.long().view(-1, 3)

        uniq, inv = torch.unique(patch_coords, dim=0, return_inverse=True)
        out = torch.zeros(xyz.shape[0], device=xyz.device, dtype=xyz.dtype)

        for uidx in range(uniq.shape[0]):
            mask = (inv == uidx)
            if not mask.any():
                continue

            sub_xyz = xyz[mask]
            key = tuple(int(v) for v in uniq[uidx].tolist())
            patch = self.patch_map.get(key, None)
            if patch is None: 
                out[mask] = 0.0
                continue

            patch["_key"] = key  # needed by seam blending
            sigma_sub = self.compute_density_patch(patch, sub_xyz)  # [N]
            patch.pop("_key", None)

            if sigma_sub.dtype != out.dtype:
                sigma_sub = sigma_sub.to(out.dtype)
            out[mask] = sigma_sub

        return out

    def compute_app_patchwise_fast(self, xyz, patch_coords):
        """
        Vectorized per-point appearance dispatch.
        Args:
            xyz:          [N,3]    (normalized coord)
            patch_coords: [N,3]    (int patch index)  
        Returns:
            out: [N, app_dim]
        """
        patch_coords = patch_coords.long().view(-1, 3)

        uniq, inv = torch.unique(patch_coords, dim=0, return_inverse=True)
        out = torch.zeros((xyz.shape[0], self.app_dim), device=xyz.device, dtype=xyz.dtype)

        for uidx in range(uniq.shape[0]):
            mask = (inv == uidx)
            if not mask.any():
                continue

            sub_xyz = xyz[mask]
            key = tuple(int(v) for v in uniq[uidx].tolist())
            patch = self.patch_map.get(key, None)
            if patch is None:  
                out[mask] = 0.0
                continue
            
            patch["_key"] = key  # needed by seam blending
            feat_sub = self.compute_app_patch(patch, sub_xyz)  # [N, app_dim]
            patch.pop("_key", None)

            if feat_sub.dtype != out.dtype:
                feat_sub = feat_sub.to(out.dtype)
            out[mask] = feat_sub

        return out
    
    def get_progressive_resolution(self, iteration, patch_importance=0.5):
        """
        漸進式解析度策略
        根據訓練進度和 patch 重要性動態調整解析度
        """
        total_iters = getattr(self, 'total_iters', 30000)
        progress = min(1.0, iteration / total_iters)
        
        # 基礎解析度調度
        if progress < 0.1:
            base_res = [8, 8, 8]
        elif progress < 0.3:
            base_res = [16, 16, 16]
        elif progress < 0.6:
            base_res = [32, 32, 32]
        else:
            base_res = [48, 48, 48]
        
        # 根據重要性調整
        if patch_importance > 0.8:  # 重要 patch
            adjusted = [min(64, int(r * 1.5)) for r in base_res]
        elif patch_importance < 0.3:  # 不重要 patch
            adjusted = [max(8, int(r * 0.7)) for r in base_res]
        else:
            adjusted = base_res
        
        # 確保是 8 的倍數（對 GPU 友好）
        return tuple((r // 8) * 8 + (8 if r % 8 >= 4 else 0) for r in adjusted)
    
    def compute_patch_complexity(self, patch):
        """
        計算 patch 內容複雜度
        用於智能 split 決策和解析度調整
        """
        complexity_score = 0.0
        
        with torch.no_grad():
            for i in range(3):
                # Density 複雜度
                d_plane = patch['density_plane'][i]
                if d_plane.numel() > 0:
                    # 計算梯度幅度
                    grad_y = torch.abs(d_plane[..., 1:, :] - d_plane[..., :-1, :]).mean()
                    grad_x = torch.abs(d_plane[..., :, 1:] - d_plane[..., :, :-1]).mean()
                    complexity_score += (grad_y + grad_x).item()
                    
                    # 計算標準差
                    complexity_score += torch.std(d_plane).item()
                
                # App 複雜度
                a_plane = patch['app_plane'][i]
                if a_plane.numel() > 0:
                    grad_y = torch.abs(a_plane[..., 1:, :] - a_plane[..., :-1, :]).mean()
                    grad_x = torch.abs(a_plane[..., :, 1:] - a_plane[..., :, :-1]).mean()
                    complexity_score += (grad_y + grad_x).item() * 0.5  # app 權重較低
                    
                    complexity_score += torch.std(a_plane).item() * 0.5
        
        # 正規化到 [0, 1]
        return min(1.0, complexity_score / 6.0)
    
    def compute_patch_importance(self, patch, iteration):
        """
        計算 patch 的重要性分數，用於動態 rank 調整
        """
        # 基礎重要性：alpha mass
        alpha_mass = float(patch.get('alpha_mass', 0.0))
        
        # 計算梯度資訊（如果可用）
        grad_score = 0.0
        if hasattr(self, '_patch_grad_history'):
            key = patch.get('_key', None)
            if key and key in self._patch_grad_history:
                grad_score = self._patch_grad_history[key]
        
        # 訓練進度因子
        progress = min(1.0, iteration / 30000)
        
        # 綜合重要性
        importance = alpha_mass * (1.0 + grad_score * 0.5)
        
        # 隨進度調整的閾值
        threshold = 0.8 * (1.0 - progress * 0.5)  # 從 0.8 降到 0.4
        
        return importance, threshold

    def aggressive_rank_autoscale(self, iteration, alpha_keep_q=0.70):
        """
        更激進的 rank 自適應策略
        """
        if len(self.patch_map) == 0:
            return 0
        
        changed = 0
        progress = min(1.0, iteration / 30000)
        
        # 更激進的參數
        gamma = 0.4 + progress * 0.3  # 從 0.4 逐漸增加到 0.7
        keep_q = alpha_keep_q - progress * 0.2  # 從 0.7 逐漸降到 0.5
        
        for key, patch in list(self.patch_map.items()):
            importance, threshold = self.compute_patch_importance(patch, iteration)
            
            # 獲取當前 ranks
            cur_sig_axes = [int(patch['density_plane'][i].shape[1]) for i in range(3)]
            cur_app_axes = [int(patch['app_plane'][i].shape[1]) for i in range(3)]
            
            # 計算目標 ranks
            if importance < threshold:
                # 激進降低
                scale = 0.5 if progress > 0.5 else 0.7
            elif importance > threshold * 1.5:
                # 選擇性增加
                scale = 1.3 if progress < 0.3 else 1.1
            else:
                scale = 1.0
            
            new_sig_axes = [max(8, int(c * scale)) for c in cur_sig_axes]
            new_app_axes = [max(8, int(c * scale)) for c in cur_app_axes]
            
            # 應用上限
            max_rank = 96 if progress < 0.5 else 64
            new_sig_axes = [min(max_rank, c) for c in new_sig_axes]
            new_app_axes = [min(max_rank, c) for c in new_app_axes]
            
            if new_sig_axes != cur_sig_axes or new_app_axes != cur_app_axes:
                # 執行 resize（使用您現有的 _resize_one_factor_block）
                # ... resize 邏輯 ...
                changed += 1
        
        return changed

    @torch.no_grad()
    def strict_evenize_once(self, target_G, vm_reso, use_progressive=True, iteration=0):
        """
        Only change patch grid res here, keeping VM res per-patch.
        If use_progressive=True, use progressive resolution based on iteration and patch importance.
        """
        device = self.aabb.device
        Gx, Gy, Gz = map(int, target_G)
        
        # 如果啟用漸進式解析度，根據迭代次數調整基礎解析度
        if use_progressive:
            base_R = list(self.get_progressive_resolution(iteration, patch_importance=0.5))
            print(f"[PPR] Using progressive resolution: {base_R} at iteration {iteration}")
        else:
            R = list(vm_reso)
            base_R = R

        if not self.patch_map:
            self.ensure_default_patch()

        cur_keys = list(self.patch_map.keys())
        keys_t = torch.tensor(cur_keys, dtype=torch.long, device=device)

        def _clone_pl(pl):
            return torch.nn.ParameterList([
                torch.nn.Parameter(p.detach().clone().to(device), requires_grad=True) for p in pl
            ])

        new_map = {}
        for i in range(Gx):
            for j in range(Gy):
                for k in range(Gz):
                    key = (i, j, k)
                    
                    # 如果啟用漸進式，根據 patch 重要性調整解析度
                    if use_progressive and key in self.patch_map:
                        patch_importance = self.patch_map[key].get('alpha_mass', 0.5)
                        R = list(self.get_progressive_resolution(iteration, patch_importance))
                    else:
                        R = base_R[:]
                    
                    if key in self.patch_map:
                        p = self._ensure_patch_device(self.patch_map[key], device)
                        
                        # 如果解析度改變，需要 upsample/downsample
                        src_R = list(p.get('res', R))
                        if tuple(src_R) != tuple(R):
                            print(f"[PPR] Patch {key}: {src_R} -> {R}")
                            dpl, dln = self.upsample_VM(p["density_plane"], p["density_line"], R)
                            apl, aln = self.upsample_VM(p["app_plane"], p["app_line"], R)
                            p["density_plane"] = dpl
                            p["density_line"] = dln
                            p["app_plane"] = apl
                            p["app_line"] = aln
                        
                        p['res'] = R[:]
                        new_map[key] = p
                        continue

                    # 創建新 patch（從最近鄰居複製）
                    q = torch.tensor([[i, j, k]], dtype=torch.long, device=device)
                    nn_idx  = self._cheby_nearest_keys(keys_t, q).item()
                    src_key = cur_keys[nn_idx]
                    src     = self.patch_map[src_key]

                    new_patch = self._create_patch(R, device)

                    src_R = list(src.get('res', R))
                    if tuple(src_R) != tuple(R):
                        dpl, dln = self.upsample_VM(src["density_plane"], src["density_line"], R)
                        apl, aln = self.upsample_VM(src["app_plane"], src["app_line"], R)
                        new_patch["density_plane"] = dpl
                        new_patch["density_line"] = dln
                        new_patch["app_plane"] = apl
                        new_patch["app_line"] = aln
                    else:
                        new_patch["density_plane"] = _clone_pl(src["density_plane"])
                        new_patch["density_line"] = _clone_pl(src["density_line"])
                        new_patch["app_plane"] = _clone_pl(src["app_plane"])
                        new_patch["app_line"] = _clone_pl(src["app_line"])

                    if "basis_mat" in src:
                        new_in  = self._app_in_dim_from_vm(new_patch["app_plane"], new_patch["app_line"])
                        new_patch["basis_mat"] = self.get_shared_basis(new_in, self.app_dim)
                    
                    new_patch["res"] = R[:]
                    new_map[key] = self._ensure_patch_device(new_patch, device)

        self.patch_map = new_map
        for k, p in self.patch_map.items():
            p['depth'] = 0
        self.base_patch_grid_reso = tuple(target_G)
        self.patch_grid_reso = (Gx, Gy, Gz)
        self.current_patch_keys = list(self.patch_map.keys())
        
        # 更新 gridSize 為最大解析度
        max_res = max(R)
        self.gridSize = torch.LongTensor([max_res, max_res, max_res]).to(device)
        self.update_stepSize([max_res, max_res, max_res])

        try:
            self.assert_zero_origin_and_contiguous()
        except Exception as e:
            print(f"[WARN] assert_zero_origin_and_contiguous failed: {e}")

    def _resolve_caps(self, args):
        cap_single = getattr(args, "max_rank", None)              # int or None
        cap_sigma  = getattr(args, "rank_cap_sigma", cap_single)  # int | [v] | [vx,vy,vz] | [] | None
        cap_app    = getattr(args, "rank_cap_app",   cap_single)

        def _norm_cap(x):
            if x is None:
                return None
            if isinstance(x, (list, tuple)):
                if len(x) == 0:
                    return None
                if len(x) == 1:
                    return int(x[0])
                if len(x) == 3:
                    return [max(1, int(v)) for v in x]  
                raise ValueError(f"rank_cap expects 1 or 3 ints, got {x}")
            return max(1, int(x))

        return _norm_cap(cap_sigma), _norm_cap(cap_app)

    def _current_ranks_from_patches(self):
        if not getattr(self, "patch_map", None):
            return list(getattr(self, "density_n_comp")), list(getattr(self, "app_n_comp"))
        
        sig_now = [max(int(p['density_plane'][i].shape[1]) for p in self.patch_map.values()) for i in range(3)]
        app_now = [max(int(p['app_plane'][i].shape[1])     for p in self.patch_map.values()) for i in range(3)]
        return sig_now, app_now
    
    @torch.no_grad()
    def _resize_one_factor_block(self, planes, lines, c_new):
        """
        planes: nn.ParameterList with 3 tensors [1, C_old, H, W] (one per mode)
        lines:  nn.ParameterList with 3 tensors [1, C_old, L, 1]
        Return new ParameterList (planes, lines) with C_new, copy-over min(C_old, C_new) channels.
        """
        new_planes, new_lines = [], []
        dev = self.aabb.device
        for i in range(len(planes)):
            p = planes[i].detach()
            l = lines[i].detach()

            c_old = p.shape[1]
            H, W  = p.shape[-2], p.shape[-1]
            L     = l.shape[-2]

            p_new = torch.zeros((1, c_new, H, W), device=dev, dtype=p.dtype)
            l_new = torch.zeros((1, c_new, L, 1), device=dev, dtype=l.dtype)

            c_copy = min(c_old, c_new)
            if c_copy > 0:
                p_new[:, :c_copy] = p[:, :c_copy]
                l_new[:, :c_copy] = l[:, :c_copy]

            new_planes.append(torch.nn.Parameter(p_new, requires_grad=True))
            new_lines.append(torch.nn.Parameter(l_new, requires_grad=True))

        return torch.nn.ParameterList(new_planes), torch.nn.ParameterList(new_lines)
    
    @torch.no_grad()
    def selective_rank_upgrade(self, sigma_target, app_target,
                            min_patch_res=8, budget_mb=0.0,
                            importance_mode='alpha',
                            verbose=True):
        """
        Upgrade ranks for patches where min(res) >= min_patch_res.
        This version also upgrades *_res (residual) blocks to the SAME target ranks
        to keep feat and rfeat aligned in channel dimension.
        """
        dev = self.aabb.device
        some_param = next((p for p in self.parameters() if p.requires_grad), None)
        dtype_bytes = 4
        if some_param is not None and some_param.dtype in (torch.float16, torch.bfloat16):
            dtype_bytes = 2

        def _to_vec(t):
            if isinstance(t, (list, tuple)):
                assert len(t) == 3
                return [int(t[0]), int(t[1]), int(t[2])]
            return [int(t), int(t), int(t)]
        
        sigma_t = _to_vec(sigma_target)
        app_t   = _to_vec(app_target)

        def _planes_lines_added_bytes(planes, lines, addC_axes):
            total = 0
            for i in range(3):
                addC = int(addC_axes[i])
                if addC <= 0:
                    continue
                H, W = int(planes[i].shape[-2]), int(planes[i].shape[-1])
                L    = int(lines[i].shape[-2])
                elems = addC * ((H * W) + L)
                total += elems
            return total * dtype_bytes
        
        candidates = []  # (importance, delta_bytes, key, tgt_sig_axes, tgt_app_axes)
        for key, p in self.patch_map.items():
            rmin = int(min(p.get('res', self.gridSize)))
            if rmin < int(min_patch_res):
                continue

            cur_sig = [int(p['density_plane'][i].shape[1]) for i in range(3)]
            cur_app = [int(p['app_plane'][i].shape[1])     for i in range(3)]

            tgt_sig_axes = [max(cur_sig[i], int(sigma_t[i])) for i in range(3)]
            tgt_app_axes = [max(cur_app[i], int(app_t[i]))   for i in range(3)]

            add_bytes = _planes_lines_added_bytes(p['density_plane'], p['density_line'],
                                                [tgt_sig_axes[i] - cur_sig[i] for i in range(3)])
            add_bytes += _planes_lines_added_bytes(p['app_plane'], p['app_line'],
                                                [tgt_app_axes[i] - cur_app[i] for i in range(3)])

            imp = float(p.get('alpha_mass', 0.0))
            candidates.append((imp, int(add_bytes), key, tgt_sig_axes, tgt_app_axes))

        if not candidates:
            if verbose:
                print(f"[rank] no eligible patches (min_patch_res={min_patch_res}).")
            return 0

        candidates.sort(key=lambda x: (-x[0], x[1]))
        byte_budget = float('inf') if (budget_mb is None or budget_mb <= 0) else int(budget_mb * 1024 * 1024)

        picked, acc = [], 0
        for imp, dbytes, key, tgt_sig_axes, tgt_app_axes in candidates:
            if acc + dbytes > byte_budget:
                continue
            picked.append((key, tgt_sig_axes, tgt_app_axes))
            acc += dbytes

        if not picked:
            if verbose:
                print(f"[rank] budget exhausted: 0 picked (budget_mb={budget_mb}).")
            return 0

        upcnt = 0
        for key, tgt_sig_axes, tgt_app_axes in picked:
            P = self.patch_map[key]

            # ---- sigma (base) ----
            new_dp, new_dl = [], []
            for i in range(3):
                curC = int(P['density_plane'][i].shape[1])
                needC = int(tgt_sig_axes[i])
                if needC != curC:
                    dpi, dli = self._resize_one_factor_block([P['density_plane'][i]], [P['density_line'][i]], c_new=needC)
                    new_dp.append(dpi[0]); new_dl.append(dli[0])
                else:
                    new_dp.append(P['density_plane'][i]); new_dl.append(P['density_line'][i])
            P['density_plane'] = new_dp; P['density_line'] = new_dl

            # ---- sigma (residual) keep in sync if present ----
            if ('density_plane_res' in P) and ('density_line_res' in P):
                new_dp_res, new_dl_res = [], []
                for i in range(3):
                    curC = int(P['density_plane_res'][i].shape[1])
                    needC = int(tgt_sig_axes[i])
                    if needC != curC:
                        dpi, dli = self._resize_one_factor_block([P['density_plane_res'][i]], [P['density_line_res'][i]], c_new=needC)
                        new_dp_res.append(dpi[0]); new_dl_res.append(dli[0])
                    else:
                        new_dp_res.append(P['density_plane_res'][i]); new_dl_res.append(P['density_line_res'][i])
                P['density_plane_res'] = torch.nn.ParameterList(new_dp_res)
                P['density_line_res']  = torch.nn.ParameterList(new_dl_res)

            # ---- app (base) ----
            new_ap, new_al = [], []
            app_changed = False
            for i in range(3):
                curC = int(P['app_plane'][i].shape[1])
                needC = int(tgt_app_axes[i])
                if needC != curC:
                    api, ali = self._resize_one_factor_block([P['app_plane'][i]], [P['app_line'][i]], c_new=needC)
                    new_ap.append(api[0]); new_al.append(ali[0])
                    app_changed = True
                else:
                    new_ap.append(P['app_plane'][i]); new_al.append(P['app_line'][i])
            P['app_plane'] = new_ap; P['app_line'] = new_al

            # ---- app (residual) keep in sync if present ----
            if ('app_plane_res' in P) and ('app_line_res' in P):
                new_ap_res, new_al_res = [], []
                for i in range(3):
                    curC = int(P['app_plane_res'][i].shape[1])
                    needC = int(tgt_app_axes[i])
                    if needC != curC:
                        api, ali = self._resize_one_factor_block([P['app_plane_res'][i]], [P['app_line_res'][i]], c_new=needC)
                        new_ap_res.append(api[0]); new_al_res.append(ali[0])
                    else:
                        new_ap_res.append(P['app_plane_res'][i]); new_al_res.append(P['app_line_res'][i])
                P['app_plane_res'] = torch.nn.ParameterList(new_ap_res)
                P['app_line_res']  = torch.nn.ParameterList(new_al_res)

            if app_changed and hasattr(self, '_app_in_dim_from_vm'):
                new_in = self._app_in_dim_from_vm(P['app_plane'], P['app_line'])
                P['basis_mat'] = self.get_shared_basis(new_in, self.app_dim)

            upcnt += 1

        if verbose:
            print(f"[rank] upgraded {upcnt} patches | Δ≈{acc/1024/1024:.2f} MB | min_res≥{min_patch_res} | budget_mb={budget_mb}")
        return upcnt

    @torch.no_grad()
    def selective_rank_autoscale(self, *, ref_res:int = 64, gamma:float = 0.60,
                                c_sig_base:Optional[int] = None, c_app_base:Optional[int] = None,
                                c_min:Optional[int] = None, c_max:Optional[int] = None,
                                cap_sig: Optional[Sequence[int]] = None, 
                                cap_app: Optional[Sequence[int]] = None,
                                round_to:int = 4,           # rank rounds to multiple of this
                                alpha_keep_q:float = 0.85,  # keep-quantile based on alpha_mass
                                allow_down:bool = True, allow_up:bool = False,    
                                verbose:bool = True):
        """
        Auto-determine target per-patch rank by min(res).
        Downsize by default; optional upgrade if allow_up=True.

        IMPORTANT: This version keeps residual blocks (*_res) in sync with base tensors
                   using the SAME keep indices so that feat and rfeat remain channel-aligned.
        """
        if cap_sig is None:
            cap_sig = getattr(self, "rank_cap_sigma", None)
        if cap_app is None:
            cap_app = getattr(self, "rank_cap_app", None)
        
        if c_sig_base is None or c_app_base is None:
            c_sig_base = max(16, int(getattr(self, "rank_base_floor_sig", 16)))
            c_app_base = max(48, int(getattr(self, "rank_base_floor_app", 48)))
        if c_min is None:
            c_min = max(8, int(getattr(self, "min_rank", 8)))
        if c_max is None:
            c_max = int(getattr(self, "max_rank", 96))

        def _apply_axis_caps(vec3, cap):
            if cap is None:
                return vec3
            if isinstance(cap, int):
                return [min(v, int(cap)) for v in vec3]
            if isinstance(cap, (list, tuple)) and len(cap) == 3:
                return [min(int(v), int(c)) for v, c in zip(vec3, cap)]
            return vec3
        
        def _round_to_mult(x: float, m: int) -> int:
            return int(max(m, m * int(round(float(x) / m))))

        def _target_from_res(rmin: int, base: int) -> int:
            raw = float(base) * (max(rmin, 1) / max(ref_res, 1))**float(gamma)
            round_raw = _round_to_mult(raw, round_to)
            return int(np.clip(round_raw, c_min, c_max))

        def _distribute_anisotropic(c_base: int, base_triplet: Sequence[int], *,
                                    cmin:int, cmax:int, round_to:int) -> List[int]:
            base = torch.tensor(base_triplet, dtype=torch.float32)
            if base.mean() <= 0:
                base = torch.ones_like(base)
            ratio = base / base.mean()
            raw = c_base * ratio

            out = []
            for v in raw.tolist():
                vv = int(round(v / round_to) * round_to)
                vv = int(max(cmin, min(cmax, vv)))
                out.append(vv)
            return out
        
        @torch.no_grad()
        def _topk_resize_one_mode_with_idx(plane_1CHW: torch.Tensor, line_1CL1: torch.Tensor, c_new: int):
            """
            Resize one axis to c_new using Top-K energy selection.
            Returns: (plane_new, line_new, keep_idx)
            - keep_idx is a 1D LongTensor of indices used when downscaling.
            - When upscaling, keep_idx is arange(c_old), i.e., original channels copied to front.
            """
            device = plane_1CHW.device
            c_old = int(plane_1CHW.shape[1])
            H, W = int(plane_1CHW.shape[-2]), int(plane_1CHW.shape[-1])
            L    = int(line_1CL1.shape[-2])

            if c_new == c_old:
                keep = torch.arange(c_old, device=device)
                return (plane_1CHW, line_1CL1, keep)

            if c_new > c_old:
                plane_new = torch.zeros((1, c_new, H, W), device=device, dtype=plane_1CHW.dtype)
                line_new  = torch.zeros((1, c_new, L, 1), device=device, dtype=line_1CL1.dtype)
                plane_new[:, :c_old] = plane_1CHW.detach().contiguous()
                line_new[:,  :c_old] = line_1CL1.detach().contiguous()
                keep = torch.arange(c_old, device=device)
                return (torch.nn.Parameter(plane_new, requires_grad=True),
                        torch.nn.Parameter(line_new,  requires_grad=True),
                        keep)

            # downsize: Top-K score = ||plane|| * ||line||
            p_norm = plane_1CHW.detach().pow(2).sum(dim=(2, 3)).sqrt().squeeze(0)  # [C_old]
            l_norm = line_1CL1.detach().pow(2).sum(dim=(2, 3)).sqrt().squeeze(0)   # [C_old]
            score  = (p_norm * l_norm).clamp_min(1e-12)
            keep   = torch.topk(score, k=int(c_new), largest=True).indices.sort().values

            plane_new = plane_1CHW.detach()[:, keep, :, :].clone().contiguous()
            line_new  = line_1CL1.detach()[:,  keep, :, :].clone().contiguous()
            return (torch.nn.Parameter(plane_new, requires_grad=True),
                    torch.nn.Parameter(line_new,  requires_grad=True),
                    keep)

        if len(self.patch_map) == 0:
            if verbose: print("[rank-autoscale] no patches; skip.")
            return 0

        # guarded by alpha_mass
        masses = [float(p.get('alpha_mass', 0.0)) for p in self.patch_map.values()]
        thr = np.quantile(masses, alpha_keep_q) if len(masses) and (0.0 < alpha_keep_q < 1.0) else float("inf")

        changed = 0
        mem_before = int(self.get_total_mem() / 1024**2) if hasattr(self, 'get_total_mem') else -1
        dev = self.aabb.device

        for key, P in list(self.patch_map.items()):
            # min(res) as scale anchor
            try:
                rmin = int(min(P['res']))
            except Exception:
                rmin = int(min(self.gridSize)) if hasattr(self, "gridSize") else 8

            csig_scalar = _target_from_res(rmin, c_sig_base)
            capp_scalar = _target_from_res(rmin, c_app_base)
            cur_sig_axes = [int(P['density_plane'][i].shape[1]) for i in range(3)]
            cur_app_axes = [int(P['app_plane'][i].shape[1])     for i in range(3)]

            tgt_sig_axes = _distribute_anisotropic(csig_scalar, cur_sig_axes, cmin=c_min, cmax=c_max, round_to=round_to)
            tgt_app_axes = _distribute_anisotropic(capp_scalar, cur_app_axes, cmin=c_min, cmax=c_max, round_to=round_to)
            tgt_sig_axes = _apply_axis_caps(tgt_sig_axes, cap_sig)
            tgt_app_axes = _apply_axis_caps(tgt_app_axes, cap_app)

            imp = float(P.get('alpha_mass', 0.0))

            def _apply_gate(cur_axes, tgt_axes):
                out = cur_axes[:]
                if imp >= thr:
                    if allow_up:
                        out = [max(o, t) for o, t in zip(out, tgt_axes)]
                else:
                    if allow_down:
                        out = [min(o, t) for o, t in zip(out, tgt_axes)]
                    if allow_up:
                        out = [max(o, t) for o, t in zip(out, tgt_axes)]
                return out

            new_sig_axes = _apply_gate(cur_sig_axes, tgt_sig_axes)
            new_app_axes = _apply_gate(cur_app_axes, tgt_app_axes)

            if new_sig_axes == cur_sig_axes and new_app_axes == cur_app_axes:
                continue

            # === Density branch (base + residual kept in sync) ===
            new_dp, new_dl = [], []
            new_dp_res, new_dl_res = [], []
            has_sig_res = ('density_plane_res' in P) and ('density_line_res' in P)

            for i in range(3):
                p_new, l_new, keep = _topk_resize_one_mode_with_idx(P['density_plane'][i], P['density_line'][i], int(new_sig_axes[i]))
                new_dp.append(p_new); new_dl.append(l_new)

                if has_sig_res:
                    # resize residual using the SAME keep indices to keep channel alignment
                    rpi = P['density_plane_res'][i]
                    rli = P['density_line_res'][i]
                    c_new = int(new_sig_axes[i])
                    c_old_res = int(rpi.shape[1])

                    if c_new == c_old_res:
                        new_dp_res.append(rpi); new_dl_res.append(rli)
                    else:
                        H_res, W_res = int(rpi.shape[-2]), int(rpi.shape[-1])
                        L_res        = int(rli.shape[-2])
                        rp_new = torch.zeros((1, c_new, H_res, W_res), device=dev, dtype=rpi.dtype)
                        rl_new = torch.zeros((1, c_new, L_res, 1),     device=dev, dtype=rli.dtype)

                        # copy-aligned channels
                        if c_old_res > 0:
                            k = min(len(keep), c_old_res)
                            rp_new[:, :k] = rpi.detach()[:, keep[:k], :, :]
                            rl_new[:, :k] = rli.detach()[:, keep[:k], :, :]

                        new_dp_res.append(torch.nn.Parameter(rp_new, requires_grad=True))
                        new_dl_res.append(torch.nn.Parameter(rl_new, requires_grad=True))

            P['density_plane'] = new_dp; P['density_line'] = new_dl
            if has_sig_res:
                P['density_plane_res'] = torch.nn.ParameterList(new_dp_res)
                P['density_line_res']  = torch.nn.ParameterList(new_dl_res)

            # === Appearance branch (base + residual kept in sync) ===
            new_ap, new_al = [], []
            new_ap_res, new_al_res = [], []
            has_app_res = ('app_plane_res' in P) and ('app_line_res' in P)
            app_changed = False

            for i in range(3):
                p_new, l_new, keep = _topk_resize_one_mode_with_idx(P['app_plane'][i], P['app_line'][i], int(new_app_axes[i]))
                app_changed |= (int(P['app_plane'][i].shape[1]) != int(new_app_axes[i]))
                new_ap.append(p_new); new_al.append(l_new)

                if has_app_res:
                    rpi = P['app_plane_res'][i]
                    rli = P['app_line_res'][i]
                    c_new = int(new_app_axes[i])
                    c_old_res = int(rpi.shape[1])

                    if c_new == c_old_res:
                        new_ap_res.append(rpi); new_al_res.append(rli)
                    else:
                        H_res, W_res = int(rpi.shape[-2]), int(rpi.shape[-1])
                        L_res        = int(rli.shape[-2])
                        rp_new = torch.zeros((1, c_new, H_res, W_res), device=dev, dtype=rpi.dtype)
                        rl_new = torch.zeros((1, c_new, L_res, 1),     device=dev, dtype=rli.dtype)

                        if c_old_res > 0:
                            k = min(len(keep), c_old_res)
                            rp_new[:, :k] = rpi.detach()[:, keep[:k], :, :]
                            rl_new[:, :k] = rli.detach()[:, keep[:k], :, :]

                        new_ap_res.append(torch.nn.Parameter(rp_new, requires_grad=True))
                        new_al_res.append(torch.nn.Parameter(rl_new, requires_grad=True))

            P['app_plane'] = new_ap; P['app_line'] = new_al
            if has_app_res:
                P['app_plane_res'] = torch.nn.ParameterList(new_ap_res)
                P['app_line_res']  = torch.nn.ParameterList(new_al_res)

            if app_changed and 'basis_mat' in P and hasattr(self, '_app_in_dim_from_vm'):
                new_in = self._app_in_dim_from_vm(P['app_plane'], P['app_line'])
                P['basis_mat'] = self.get_shared_basis(new_in, self.app_dim)

            changed += 1

        mem_after = int(self.get_total_mem() / 1024**2) if hasattr(self, 'get_total_mem') else -1
        if verbose:
            print(f"[rank-autoscale] changed={changed} | mem {mem_before} → {mem_after} MB | keep_q={alpha_keep_q:.2f} ref_res={ref_res} γ={gamma}")
        return changed

    @torch.no_grad()
    def rank_prune_once(self, keep_ratio: float = 0.75):
        for k, p in self.patch_map.items():
            if ('mix_W' not in p) or ('basis_B' not in p): 
                continue
            W = p['mix_W']  # [r, in_dim]
            r, d = W.shape
            keep = max(1, int(round(r * keep_ratio)))
         
            l2 = torch.norm(W, dim=1)  # [r]
            idx = torch.topk(l2, k=keep, largest=True).indices
          
            W_new = torch.nn.Parameter(W[idx].clone(), requires_grad=True)  # [keep, d]
            B_old = p['basis_B']  # Linear(in=rank, out=app_dim)
            B_new = torch.nn.Linear(keep, B_old.out_features, bias=False, device=W.device, dtype=W.dtype)
    
            with torch.no_grad():
                B_new.weight.copy_(B_old.weight[:, idx])
            p['mix_W']  = W_new
            p['basis_B'] = B_new

    def rank_regrow(self, add_rows: int = 0, std: float = 1e-3):
        if add_rows <= 0: return
        for k, p in self.patch_map.items():
            if ('mix_W' not in p) or ('basis_B' not in p): 
                continue
            W = p['mix_W']
            r, d = W.shape
            extra = torch.randn((add_rows, d), device=W.device, dtype=W.dtype) * std
            p['mix_W'] = torch.nn.Parameter(torch.cat([W, extra], dim=0), requires_grad=True)
       
            B = p['basis_B']
            B_enlarged = torch.nn.Linear(r + add_rows, B.out_features, bias=False, device=W.device, dtype=W.dtype)
            with torch.no_grad():
                B_enlarged.weight.zero_()
                B_enlarged.weight[:, :r].copy_(B.weight)
            p['basis_B'] = B_enlarged

    def _current_ranks_from_patches(self):
        """
        Obtain current max rank along each axis from patch map.

        Return:
          (cur_sigma[3], cur_app[3]).
        """
        if getattr(self, "patch_map", None):
            sig = [0, 0, 0]
            app = [0, 0, 0]
            for p in self.patch_map.values():
                for i in range(3):
                    sig[i] = max(sig[i], int(p['density_plane'][i].shape[1]))
                    app[i] = max(app[i], int(p['app_plane'][i].shape[1]))
            return sig, app
        return [int(x) for x in getattr(self, "density_n_comp")], [int(x) for x in getattr(self, "app_n_comp")]

    def try_rank_upgrade(self, args, *,
                         reso_hint: Optional[Sequence[int]] = None,
                         iteration: Optional[int] = None) -> Dict[str, Any]:
        """
        Only upgrade rank without any downscaling.
        
        Return dict: 
                {'up': int, 'down': 0, 'promoted': int, 'reso_hint': (x,y,z),
                'sigma_before': (..), 'sigma_after': (..), 'app_before': (..), 'app_after': (..)}
        """
        if not bool(getattr(args, "dynamic_rank", True)):
            return {"up": 0, "down": 0, "promoted": 0, "skipped": "disabled"}

        it = int(iteration) if iteration is not None else None

        base_warmup  = getattr(args, "rank_warmup", 0)
        base_cooldown= getattr(args, "rank_cooldown", 600)
        freeze_after = getattr(args, "rank_freeze_after", None)

        first_split = min(getattr(args, "split_even_kicks", []) or [10**9])
        first_ups   = min(getattr(args, "vm_upsamp_list",   []) or [10**9])
        first_event = min(first_split, first_ups, 10**9)
 
        margin = int(getattr(args, "rank_warmup_margin", 200))
        eff_warmup = int(base_warmup if base_warmup is not None else 0)
        if first_event < 10**9:
            eff_warmup = min(eff_warmup, max(0, first_event - margin))
        eff_cooldown = int(base_cooldown if base_cooldown is not None else 0)

        last_it = getattr(self, "_last_rank_resize_iter", None)

        if it is not None:
            if it < eff_warmup:
                return {"up": 0, "down": 0, "promoted": 0, "skipped": "warmup",
                        "warmup_until": int(eff_warmup), "iter": int(it)}
            if freeze_after is not None and it >= int(freeze_after):
                return {"up": 0, "down": 0, "promoted": 0, "skipped": "frozen",
                        "freeze_after": int(freeze_after), "iter": int(it)}
            if last_it is not None and (it - int(last_it)) < eff_cooldown:
                return {"up": 0, "down": 0, "promoted": 0, "skipped": "cooldown",
                        "cooldown_left": int(eff_cooldown - (it - int(last_it))),
                        "last_resize_iter": int(last_it), "iter": int(it)}

        def _vec3(v) -> Tuple[int, int, int]:
            if isinstance(v, (list, tuple)):
                if len(v) == 3: return int(v[0]), int(v[1]), int(v[2])
                if len(v) == 1: return int(v[0]), int(v[0]), int(v[0])
            if isinstance(v, int): return (v, v, v)
            raise ValueError(f"Cannot vec3: {v}")

        def _infer_R() -> Tuple[int, int, int]:
            if reso_hint is not None:
                try: return _vec3(reso_hint)
                except Exception: pass
            for a in ("reso_cur", "grid_size", "gridSize", "vm_reso", "voxel_reso", "reso"):
                if hasattr(self, a):
                    try: return _vec3(getattr(self, a))
                    except Exception: pass
            return (64, 64, 64)  
        
        R = _infer_R()
        Rmin = int(min(R))

        cur_sigma, cur_app = self._current_ranks_from_patches()

        if not hasattr(self, "rank_policy"):
            return {"up": 0, "down": 0, "promoted": 0, "skipped": "no_rank_policy"}

        tgt_sigma, tgt_app = self.rank_policy(R)

        def _norm3(x, fallback):
            if isinstance(x, (list, tuple)):
                if len(x) == 3: return [int(v) for v in x]
                if len(x) == 1: return [int(x[0])] * 3
            if isinstance(x, int): return [x, x, x]
            return list(fallback)

        policy_sigma_raw = _norm3(tgt_sigma, cur_sigma)
        policy_app_raw   = _norm3(tgt_app,   cur_app)

        tgt_sigma = list(policy_sigma_raw)
        tgt_app   = list(policy_app_raw)

        # ===== Flexible dynamic floors (args-driven, baseline-aware) =====
        def _scalar_from_triplet(x, default):
            if x is None:
                return int(default)
            if isinstance(x, int):
                return int(x)
            if isinstance(x, (list, tuple)):
                if len(x) == 1:
                    return int(x[0])
                if len(x) == 3:
                    return int(max(x))
            try:
                return int(x)
            except Exception:
                return int(default)

        sig_src = getattr(args, "rank_base_floor_sig", None)
        app_src = getattr(args, "rank_base_floor_app", None)

        base_floor_sig = _scalar_from_triplet(sig_src, 16)
        base_floor_app = _scalar_from_triplet(app_src, 48)

        floor_mode = str(getattr(args, "rank_floor_mode", "steps")).lower()
        round_to   = int(getattr(args, "rank_floor_round_to", 4))
        ref_res    = int(getattr(args, "vm_reso_max", 64))
        Rmin       = int(min(R))  

        steps_arg = getattr(args, "rank_floor_steps", "")
        step_table = []
        if isinstance(steps_arg, str) and steps_arg.strip():
            try:
                for item in steps_arg.split("|"):
                    key, pair = item.split(":")
                    s,a = pair.split(",")
                    step_table.append((int(key), int(s), int(a)))
                step_table.sort(key=lambda x: x[0])
            except Exception:
                step_table = []
        if not step_table:
            step_table = [
                (16, base_floor_sig,      base_floor_app),
                (32, base_floor_sig + 4,  base_floor_app + 8),
                (64, base_floor_sig + 12, base_floor_app + 24),
            ]

        def _round_clip(v, mx):
            v = int(round(v / round_to) * round_to)
            return max(round_to, min(v, mx))

        if floor_mode == "scale":
            anchor = int(getattr(args, "rank_floor_scale_anchor", 16))
            beta   = float(getattr(args, "rank_floor_scale_beta", 0.5))
            scale  = (max(Rmin, anchor) / float(anchor)) ** beta
            dyn_sig = int(round(base_floor_sig * scale))
            dyn_app = int(round(base_floor_app * scale))
        else:
            dyn_sig, dyn_app = base_floor_sig, base_floor_app
            for k, s, a in step_table:
                if Rmin >= k:
                    dyn_sig, dyn_app = s, a
                else:
                    break

        max_rank_glob = int(getattr(args, "max_rank", 96))
        cap_sigma, cap_app = self._resolve_caps(args)  

        def _cap3(val, cap):
            if cap is None: return [val]*3
            if isinstance(cap, int): return [min(val, cap)]*3
            if isinstance(cap, (list, tuple)) and len(cap) == 3:
                return [min(val, int(c)) for c in cap]
            return [val]*3

        def _apply_cap(vec, cap):
            # vec: List[int] length=3
            # cap: None | int | length=3 (list/tuple)
            if cap is None:
                return list(vec)
            if isinstance(cap, int):
                return [min(int(v), cap) for v in vec]
            if isinstance(cap, (list, tuple)) and len(cap) == 3:
                return [min(int(v), int(c)) for v, c in zip(vec, cap)]
            return list(vec)

        def _norm_cap_for_log(cap):
            if cap is None: return None
            if isinstance(cap, int): return (int(cap), int(cap), int(cap))
            if isinstance(cap, (list, tuple)) and len(cap) == 3:
                return (int(cap[0]), int(cap[1]), int(cap[2]))
            return None

        floor_sig_vec = _cap3(dyn_sig, cap_sigma)
        floor_app_vec = _cap3(dyn_app, cap_app)
        floor_sig_vec = [_round_clip(v, max_rank_glob) for v in floor_sig_vec]
        floor_app_vec = [_round_clip(v, max_rank_glob) for v in floor_app_vec]

        tgt_sigma_capped = _apply_cap(tgt_sigma, cap_sigma)
        tgt_app_capped   = _apply_cap(tgt_app,   cap_app)
        tgt_sigma = [max(c, f, t) for c, f, t in zip(cur_sigma, floor_sig_vec, tgt_sigma_capped)]
        tgt_app   = [max(c, f, t) for c, f, t in zip(cur_app,   floor_app_vec, tgt_app_capped)]

        min_res_arg  = int(getattr(args, "rank_min_res", 8))
        min_res_eff  = int(min(min_res_arg, Rmin))

        def _any_patch_needs_upgrade(target_sig3, target_app3):
            for p in self.patch_map.values():
                cur_sig_axes = [int(p['density_plane'][i].shape[1]) for i in range(3)]
                cur_app_axes = [int(p['app_plane'][i].shape[1])     for i in range(3)]
                for i in range(3):
                    if cur_sig_axes[i] < int(target_sig3[i]) or cur_app_axes[i] < int(target_app3[i]):
                        return True
            return False

        if not _any_patch_needs_upgrade(tgt_sigma, tgt_app):
            return {"up": 0, "down": 0, "promoted": 0, "skipped": "unchanged",
                    "reso_hint": tuple(R),
                    "sigma_before": tuple(cur_sigma), "sigma_after": tuple(cur_sigma),
                    "app_before": tuple(cur_app),     "app_after": tuple(cur_app),
                    "base_floor": (base_floor_sig, base_floor_app),
                    "floor_eff_sig": tuple(int(v) for v in floor_sig_vec),
                    "floor_eff_app": tuple(int(v) for v in floor_app_vec),
                    "policy_sigma_raw": tuple(int(v) for v in policy_sigma_raw),
                    "policy_app_raw":   tuple(int(v) for v in policy_app_raw),
                    "tgt_sigma_capped": tuple(int(v) for v in tgt_sigma_capped),
                    "tgt_app_capped":   tuple(int(v) for v in tgt_app_capped),
                    "cap_sigma_vec": _norm_cap_for_log(cap_sigma),
                    "cap_app_vec":   _norm_cap_for_log(cap_app),
                    "min_res_eff": int(min_res_eff),
                    "round_to": int(round_to),
                    "max_rank_glob": int(max_rank_glob),
                    "floor_mode": floor_mode,
                    "steps": steps_arg,
            }

        min_res_arg = int(getattr(args, "rank_min_res", 8))
        promoted = self.selective_rank_upgrade(sigma_target=tgt_sigma, app_target=tgt_app,
                                               min_patch_res=min(min_res_arg, Rmin),
                                               budget_mb=float(getattr(args, "rank_budget_mb", 0.0)),
                                               importance_mode=str(getattr(args, "rank_importance", "alpha")),
                                               verbose=bool(getattr(args, "rank_verbose", True)))
        promoted_count = int(promoted.get("promoted", promoted.get("up", 0))) if isinstance(promoted, dict) else int(promoted)

        if promoted_count > 0:
            for hook in ("post_rank_resize", "on_rank_resize", "clean_caches"):
                if hasattr(self, hook):
                    try: getattr(self, hook)()
                    except Exception: pass
            if it is not None:
                self._last_rank_resize_iter = it  # cooldown ref

        if promoted_count > 0:
            new_sigma, new_app = self._current_ranks_from_patches()
            self.density_n_comp = list(new_sigma)
            self.app_n_comp     = list(new_app)
        else:
            new_sigma, new_app = cur_sigma, cur_app

        return {"up": promoted_count, "down": 0, "promoted": promoted_count,
                "reso_hint": tuple(R), "base_floor": (base_floor_sig, base_floor_app),
                "cur_sigma": tuple(cur_sigma), "tgt_sigma": tuple(tgt_sigma),
                "cur_app":   tuple(cur_app),   "tgt_app":   tuple(tgt_app),
                "Rmin": Rmin, "steps": steps_arg, "floor_mode": floor_mode,
                "sigma_before": tuple(cur_sigma), "sigma_after": tuple(new_sigma),
                "app_before":   tuple(cur_app),   "app_after":   tuple(new_app),
                "base_floor": (base_floor_sig, base_floor_app),
                "floor_eff_sig": tuple(int(v) for v in floor_sig_vec),
                "floor_eff_app": tuple(int(v) for v in floor_app_vec),
                "policy_sigma_raw": tuple(int(v) for v in policy_sigma_raw),
                "policy_app_raw":   tuple(int(v) for v in policy_app_raw),
                "tgt_sigma_capped": tuple(int(v) for v in tgt_sigma_capped),
                "tgt_app_capped":   tuple(int(v) for v in tgt_app_capped),
                "cap_sigma_vec": _norm_cap_for_log(cap_sigma),
                "cap_app_vec":   _norm_cap_for_log(cap_app),
                "min_res_eff": int(min_res_eff),
                "round_to": int(round_to),
                "max_rank_glob": int(max_rank_glob),
                "floor_mode": floor_mode,
                "steps": steps_arg,
        }
    
    def _app_in_dim_from_vm(self, app_plane_list, app_line_list):
        """正確的 app 特徵輸入維度 = 3 個 planes 的 C 總和 + 3 條 lines 的 C 總和"""
        return int(sum(p.shape[1] for p in app_plane_list) + sum(l.shape[1] for l in app_line_list))

    def _ensure_basis_for_feat(self, patch: dict, feat_dim: int):
        """確保 patch['basis_mat'] 的 in_features == feat_dim；不符就重建（盡量沿用舊權重）"""
        lin = patch.get('basis_mat', None)
        if lin is None:
            patch['basis_mat'] = self.get_shared_basis(feat_dim, self.app_dim)
            return
        if getattr(lin, 'in_features', None) != int(feat_dim):
            if hasattr(self, '_rebuild_basis_like'):
                patch['basis_mat'] = self._rebuild_basis_like(lin, int(feat_dim))
            else:
                patch['basis_mat'] = self.get_shared_basis(int(feat_dim), self.app_dim)

    @classmethod
    def _gs2d_by_grid(cls, plane_1CHW: torch.Tensor, grid_1HW2: torch.Tensor) -> torch.Tensor:
        """用你準備好的 (1,H,W,2) grid 對 (1,C,H,W) 做取樣，回 (1,C,H,W)"""
        p = cls._to_1CHW(plane_1CHW)
        return F.grid_sample(p, grid_1HW2, mode="bilinear", padding_mode="border", align_corners=False).contiguous()

    @classmethod
    def _gs1d_by_grid(cls, line_1CL1: torch.Tensor, y_1L: torch.Tensor) -> torch.Tensor:
        """
        對 (1,C,L,1) 的 line 沿著 L 軸做 grid_sample：
        - y_1L: (1,L) ，數值域在 [-1,1]，代表「高度」座標；寬度固定 0
        回傳 (1,C,L,1)
        """
        l = cls._to_1CL1(line_1CL1)
        # 建 grid: (1, L, 1, 2)，第二維是 y（高度）；x 固定 0
        Y = y_1L.view(1, -1, 1, 1)
        X = torch.zeros_like(Y)
        grid = torch.cat([X, Y], dim=-1)  # (1,L,1,2) in [-1,1]
        return F.grid_sample(l, grid, mode="bilinear", padding_mode="border", align_corners=False).contiguous()

    @classmethod
    def _to_1CHW(cls, x: torch.Tensor) -> torch.Tensor:
        # plane: (C,H,W)|(1,C,H,W) -> (1,C,H,W)
        if x.dim() == 4 and x.shape[0] == 1: 
            return x
        if x.dim() == 3: 
            return x.unsqueeze(0)
        raise ValueError(f"_to_1CHW: bad shape {tuple(x.shape)}")

    @classmethod
    def _to_1CL1(cls, x: torch.Tensor) -> torch.Tensor:
        # line: (C,L)|(C,L,1)|(1,C,L,1)|(C,1,L)|(1,C,1,L) -> (1,C,L,1)
        # 4D, batch=1
        if x.dim() == 4 and x.shape[0] == 1:
            H, W = x.shape[-2], x.shape[-1]
            if W == 1:                    # (1,C,L,1) 
                return x
            if H == 1:                    # (1,C,1,L) -> (1,C,L,1)
                return x.permute(0, 1, 3, 2).contiguous()
            raise ValueError(f"_to_1CL1: both H and W > 1: {tuple(x.shape)}")

        # 3D (no batch)
        if x.dim() == 3:
            H, W = x.shape[-2], x.shape[-1]
            if W == 1:                    # (C,L,1) -> (1,C,L,1)
                return x.unsqueeze(0)
            if H == 1:                    # (C,1,L) -> (1,C,L,1)
                return x.unsqueeze(0).permute(0, 1, 3, 2).contiguous()
            raise ValueError(f"_to_1CL1: both H and W > 1: {tuple(x.shape)}")

        # 2D
        if x.dim() == 2:                  # (C,L) -> (1,C,L,1)
            return x.unsqueeze(0).unsqueeze(-1)

        raise ValueError(f"_to_1CL1: bad shape {tuple(x.shape)}")

    @classmethod
    def _gs2d_1CHW(self, plane_1CHW: torch.Tensor, uv_01: torch.Tensor) -> torch.Tensor:
        """
        plane_1CHW: (1,C,H,W)
        uv_01:      (N,2)  in [0,1] for the 2 axes of this plane
        return:     (N,C)
        """
        p = self._to_1CHW(plane_1CHW)                     # (1,C,H,W)
        uv = uv_01.to(p.device, p.dtype).clamp(0, 1)      # (N,2) in [0,1]
        grid = uv.mul(2).sub(1).view(1, -1, 1, 2)         # (1,N,1,2) -> [-1,1]
        samp = F.grid_sample(p, grid, mode="bilinear",
                             padding_mode="border", align_corners=False)  # (1,C,N,1)
        return samp.squeeze(0).squeeze(-1).transpose(0,1).contiguous()  # (N,C)

    @classmethod
    def _gs1d_1CL1(self, line_1CL1: torch.Tensor, x_01: torch.Tensor) -> torch.Tensor:
        """
        line_1CL1: (1,C,L,1)
        x_01:      (N,1) in [0,1]
        return:    (N,C)
        """
        l = self._to_1CL1(line_1CL1)                      # (1,C,L,1)
        x = x_01.to(l.device, l.dtype).clamp(0, 1)        # (N,1)
        # 1D → 2D 的 grid，第二維固定 0
        zeros = torch.zeros_like(x)
        grid = torch.cat([x, zeros], dim=1).mul(2).sub(1).view(1, -1, 1, 2)  # (1,N,1,2)
        samp = F.grid_sample(l, grid, mode="bilinear",
                             padding_mode="border", align_corners=False)  # (1,C,N,1)
        return samp.squeeze(0).squeeze(-1).transpose(0,1).contiguous()  # (N,C)

    @classmethod
    def _interp2d_1CHW(cls, plane: torch.Tensor, H_out: int, W_out: int) -> torch.Tensor:
        # (1,C,H,W) -> (1,C,H_out,W_out)
        x = cls._to_1CHW(plane)
        return F.interpolate(x, size=(int(H_out), int(W_out)), mode="bilinear", align_corners=False).contiguous()

    @classmethod
    def _interp1d_1CL1(cls, line: torch.Tensor, L_out: int) -> torch.Tensor:
        # (1,C,L,1) -> (1,C,L_out,1)；用 2D bilinear 沿 L 維插值
        x = cls._to_1CL1(line)
        return F.interpolate(x, size=(int(L_out), 1), mode="bilinear", align_corners=False).contiguous()

    def upsample_VM(self, plane_coef, line_coef, res_target):
        """
        Upsample the plane & line coefficient tensors of a single patch 
        under VM decomposition to a new target resolution:
            - plane: [1,C,H,W] -> up to new H, W
            - line: [1,C,L,1] -> squeeze to [1,C,L] -> up -> unsqueeze back
        
        Returns two nn.ParameterLists: (upsampled_planes, upsampled_lines), each entry requiring grad.
        """
        dev = self.aabb.device
        new_planes, new_lines = [], []
        with torch.no_grad():
            for i, plane in enumerate(plane_coef):
                line = line_coef[i]
                Ht = res_target[self.matMode[i][1]]
                Wt = res_target[self.matMode[i][0]]
                Lt = res_target[self.vecMode[i]]

                up_plane = self._interp2d_1CHW(plane, Ht, Wt)  # -> (C,Ht,Wt)
                up_line = self._interp1d_1CL1(line, Lt)        # -> (1,C,Lt,1)

                new_planes.append(up_plane.to(dev))
                new_lines.append(up_line.to(dev))

        param_planes = torch.nn.ParameterList([torch.nn.Parameter(t.detach().clone(), requires_grad=True) for t in new_planes])
        param_lines  = torch.nn.ParameterList([torch.nn.Parameter(t.detach().clone(), requires_grad=True) for t in new_lines])
        return param_planes, param_lines

    def upsample_volume_grid(self, reso_new):
        """
        Upsample every patch in self.patch_map to the new resolution.
        Each patch’s VM components are replaced with fresh Parameters.
        """
        device = self.aabb.device
        print(f"[INFO] Upsampling all patches to resolution {reso_new}...")

        new_map = {}
        for key, patch in self.patch_map.items():
            d_plane_up, d_line_up = self.upsample_VM(patch["density_plane"], patch["density_line"], reso_new)
            a_plane_up, a_line_up = self.upsample_VM(patch["app_plane"],     patch["app_line"],     reso_new)

            new_patch = self._create_patch(reso_new, self.aabb.device)
            new_patch["density_plane"] = d_plane_up
            new_patch["density_line"]  = d_line_up
            new_patch["app_plane"]     = a_plane_up
            new_patch["app_line"]      = a_line_up
            new_patch['res']           = list(reso_new)

            if "basis_mat" in patch:
                new_in = self._app_in_dim_from_vm(a_plane_up, a_line_up)
                new_patch["basis_mat"] = self.get_shared_basis(new_in, self.app_dim)
            
            new_map[key] = self._ensure_patch_device(new_patch, device)

        self.patch_map = new_map
        self.gridSize = torch.LongTensor(reso_new).to(self.aabb.device)
        self.update_stepSize(reso_new)
        self.current_patch_keys = list(self.patch_map.keys())
    
    @torch.no_grad()
    def _resample_parent_to_child(self, parent_patch: dict, child_res: tuple, octant: tuple):
        dev = self.aabb.device
        Rx, Ry, Rz = [int(x) for x in child_res]
        dx, dy, dz = [int(v) for v in octant]

        # ------------- 2D subgrid（[-1,1] 半域）-------------
        def _subgrid_2d(H, W, use_x, use_y):
            x0, x1 = (-1.0 + use_x * 1.0), (-1.0 + (use_x + 1) * 1.0)
            y0, y1 = (-1.0 + use_y * 1.0), (-1.0 + (use_y + 1) * 1.0)
            xs = torch.linspace(x0, x1, W, device=dev)
            ys = torch.linspace(y0, y1, H, device=dev)
            Y, X = torch.meshgrid(ys, xs, indexing='ij')
            return torch.stack([X, Y], dim=-1).unsqueeze(0).contiguous()  # (1,H,W,2) in [-1,1]

        # ------------- 1D subgrid（沿 L 軸；x 固定 0，y 走 [-1,1] 半域）-------------
        def _subgrid_1d(L, use_axis_half):
            y0, y1 = (-1.0 + use_axis_half * 1.0), (-1.0 + (use_axis_half + 1) * 1.0)
            ys = torch.linspace(y0, y1, L, device=dev)  # (L,) in [-1,1]
            return ys.view(1, -1)  # (1, L)

        # --------- density ---------
        dplanes_new, dlines_new = [], []
        # 對應 (XY), (XZ), (YZ)
        plane_use = [(dx, dy), (dx, dz), (dy, dz)]
        target_hw = [(Ry, Rx), (Rz, Rx), (Rz, Ry)]
        for i in range(3):
            H, W = target_hw[i]
            grid2d = _subgrid_2d(H, W, plane_use[i][0], plane_use[i][1])   # (1,H,W,2)
            dp = parent_patch['density_plane'][i]                           # (1,C,Hp,Wp) or (C,Hp,Wp)
            dpn = self._gs2d_by_grid(dp, grid2d)                            # (1,C,H,W)
            dplanes_new.append(self._wrap_param(dpn.to(dev)))

        Ls = [Rx, Ry, Rz]
        line_use = [dx, dy, dz]
        for i in range(3):
            dl = parent_patch['density_line'][i]                            # (1,C,Lp,1) or (C,Lp,1)
            yL = _subgrid_1d(Ls[i], line_use[i])                            # (1,L)
            dln = self._gs1d_by_grid(dl, yL)                                # (1,C,L,1)
            dlines_new.append(self._wrap_param(dln.to(dev)))

        # --------- appearance ---------
        aplanes_new, alines_new = [], []
        for i in range(3):
            H, W = target_hw[i]
            grid2d = _subgrid_2d(H, W, plane_use[i][0], plane_use[i][1])
            ap = parent_patch['app_plane'][i]
            apn = self._gs2d_by_grid(ap, grid2d)                            # (1,C,H,W)
            aplanes_new.append(self._wrap_param(apn.to(dev)))

        for i in range(3):
            al = parent_patch['app_line'][i]
            yL = _subgrid_1d(Ls[i], line_use[i])
            aln = self._gs1d_by_grid(al, yL)                                # (1,C,L,1)
            alines_new.append(self._wrap_param(aln.to(dev)))

        return {
            'density_plane': torch.nn.ParameterList(dplanes_new),
            'density_line':  torch.nn.ParameterList(dlines_new),
            'app_plane':     torch.nn.ParameterList(aplanes_new),
            'app_line':      torch.nn.ParameterList(alines_new),
        }

    @torch.no_grad()
    def split_patch(self, parent_key, parent_patch, res_child):
        """
        把一個 parent patch 切成 8 個子 patch（octants）。
        重要：每個子 patch 的 VM 內容，必須是父 patch 在對應 octant 的「子區域重取樣」，
            不是把同一份上採樣後的張量複製 8 份。
        """
        if not hasattr(self, "_split_patch_banner"):
            print("[SPLIT] using octant-aware resampling")
            self._split_patch_banner = True

        device = self.aabb.device
        i0, j0, k0 = parent_key
        depth = int(parent_patch.get('depth', 0)) + 1

        # 決定 child 的「有效解析度」：預設沿用呼叫者傳入（完全不改你既有行為）
        parent_res = list(parent_patch.get('res', res_child))
        eff_res = list(res_child)

        pol = getattr(self, "split_child_res_policy", "arg")
        if pol == "half":
            # 以 parent_res 的一半為目標，避免 8 倍體積；每軸至少 split_child_min
            eff_res = [max(self.split_child_min, int(math.ceil(r/2))) for r in parent_res]
        elif pol == "scale":
            # 以 parent_res * scale 作為目標；每軸至少 split_child_min
            sc = float(getattr(self, "split_child_scale", 1.0))
            eff_res = [max(self.split_child_min, int(round(r*sc))) for r in parent_res]
        # 其他情況（"arg"）：直接用呼叫者提供的 res_child

        eff_res = tuple(int(x) for x in eff_res)

        children = {}
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    new_key = (i0 * 2 + dx, j0 * 2 + dy, k0 * 2 + dz)
                    # octant-aware 子區域重採樣（對應到 [-1,1]^3 的 1/8 區塊）
                    child = self._resample_parent_to_child(parent_patch, tuple(eff_res), (dx, dy, dz))
                    child['res']   = list(eff_res)
                    child['depth'] = depth

                    # 繼承/共用 basis_mat
                    if 'basis_mat' in parent_patch:
                        child['basis_mat'] = parent_patch['basis_mat']
                    else:
                        in_f = self._app_in_dim_from_vm(child['app_plane'], child['app_line'])
                        child['basis_mat'] = self.get_shared_basis(in_f, self.app_dim)

                    children[new_key] = child

        return children

    @torch.no_grad()
    def _resize_patch_to_res(self, patch: dict, res_new: tuple, *,
                             mode: str = "bilinear", align_corners: bool = False) -> dict:
        """
        把單一 patch 的 density/app planes+lines 重新取樣到 res_new=(Rx,Ry,Rz)。
        會同步更新 patch['res']、並確保 basis_mat.in_features 正確。
        """
        device = self.aabb.device
        Rx, Ry, Rz = [int(x) for x in res_new]

        # ---- density ----
        dplanes_new, dlines_new = [], []
        for i in range(3):
            H, W = self._plane_target_hw_for_res(res_new, i)
            # planes: [1, C, H, W]
            dp = patch['density_plane'][i]
            dpn = self._interp2d_1CHW(dp, H, W)
            dplanes_new.append(self._wrap_param(dpn.to(device)))

        # lines: [1, C, L, 1]，三條分別對應 X/Y/Z 軸長度
        Ls = [Rx, Ry, Rz]
        for i in range(3):
            dl = patch['density_line'][i]
            dln = self._interp1d_1CL1(dl, Ls[i])
            dlines_new.append(self._wrap_param(dln.to(device)))

        # ---- appearance ----
        aplanes_new, alines_new = [], []
        for i in range(3):
            H, W = self._plane_target_hw_for_res(res_new, i)
            ap = patch['app_plane'][i]
            apn = self._interp2d_1CHW(ap, H, W)
            aplanes_new.append(self._wrap_param(apn.to(device)))

        for i in range(3):
            al = patch['app_line'][i]
            aln = self._interp1d_1CL1(al, Ls[i])
            alines_new.append(self._wrap_param(aln.to(device)))

        new_patch = dict(patch)
        new_patch['density_plane'] = dplanes_new
        new_patch['density_line']  = dlines_new
        new_patch['app_plane']     = aplanes_new
        new_patch['app_line']      = alines_new
        new_patch['res']           = [Rx, Ry, Rz]

        # ---- basis_mat in_features 修正 ----
        if 'basis_mat' in patch and hasattr(self, '_app_in_dim_from_vm'):
            new_in = self._app_in_dim_from_vm(aplanes_new, alines_new)
            old_lin = patch['basis_mat']
            if getattr(old_lin, 'in_features', None) != new_in:
                if hasattr(self, '_rebuild_basis_like'):
                    new_patch['basis_mat'] = self.get_shared_basis(new_in, self.app_dim)
                else:
                    out_dim = self.app_dim
                    new_lin = self.get_shared_basis(new_in, out_dim)
                    new_patch['basis_mat'] = new_lin
        else:
            if hasattr(self, '_app_in_dim_from_vm'):
                new_in = self._app_in_dim_from_vm(aplanes_new, alines_new)
                out_dim = self.app_dim
                new_lin = self.get_shared_basis(new_in, out_dim)
                new_patch['basis_mat'] = new_lin

        return new_patch
    
    @torch.no_grad()
    def upsample_patches(self, keys, res_new: tuple, *, mode: str = "bilinear", 
                         align_corners: bool = False, verbose: bool = True) -> int:
        """
        只對指定 keys（list of tuple，例如 (gx,gy,gz)）的 patch 進行 upsample。
        其他 patch 完全不變。
        回傳修改的 patch 數量。
        """
        if not isinstance(res_new, (list, tuple)) or len(res_new) != 3:
            raise ValueError("res_new must be a tuple/list of length 3, e.g. (64,64,64)")

        if len(self.patch_map) == 0:
            if verbose:
                print("[upsample_patches] patch_map is empty; nothing to do.")
            return 0

        cnt = 0
        new_map = {}
        keyset = set([tuple(map(int, k)) for k in keys])
        for k, p in self.patch_map.items():
            if tuple(k) in keyset:
                new_map[k] = self._resize_patch_to_res(p, tuple(res_new), mode=mode, align_corners=align_corners)
                cnt += 1
            else:
                new_map[k] = p  # untouched

        self.patch_map = new_map
        self.current_patch_keys = list(self.patch_map.keys())
        if verbose:
            print(f"[upsample_patches] upgraded {cnt} / {len(self.patch_map)} patches → res={tuple(res_new)}")
        return cnt

    @torch.no_grad()
    def select_topk_patches_by_density(self, topk=8, *, min_res: int = 0, exclude_if_res_ge: int = 0):
        """
        以 density 的平均絕對值當作重要度，挑出 top-K patch 的 key。
        - min_res: 小於此最小解析度的 patch 可直接略過（避免對超小 patch 升級）
        - exclude_if_res_ge: 若 patch 的 min(res) 已 ≥ 此值，就不再升（避免重覆升級）
        """
        items = []
        for k, p in self.patch_map.items():
            res = p.get('res', [0,0,0])
            if min_res > 0 and min(res) < min_res:
                continue
            if exclude_if_res_ge > 0 and min(res) >= exclude_if_res_ge:
                continue
            imp = 0.0
            for t in p['density_plane']:
                imp += t.abs().mean().item()
            for t in p['density_line']:
                imp += t.abs().mean().item()
            items.append((imp, tuple(k)))
        items.sort(key=lambda x: x[0], reverse=True)
        keys = [k for _, k in items[:max(0, int(topk))]]
        return keys

    @torch.no_grad()
    def select_patches_by_alpha_mass(self, target_res, *, ratio=0.0, topk=None, min_k=4, max_k=64):
        """
        Consider alpha mass as importance to select top-K patches with certain ratio.
        If no ratio, fallback to select by fixed number of top-K.
        """
        sel = []
        for k, p in self.patch_map.items():
            res = p.get('res', (0,0,0))
            if int(min(res)) >= int(min(target_res)):
                continue
            imp = float(p.get('alpha_mass', 0.0))
            sel.append((imp, tuple(k)))
        sel.sort(key=lambda x: x[0], reverse=True)
        eligible = len(sel)

        if ratio and ratio > 0:
            k = max(min_k, min(max_k, int(np.ceil(ratio * eligible))))
        else:
            k = int(topk or 12)

        return [k_ for _, k_ in sel[:max(0, k)]], eligible

    @torch.no_grad()
    def update_alpha_mass_per_patch(self, n_per: int = 512):
        """
        Estimate alpha_mass per patch by random sampling in patch-local coords.
        Writes patch['alpha_mass'] = mean alpha in [0,1].
        Cheap and robust; consistent with training alpha formula.
        """
        dev = self.aabb.device
        scale = float(getattr(self, "alpha_gate_scale", 1.0))

        for key, patch in self.patch_map.items():
            R = torch.tensor(patch['res'], device=dev, dtype=self.aabb.dtype)  # [Rx,Ry,Rz]
            units = (self.aabb[1] - self.aabb[0]) / (R - 1).clamp(min=1)       # world-steps per axis
            step = float(units.mean() * self.step_ratio)                       # per-patch step

            xyz = torch.rand((n_per, 3), device=dev) * 2 - 1
            dens_feat = self.compute_density_patch(patch, xyz)
            sigma = self.feature2density(dens_feat)
            alpha = 1.0 - torch.exp(-sigma * step * scale)
            alpha = torch.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
            patch['alpha_mass'] = float(alpha.mean().item())

        self.current_patch_keys = list(self.patch_map.keys())

    @torch.no_grad()
    def update_alpha_mass_via_dense(self, res: int = 96, chunk: int = 262144, apply_gate: bool = True):
        """
        Estimate alpha_mass per patch by aggregating dense alpha volume samples.
        More global/stable; a bit heavier than per-patch sampling.
        """
        dev = self.aabb.device
        a0, a1 = self.aabb[0], self.aabb[1]

        # build grid in world coords
        xs = torch.linspace(0, 1, res, device=dev)
        ys = torch.linspace(0, 1, res, device=dev)
        zs = torch.linspace(0, 1, res, device=dev)
        X, Y, Z = torch.meshgrid(xs, ys, zs, indexing='ij')
        grid = torch.stack([X, Y, Z], dim=-1).view(-1, 3)  # [res^3,3] in [0,1]
        world = a0 + (a1 - a0) * grid                      # [N,3] world coords

        # map once to patches
        patch_coords, exists = self._map_coords_to_patch(world)
        norm = self.normalize_coord(world)

        # step & gate
        step  = float(self._step_size_scalar())
        scale = float(getattr(self, "alpha_gate_scale", 1.0)) if apply_gate else 1.0

        # compute alpha in chunks
        alpha_all = torch.zeros((world.shape[0],), device=dev)
        for s in range(0, norm.shape[0], chunk):
            e = min(s + chunk, norm.shape[0])
            sigma = self.feature2density(self.compute_density_patchwise_fast(norm[s:e], patch_coords[s:e]))
            alpha = 1.0 - torch.exp(-sigma * step * scale)
            alpha = torch.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
            alpha_all[s:e] = alpha.view(-1)

        # bin by patch key
        sums   = {}
        counts = {}
        idxs   = patch_coords.long()
        valids = exists.nonzero(as_tuple=False).squeeze(-1)
        for i in valids.tolist():
            k = tuple(int(x) for x in idxs[i].tolist())
            sums[k]   = sums.get(k,   0.0) + float(alpha_all[i].item())
            counts[k] = counts.get(k, 0  ) + 1

        for key, patch in self.patch_map.items():
            c = counts.get(key, 0)
            m = sums.get(key, 0.0) / c if c > 0 else 0.0
            patch['alpha_mass'] = float(m)

        self.current_patch_keys = list(self.patch_map.keys())

    def _spawn_child_from(self, parent_patch):
        device = self.aabb.device
        child = self._create_patch(self.gridSize.tolist(), device)
        child['res']   = parent_patch['res']
        child['depth'] = int(parent_patch.get('depth', 0)) + 1
        return child

    def get_shared_basis(self, in_dim: int, out_dim: int, *, device=None, dtype=None) -> torch.nn.Linear:
        """
        Return shared nn.Linear(in_dim, out_dim).
        Only build one if the same in/out dim + same device/dtype, putting in self._basis_registry to use repeatedly.
        """
        in_dim  = int(in_dim)
        out_dim = int(out_dim)
        device  = device or self.aabb.device
        if dtype is None:
            dtype = getattr(self, "basis_dtype", None) or getattr(self.aabb, "dtype", torch.float32)

        # registry key
        dev_key = (device.type, device.index)
        key = (in_dim, out_dim, dev_key, dtype)

        lin = self._basis_registry.get(key, None)
        if lin is None:
            lin = torch.nn.Linear(in_dim, out_dim, bias=True).to(device=device, dtype=dtype)
            self._basis_registry[key] = lin
        return lin

    def _app_in_dim_from_planes(self, app_plane_list):
        return int(sum(p.shape[1] for p in app_plane_list))

    def get_optparam_groups(self, lr_spatial=0.02, lr_network=0.001):
        spatial_params = []
        network_params = []

        for patch in self.patch_map.values():
            spatial_params += list(patch['density_plane'])
            spatial_params += list(patch['density_line'])
            spatial_params += list(patch['app_plane'])
            spatial_params += list(patch['app_line'])

            if 'basis_mat' in patch:
                network_params += list(patch['basis_mat'].parameters())

        if hasattr(self, "_seam_banks"):
            for sid, bank in self._seam_banks.items():
                for name, p in bank.named_parameters(recurse=True):
                    if p is not None and p.requires_grad:
                        network_params.append(p)

        if hasattr(self, "renderModule") and isinstance(self.renderModule, torch.nn.Module):
            network_params += list(self.renderModule.parameters())

        def unique(params_list):
            seen = set(); out = []
            for p in params_list:
                if id(p) not in seen:
                    seen.add(id(p)); out.append(p)
            return out

        spatial_params = unique(spatial_params)
        network_params = unique(network_params)

        if 'basis_B' in patch and isinstance(patch['basis_B'], torch.nn.Module):
            network_params += list(patch['basis_B'].parameters())
        if 'mix_W' in patch and isinstance(patch['mix_W'], torch.nn.Parameter):
            network_params.append(patch['mix_W'])

        return [
            {'params': spatial_params, 'lr': lr_spatial},
            {'params': network_params, 'lr': lr_network}
        ]

    def _app_feat_raw(self, patch, xyz_sampled):
        """
        Build residual appearance features BEFORE basis projection.
        Returns: [N_sub, D_in] with the SAME layout as base feat:
                 concat of [rfx, rfy, rfz, rgx, rgy, rgz].
        """
        coord_plane, coord_line = self._get_patch_coords(xyz_sampled)  # [3,N,1,2] / [3,N,1,1]
        rpl = patch.get('app_plane_res', None)
        rln = patch.get('app_line_res',  None)
        if (rpl is None) or (rln is None):
            in_dim = self._app_in_dim_from_vm(patch['app_plane'], patch['app_line'])
            return torch.zeros((xyz_sampled.shape[0], in_dim),
                               device=xyz_sampled.device, dtype=xyz_sampled.dtype)

        comps = []
        for i, (rplane, rline) in enumerate(zip(rpl, rln)):
            rp = self._gs2d_1CHW(rplane, coord_plane[i])  # [N,Ci]
            rl = self._gs1d_1CL1(rline,  coord_line[i])   # [N,Ci]
            comps.extend([rp, rl])                      

        feat = torch.cat(comps, dim=1)  # [N, 2*(Cx+Cy+Cz)]
        return feat

    def _infer_app_latent_dim_from_patch(self, patch) -> int:
        """
        Estimate the raw appearance feat dim (Linear *input* dim) for this patch.
        Default: sum of channels of app_plane tensors (robust for concat-style).
        Fallback: run a tiny forward of _app_feat_raw if shapes are not accessible.
        """
        try:
            return int(sum(p.shape[1] for p in patch["app_plane"]))
        except Exception:
            with torch.no_grad():
                sub = torch.zeros(1, 3, device=self.aabb.device)
                feat_raw = self._app_feat_raw(patch, sub)
                return int(feat_raw.shape[-1])
    
    def _rebuild_basis_like(self, old_linear, new_in_dim):
        out_dim = old_linear.out_features
        dev = old_linear.weight.device
        new_linear = torch.nn.Linear(new_in_dim, out_dim, bias=True).to(dev)
        with torch.no_grad():
            k = min(new_in_dim, old_linear.in_features)
            new_linear.weight[:, :k].copy_(old_linear.weight[:, :k])
            if old_linear.bias is not None and new_linear.bias is not None:
                new_linear.bias.copy_(old_linear.bias)
        return new_linear
    
    def infer_patch_grid_reso_from_keys(self, patch_map):
        if not patch_map: 
            return [1,1,1]
        
        ks = torch.tensor(list(patch_map.keys()), dtype=torch.long)  # [N,3]
        mink, maxk = ks.min(0).values, ks.max(0).values
        G = (maxk - mink + 1).tolist()
        return [int(g) for g in G]

    def ensure_min_coverage(self, target_miss=0.10, seed_cells=6):
        """
        Depth-aware coverage repair.

        Strategy:
        - Sample random xyz in the AABB.
        - Use hierarchical `_map_coords_to_patch(xyz, snap_missing=False)` to detect real holes.
        - If miss ratio <= target, return 0.
        - Otherwise, create new *parent-level* patches at base grid keys for a small
          subset of missing locations (<= seed_cells). Initialize each new patch
          by cloning the nearest existing patch (Chebyshev distance in base grid).
        - New patches start at depth=0 and keep current VM res (`self.gridSize`).
        """
        dev = self.aabb.device
        self.ensure_default_patch()

        n = 8192
        u = torch.rand((n,3), device=dev)
        xyz = self.aabb[0] + (self.aabb[1]-self.aabb[0]) * u

        # Depth-aware existence check
        _, exists = self._map_coords_to_patch(xyz, snap_missing=False)
        miss_ratio = float((~exists).float().mean().item())

        if miss_ratio <= float(target_miss):
            return 0

        # Base grid index for missing points
        a0, a1 = self.aabb[0], self.aabb[1]
        extent = (a1 - a0).clamp_min(1e-8)
        p = (xyz - a0) / extent
        baseG = torch.tensor(getattr(self, "base_patch_grid_reso", self._get_patch_grid_reso()), device=dev, dtype=torch.long)
        kd0 = torch.floor(p * baseG.float()).long()
        kd0[...,0].clamp_(0, baseG[0]-1); kd0[...,1].clamp_(0, baseG[1]-1); kd0[...,2].clamp_(0, baseG[2]-1)
        kd0 = kd0[(~exists)]  # only for missing

        # Unique keys to add (cap by seed_cells)
        uniq = torch.unique(kd0, dim=0)
        if uniq.shape[0] > seed_cells:
            perm = torch.randperm(uniq.shape[0], device=dev)[:seed_cells]
            uniq = uniq[perm]

        # Build a tensor of existing parent-level keys (depth-agnostic set of actual keys)
        cur_keys = torch.tensor(
            [k for k in self.patch_map.keys() if isinstance(k, tuple)], 
            device=dev, dtype=torch.long
        )
        added = 0

        if cur_keys.numel() == 0:
            # no patches yet -> seed the center
            uniq = uniq[:max(1, seed_cells)]

        for v in uniq.tolist():
            v_t = torch.tensor(v, device=dev, dtype=torch.long) 
            key = tuple(int(x) for x in v)
            if key in self.patch_map:
                continue

            # nearest existing key to clone from (Chebyshev)
            if cur_keys.numel() > 0:
                dist = (cur_keys - v_t).abs().amax(dim=-1)
                src_key = tuple(int(x) for x in cur_keys[dist.argmin().item()].tolist())
                src = self.patch_map[src_key]
                # upsample or clone to current VM res
                R = self.gridSize.tolist()
                src_R = list(src.get('res', R))
                if tuple(src_R) != tuple(R):
                    dpl, dln = self.upsample_VM(src["density_plane"], src["density_line"], R)
                    apl, aln = self.upsample_VM(src["app_plane"],     src["app_line"],     R)
                    patch = {
                        'res': R[:],
                        'density_plane': dpl, 'density_line': dln,
                        'app_plane': apl, 'app_line': aln,
                    }
                    if "basis_mat" in src:
                        new_in = self._app_in_dim_from_vm(patch["app_plane"], patch["app_line"])
                        patch["basis_mat"] = self.get_shared_basis(new_in, self.app_dim)
                else:
                    # deep clone
                    def _clone_pl(pl):
                        return torch.nn.ParameterList([torch.nn.Parameter(p.detach().clone().to(dev), requires_grad=True) for p in pl])
                    patch = {
                        'res': R[:],
                        'density_plane': _clone_pl(src["density_plane"]),
                        'density_line':  _clone_pl(src["density_line"]),
                        'app_plane':     _clone_pl(src["app_plane"]),
                        'app_line':      _clone_pl(src["app_line"]),
                    }
                    if "basis_mat" in src:
                        in_f  = int(src["basis_mat"].in_features)
                        out_f = int(src["basis_mat"].out_features)
                        patch["basis_mat"] = self.get_shared_basis(in_f, out_f)
            else:
                # very first seed
                patch = self._create_patch(self.gridSize.tolist(), dev)

            patch['depth'] = 0
            self.patch_map[key] = self._ensure_patch_device(patch, dev)
            added += 1

        # refresh helpers
        if hasattr(self, "infer_patch_grid_reso_from_keys"):
            self.patch_grid_reso = tuple(self.infer_patch_grid_reso_from_keys(self.patch_map))
        self.current_patch_keys = list(self.patch_map.keys())

        return added

    @torch.no_grad()
    def debug_dump_basis_stats(self, verbose: bool = True):
        uniq = {}
        refs = 0
        for p in self.patch_map.values():
            lin = p.get('basis_mat', None)
            if lin is None: continue
            refs += 1
            k = (lin.in_features, lin.out_features)
            uniq.setdefault(k, set()).add(id(lin))
        n_unique_lin = sum(len(s) for s in uniq.values())
        if verbose:
            print(f"[basis] unique Linear modules: {n_unique_lin}  |  total patch refs: {refs}")
            for k, s in uniq.items():
                print(f"  - shape={k} : {len(s)} unique objects (referenced by multiple patches)")
        return n_unique_lin, refs

    def _make_zero_param_like(self, shape, device):
        return torch.nn.Parameter(torch.zeros(shape, device=device), requires_grad=False)

    def deactivate_patch_(self, key):
        """
        Soft prune: keep key and squeeze to min capcity (mark dead=True)
        """
        if key not in self.patch_map: 
            return False
        p = self.patch_map[key]
        dev = self.aabb.device
        Csig = p['density_plane'][0].shape[1]
        Capp = p['app_plane'][0].shape[1]

        def _mk_planes_lines(C, R):
            Rx,Ry,Rz = R
            planes = torch.nn.ParameterList([
                self._make_zero_param_like((1, C, Ry, Rz), dev),
                self._make_zero_param_like((1, C, Rx, Rz), dev),
                self._make_zero_param_like((1, C, Rx, Ry), dev),
            ])
            lines  = torch.nn.ParameterList([
                self._make_zero_param_like((1, C, Rx, 1), dev),
                self._make_zero_param_like((1, C, Ry, 1), dev),
                self._make_zero_param_like((1, C, Rz, 1), dev),
            ])
            return planes, lines

        # 1^3 as placeholder 
        p['density_plane'], p['density_line'] = _mk_planes_lines(Csig, (1,1,1))
        p['app_plane'],     p['app_line']     = _mk_planes_lines(Capp, (1,1,1))
        p['res']  = [1,1,1]
        p['dead'] = True
  
        return True

    def _get_patch_coords(self, xyz_sampled: torch.Tensor):
        """
        回傳:
        coord_plane: [3, N, 1, 2]  三張 plane 的 (u,v)
        coord_line:  [3, N, 1, 1]  三條 line 的 t
        """
        # planes
        cps = []
        for i in range(3):
            ax0, ax1 = self.matMode[i]  # e.g. (1,2), (0,2), (0,1)
            uv = torch.stack([xyz_sampled[..., ax0], xyz_sampled[..., ax1]], dim=-1)  # [N,2]
            cps.append(uv.view(-1, 1, 2))  # [N,1,2]
        coord_plane = torch.stack(cps, dim=0)  # [3,N,1,2]

        # lines
        cls = []
        for i in range(3):
            ax = self.vecMode[i]  # 0 or 1 or 2
            t = xyz_sampled[..., ax]       # [N]
            cls.append(t.view(-1, 1, 1))   # [N,1,1]
        coord_line = torch.stack(cls, dim=0)  # [3,N,1,1]
        return coord_plane, coord_line

    @torch.no_grad()
    def _ray_aabb_midpoint(self, rays_o, rays_d, eps=1e-8):
        """
        Compute midpoints of ray–AABB intersections.
        Returns:
            mid: [N,3] world coords of midpoint along intersection segment (undefined if !hit)
            hit: [N] bool mask whether each ray hits the AABB
        """
        if rays_o.numel() == 0:
            return rays_o, torch.zeros((0,), dtype=torch.bool, device=rays_o.device)

        aabb_min, aabb_max = self.aabb[0], self.aabb[1]
        # avoid div by 0 in slab test
        safe_d = torch.where(rays_d.abs() < eps, torch.sign(rays_d) * eps, rays_d)
        invd = 1.0 / safe_d

        t0 = (aabb_min - rays_o) * invd
        t1 = (aabb_max - rays_o) * invd
        tmin = torch.minimum(t0, t1).amax(dim=-1)  
        tmax = torch.maximum(t0, t1).amin(dim=-1)  

        hit = tmax >= tmin
        tmid = 0.5 * (tmin + tmax)
        mid = rays_o + tmid.unsqueeze(-1) * rays_d
        return mid, hit

    def get_patch_storage(self, patch_key=None, plane_list=None, line_list=None):
        """
        Flexible memory usage query:
            - If patch_key is given, return memory usage of that patch.
            - If plane_list + line_list are given, compute memory of the custom set.
            - If neither is given, return total memory of all patches.
        """
        self.ensure_default_patch()
        assert self.patch_map, "patch_map is empty before get_patch_storage!"

        total = 0
        if patch_key is not None and patch_key in self.patch_map:
            patch = self.patch_map[patch_key]
            for plist in [patch['density_plane'], patch['density_line'], patch['app_plane'], patch['app_line']]:
                for t in plist:
                    total += t.numel() * t.element_size()
        elif plane_list is not None and line_list is not None:
            for plist in [plane_list, line_list]:
                for t in plist:
                    total += t.numel() * t.element_size()
        else:
            for patch in self.patch_map.values():
                for plist in [patch['density_plane'], patch['density_line'], patch['app_plane'], patch['app_line']]:
                    for t in plist:
                        total += t.numel() * t.element_size()
        return total
    
    @torch.no_grad()
    def patch_grid_coverage_snapshot(self, G=32, verbose=True):
        """
        Uniformly sample a G^3 grid inside AABB, map to patch keys, and report coverage.
        Returns a dict of summary stats.
        """
        device = self.aabb.device
        a0, a1 = self.aabb[0], self.aabb[1]
        extent = (a1 - a0).clamp_min(1e-8)

        # grid xyz in [0,1]^3 -> world
        t = torch.linspace(0.0, 1.0, G, device=device)
        zz, yy, xx = torch.meshgrid(t, t, t, indexing='ij')
        p = torch.stack([xx, yy, zz], dim=-1)  # [G,G,G,3]
        xyz = a0 + p * extent
        xyz = xyz.reshape(-1, 3)

        coords, exists = self._map_coords_to_patch(xyz)
        total = xyz.shape[0]
        frac_exists = exists.float().mean().item()

        # per-depth coverage
        uniq, inv = torch.unique(coords[exists], dim=0, return_inverse=True)
        if uniq.numel() == 0:
            depth_fracs = {}
        else:
            # counts per unique key
            counts = torch.bincount(inv, minlength=uniq.shape[0]).float()
            depths = []
            for k in uniq:
                patch = self.patch_map.get(tuple(k.tolist()), {})
                depths.append(int(patch.get('depth', 0)))
            depths = torch.tensor(depths, device=device)
            depth_fracs = {}
            for d in torch.unique(depths).tolist():
                idx = (depths == d).nonzero(as_tuple=False).squeeze(1)
                frac_d = (counts[idx].sum() / float(total)).item()
                depth_fracs[int(d)] = frac_d

        # boundary-miss diagnostics 
        p_raw = (xyz - a0) / extent
        miss = ~exists
        if miss.any():
            pr = p_raw[miss]
            near_hi = (pr >= 1.0).any(dim=-1).float().mean().item()
            near_lo = (pr <= 0.0).any(dim=-1).float().mean().item()
        else:
            near_hi = near_lo = 0.0

        out = {
            "G": G,
            "exists_frac": frac_exists,        # overall coverage
            "per_depth_frac": depth_fracs,     # coverage of each depth (sum≈exists_frac)
            "boundary_miss_hi_frac": near_hi,  
            "boundary_miss_lo_frac": near_lo   
        }

        if verbose:
            depth_str = " ".join([f"d{d}:{v:.1%}" for d,v in sorted(depth_fracs.items())]) if depth_fracs else "n/a"
            print(f"[coverage] G={G} | exists={frac_exists:.1%} | {depth_str} | miss@hi={near_hi:.1%} | miss@lo={near_lo:.1%}")

        return out

    def _validate_one_patch_shapes(self, pid, patch):
        # planes: (C,H,W) or (1,C,H,W)
        for t in list(patch['density_plane']) + list(patch['app_plane']):
            if t.dim() == 3:
                assert t.shape[1] > 1 and t.shape[2] > 1, f"[{pid}] plane CHW bad shape {tuple(t.shape)}"
            elif t.dim() == 4:
                assert t.shape[0] == 1, f"[{pid}] plane batch must be 1, got {tuple(t.shape)}"
            else:
                raise ValueError(f"[{pid}] plane bad dim {t.dim()} with shape {tuple(t.shape)}")

        # lines: (C,L,1) or (1,C,L,1)
        for t in list(patch['density_line']) + list(patch['app_line']):
            if t.dim() == 3:
                assert t.shape[2] == 1, f"[{pid}] line CL1 bad shape {tuple(t.shape)}"
            elif t.dim() == 4:
                assert t.shape[0] == 1 and t.shape[3] == 1, f"[{pid}] line 1CL1 bad shape {tuple(t.shape)}"
            else:
                raise ValueError(f"[{pid}] line bad dim {t.dim()} with shape {tuple(t.shape)}")

    def estimate_bytes_if_upsample(self, reso_target):
        import math
        # dtype bytes
        def nbytes_of(t):
            if t.dtype.is_floating_point:
                return torch.finfo(t.dtype).bits // 8
            return torch.iinfo(t.dtype).bits // 8

        total_elems = 0
        byte_per = None

        for p in self.patch_map.values():
            if byte_per is None:
                ref = p["density_plane"][0]
                byte_per = nbytes_of(ref)

            # let H/W/L become aligned axis of reso_target 
            for i, plane in enumerate(p["density_plane"]):
                C = plane.shape[1]
                Ht = reso_target[self.matMode[i][1]]
                Wt = reso_target[self.matMode[i][0]]
                total_elems += 1 * C * Ht * Wt

            for i, line in enumerate(p["density_line"]):
                C = line.shape[1]
                Lt = reso_target[self.vecMode[i]]
                total_elems += 1 * C * Lt

            for i, plane in enumerate(p["app_plane"]):
                C = plane.shape[1]
                Ht = reso_target[self.matMode[i][1]]
                Wt = reso_target[self.matMode[i][0]]
                total_elems += 1 * C * Ht * Wt

            for i, line in enumerate(p["app_line"]):
                C = line.shape[1]
                Lt = reso_target[self.vecMode[i]]
                total_elems += 1 * C * Lt

            # basis_mat doesn't change as different reso -> fixed or add certain values
            # total_elems += p["basis_mat"].weight.numel()

        return int(total_elems * (byte_per if byte_per is not None else 4))

    def vector_comp_diffs(self):
        total = 0.0
        for patch in self.patch_map.values():
            # density_plane: [1,C,H,W]
            for plane in patch['density_plane']:
                H, W = plane.shape[-2], plane.shape[-1]
                if H > 1:
                    dx = plane[..., 1:, :] - plane[..., :-1, :]
                    total += torch.mean(torch.sqrt(dx.pow(2) + 1e-6))
                if W > 1:
                    dy = plane[..., :, 1:] - plane[..., :, :-1]
                    total += torch.mean(torch.sqrt(dy.pow(2) + 1e-6))
            # density_line: [1,C,L,1]
            for line in patch['density_line']:
                L = line.shape[-2]
                if L > 1:
                    dz = line[..., 1:, :] - line[..., :-1, :]
                    total += torch.mean(torch.sqrt(dz.pow(2) + 1e-6))
        return total

    """
    L1 & TV reg loss terms: 
        return mean() instead of sum() to avoid progressively increasing as split ocurrs.
    """
    def density_L1(self):
        terms = []
        for patch in self.patch_map.values():
            for plane in patch['density_plane']:
                terms.append((plane ** 2 + 1e-6).sqrt().mean())  # numerically stable L1
            for line in patch['density_line']:
                terms.append((line ** 2 + 1e-6).sqrt().mean())
        return torch.stack(terms).mean() if terms else torch.tensor(0.0, device=self.aabb.device)

    def TV_loss_density(self, tvreg, subsample=4, max_pixels=262144):
        s = int(getattr(self, "tv_subsample", subsample))
        cap = int(getattr(self, "tv_max_pixels", max_pixels))
        terms = []
        
        for p in self.patch_map.values():
            # planes [1,C,H,W]
            for t in p["density_plane"]:
                H, W = int(t.shape[-2]), int(t.shape[-1])
                step = max(1, s)
                while (math.ceil(H/step) * math.ceil(W/step)) > cap and step < max(H, W):
                    step *= 2
                sub = t[..., ::step, ::step]
                terms.append(tvreg(sub).mean())

            # lines [1,C,L,1] 
            for l in p["density_line"]:
                L = int(l.shape[-2])  # reshape as 1D
                step = max(1, s)
                while math.ceil(L/step) > cap and step < L:
                    step *= 2
                sub = l[:, :, ::step, :]               # [1, C, L', 1]
                if sub.shape[-2] > 1:                  # L' >= 2
                    dx = (sub[:, :, 1:, :] - sub[:, :, :-1, :]).pow(2).mean()
                    terms.append(dx)

        if not terms:
            return torch.tensor(0.0, device=self.aabb.device)
        return torch.stack(terms).mean()

    def TV_loss_app(self, tvreg, subsample=4, max_pixels=262144):
        s = int(getattr(self, "tv_subsample", subsample))
        cap = int(getattr(self, "tv_max_pixels", max_pixels))
        terms = []
        
        for p in self.patch_map.values():
            # planes [1,C,H,W]
            for t in p["app_plane"]:
                H, W = int(t.shape[-2]), int(t.shape[-1])
                step = max(1, s)
                while (math.ceil(H/step) * math.ceil(W/step)) > cap and step < max(H, W):
                    step *= 2
                sub = t[..., ::step, ::step]
                terms.append(tvreg(sub).mean())
            # lines [1,C,L,1] 
            for l in p["app_line"]:
                L = int(l.shape[-2])  # reshape as 1D
                step = max(1, s)
                while math.ceil(L/step) > cap and step < L:
                    step *= 2
                sub = l[:, :, ::step, :]               # [1, C, L', 1]
                if sub.shape[-2] > 1:                  # L' >= 2
                    dx = (sub[:, :, 1:, :] - sub[:, :, :-1, :]).pow(2).mean()
                    terms.append(dx)
        
        if not terms:
            return torch.tensor(0.0, device=self.aabb.device)
        return torch.stack(terms).mean()

    @torch.no_grad()
    def _plane_target_hw_for_res(self, res, plane_idx: int):
        """
        給定三軸解析度 res=(Rx,Ry,Rz)，回傳第 plane_idx 個 2D 平面的(H,W)。
        約定：0->YZ, 1->XZ, 2->XY（與 TensoRF 常見對應一致）
        """
        Rx, Ry, Rz = int(res[0]), int(res[1]), int(res[2])
        if plane_idx == 0:   # YZ
            return (Ry, Rz)
        elif plane_idx == 1: # XZ
            return (Rx, Rz)
        else:                # XY
            return (Rx, Ry)

    def _wrap_param(self, t: torch.Tensor) -> torch.nn.Parameter:
        return torch.nn.Parameter(t.contiguous(), requires_grad=True)

    @torch.no_grad()
    def shrink(self, new_aabb):
        """
        Patch-aware shrink (safe version):
        - Drop entire patches that lie completely outside the new AABB
        - Do NOT slice per-patch tensors (avoids geometric corruption)
        - Update AABB and step size for consistent sampling
        """
        a0_old, a1_old = self.aabb[0], self.aabb[1]
        a0_new, a1_new = new_aabb[0], new_aabb[1]
        extent_old = (a1_old - a0_old).clamp_min(1e-8)

        try:
            Gx, Gy, Gz = map(int, self.patch_grid_reso)
        except Exception:
            Gx, Gy, Gz = map(int, self.infer_patch_grid_reso_from_keys(self.patch_map))

        def patch_world_bounds(i, j, k):
            lo = a0_old + extent_old * torch.tensor([i / Gx,     j / Gy,     k / Gz    ], device=a0_old.device, dtype=a0_old.dtype)
            hi = a0_old + extent_old * torch.tensor([(i+1) / Gx, (j+1) / Gy, (k+1) / Gz], device=a0_old.device, dtype=a0_old.dtype)
            return lo, hi

        def overlap_3d(lo1, hi1, lo2, hi2):
            # 判斷 [lo1,hi1] 與 [lo2,hi2] 是否在三個軸皆有重疊
            return bool(((hi1 >= lo2) & (lo1 <= hi2)).all().item())

        kept = {}
        removed = 0
        for (i, j, k), patch in self.patch_map.items():
            p_lo, p_hi = patch_world_bounds(i, j, k)
            if overlap_3d(p_lo, p_hi, a0_new, a1_new):
                kept[(i, j, k)] = patch   # 部分或完全重疊 → 保留原參數（不切）
            else:
                removed += 1              # 完全在外 → 刪除整塊 patch

        self.patch_map = kept
        self.current_patch_keys = list(self.patch_map.keys())

        self.aabb = torch.stack([a0_new, a1_new], dim=0)

        # 更新 stepSize（僅作 fallback/全域用；真正 per-patch 步長由 compute_alpha/sample_ray 決定）
        if isinstance(self.gridSize, torch.Tensor):
            gs = self.gridSize.tolist()
        else:
            gs = list(self.gridSize)
        self.update_stepSize(gs)

        print(f"[shrink] removed {removed} patches; new AABB = {self.aabb.tolist()}")
    
    @torch.no_grad()
    def shrink_inside_patches(self, new_aabb, pad=1, min_len=2):
        """
        For each patch that intersects new_aabb, crop its spatial extent (planes/lines)
        to the intersection. 'pad' keeps a 1-voxel margin; 'min_len' avoids degenerate dims.
        """
        if not hasattr(self, "patch_map") or len(self.patch_map) == 0:
            return 0

        old_aabb = self.aabb
        a0, a1 = old_aabb[0], old_aabb[1]
        na0, na1 = new_aabb[0], new_aabb[1]
        span = (a1 - a0).clamp_min(1e-12)

        G = tuple(self.patch_grid_reso)
        device = a0.device
        n_cropped = 0

        def _axis_slice(ext, axis):
            l, r = ext[axis]
            return slice(int(l), int(r) + 1)

        for key, P in list(self.patch_map.items()):
            # world bbox of this patch cell
            k = torch.tensor(key, dtype=a0.dtype, device=device)
            cell_min = a0 + (k / torch.tensor(G, dtype=a0.dtype, device=device)) * span
            cell_max = a0 + ((k + 1) / torch.tensor(G, dtype=a0.dtype, device=device)) * span

            # intersection with new_aabb
            lo = torch.maximum(cell_min, na0)
            hi = torch.minimum(cell_max, na1)
            if not bool((hi > lo).all()):
                # 完全在外側的應該已被你前面的 shrink() 移除了
                continue

            # map to local [0,1], then to index [0..R-1]
            Rx, Ry, Rz = [int(x) for x in P['res']]
            R = [Rx, Ry, Rz]
            loc0 = ((lo - cell_min) / (cell_max - cell_min)).clamp(0, 1)
            loc1 = ((hi - cell_min) / (cell_max - cell_min)).clamp(0, 1)

            exts = []
            for ax in range(3):
                L = max(0, int(torch.floor(loc0[ax] * (R[ax] - 1)).item()) - int(pad))
                Rax = min(R[ax] - 1, int(torch.ceil(loc1[ax] * (R[ax] - 1)).item()) + int(pad))
                if Rax < L:
                    L, Rax = 0, R[ax] - 1
                # 保證至少 min_len
                if (Rax - L + 1) < min_len:
                    need = min_len - (Rax - L + 1)
                    Rax = min(R[ax] - 1, Rax + (need // 2 + need % 2))
                    L = max(0, Rax - (min_len - 1))
                exts.append((L, Rax))

            # matMode/vecMode 決定 planes/lines 的對應軸；注意 plane 的維度順序是 [1, C, H, W] = [mat_id_1, mat_id_0]
            matMode = getattr(self, "matMode", [[0,1],[0,2],[1,2]])
            vecMode = getattr(self, "vecMode", [2,1,0])

            def _crop_planes(pl_list, exts):
                out = []
                for i, (a, b) in enumerate(matMode):
                    Wsl = _axis_slice(exts, a)  # mat_id_0 → W
                    Hsl = _axis_slice(exts, b)  # mat_id_1 → H
                    T = pl_list[i]
                    sliced = T.data[..., Hsl, Wsl].contiguous()
                    out.append(torch.nn.Parameter(sliced))
                return out

            def _crop_lines(li_list, exts):
                out = []
                for i, v in enumerate(vecMode):
                    Lsl = _axis_slice(exts, v)
                    T = li_list[i]
                    sliced = T.data[..., Lsl, :].contiguous()
                    out.append(torch.nn.Parameter(sliced))
                return out

            P['density_plane'] = torch.nn.ParameterList(_crop_planes(P['density_plane'], exts))
            P['density_line']  = torch.nn.ParameterList(_crop_lines(P['density_line'], exts))
            P['app_plane']     = torch.nn.ParameterList(_crop_planes(P['app_plane'], exts))
            P['app_line']      = torch.nn.ParameterList(_crop_lines(P['app_line'], exts))

            newRx = exts[0][1] - exts[0][0] + 1
            newRy = exts[1][1] - exts[1][0] + 1
            newRz = exts[2][1] - exts[2][0] + 1
            if (newRx, newRy, newRz) != (Rx, Ry, Rz):
                P['res'] = [int(newRx), int(newRy), int(newRz)]
                n_cropped += 1

        return n_cropped

    @torch.no_grad()
    def _alpha_max_of_patch(self, patch, n=200):
        xyz = torch.rand((n, 3), device=self.aabb.device) * 2 - 1

        feat = self.compute_density_patch(patch, xyz)         # [n]
        sigma = self.feature2density(feat)                    # [n] >= 0
        step = float(self.stepSize.detach().clamp(min=1e-6))  
        scale = float(self.alpha_gate_scale)     # 1.0 by default

        alpha = 1.0 - torch.exp(-sigma * step * scale)                # [n]
        # clear NaN/Inf out
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)

        return float(alpha.max().item())

    @torch.no_grad()
    def _step_size_scalar(self):
        """
        Safe scalar step size for converting sigma -> alpha.
        """
        if hasattr(self, "stepSize") and isinstance(self.stepSize, torch.Tensor):
            return float(self.stepSize.detach().clamp(min=1e-4))
        aabb_extent = (self.aabb[1] - self.aabb[0]).abs()
        L = float(aabb_extent.norm()) + 1e-6
        return max(L / 1024.0, 1e-4)

    @torch.no_grad()
    def alpha_has_signal(self, eps: float = 1e-5, n: int = 4096):
        """
        Quick global check: take n random samples to test whether exist visible alpha signal already.
        """
        dev = self.aabb.device
        u = torch.rand((n, 3), device=dev)
        xyz = self.aabb[0] + (self.aabb[1] - self.aabb[0]) * u

        xyz_norm = self.normalize_coord(xyz)

        patch_coords, exists = self._map_coords_to_patch(xyz)

        sigma_feat = self.compute_density_patchwise_fast(xyz_norm, patch_coords)  # [n]
        sigma = self.feature2density(sigma_feat)                                  # [n]
        alpha = 1.0 - torch.exp(-sigma * self._step_size_scalar())

        amax = float(alpha.max().item())
        return (amax >= eps)

    @torch.no_grad()
    def any_patch_confident(self, alpha_quantile: float = 0.90, min_val: float = 1e-3, n: int = 4096):
        """
        Ouick random sample check: whether alpha quatile of one containing at least one patch is over threshold.
        Set a reasonable gate for staged alpha/shrink/prune to avoid process too early.
        """
        dev = self.aabb.device

        u = torch.rand((n, 3), device=dev)
        xyz = self.aabb[0] + (self.aabb[1] - self.aabb[0]) * u
        xyz_norm = self.normalize_coord(xyz)
        coords, exists = self._map_coords_to_patch(xyz)

 
        sigma_feat = self.compute_density_patchwise_fast(xyz_norm, coords)  # [n]
        sigma = self.feature2density(sigma_feat)                            # [n]
        alpha = 1.0 - torch.exp(-sigma * self._step_size_scalar())          # [n]

        # build unique id of each patch by mixed radix for grouping
        try:
            Gx, Gy, Gz = map(int, self.patch_grid_reso)
        except Exception:
            Gx, Gy, Gz = map(int, self.infer_patch_grid_reso_from_keys(self.patch_map))
        baseZ = 1
        baseY = Gz
        baseX = Gy * Gz
        ids = coords[:, 0] * baseX + coords[:, 1] * baseY + coords[:, 2] * baseZ  # [n]

        uniq, inv = torch.unique(ids, return_inverse=True)
        confident = False

        for uid in uniq:
            m = (inv == uid)
            if not m.any():
                continue
            a = alpha[m]
            qv = torch.quantile(a, q=alpha_quantile)
            if float(qv.item()) >= min_val:
                confident = True
                break

        return confident

    def prune_empty_patches(self, alpha_thres=None, min_reso=16, group_size=2):
        """
        Density-based pruning:
            - Only run if min(gridSize) >= min_reso.
            - Dynamic threshold: th_eff = max(stage_thres, global p10)
            - Coarse group safeguard: each coarse cell preserves at least one patch.
        """
        cur_min = int(min(self.gridSize).item() if isinstance(self.gridSize, torch.Tensor) else min(self.gridSize))
        if cur_min < min_reso:
            print("[INFO] Skipping patch pruning: current voxel reso too low")
            return 0

        def stage_thres(r):
            if r < 16:  return None   
            if r < 32:  return 0.0003
            if r < 64:  return 0.0005
            return 0.0008

        st = stage_thres(cur_min)
        if st is None:
            print("[INFO] Skipping prune at this stage (reso too low).")
            return 0

        alpha_max = {}
        vals = []
        for key, patch in self.patch_map.items():
            amax = self._alpha_max_of_patch(patch, n=200)
            alpha_max[key] = amax
            vals.append(amax)

        if len(vals) == 0:
            print("[WARN] No patches to prune.")
            return 0

        # global distribution ref.
        vals_np = np.asarray(vals, dtype=np.float32)
        q50, q75, q90, q95 = np.quantile(vals_np, [0.5, 0.75, 0.9, 0.95]).tolist()
        p10 = float(np.quantile(vals_np, 0.10))
        
        th_eff = max(st, p10)
        print(f"[alpha dist] p50={q50:.4f} p75={q75:.4f} p90={q90:.4f} p95={q95:.4f} | stage={st:.4f} p10={p10:.4f} → th_eff={th_eff:.4f}")

        # grouping by coarse cell 
        groups = defaultdict(list)
        for (i, j, k) in self.patch_map.keys():
            gi, gj, gk = i // group_size, j // group_size, k // group_size
            groups[(gi, gj, gk)].append((i, j, k))

        to_delete = []
        for gkey, members in groups.items():
            below = [m for m in members if alpha_max[m] < th_eff]
            if len(below) == len(members):
                keep = max(members, key=lambda m: alpha_max[m])
                for m in members:
                    if m != keep:
                        to_delete.append(m)
            else:
                to_delete.extend(below)

        for k in to_delete:
            self.patch_map.pop(k, None)

        self.current_patch_keys = list(self.patch_map.keys())
        print(f"[prune_empty_patches_safe] removed {len(to_delete)} — remain {len(self.patch_map)}")
        return len(to_delete)

    @torch.no_grad()
    def get_dense_alpha_from_patch(self, res=256, chunk=262144, apply_gate=True):
        """
        Aggregate alpha values from all patches into a dense volume [res, res, res].
        - apply_gate: whether to apply the current alpha-gate (keeps consistency with training).
        - chunk: process in chunks to avoid OOM.
        """
        dev = self.aabb.device
        aabb0, aabb1 = self.aabb[0], self.aabb[1]

        # precompute step & gate scale
        step = float(self.stepSize.detach().clamp(min=1e-4))
        scale = float(self.alpha_gate_scale) if apply_gate else 1.0

        # prepare grid in world coords
        xs = torch.linspace(0, 1, res, device=dev)
        ys = torch.linspace(0, 1, res, device=dev)
        zs = torch.linspace(0, 1, res, device=dev)
        X, Y, Z = torch.meshgrid(xs, ys, zs, indexing='ij')
        grid_coords = torch.stack([X, Y, Z], dim=-1).view(-1, 3)  # [res^3, 3] in [0,1]^3
        world_coords = aabb0 + (aabb1 - aabb0) * grid_coords

        # map to patches once (for warnings / optional masking)
        patch_coords, exists = self._map_coords_to_patch(world_coords)
        miss_ratio = 1.0 - exists.float().mean().item()
        if miss_ratio > 0.3:
            print(f"[WARNING] {miss_ratio:.0%} of dense alpha samples map to MISSING patches (fallback risk)!")

        # compute sigma in chunks (compute_density_patchwise_fast expects normalized coords)
        norm_coords = self.normalize_coord(world_coords)
        out = torch.empty((world_coords.shape[0],), device=dev, dtype=torch.float32)

        for s in range(0, norm_coords.shape[0], chunk):
            e = min(s + chunk, norm_coords.shape[0])
            sigma = self.feature2density(
                self.compute_density_patchwise_fast(norm_coords[s:e], patch_coords[s:e])
            )
            # sigma -> alpha with step & (optional) gate
            alpha = 1.0 - torch.exp(-sigma * step * scale)
            # safety
            alpha = torch.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
            out[s:e] = alpha.view(-1)

        out[~exists] = 0.0

        return out.view(res, res, res)

    def get_total_voxels(self):
        """
        Compute the total number of voxels across all patches.
        If patch_map is empty, return the total number of voxels in the global grid.
        """
        if not self.patch_map:
            if isinstance(self.gridSize, torch.Tensor):
                gs = self.gridSize.tolist()
            else:
                gs = list(self.gridSize)
            return gs[0] * gs[1] * gs[2]

        total = 0
        for p in self.patch_map.values():
            res = p.get('res', None)
            if res is None:
                if isinstance(self.gridSize, torch.Tensor):
                    gs = self.gridSize.tolist()
                else:
                    gs = list(self.gridSize)
                total += gs[0] * gs[1] * gs[2]
            else:
                total += res[0] * res[1] * res[2]
        return total

    def get_total_mem(self):
        """
        Compute total memory (bytes) used by all patches.
        If patch_map is empty, first create the default fallback patch.
        """
        self.ensure_default_patch()

        total = 0
        for p in self.patch_map.values():
            # element_size(): bytes used by each element
            for k in ['density_plane', 'density_line', 'app_plane', 'app_line']:
                for t in p.get(k, []):
                    total += t.numel() * t.element_size()

            bm = p.get('basis_mat', None)
            if bm is not None and hasattr(bm, 'weight'):
                w = bm.weight
                total += w.numel() * w.element_size()

        return total
    
    @torch.no_grad()
    def save(self, path, extra_meta=None):
        ckpt = {"kwargs": self.get_kwargs() , "state_dict": self.state_dict()}

        if hasattr(self, "alphaMask") and (self.alphaMask is not None):
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape':alpha_volume.shape})
            ckpt.update({'alphaMask.mask': np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})

        if extra_meta is not None:
            ckpt["meta"] = extra_meta

        torch.save(ckpt, path)

    @torch.no_grad()
    def load(self, ckpt_or_path, map_location=None):
        if isinstance(ckpt_or_path, (str, os.PathLike)):
            ckpt = torch.load(ckpt_or_path, map_location="cpu")
        else:
            ckpt = ckpt_or_path

        device = map_location or (self.aabb.device if hasattr(self, "aabb") else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        if ("alphaMask.shape" in ckpt) and ("alphaMask.mask" in ckpt):
            from models.tensorBase import AlphaGridMask
            shape = tuple(ckpt["alphaMask.shape"])
            bits  = ckpt["alphaMask.mask"]
            aabb  = ckpt.get("alphaMask.aabb", self.aabb).to(device)
            flat  = np.unpackbits(bits)[:np.prod(shape)]
            mask  = torch.from_numpy(flat.reshape(shape)).bool().to(device)
            self.alphaMask = AlphaGridMask(device, aabb, mask)

        if "patch_map" in ckpt:
            pm = {}
            for k_str, P in ckpt["patch_map"].items():
                try:
                    key = eval(k_str) if isinstance(k_str, str) else tuple(k_str)
                except Exception:
                    key = tuple(int(x) for x in str(k_str).replace("(", "").replace(")", "").split(",") if x != "")
                pm[key] = self._deserialize_patch(P, device)

            self.patch_map = pm
            self.current_patch_keys = list(self.patch_map.keys())
            self.patch_grid_reso = tuple(ckpt.get("patch_grid_reso", (1,1,1)))
            self.gridSize = torch.LongTensor(list(ckpt.get("vm_reso", self.gridSize.tolist()))).to(device)
            self.update_stepSize(self.gridSize.tolist())
            self.density_n_comp = list(ckpt.get("density_n_comp", self.density_n_comp))
            self.app_n_comp = list(ckpt.get("app_n_comp", self.app_n_comp))

            if hasattr(self, "assert_zero_origin_and_contiguous"):
                try: self.assert_zero_origin_and_contiguous()
                except Exception as e: print(f"[WARN] assert_zero_origin_and_contiguous failed: {e}")

        elif "state_dict" in ckpt:
            self.load_state_dict(ckpt["state_dict"], strict=False)
            print("[load] loaded from state_dict; rebuilt patches from state_dict.patch_map")
        else:
            raise RuntimeError("[load] checkpoint missing both 'patch_map' and 'state_dict'")

    @torch.no_grad()
    def log_patch_distribution(self, rays, iteration=None):
        """
        Print how many rays pass through each patch.
        Accepts either:
            - rays: [N,6] (origins + dirs) → use AABB hit midpoints to assign patches
            - rays: [N,3] (origins only)   → directly map origins to patches (fallback)
        """
        if rays.shape[-1] == 6:
            rays_o, rays_d = rays[:, :3], rays[:, 3:6]
            mid, hit = self._ray_aabb_midpoint(rays_o, rays_d)
            if hit.any():
                coords, exists = self._map_coords_to_patch(mid[hit])
                miss_ratio = 1.0 - exists.float().mean().item()
                if miss_ratio > 0.3:
                    print(f"[WARNING] {miss_ratio:.0%} of patch log samples map to MISSING patches (fallback risk)!")
                keys = [tuple(c.tolist()) for c in coords]
            else:
                keys = []
        elif rays.shape[-1] == 3:
            # Fallback: treat inputs as world-space positions (no dir info)
            coords, exists = self._map_coords_to_patch(rays)
            miss_ratio = 1.0 - exists.float().mean().item()
            if miss_ratio > 0.3:
                print(f"[WARNING] {miss_ratio:.0%} of patch log samples map to MISSING patches (fallback risk)!")
            keys = [tuple(c.tolist()) for c in coords]
        else:
            print("[log_patch_distribution] unexpected ray shape:", tuple(rays.shape))
            keys = []

        from collections import Counter
        cnt = Counter(keys)
        header = f"[PatchDist{' @'+str(iteration) if iteration is not None else ''}]"
        body = ", ".join(f"{k}:{v}" for k, v in sorted(cnt.items(), key=lambda x: x[0]))
        print(f"{header} {body if body else '(no hits)'}")

    def _serialize_patch(self, patch):
        lin = patch["basis_mat"]
        return {
            "res": list(patch["res"]),
            "density_plane": [p.detach().cpu() for p in patch["density_plane"]],
            "density_line":  [p.detach().cpu() for p in patch["density_line"]],
            "app_plane":     [p.detach().cpu() for p in patch["app_plane"]],
            "app_line":      [p.detach().cpu() for p in patch["app_line"]],
            "basis_w":       lin.weight.detach().cpu(),
            "basis_b":       (lin.bias.detach().cpu() if lin.bias is not None else None),
            "app_dim":       int(lin.out_features),
        }

    def _deserialize_patch(self, P, device):
        R = list(P["res"])
        new_patch = self._create_patch(R, device)

        def _copy(dst_pl, src_list):
            for di, si in zip(dst_pl, src_list):
                di.data.copy_(si.to(device))

        _copy(new_patch["density_plane"], P["density_plane"])
        _copy(new_patch["density_line"],  P["density_line"])
        _copy(new_patch["app_plane"],     P["app_plane"])
        _copy(new_patch["app_line"],      P["app_line"])

        in_dim  = P["basis_w"].shape[1]
        out_dim = int(P.get("app_dim", P["basis_w"].shape[0]))
        has_b   = (P["basis_b"] is not None)
        new_lin = torch.nn.Linear(in_dim, out_dim, bias=has_b).to(device)
        new_lin.weight.data.copy_(P["basis_w"].to(device))
        if has_b:
            new_lin.bias.data.copy_(P["basis_b"].to(device))
        new_patch["basis_mat"] = new_lin
        new_patch["res"] = R
        return new_patch

    def state_dict(self):
        """
        Save everything needed to reconstruct patches.
        """
        state = super().state_dict()
        patch_state = {}

        for (gx, gy, gz), patch in self.patch_map.items():
            key = f"{gx}_{gy}_{gz}"

            res = torch.tensor(patch['res'], dtype=torch.int32)
            patch_state[f"{key}.res"] = res

            for name in ['density_plane', 'density_line', 'app_plane', 'app_line']:
                for idx, param in enumerate(patch[name]):
                    patch_state[f"{key}.{name}.{idx}"] = param.detach().cpu()

            lin = patch['basis_mat']
            patch_state[f"{key}.basis_mat.in"]  = torch.tensor(lin.in_features, dtype=torch.int32)
            patch_state[f"{key}.basis_mat.out"] = torch.tensor(lin.out_features, dtype=torch.int32)
            has_bias = (lin.bias is not None)
            patch_state[f"{key}.basis_mat.has_bias"] = torch.tensor(1 if has_bias else 0, dtype=torch.uint8)
            patch_state[f"{key}.basis_mat.weight"] = lin.weight.detach().cpu()
            if has_bias:
                patch_state[f"{key}.basis_mat.bias"] = lin.bias.detach().cpu()

        state['patch_map'] = patch_state
        state['patch_grid_reso'] = torch.tensor(list(self.patch_grid_reso), dtype=torch.int32)
        state['aabb'] = self.aabb.detach().cpu()
        state['vm_reso_cur'] = torch.tensor(self.gridSize.tolist(), dtype=torch.int32)  
        return state

    def load_state_dict(self, state, strict=False):
        """
        Compatible ckpt:
        a) only basis_mat.weight (and maybe .bias) saved
        b) also save basis_mat.in / .out / .has_bias
        """
        super().load_state_dict(state, strict=False)

        device = self.aabb.device if hasattr(self, "aabb") else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        patch_state = state.get("patch_map", None)
        if patch_state is None:
            raise RuntimeError("[load_state_dict] missing 'patch_map' in state_dict")

        if "aabb" in state:
            self.aabb = state["aabb"].to(device)
        if "vm_reso_cur" in state:
            gs = state["vm_reso_cur"].tolist() if hasattr(state["vm_reso_cur"], "tolist") else list(state["vm_reso_cur"])
            self.gridSize = torch.LongTensor(gs).to(device)
            self.update_stepSize(gs)
        if "density_n_comp" in state:
            self.density_n_comp = list(map(int, state["density_n_comp"]))
        if "app_n_comp" in state:
            self.app_n_comp = list(map(int, state["app_n_comp"]))

        keys_by_patch = {}
        for full_key in patch_state.keys():
            pid = full_key.split(".")[0]  
            keys_by_patch.setdefault(pid, []).append(full_key)

        def _collect_paramlist(pid, name):
            plist, idx = [], 0
            while f"{pid}.{name}.{idx}" in patch_state:
                ten = patch_state[f"{pid}.{name}.{idx}"].to(device)
                plist.append(torch.nn.Parameter(ten, requires_grad=True))
                idx += 1
            return torch.nn.ParameterList(plist)

        new_map = {}
        for pid in keys_by_patch.keys():
            pi, pj, pk = map(int, pid.split("_"))

            res_t = patch_state[f"{pid}.res"]
            res = res_t.tolist() if hasattr(res_t, "tolist") else list(res_t)

            dpl = _collect_paramlist(pid, "density_plane")
            dln = _collect_paramlist(pid, "density_line")
            apl = _collect_paramlist(pid, "app_plane")
            aln = _collect_paramlist(pid, "app_line")

            has_w   = (f"{pid}.basis_mat.weight" in patch_state)
            has_b   = (f"{pid}.basis_mat.bias"   in patch_state)
            has_in  = (f"{pid}.basis_mat.in"     in patch_state)
            has_out = (f"{pid}.basis_mat.out"    in patch_state)

            if has_w: 
                W = patch_state[f"{pid}.basis_mat.weight"]
                out_dim, in_dim = int(W.shape[-2]), int(W.shape[-1])
            elif has_in and has_out:
                in_dim  = int(patch_state[f"{pid}.basis_mat.in"])
                out_dim = int(patch_state[f"{pid}.basis_mat.out"])
            else:
                in_dim  = self._app_in_dim_from_vm(apl, aln)
                out_dim = getattr(self, "app_dim", 27)

            bm = self.get_shared_basis(in_dim, out_dim)

            with torch.no_grad():
                if has_w:
                    W = patch_state[f"{pid}.basis_mat.weight"].to(device)
                    if tuple(W.shape) == tuple(bm.weight.shape):
                        bm.weight.copy_(W)
                    else:
                        print(f"[load_state_dict][WARN] basis weight shape mismatch: "
                            f"ckpt={tuple(W.shape)} vs cur={tuple(bm.weight.shape)}; skip copy.")
                if has_b and (bm.bias is not None):
                    B = patch_state[f"{pid}.basis_mat.bias"].to(device)
                    if tuple(B.shape) == tuple(bm.bias.shape):
                        bm.bias.copy_(B)
                    else:
                        print(f"[load_state_dict][WARN] basis bias shape mismatch: "
                            f"ckpt={tuple(B.shape)} vs cur={tuple(bm.bias.shape)}; skip copy.")

            patch = {
                "res": res,
                "density_plane": dpl,
                "density_line":  dln,
                "app_plane":     apl,
                "app_line":      aln,
                "basis_mat":     bm,
            }
            new_map[(pi, pj, pk)] = self._ensure_patch_device(patch, device)

        self.patch_map = new_map
        if "patch_grid_reso" in state:
            pgr = state["patch_grid_reso"]
            self.patch_grid_reso = tuple(pgr.tolist() if hasattr(pgr, "tolist") else list(pgr))
        else:
            self.patch_grid_reso = tuple(self.infer_patch_grid_reso_from_keys(self.patch_map))

        self.current_patch_keys = list(self.patch_map.keys())

        try:
            self.assert_zero_origin_and_contiguous()
        except Exception as e:
            print(f"[WARN] assert_zero_origin_and_contiguous failed: {e}")

    def _to_device_paramlist(self, pl, device):
        return torch.nn.ParameterList([
            torch.nn.Parameter(p.detach().to(device), requires_grad=p.requires_grad) for p in pl
        ])

    def _move_one_patch_to(self, patch, device):
        patch['density_plane'] = self._to_device_paramlist(patch['density_plane'], device)
        patch['density_line']  = self._to_device_paramlist(patch['density_line'],  device)
        patch['app_plane']     = self._to_device_paramlist(patch['app_plane'],     device)
        patch['app_line']      = self._to_device_paramlist(patch['app_line'],      device)

        if 'basis_mat' in patch and isinstance(patch['basis_mat'], torch.nn.Module):
            patch['basis_mat'] = patch['basis_mat'].to(device)
        if 'basis_B' in patch and isinstance(patch['basis_B'], torch.nn.Module):
            patch['basis_B'] = patch['basis_B'].to(device)
        if 'mix_W' in patch and isinstance(patch['mix_W'], torch.nn.Parameter):
            patch['mix_W'] = torch.nn.Parameter(patch['mix_W'].detach().to(device), requires_grad=True)

        return patch

    @torch.no_grad()
    def move_all_patches_to(self, device):
        for k in list(self.patch_map.keys()):
            self.patch_map[k] = self._move_one_patch_to(self.patch_map[k], device)

    def forward(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):
        """
        Ray marching over VM patches. Keeps a fallback patch so it won't crash
        when mapped coords miss current keys.
        """
        dev = self.aabb.device
        rays_chunk = rays_chunk.to(dev)
        rays_o, viewdirs = rays_chunk[:, :3], rays_chunk[:, 3:6]

        xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_o, viewdirs, is_train=is_train, N_samples=N_samples)
        dists = torch.cat([z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])], dim=-1)
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)

        sigma = xyz_sampled.new_zeros(xyz_sampled.shape[:-1])       # [N_rays, N_samples]
        rgb   = xyz_sampled.new_zeros((*xyz_sampled.shape[:2], 3))  # [N_rays, N_samples, 3]

        self.ensure_default_patch()

        # ========== density ==========
        if ray_valid.any():
            pts_world = xyz_sampled[ray_valid]  # world coords for valid samples
            pts_norm  = self.normalize_coord(pts_world)
            coords, exists, snapped = self._map_coords_to_patch(pts_world,
                                                                snap_missing=bool(getattr(self, "repair_enable", True)),
                                                                snap_tau=float(getattr(self, "repair_tau", 1.0)),
                                                                adjacent_only=bool(getattr(self, "repair_adjacent_only", True)),
                                                                return_snapped=True)

            if is_train and torch.rand(()) < 0.01:
                print(f"[repair] snapped(density)={snapped.float().mean().item():.3f}")
            miss_ratio = 1.0 - exists.float().mean().item()
            if miss_ratio > 0.3:
                print(f"[WARNING] {miss_ratio:.0%} of forward density map to MISSING patches (fallback risk)!")

            sigma_feat = self.compute_density_patchwise_fast(pts_norm, coords)
            sigma_sub  = self.feature2density(sigma_feat)
            sigma_sub  = torch.nan_to_num(sigma_sub, nan=0.0, posinf=1e6, neginf=0.0)

            if snapped.any():
                g = torch.ones_like(sigma_sub)
                g[snapped] = float(getattr(self, "repair_grad_scale_sigma", 0.0))
                sigma_sub = self._grad_scale(sigma_sub, g)

            gate = float(self.alpha_gate_scale)
            sigma_sub = sigma_sub * gate
            sigma = sigma.masked_scatter(ray_valid, sigma_sub)

        # ========== alpha & weights ==========
        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        # ========== appearance ==========
        app_mask   = (weight > self.rayMarch_weight_thres)  # [N_rays, N_samples]
        keep_count = app_mask.sum(dim=1)                    # [N_rays]
        no_keep    = (keep_count == 0)

        if no_keep.any():
            best_idx = weight.argmax(dim=1)
            app_mask[no_keep, best_idx[no_keep]] = True

        if not app_mask.any():
            app_mask = (weight > 0.0)

        if app_mask.any():
            pts_world = xyz_sampled[app_mask]
            pts_norm  = self.normalize_coord(pts_world)
            dirs_sub  = viewdirs[app_mask]
            coords, exists, snapped = self._map_coords_to_patch(pts_world,
                                                                snap_missing=bool(getattr(self, "repair_enable", True)),
                                                                snap_tau=float(getattr(self, "repair_tau", 1.0)),
                                                                adjacent_only=bool(getattr(self, "repair_adjacent_only", True)),
                                                                return_snapped=True)
            if is_train and torch.rand(()) < 0.01:
                print(f"[repair] snapped(appearance)={snapped.float().mean().item():.3f}")
            miss_ratio = 1.0 - exists.float().mean().item()
            if miss_ratio > 0.3:
                print(f"[WARNING] {miss_ratio:.0%} of forward appearance map to MISSING patches (fallback risk)!")

            app_feat  = self.compute_app_patchwise_fast(pts_norm, coords)  # [M_kept, app_dim]
            
            if snapped.any():
                g = torch.ones((app_feat.shape[0], 1), device=app_feat.device, dtype=app_feat.dtype)
                g[snapped] = float(getattr(self, "repair_grad_scale_app", 0.3))
                app_feat = self._grad_scale(app_feat, g)

            valid_rgbs = self.renderModule(pts_norm, dirs_sub, app_feat)  # [M_kept, 3]
            if valid_rgbs.dtype != rgb.dtype:
                valid_rgbs = valid_rgbs.to(rgb.dtype)
            rgb = rgb.masked_scatter(app_mask.unsqueeze(-1), valid_rgbs)  # [N_rays, N_samples, 3]

        # ========== synthesis ==========
        acc_map = torch.sum(weight, dim=-1)                  # [N_rays]
        rgb_map = torch.sum(weight[..., None] * rgb, dim=1)  # [N_rays, 3]

        if white_bg or (is_train and torch.rand(()) < 0.5):
            rgb_map = rgb_map + (1.0 - acc_map)[..., None]
        rgb_map = rgb_map.clamp(0.0, 1.0)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, dim=-1)
            depth_map = depth_map + (1.0 - acc_map) * rays_chunk[..., -1]

        return rgb_map, alpha, depth_map, weight, bg_weight
