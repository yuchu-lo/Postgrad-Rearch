import math
import torch
import torch.nn.functional as F
from collections import defaultdict

def _aabb_minmax(aabb: torch.Tensor):
    """
    Returns (aabb_min[3], aabb_max[3]) regardless of input layout:
      - (6,)  : [xmin,ymin,zmin, xmax,ymax,zmax]
      - (2,3) : [[xmin,ymin,zmin],
                 [xmax,ymax,zmax]]
    """
    if aabb is None:
        raise ValueError("AABB is None")
    if aabb.ndim == 1 and aabb.numel() == 6:
        return aabb[:3], aabb[3:6]
    if tuple(aabb.shape) == (2, 3):
        return aabb[0], aabb[1]
    raise ValueError(f"Unexpected AABB shape {tuple(aabb.shape)}; expected (6,) or (2,3).")

@torch.no_grad()
def uneven_critrn(test_dataset, tensorf, res_target, args, renderer, step, device="cuda"):    
    """
    Patch-wise refinement using PUF:
        P = MSE + λ * normalized_memory (+ α_b * boundary_cost_for_split).

    This version augments SPLIT decisions with a boundary capacity cost proxy:
      boundary_dof ≈ ry*rz + rx*rz + rx*ry, scaled by (1 - roughness_proxy),
      where roughness_proxy is the average relative improvement (coarse->fine)
      on the sampled rays for this patch.

    VM upgrades keep the original ranking (margin-based); boundary cost only
    affects split ranking to suppress over-splitting in smooth regions.

    Notes:
      - No model-class changes required.
      - align_corners is consistently False up the stack.
    """
    print("[uneven_critrn] Applying criterion ...")

    # ----------------- hyperparams & knobs -----------------
    lam      = float(args.critrn_lambda)             # λ for memory
    tau      = float(args.logmargin_tau)             # acceptance threshold on log-margin
    acc_r    = float(args.critrn_accept_ratio)       # fraction of views that must accept
    frac     = float(args.critrn_refine_frac)        # final refine fraction among candidates

    focus_start      = int(getattr(args, "critrn_focus_start", 8000))
    halo_cheby       = int(getattr(args, "critrn_focus_halo", 1))
    focus_samples    = int(getattr(args, "critrn_focus_samples", 8))
    min_total_rays   = int(getattr(args, "critrn_min_total_rays", 2000))
    global_mix_ratio = float(getattr(args, "critrn_global_mix_ratio", 0.3))
    sample_per_view  = int(getattr(args, "critrn_sample_rays", 2048))

    # boundary cost controls (safe defaults keep behavior similar if unset)
    alpha_boundary        = float(getattr(args, "puf_alpha_boundary", 0.3))  # 0.0 = disabled
    bcost_mode            = str(getattr(args, "boundary_cost_mode", "dof"))  # "dof" (current)
    bcost_smooth_strength = float(getattr(args, "boundary_smooth_strength", 1.0))  # scale on (1 - roughness)
    # ------------------------------------------------------------------------

    if   step < 5_000:  n_view = 3
    elif step < 10_000: n_view = 5
    else:               n_view = 8

    views  = min(n_view, len(test_dataset.all_rgbs))
    stride = max(len(test_dataset.all_rgbs) // max(views, 1), 1)

    rays_views = test_dataset.all_rays[::stride]
    gts_views  = [g.view(-1, 3).to(device) for g in test_dataset.all_rgbs[::stride]]

    tensorf.ensure_default_patch()
    patch_keys = list(tensorf.patch_map.keys())

    # ----------------- memory estimation helpers -----------------
    def _patch_mem(p):
        total = 0
        # planes / lines
        for k in ("density_plane", "density_line", "app_plane", "app_line"):
            for T in p.get(k, []):
                total += T.numel() * T.element_size()
        # basis (rank 相關)
        bm = p.get("basis_mat", None)
        if bm is not None and hasattr(bm, "weight"):
            total += bm.weight.numel() * bm.weight.element_size()
            if getattr(bm, "bias", None) is not None:
                total += bm.bias.numel() * bm.bias.element_size()
        return total
    
    max_mem = None

    # boundary cost proxy (per *current* patch; used for SPLIT ranking)
    def _boundary_cost_proxy_for_split(patch, roughness_avg: float, tensorf=None) -> float:
        """
        改進的邊界成本，考慮內容複雜度
        """
        if alpha_boundary <= 0.0:
            return 0.0
        
        rx, ry, rz = [int(x) for x in patch.get("res", (0, 0, 0))]
        
        if bcost_mode == "dof":
            # 基礎 DOF：三個內部切面的自由度
            base = float(ry * rz + rx * rz + rx * ry)
        else:
            base = 1.0
        
        # 計算內容複雜度（使用 VM 分解的能量分布）
        complexity = 1.0
        if tensorf is not None:
            # 計算 plane/line 的能量分布
            density_energy = 0.0
            app_energy = 0.0
            
            for i in range(3):
                # Density 複雜度
                d_plane = patch['density_plane'][i].detach()
                d_line = patch['density_line'][i].detach()
                density_energy += torch.std(d_plane).item() + torch.std(d_line).item()
                
                # App 複雜度
                a_plane = patch['app_plane'][i].detach()
                a_line = patch['app_line'][i].detach()
                app_energy += torch.std(a_plane).item() + torch.std(a_line).item()
            
            # 正規化複雜度到 [0.5, 2.0]
            complexity = 0.5 + min(1.5, (density_energy + app_energy) / 12.0)
        
        # 平滑因子：平滑區域懲罰更高（使用平方加強）
        rough = float(max(0.0, min(1.0, roughness_avg)))
        smooth_factor = (1.0 - rough) ** 2  # 平方加強平滑區域的懲罰
        
        # 最終成本
        cost = alpha_boundary * base * (bcost_smooth_strength * smooth_factor) * complexity
        
        return cost

    # ----------------- build focus maps per view -----------------
    focus_map     = []  # focus_map[v] : dict[patch_key(tuple int,int,int)] -> LongTensor(ray_idx_for_view_v)
    view_raycount = []

    def _build_focus_for_view(rays_all, halo=1, n_samples=8):
        """
        Returns:
          focus_dict: dict[patch_key] -> LongTensor(ray_idx) (unique, sorted)
          N:          total rays in this view
        Vectorized & GPU-only version; avoids CPU sets & per-item loops.
        """
        N = rays_all.shape[0]
        if step < focus_start:
            return {}, N

        rays_all = rays_all.to(device)
        rays_o, rays_d = rays_all[:, :3], rays_all[:, 3:6]
        xyz_s, _, valid = tensorf.sample_ray(rays_o, rays_d, is_train=False, N_samples=n_samples)  # [N,S,3], [N,S]
        if not valid.any():
            return {}, N

        # compact valid rays/samples
        idx_pairs = valid.nonzero(as_tuple=False)     # [M,2] -> (ray_idx, sample_idx)
        ray_idx   = idx_pairs[:, 0]                   # [M]
        xyz_flat  = xyz_s[valid]                      # [M,3]

        patch_idx, exists = tensorf._map_coords_to_patch(xyz_flat)  # [M,3], [M]
        if exists.numel() > 0:
            miss_ratio = 1.0 - exists.float().mean().item()
            if miss_ratio > 0.30:
                print(f"[WARN] view focus build: {miss_ratio:.0%} samples map to MISSING patches")

        # clamp to grid for halo expansion
        Gx, Gy, Gz = map(int, getattr(tensorf, "patch_grid_reso", (0,0,0)))
        if (Gx == 0) or (Gy == 0) or (Gz == 0):
            try:
                tensorf.patch_grid_reso = tuple(tensorf.infer_patch_grid_reso_from_keys(tensorf.patch_map))
                Gx, Gy, Gz = map(int, tensorf.patch_grid_reso)
            except Exception:
                pass

        if halo <= 0:
            # group rays by patch_idx (GPU)
            uniq, inv = torch.unique(patch_idx, dim=0, return_inverse=True)
            focus_dict = {}
            for gi in range(uniq.shape[0]):
                sel = (inv == gi).nonzero(as_tuple=False).view(-1)
                rsel = ray_idx.index_select(0, sel)
                focus_dict[tuple(int(x) for x in uniq[gi].tolist())] = torch.unique(rsel, sorted=True)
            return focus_dict, N

        # halo > 0 : expand Chebyshev neighborhood
        o = torch.arange(-halo, halo+1, device=device)
        ox, oy, oz = torch.meshgrid(o, o, o, indexing="ij")   # [(2h+1)^3,]
        offsets = torch.stack([ox.reshape(-1), oy.reshape(-1), oz.reshape(-1)], dim=1)  # [K,3]
        K = offsets.shape[0]

        # [M,1,3] + [1,K,3] -> [M,K,3]
        neigh = (patch_idx.view(-1,1,3) + offsets.view(1,-1,3))
        if Gx > 0 and Gy > 0 and Gz > 0:
            neigh[..., 0].clamp_(0, Gx-1)
            neigh[..., 1].clamp_(0, Gy-1)
            neigh[..., 2].clamp_(0, Gz-1)

        # flatten pairs to group by neighbor key
        neigh_flat = neigh.view(-1, 3)               # [M*K,3]
        rays_rep   = ray_idx.view(-1,1).repeat(1, K).reshape(-1)  # [M*K]

        uniq, inv = torch.unique(neigh_flat, dim=0, return_inverse=True)
        focus_dict = {}
        # group on CPU dict (per-unique-key loop; patch-level not per-ray set)
        for gi in range(uniq.shape[0]):
            sel = (inv == gi).nonzero(as_tuple=False).view(-1)
            rsel = rays_rep.index_select(0, sel)
            focus_dict[tuple(int(x) for x in uniq[gi].tolist())] = torch.unique(rsel, sorted=True)
        return focus_dict, N


    for v in range(views):
        fdict, n_rays = _build_focus_for_view(rays_views[v], halo=halo_cheby, n_samples=focus_samples)
        focus_map.append(fdict)
        view_raycount.append(int(n_rays))

    # ----------------- evaluate candidates -----------------
    def _sample_k_without_replacement(n: int, k: int, device):
        k = int(k)
        if k <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if k >= n:
            return torch.arange(n, device=device, dtype=torch.long)
        return torch.rand(n, device=device).topk(k, largest=False).indices

    candidates = []  # list of dict: {key, avg_margin, roughness_avg, split_cost}
    _up_cache = {}   # (key, res_target) -> (dp_up, dl_up, ap_up, al_up)
    _res_key = tuple(int(x) for x in res_target)

    for key in patch_keys:
        p = tensorf.patch_map[key]
        dp, dl  = p["density_plane"], p["density_line"]
        ap, al  = p["app_plane"],     p["app_line"]
        complexity = tensorf.compute_patch_complexity(p)

        _uk = (key, _res_key)
        if _uk in _up_cache:
            dp_up, dl_up, ap_up, al_up = _up_cache[_uk]
        else:
            if getattr(args, 'use_progressive_resolution', False):
                # 計算此 patch 的 PPR 目標解析度
                patch_importance = p.get('alpha_mass', 0.5)
                if hasattr(p, '_complexity_cache'):
                    patch_complexity = p._complexity_cache
                else:
                    patch_complexity = tensorf.compute_patch_complexity(p)
                    p._complexity_cache = patch_complexity
                
                ppr_target = tensorf.get_progressive_resolution(step, max(patch_importance, patch_complexity))
                # 使用 PPR 目標但不超過 res_target
                actual_target = tuple(min(p, r) for p, r in zip(ppr_target, res_target))
            else:
                actual_target = res_target
            
            dp_up, dl_up = tensorf.upsample_VM(dp, dl, actual_target)
            ap_up, al_up = tensorf.upsample_VM(ap, al, actual_target)
            _up_cache[_uk] = (dp_up, dl_up, ap_up, al_up)


        mem_c = _patch_mem({"density_plane": dp,    "density_line": dl,
                            "app_plane":     ap,    "app_line":     al,
                            "basis_mat":     p.get("basis_mat", None)})
        
        mem_f = _patch_mem({"density_plane": dp_up, "density_line": dl_up,
                            "app_plane":     ap_up, "app_line":     al_up,
                            "basis_mat":     p.get("basis_mat", None)})

        if max_mem is None:
            max_mem = max(1, mem_f * max(1, len(patch_keys)))

        # build batch across views (focus + global mix)
        batch_rays, batch_gt, seg_lengths = [], [], []

        for v in range(views):
            rays_all = rays_views[v].to(device)
            gt_all   = gts_views[v]
            Nv       = view_raycount[v]

            if step >= focus_start:
                fidx = focus_map[v].get(tuple(key), None)
                if fidx is None or fidx.numel() == 0:
                    seg_lengths.append(0)
                    continue

                k_focus  = min(fidx.numel(), max(min_total_rays, sample_per_view))
                idx      = _sample_k_without_replacement(fidx.numel(), k_focus, device)
                foc_sel  = fidx.index_select(0, idx)

                k_global = int(k_focus * global_mix_ratio)
                if k_global > 0:
                    mask = torch.zeros(Nv, dtype=torch.bool, device=device)
                    mask[foc_sel] = True
                    pool_other = (~mask).nonzero(as_tuple=False).squeeze(-1)
                    if pool_other.numel() > 0:
                        gidx  = _sample_k_without_replacement(pool_other.numel(), k_global, device)
                        sel   = torch.cat([foc_sel, pool_other.index_select(0, gidx)], 0)
                    else:
                        sel = foc_sel
                else:
                    sel = foc_sel
            else:
                k = min(sample_per_view, rays_all.shape[0])
                sel = torch.arange(k, device=device)

            if sel.numel() == 0:
                seg_lengths.append(0)
                continue

            batch_rays.append(rays_all[sel])
            batch_gt.append(gt_all[sel])
            seg_lengths.append(int(sel.numel()))

        total = sum(seg_lengths)
        if total == 0:
            continue

        rays_cat = torch.cat(batch_rays, dim=0)
        gt_cat   = torch.cat(batch_gt,   dim=0)

        # render (coarse then fine)
        with tensorf.no_seam_blend():
            with tensorf.fast_patch_vm(key, dp, dl, ap, al):
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        rgb_c = renderer(rays_cat)[0]

            with tensorf.fast_patch_vm(key, dp_up, dl_up, ap_up, al_up):
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        rgb_f = renderer(rays_cat)[0]

        # per-view margins & acceptance; also collect roughness proxy
        margins = []
        agree = 0
        rough_vals = []
        offset = 0
        gains = []
        norm_c = (mem_c / max_mem)
        norm_f = (mem_f / max_mem)

        for v in range(views):
            L = seg_lengths[v]
            if L == 0:
                continue

            rc = rgb_c[offset: offset + L]
            rf = rgb_f[offset: offset + L]
            gt = gt_cat[offset: offset + L]
            offset += L

            mse_c = F.mse_loss(rc, gt)
            mse_f = F.mse_loss(rf, gt)

            # PUF per-view
            Pc = mse_c + lam * norm_c
            Pf = mse_f + lam * norm_f
            
            margin = float(torch.log((Pf + 1e-12) / (Pc + 1e-12)))
            margins.append(margin)
            
            agree += int(margin < tau)

            # roughness proxy in [0,1]
            rel_improve = torch.clamp((mse_c - mse_f) / (mse_c + 1e-8), min=0.0, max=1.0)
            rough_vals.append(float(rel_improve))
            
            gains.append(float((Pc - Pf).detach().item()))

        used_views = len(margins)
        if used_views == 0:
            continue

        agree_ratio = agree / used_views
        if agree_ratio < acc_r:
            # skip patches that don't pass acceptance under current PUF (as before)
            continue

        avg_margin = sum(margins) / used_views
        rough_avg = (sum(rough_vals) / used_views) if len(rough_vals) > 0 else 0.0
        delta_mem_MB = max(1e-6, float(mem_f - mem_c) / (1024.0**2))
        avg_gain  = float(sum(gains) / used_views) if len(gains) == used_views else 0.0
        gain_per_mem = avg_gain / delta_mem_MB

        split_cost = _boundary_cost_proxy_for_split(p, rough_avg)
        complexity_factor = 1.0 - complexity * 0.5  # 複雜區域 cost 降低最多 50%
        split_cost = split_cost * complexity_factor

        candidates.append({
            "key": key,
            "avg_margin": avg_margin,
            "roughness": rough_avg,
            "split_cost": split_cost,
            "complexity": complexity,
            "avg_gain": avg_gain,
            "delta_mem_MB": delta_mem_MB,
            "gain_per_mem": gain_per_mem,
        })

    # ----------------- rank candidates & decide ops -----------------
    # original knobs
    mode          = str((getattr(args, "critrn_mode", "hybrid") or "hybrid")).strip().lower()  #  "vm" | "split" | "hybrid"
    vm_topk       = int(getattr(args, "critrn_vm_topk", 8))
    split_topk    = int(getattr(args, "critrn_split_topk", 4))
    VM_MAX        = int(getattr(args, "vm_reso_max", 128))
    vox_budget    = int(getattr(args, "voxel_budget", 1e12))
    vram_budgetMB = float(getattr(args, "vram_budget_MB", 1e9))

    vm_metric         = str(getattr(args, "critrn_vm_metric", "gain_per_mem"))  # "gain_per_mem" | "margin"
    floor_min_res     = int(getattr(args, "critrn_vm_floor_min", 0))            # e.g., 24 (0=disabled)
    floor_share       = float(getattr(args, "critrn_vm_floor_share", 0.35))   
    floor_min_count   = int(getattr(args, "critrn_vm_floor_min_count", 1))      

    vox_now       = int(getattr(tensorf, "get_total_voxels", lambda: 0)())
    mem_now_MB    = float(getattr(tensorf, "get_total_mem",   lambda: 0)() / 1024**2)
    headroom_vox  = vox_now < 0.95 * vox_budget
    headroom_mem  = mem_now_MB < 0.95 * vram_budgetMB

    def _can_vm_upgrade(patch):
        r = patch.get("res", [0, 0, 0])
        return (min(r) < min(res_target)) and headroom_vox and headroom_mem

    patch_keys = list(tensorf.patch_map.keys())
    if getattr(args, 'use_progressive_resolution', False):
        print("[uneven_critrn] Pre-computing patch complexity for PPR...")
        for key, patch in tensorf.patch_map.items():
            if not hasattr(patch, '_complexity_cache') or patch.get('_last_complexity_update', 0) < step - 1000:
                complexity = tensorf.compute_patch_complexity(patch)
                patch._complexity_cache = complexity
                patch._last_complexity_update = step

    max_by_ratio = max(1, int(len(candidates) * frac))
    max_by_abs   = max(1, min(6, int(0.10 * max(1, len(patch_keys)))))
    split_budget = min(max_by_ratio, max_by_abs)
    vm_budget    = vm_topk

    # --- Build rankings ---
    if vm_metric == "gain_per_mem":
        cand_vm_sorted = sorted(candidates, key=lambda d: (-d.get("gain_per_mem", 0.0), d["avg_margin"]))
    else:  
        cand_vm_sorted = sorted(candidates, key=lambda d: d["avg_margin"])

    def split_score(d):
        base_score = d["avg_margin"] + d["split_cost"]
        # 複雜區域加分（優先 split）
        complexity_bonus = -d["complexity"] * 0.3
        return base_score + complexity_bonus
    
    cand_split_sorted = sorted(candidates, key=split_score)

    to_promote, to_split = [], []

    if floor_min_res > 0 and vm_budget > 0:
        low = []
        for d in candidates:
            k = d["key"]; p = tensorf.patch_map.get(k, {})
            r = p.get("res", (0,0,0))
            if min(r) < floor_min_res and _can_vm_upgrade(p):
                # 重要度：優先用 gain_per_mem，否則 roughness+反向margin
                imp = d.get("gain_per_mem", None)
                if imp is None:
                    imp = (1.0 - d.get("avg_margin", 0.0)) + 0.1 * d.get("roughness", 0.0)
                low.append((imp, k))
        low.sort(key=lambda x: x[0], reverse=True)
        quota = max(floor_min_count, int(round(vm_budget * floor_share)))
        for _, k in low[:quota]:
            if k not in to_promote:
                to_promote.append(k)

    if mode in ("vm", "hybrid"):
        for d in cand_vm_sorted:
            if len(to_promote) >= vm_budget:
                break
            k = d["key"]
            if k in to_promote:
                continue
            p = tensorf.patch_map[k]
            if _can_vm_upgrade(p):
                to_promote.append(k)

    if mode in ("split", "hybrid"):
        for d in cand_split_sorted[:split_budget]:
            to_split.append(d["key"])

    n_ops = 0

    assert len(set(to_promote) & set(to_split)) == 0, "promote/split overlap"

    # ----- apply VM upgrades (batched with PPR) -----
    if len(to_promote) > 0:
        use_ppr = bool(getattr(args, 'use_progressive_resolution', False))
        
        if use_ppr:
            # 使用漸進式解析度，每個 patch 可能有不同的目標
            actually_promoted = 0
            for key in to_promote:
                patch = tensorf.patch_map[key]
                current_res = tuple(patch.get('res', [8, 8, 8]))
                
                # 根據 patch 重要性和複雜度獲取目標解析度
                importance = patch.get('alpha_mass', 0.5)
                
                # 使用已計算的複雜度（如果有）
                if hasattr(patch, '_complexity_cache'):
                    complexity = patch._complexity_cache
                else:
                    complexity = tensorf.compute_patch_complexity(patch)
                    patch._complexity_cache = complexity
                
                combined_score = max(importance, complexity)
                
                # 獲取 PPR 目標解析度
                target_res_ppr = tensorf.get_progressive_resolution(step, combined_score)
                
                # 確保不超過全局 res_target
                target_res_final = tuple(min(t, r) for t, r in zip(target_res_ppr, res_target))
                
                # 只有當目標解析度大於當前解析度時才升級
                if any(t > c for t, c in zip(target_res_final, current_res)):
                    print(f"[PUF-PPR] Patch {key}: {current_res} -> {target_res_final} (imp={importance:.3f}, comp={complexity:.3f})")
                    
                    # 執行 upsample
                    dp_up, dl_up = tensorf.upsample_VM(patch["density_plane"], patch["density_line"], target_res_final)
                    ap_up, al_up = tensorf.upsample_VM(patch["app_plane"], patch["app_line"], target_res_final)
                    
                    patch["density_plane"] = dp_up
                    patch["density_line"] = dl_up
                    patch["app_plane"] = ap_up
                    patch["app_line"] = al_up
                    patch["res"] = list(target_res_final)
                    
                    actually_promoted += 1
            
            n_ops += actually_promoted
            print(f"[uneven_critrn] PPR VM-upgraded {actually_promoted} patches")
        else:
            # 原始邏輯：批量升級到固定解析度
            promoted = tensorf.upsample_patches(
                to_promote, tuple(res_target), mode="bilinear", align_corners=False, verbose=True
            )
            n_ops += int(promoted)
            print(f"[uneven_critrn] VM-upgraded {promoted} patches to {tuple(res_target)}")

    # ----- apply splits + build field-KD buffers (teacher = pre-split parent) -----
    if len(to_split) > 0:
        def _child_keys_of(parent_key):
            x, y, z = int(parent_key[0]), int(parent_key[1]), int(parent_key[2])
            return [(2*x+dx, 2*y+dy, 2*z+dz) for dz in (0,1) for dy in (0,1) for dx in (0,1)]

        def _cell_world_aabb(key_xyz, G_before, aabb):
            """Return (lo, hi), world AABB of the single cell indexed by key_xyz under grid size G_before*2 (children)."""
            device = aabb.device
            key = torch.tensor(key_xyz, dtype=torch.float32, device=device)
            Gf  = torch.tensor(G_before, dtype=torch.float32, device=device) * 2.0
            lo01 = key / Gf
            hi01 = (key + 1.0) / Gf
            a_min, a_max = _aabb_minmax(aabb)
            lo = a_min + lo01 * (a_max - a_min)
            hi = a_min + hi01 * (a_max - a_min)
            return lo, hi

        def _sample_in_aabb(lo, hi, K):
            """Uniform samples in a world-space AABB."""
            u = torch.rand(K, 3, device=device)
            return lo.view(1,3) + u * (hi - lo).view(1,3)

        # build field-KD buffers *before* actually mutating patch_map
        kd_enable   = bool(int(getattr(args, "post_event_kd", 1)) == 1)
        kd_K        = int(getattr(args, "kd_pts_per_child", 256))
        kd_horizon  = int(getattr(args, "post_event_kd_horizon", 600))
        kd_sigma_w  = float(getattr(args, "kd_sigma_weight", 1.0))
        kd_app_w    = float(getattr(args, "kd_app_weight", 1.0))

        if kd_enable and not hasattr(tensorf, "_kd_buffers"):
            tensorf._kd_buffers = []  # list of dicts

        # cache current grid size (before split) to compute child-world AABB
        try:
            G_before = tuple(tensorf.patch_grid_reso)
        except Exception:
            G_before = tuple(tensorf.infer_patch_grid_reso_from_keys(tensorf.patch_map))

        aabb = tensorf.aabb  # world AABB, tensor of shape [6] on cuda

        # for each parent, prepare KD targets from the *parent* field
        kd_records_children = []
        if kd_enable:
            for pkey in to_split:
                parent = tensorf.patch_map[pkey]
                child_keys = _child_keys_of(pkey)
                xyz_list, sig_tgt_list, app_tgt_list = [], [], []

                # Evaluate the parent's field in its local coords:
                # map world xyz -> parent-local [-1,1]^3, then use compute_*_patch(parent, ...)
                # Get parent cell world AABB once
                #   parent cell indices are pkey in grid G_before
                #   its lo/hi (world) is just child AABB merged; but only need normalization,
                #   so derive directly from pkey & G_before:
                key = torch.tensor(pkey, dtype=torch.float32, device=device)
                Gf  = torch.tensor(G_before, dtype=torch.float32, device=device)
                plo01 = key / Gf
                phi01 = (key + 1.0) / Gf
                a_min, a_max = _aabb_minmax(aabb)
                parent_lo = a_min + plo01 * (a_max - a_min)
                parent_hi = a_min + phi01 * (a_max - a_min)

                for ckey in child_keys:
                    lo, hi = _cell_world_aabb(ckey, G_before, aabb)
                    xyz_w = _sample_in_aabb(lo, hi, kd_K)  # [K,3] world
                    # world -> parent-local [-1,1]
                    xyz_loc01 = (xyz_w - parent_lo.view(1,3)) / (parent_hi - parent_lo).view(1,3)
                    xyz_loc   = xyz_loc01 * 2.0 - 1.0

                    # teacher targets from *parent* patch params
                    sig_feat = tensorf.compute_density_patch(parent, xyz_loc)       # [K, C_sigma]
                    sigma_t  = tensorf.feature2density(sig_feat).detach()           # [K, 1] or [K]
                    app_t    = tensorf.compute_app_patch(parent, xyz_loc).detach()  # [K, C_app]

                    xyz_list.append(xyz_w)
                    sig_tgt_list.append(sigma_t.view(-1, 1))
                    app_tgt_list.append(app_t)

                if xyz_list:
                    kd_entry = dict(
                        children=set(map(tuple, child_keys)),
                        world_xyz=torch.cat(xyz_list, dim=0).cpu(),
                        sigma_tgt=torch.cat(sig_tgt_list, dim=0).cpu(),
                        app_tgt=torch.cat(app_tgt_list, dim=0).cpu(),
                        sigma_w=float(kd_sigma_w),
                        app_w=float(kd_app_w),
                        expires_at=int(step) + int(kd_horizon),
                    )
                    tensorf._kd_buffers.append(kd_entry)
                    kd_records_children.extend(child_keys)

        # apply the real split (mutate patch_map)
        if (step >= args.split_warmup_iters) and (len(to_split) > 0):
            use_ppr = bool(getattr(args, 'use_progressive_resolution', False))
            new_map = dict(tensorf.patch_map)
            
            for key in to_split:
                parent = new_map.pop(key)
                
                if use_ppr:
                    # 使用 PPR 決定子 patch 的解析度
                    parent_importance = parent.get('alpha_mass', 0.5)
                    
                    # 獲取父 patch 複雜度
                    if hasattr(parent, '_complexity_cache'):
                        parent_complexity = parent._complexity_cache
                    else:
                        parent_complexity = tensorf.compute_patch_complexity(parent)
                    
                    # 子 patch 的目標解析度基於父 patch 的重要性和訓練進度
                    child_res = tensorf.get_progressive_resolution(step, max(parent_importance, parent_complexity))
                    
                    # 確保子 patch 解析度合理
                    parent_res = parent.get('res', [16, 16, 16])
                    
                    # 策略：子 patch 不應該比父 patch 小太多（最多減半）
                    min_child_res = tuple(max(8, r // 2) for r in parent_res)
                    child_res = tuple(max(c, m) for c, m in zip(child_res, min_child_res))
                    
                    # 也不應該比父 patch 大太多（最多 1.5 倍）
                    max_child_res = tuple(min(64, int(r * 1.5)) for r in parent_res)
                    child_res = tuple(min(c, m) for c, m in zip(child_res, max_child_res))
                    
                    print(f"[PPR-split] Parent {key} (res={parent_res}, imp={parent_importance:.3f}) -> children (res={child_res})")
                else:
                    # 原始邏輯：子 patch 保持父 patch 的解析度
                    child_res = parent["res"]
                
                children = tensorf.split_patch(key, parent, child_res)
                new_map.update(children)
            
            tensorf.patch_map = new_map
            n_ops += len(to_split)
            
            if use_ppr:
                print(f"[uneven_critrn] split {len(to_split)} patches with PPR-based resolution")
            else:
                print(f"[uneven_critrn] split {len(to_split)} patches (children keep parent res)")

            try:
                tensorf._last_split_children = list(set(kd_records_children)) if kd_records_children else []
                tensorf._last_split_iter = int(step)
            except Exception:
                pass

    # ----------------- optional tiny child boost (disabled by default) -----------------
    SPLIT_BOOST = int(getattr(args, "split_boost_enable", 0)) == 1
    if SPLIT_BOOST:
        boost_start   = int(getattr(args, "split_boost_start", 12000))
        boost_topq    = float(getattr(args, "split_boost_topq", 0.20))   # top 20% children by alpha_mass
        boost_factor  = float(getattr(args, "split_boost_factor", 1.5))  # res ×1.5 (ceil)
        boost_vm_cap  = int(getattr(args, "split_boost_vm_cap", 8))      # max boosted children per round
        VM_MAX        = int(getattr(args, "vm_reso_max", 128))

        if step >= boost_start and headroom_vox and headroom_mem and len(to_split) > 0:
            new_child_keys = []
            for key in to_split:
                for dz in (0, 1):
                    for dy in (0, 1):
                        for dx in (0, 1):
                            k = (key[0]*2 + dx, key[1]*2 + dy, key[2]*2 + dz)
                            if k in tensorf.patch_map:
                                new_child_keys.append(k)

            tensorf.update_alpha_mass_per_patch(n_per=getattr(args, "alpha_mass_n", 1024))

            scored = []
            for k in new_child_keys:
                p = tensorf.patch_map.get(k, None)
                if p is None: continue
                am = float(p.get("alpha_mass", 0.0))
                scored.append((am, k))
            scored.sort(reverse=True)

            n_pick = min(boost_vm_cap, max(0, int(math.ceil(len(scored) * boost_topq))))
            if n_pick > 0:
                keys_boost = [k for _, k in scored[:n_pick]]
                any_child = tensorf.patch_map[keys_boost[0]]
                r0 = tuple(int(x) for x in any_child["res"])
                r_boost = tuple(int(min(VM_MAX, max(x, math.ceil(x * boost_factor)))) for x in r0)

                promoted = tensorf.upsample_patches(
                    keys_boost, r_boost, mode="bilinear", align_corners=False, verbose=False
                )
                print(f"[split-boost] promoted {promoted}/{len(keys_boost)} children -> {r_boost}")

    tensorf.current_patch_keys = list(tensorf.patch_map.keys())
    try:
        tensorf.patch_grid_reso = tuple(tensorf.infer_patch_grid_reso_from_keys(tensorf.patch_map))
    except Exception as e:
        print(f"[WARN] infer_patch_grid_reso_from_keys failed: {e}")

    if hasattr(tensorf, "assert_zero_origin_and_contiguous"):
        try:
            tensorf.assert_zero_origin_and_contiguous()
        except Exception as e:
            print(f"[WARN] assert_zero_origin_and_contiguous failed: {e}")

    print(f"[uneven_criterion] done: ops={n_ops} (mode={mode}) — total patches={len(tensorf.patch_map)}")
    return int(n_ops)

