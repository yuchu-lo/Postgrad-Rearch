import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["figure.max_open_warning"] = 0  
HEADLESS = True
from matplotlib import cm
import matplotlib.colors as mcolors

def _compute_alpha_metric_for_keys(tensorf, keys, aabb_min, cell, alpha_points=2048,
                                   mode="alpha_occ", alpha_tau=1e-2):
    vals = []
    dev = tensorf.aabb.device
    step = float(tensorf.stepSize.detach().clamp(min=1e-3))
    gate = float(getattr(tensorf, "alpha_gate_scale", 1.0))
    with torch.no_grad():
        for (i, j, k) in keys:
            u = torch.rand((alpha_points, 3), device=dev)
            lo = torch.tensor(aabb_min + np.array([i, j, k]) * cell, device=dev, dtype=torch.float32)
            hi = torch.tensor(aabb_min + (np.array([i, j, k]) + 1) * cell, device=dev, dtype=torch.float32)
            xyz  = lo + u * (hi - lo)
            xyz_n = tensorf.normalize_coord(xyz)
            coords = torch.tensor([i, j, k], device=dev, dtype=torch.long).view(1, 3).repeat(alpha_points, 1)

            feat  = tensorf.compute_density_patchwise_fast(xyz_n, coords)
            sigma = tensorf.feature2density(feat) * gate
            alpha = 1.0 - torch.exp(-sigma * step)

            if mode == "alpha_occ":
                val = float((alpha > alpha_tau).float().mean().item())
            elif mode == "alpha_p90":
                val = float(torch.quantile(alpha, 0.9).item())
            else:
                raise ValueError(f"Unknown alpha mode: {mode}")
            vals.append(val)
    return np.array(vals, dtype=np.float64)

def _compute_color_values(color_by, voxels_res, depth_list, dmax,
                          keys, tensorf, aabb_min, cell,
                          alpha_points=2048, alpha_tau=1e-2):
    """
    Return: vals, scale, vmin, vmax, label
    scale: 'linear' or 'log'
    """
    if color_by == "res":
        vals  = voxels_res.astype(np.float64)
        scale = "log"
        label = "per-patch VM voxels (prod of res)"
        vmin  = max(float(vals.min()) if vals.size else 1.0, 1.0)
        vmax  = float(vals.max()) if vals.size else 1.0
    elif color_by == "depth":
        vals  = np.array(depth_list, dtype=np.float64)
        scale = "linear"
        label = "patch split depth"
        vmin, vmax = 0.0, max(1.0, float(dmax))
    elif color_by in ("alpha_occ", "alpha_p90"):
        mode  = color_by
        vals  = _compute_alpha_metric_for_keys(
            tensorf, keys, aabb_min, cell,
            alpha_points=alpha_points, mode=mode, alpha_tau=alpha_tau
        )
        scale = "linear"
        label = "alpha occupancy (> {:.3g})".format(alpha_tau) if mode=="alpha_occ" else "alpha 90th percentile"
        vmin, vmax = 0.0, 1.0
    else:
        raise ValueError(f"Unknown color_by={color_by}")
    return vals, scale, vmin, vmax, label

def _map_colors(vals, scale, vmin, vmax, cmap_name, running_key=None):
    if not hasattr(_map_colors, "_running_max"):
        _map_colors._running_max = {}
    if running_key is not None:
        vmax_hist = _map_colors._running_max.get(running_key, vmax)
        vmax = max(float(vmax_hist), float(vmax))
        _map_colors._running_max[running_key] = vmax

    if scale == "log":
        norm = mcolors.LogNorm(vmin=max(vmin, 1e-8), vmax=max(vmax, 1e-8))
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    cmap = cm.get_cmap(cmap_name)
    colors = cmap(norm(vals))
    return colors, norm, vmax

def _save_and_close(fig, path=None, *, dpi=200):
    try:
        if path is not None:
            fig.savefig(path, dpi=dpi, bbox_inches='tight')
        elif not HEADLESS:
            plt.show()
    finally:
        plt.close(fig)

def draw_true_patch_boxes(ax, tensorf, *, step: int = 1, lw: float = 0.7, alpha: float = 0.5):
    """
    Draw per-patch boxes using base grid and each patch's depth.
    Smaller boxes = refined (depth>0). Keeps the view clean and faithful to uneven splits.
    """
    a0 = tensorf.aabb[0].detach().cpu().numpy()
    a1 = tensorf.aabb[1].detach().cpu().numpy()
    extent = np.maximum(a1 - a0, 1e-8)
    baseG = np.array(getattr(tensorf, "base_patch_grid_reso", tensorf._get_patch_grid_reso()))
    keys = list(tensorf.patch_map.keys())
    if not keys: return

    def _draw_wire_box(lo, hi):
        xs = [lo[0], hi[0]]; ys = [lo[1], hi[1]]; zs = [lo[2], hi[2]]
        # 12 edges
        segs = [
            ([xs[0], xs[1]], [ys[0], ys[0]], [zs[0], zs[0]]),
            ([xs[0], xs[1]], [ys[1], ys[1]], [zs[0], zs[0]]),
            ([xs[0], xs[1]], [ys[0], ys[0]], [zs[1], zs[1]]),
            ([xs[0], xs[1]], [ys[1], ys[1]], [zs[1], zs[1]]),
            ([xs[0], xs[0]], [ys[0], ys[1]], [zs[0], zs[0]]),
            ([xs[1], xs[1]], [ys[0], ys[1]], [zs[0], zs[0]]),
            ([xs[0], xs[0]], [ys[0], ys[1]], [zs[1], zs[1]]),
            ([xs[1], xs[1]], [ys[0], ys[1]], [zs[1], zs[1]]),
            ([xs[0], xs[0]], [ys[0], ys[0]], [zs[0], zs[1]]),
            ([xs[1], xs[1]], [ys[0], ys[0]], [zs[0], zs[1]]),
            ([xs[0], xs[0]], [ys[1], ys[1]], [zs[0], zs[1]]),
            ([xs[1], xs[1]], [ys[1], ys[1]], [zs[0], zs[1]]),
        ]
        for X, Y, Z in segs:
            ax.plot(X, Y, Z, color="C0", linewidth=lw, alpha=alpha)

    for idx, key in enumerate(keys[::max(1, int(step))]):
        d = int(tensorf.patch_map[key].get('depth', 0))
        den = baseG * (2 ** d)  # cells per axis at this depth
        ijk = np.array(key, dtype=np.int64)
        lo = a0 + (ijk / den) * extent
        hi = a0 + ((ijk + 1) / den) * extent
        _draw_wire_box(lo, hi)

def plot_patch_scatter_and_grid(
    patch_map,
    tensorf,
    show_global_grid: bool = True,
    save_prefix: str = None,
    # VM res → center size
    encode_vm: bool = True,
    size_base: float = 60.0,
    size_gamma: float = 0.35,
    # color (content indicator or depth/res)
    color_by: str = "alpha_occ",       # "res" | "alpha_occ" | "alpha_p90" | "depth"
    alpha_tau: float = 1e-2,
    alpha_points: int = 2048,
    cmap_name: str = "viridis",
    cb_vmin: float = None,
    cb_vmax: float = None,
    # inner VM-grid lines
    inner_grid_for: str = "refined",   # "refined" | "topk" | "all" | "none"
    draw_inner_grid_k: int = 5,        # used when inner_grid_for="topk"
    # two strategies (choose one):
    # 1) explicit map: min(res) → lines per axis
    inner_grid_lines_map: dict = None,
    # 2) auto-scaling: base * (min(res)/ref)^gamma, clamped
    inner_grid_ref_res: int = 128,
    inner_grid_base_lines: int = 9,
    inner_grid_gamma: float = 0.6,
    inner_grid_min_lines: int = 3,
    inner_grid_max_lines: int = 12,
    slice_axis: str = None,            # 'x'|'y'|'z' or None
    slice_lo: float = 0.0,             # normalized [0,1] within AABB
    slice_hi: float = 1.0,
    # draw options
    draw_true_boxes: bool = True,      # draw depth-aware blue boxes (true patch cells)
    draw_centers: bool = True,         # draw patch centers (size=VM res; color per 'color_by')
    fig1_overlay: str = "centers",     # "none" | "true" | "centers"
    # extra views for fig.2 (list of (elev, azim))
    views2: list = None,
    save_tag: str = "",                # "clean" / "debug" / "oview"
):
    """
    Make two figures:
    (1) Global wireframe (optionally with light overlay).
    (2) Wireframe + (optional) true boxes, patch centers, and downsampled VM inner-grids.

    Tips:
    - color_by: "res" | "alpha_occ" | "alpha_p90" | "depth"
    - inner_grid_for: "refined" (depth>0) | "topk" | "all" | "none"
    - For clean slides: draw_centers=False, inner_grid_for="none", draw_true_boxes=True

    
    Recommended settings:
    (1) For clear representation:
        plot_patch_scatter_and_grid(...,
            draw_centers=False, inner_grid_for="none",
            draw_true_boxes=True, fig1_overlay="true", color_by="depth"
        )
    (2) For debugging and details:
        plot_patch_scatter_and_grid(...,
            draw_centers=True, color_by="depth",
            inner_grid_for="refined", draw_true_boxes=True, fig1_overlay="centers"
        )
    """
    def _infer_G(pmap):
        if len(pmap) == 0: return (1, 1, 1)
        mi = max(k[0] for k in pmap.keys()) + 1
        mj = max(k[1] for k in pmap.keys()) + 1
        mk = max(k[2] for k in pmap.keys()) + 1
        return (mi, mj, mk)

    def _set_axes(ax, lo, hi):
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    def _draw_wire(ax, lo, hi, Gx, Gy, Gz):
        xs = np.linspace(lo[0], hi[0], Gx + 1)
        ys = np.linspace(lo[1], hi[1], Gy + 1)
        zs = np.linspace(lo[2], hi[2], Gz + 1)
        lw, alp = 0.5, 0.25
        for x in xs:
            for y in ys:
                ax.plot([x, x], [y, y], [lo[2], hi[2]], color="gray", linewidth=lw, alpha=alp)
        for x in xs:
            for z in zs:
                ax.plot([x, x], [lo[1], hi[1]], [z, z], color="gray", linewidth=lw, alpha=alp)
        for y in ys:
            for z in zs:
                ax.plot([lo[0], hi[0]], [y, y], [z, z], color="gray", linewidth=lw, alpha=alp)
    
    keys = list(patch_map.keys())
    ctrs = []

    if not isinstance(patch_map, dict) or len(patch_map) == 0:
        return

    aabb_min = tensorf.aabb[0].detach().cpu().numpy()
    aabb_max = tensorf.aabb[1].detach().cpu().numpy()

    try:
        Gx, Gy, Gz = [int(x) for x in getattr(tensorf, "patch_grid_reso", [])]
    except Exception:
        Gx = Gy = Gz = None
    if not (Gx and Gy and Gz):
        Gx, Gy, Gz = _infer_G(patch_map)

    G = np.array([Gx, Gy, Gz], dtype=np.float64)
    cell = (aabb_max - aabb_min) / G

    # ---------- fig.1：wireframe ----------
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    if show_global_grid:
        _draw_wire(ax, aabb_min, aabb_max, Gx, Gy, Gz)
    
    if fig1_overlay == "true":
        draw_true_patch_boxes(ax, tensorf, step=2, lw=0.6, alpha=0.5) 
    elif fig1_overlay == "centers" and len(patch_map) > 0:
        keys = sorted(list(patch_map.keys()))
        G = np.array([Gx, Gy, Gz], dtype=np.float64)
        cell = (aabb_max - aabb_min) / G
        ctrs = np.array([aabb_min + (np.array([i, j, k], dtype=np.float64) + 0.5) * cell for (i, j, k) in keys])
        ax.scatter(ctrs[:,0], ctrs[:,1], ctrs[:,2], s=6, c="#666666", alpha=0.7, depthshade=False, linewidths=0)

    _set_axes(ax, aabb_min, aabb_max)
    tag = (f"_{save_tag}" if save_tag else "")
    if save_prefix:
        _save_and_close(fig, f"{save_prefix}{tag}_grid.png", dpi=200)
    else:
        _save_and_close(fig, None) 

    # ---------- fig.2：wireframe + patches ----------
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    if show_global_grid:
        _draw_wire(ax, aabb_min, aabb_max, Gx, Gy, Gz)

    keys = sorted(list(patch_map.keys()))
    if len(keys) > 0:
        ctrs = np.array([aabb_min + (np.array([i, j, k], dtype=np.float64) + 0.5) * cell
                         for (i, j, k) in keys])
        
        # optional slab slicing on centers & inner grids
        mask = np.ones(len(keys), dtype=bool)
        if slice_axis in ('x','y','z'):
            axis_idx = {'x':0,'y':1,'z':2}[slice_axis]
            lo = aabb_min[axis_idx] + (aabb_max[axis_idx] - aabb_min[axis_idx]) * float(slice_lo)
            hi = aabb_min[axis_idx] + (aabb_max[axis_idx] - aabb_min[axis_idx]) * float(slice_hi)
            mask = (ctrs[:, axis_idx] >= lo) & (ctrs[:, axis_idx] <= hi)
       
            keys = [k for (k, m) in zip(keys, mask) if m]
            ctrs = ctrs[mask]

        depth_list = [int(patch_map[k].get("depth", 0)) for k in keys]
        dmax = int(max(depth_list)) if len(depth_list) > 0 else 0

        res_list = [np.array(patch_map[k].get("res", [1, 1, 1]), dtype=np.float64) for k in keys]
        voxels_res = np.array([np.prod(r) for r in res_list], dtype=np.float64)

        vals, scale, vmin, vmax, cbar_label = _compute_color_values(color_by, voxels_res, depth_list, dmax,
                                                                    keys, tensorf, aabb_min, cell,
                                                                    alpha_points=alpha_points, alpha_tau=alpha_tau
                                                                    )

        # point size
        if encode_vm and np.any(voxels_res > 0):
            base = float(max(voxels_res.min(), 1.0))
            sizes = size_base * np.power(np.maximum(voxels_res / base, 1.0), size_gamma)
        else:
            sizes = np.full((len(keys),), size_base)

        running_key = (color_by, cmap_name)
        colors, norm, vmax_used = _map_colors(vals, scale, vmin, vmax, cmap_name, running_key)

        if draw_true_boxes:
            draw_true_patch_boxes(ax, tensorf, step=1, lw=0.8, alpha=0.7)

        if draw_centers and len(keys) > 0:
            ax.scatter(ctrs[:, 0], ctrs[:, 1], ctrs[:, 2],
                       s=sizes, c=colors, alpha=0.75, depthshade=True,
                       edgecolors="none", linewidths=0.0
            )
            mappable = cm.ScalarMappable(norm=norm, cmap=cmap_name)
            mappable.set_array(vals)
            cbar = plt.colorbar(mappable, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label(cbar_label)

        if inner_grid_for.lower() == "none":
            targets = []
        elif inner_grid_for.lower() == "all":
            targets = np.arange(len(keys))
        elif inner_grid_for.lower() == "refined":
            targets = np.where(np.array(depth_list) > 0)[0]
        else:  # "topk"
            targets = np.argsort(-voxels_res)[:draw_inner_grid_k]

        def _lines_per_axis(min_res_val: int) -> int:
            if inner_grid_lines_map:
                ks = sorted(inner_grid_lines_map.keys())
                nearest = min(ks, key=lambda x: abs(int(x) - int(min_res_val)))
                val = int(inner_grid_lines_map[nearest])
            else:
                scale = max(min_res_val / float(inner_grid_ref_res), 1e-6)
                val = int(round(inner_grid_base_lines * (scale ** inner_grid_gamma)))
            return int(np.clip(val, inner_grid_min_lines, inner_grid_max_lines))

        lw, alp = 0.6, 0.35
        for ti in targets:
            (i, j, k) = keys[ti]
            r = res_list[ti].astype(int)  # actual VM res（Hx,Hy,Hz）
            lo = aabb_min + np.array([i, j, k], dtype=np.float64) * cell
            hi = lo + cell
            want = _lines_per_axis(int(r.min()))
            strides = np.maximum(1, np.floor(r / max(want,1)).astype(int))
            xs = np.linspace(lo[0], hi[0], r[0] + 1)[::strides[0]]
            ys = np.linspace(lo[1], hi[1], r[1] + 1)[::strides[1]]
            zs = np.linspace(lo[2], hi[2], r[2] + 1)[::strides[2]]
            for x in xs:
                for y in ys:
                    ax.plot([x, x], [y, y], [lo[2], hi[2]], color="tab:orange", linewidth=lw, alpha=alp)
            for x in xs:
                for z in zs:
                    ax.plot([x, x], [lo[1], hi[1]], [z, z], color="tab:orange", linewidth=lw, alpha=alp)
            for y in ys:
                for z in zs:
                    ax.plot([lo[0], hi[0]], [y, y], [z, z], color="tab:orange", linewidth=lw, alpha=alp)

        cmin, cmax = ctrs.min(axis=0), ctrs.max(axis=0)
        print(f"[viz] G=({Gx},{Gy},{Gz}), centers x:[{cmin[0]:.3f},{cmax[0]:.3f}] "
              f"y:[{cmin[1]:.3f},{cmax[1]:.3f}] z:[{cmin[2]:.3f},{cmax[2]:.3f}]")

    _set_axes(ax, aabb_min, aabb_max)

    try:
        R = tuple(np.array(res_list[0]).astype(int)) if len(res_list) else (0,0,0)
        ax.set_title(f"VM res (global): {R} | inner-grid: {inner_grid_for}", fontsize=9)
    except Exception:
        pass
    
    # fig.2: multi-view snap shot
    if views2:
        for vi, (elev, azim) in enumerate(views2, 1):
            ax.view_init(elev=elev, azim=azim)
            if save_prefix:
                _save_and_close(fig, f"{save_prefix}{f'_{save_tag}' if save_tag else ''}_grid_patches_v{vi}.png", dpi=200)
            else:
                _save_and_close(fig, None)
          
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection="3d")
            if show_global_grid:
                _draw_wire(ax, aabb_min, aabb_max, Gx, Gy, Gz)

    # fig.2: main viewpoint
    tag = (f"_{save_tag}" if save_tag else "")
    if save_prefix:
        _save_and_close(fig, f"{save_prefix}{tag}_grid_patches.png", dpi=200)
    else:
        _save_and_close(fig, None)

def export_patch_viz_bundle(patch_map, tensorf, save_prefix):
    if not isinstance(patch_map, dict) or len(patch_map) == 0:
        return

    patch_keys = list(patch_map.keys())
    if len(patch_keys) == 0:
        return

    # 1) overview-only
    plot_patch_scatter_and_grid(
        patch_map, tensorf,
        save_prefix=save_prefix, save_tag="oview",
        draw_centers=False, inner_grid_for="none",
        draw_true_boxes=False, fig1_overlay="none",
        views2=[(20,30)]
    )
    # 2) clean (thesis/representation)
    plot_patch_scatter_and_grid(
        patch_map, tensorf,
        save_prefix=save_prefix, save_tag="clean",
        draw_centers=False, inner_grid_for="none",
        draw_true_boxes=True, fig1_overlay="true",
        color_by="depth",
        views2=[(20,30),(20,120),(30,-60)]
    )
    # 3) debug (colored centers + refined wireframes + slices + multi-views)
    plot_patch_scatter_and_grid(
        patch_map, tensorf,
        save_prefix=save_prefix, save_tag="debug",
        draw_centers=True, color_by="depth",
        inner_grid_for="refined", draw_true_boxes=True, fig1_overlay="centers",
        slice_axis='z', slice_lo=0.35, slice_hi=0.65,
        views2=[(15,40),(25,120)]
    )
