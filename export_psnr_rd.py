#!/usr/bin/env python3
import argparse, sys, re, glob, os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PSNR_COL_CAND = [
    "rd/psnr_full", "rd/psnr_fast", "rd/psnr_train",   
    "test/psnr", "test/psnr_fast", "eval/psnr", "train/PSNR", "train/psnr", "psnr", "val/psnr",
]
MEM_COL_CAND  = [
    "rd/memory_MB",                                    
    "stats/total_memory_MB", "total_memory_MB", "model_memory_MB", "memory_MB",
]
VOX_COL_CAND  = [
    "rd/voxels",                                      
    "stats/total_voxels", "total_voxels", "voxels", "model_voxels",
]
ITER_COL_CAND = ["iter", "step", "global_step", "iteration"]

def find_first_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    # fallback: case-insensitive
    lc = {c.lower(): c for c in df.columns}
    for k in cands:
        if k.lower() in lc:
            return lc[k.lower()]
    return None

def load_metrics_from_path(path: Path, psnr_col_arg=None):
    files = []
    if path.is_file() and path.suffix.lower() in [".csv"]:
        files = [path]
    elif path.is_dir():
        for pat in ["**/metrics*.csv", "**/*metrics*.csv", "**/*.csv"]:
            files.extend(path.glob(pat))
    best = None; best_rows = -1
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.shape[0] > best_rows and df.shape[1] >= 2:
            best = f; best_rows = df.shape[0]
    if best is None:
        raise FileNotFoundError(f"No CSV metrics found under {path}")
    df = pd.read_csv(best)
    iter_col = find_first_col(df, ITER_COL_CAND) or df.columns[0]
    psnr_col = psnr_col_arg or find_first_col(df, PSNR_COL_CAND)
    mem_col  = find_first_col(df, MEM_COL_CAND)
    vox_col  = find_first_col(df, VOX_COL_CAND)
    for c in [iter_col, psnr_col, mem_col, vox_col]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if psnr_col is None:
        raise ValueError(f"Could not find PSNR column in {best}; cols: {list(df.columns)[:15]}...")
    df = df[[c for c in [iter_col, psnr_col, mem_col, vox_col] if c]].copy()
    df = df.dropna(subset=[psnr_col]).reset_index(drop=True)
    meta = {"file": str(best), "iter_col": iter_col, "psnr_col": psnr_col, "mem_col": mem_col, "vox_col": vox_col}
    return df, meta

def jitter_series(x, scale=1e-6):
    if x is None: return None
    x = np.asarray(x, dtype=float)
    if len(x) == 0: return x
    if np.unique(x).size <= max(1, len(x)//10):
        noise = (np.random.rand(len(x)) - 0.5) * scale * (1.0 + (np.nanmax(x) - np.nanmin(x) + 1.0))
        return x + noise
    return x

def plot_psnr_vs_iter(runs, out):
    plt.figure()
    for name, (df, meta) in runs.items():
        i, p = meta["iter_col"], meta["psnr_col"]
        if i not in df.columns: continue
        plt.plot(df[i], df[p], label=name)
    plt.xlabel("Iteration"); plt.ylabel("PSNR (dB)"); plt.title("PSNR vs Iteration")
    if len(runs) > 1: plt.legend()
    plt.grid(True, alpha=0.2); plt.savefig(out, bbox_inches="tight", dpi=150); plt.close()

def plot_scatter(runs, xkey, title, out):
    plt.figure()
    has_any = False
    for name, (df, meta) in runs.items():
        p = meta["psnr_col"]; xcol = meta[xkey]
        if xcol and xcol in df.columns and df[xcol].notna().any():
            x = jitter_series(df[xcol].values, scale=1e-3 if xkey=="mem_col" else 1e-6)
            y = df[p].values
            plt.scatter(x, y, s=10, label=name)
            i = meta["iter_col"]
            if i in df.columns:
                ord_df = df.sort_values(i)
                plt.plot(ord_df[xcol].values, ord_df[p].values, linewidth=0.8)
            has_any = True
    if not has_any:
        plt.close(); return False
    plt.xlabel("Memory (MB)" if xkey=="mem_col" else "Voxels (count)")
    plt.ylabel("PSNR (dB)"); plt.title(title)
    if len(runs) > 1: plt.legend()
    plt.grid(True, alpha=0.2); plt.savefig(out, bbox_inches="tight", dpi=150); plt.close(); return True

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Export PSNR–Memory / PSNR–Voxels plots from metrics.csv")
    ap.add_argument("--run", action="append", required=True, help="Path to run folder or metrics CSV; repeatable")
    ap.add_argument("--name", action="append", help="Optional run labels (must match count)")
    ap.add_argument("--psnr-col", default=None, help="Override PSNR column (e.g., test/psnr)")
    ap.add_argument("--outdir", default=".", help="Output directory")
    ap.add_argument("--align-ffill", action="store_true", help="Forward-fill memory/voxels per run and sample only on rows where PSNR exists")
    args = ap.parse_args()

    if args.name and len(args.name) != len(args.run):
        print("--name count must match --run count", file=sys.stderr); sys.exit(2)

    runs = {}
    for idx, r in enumerate(args.run):
        path = Path(r); label = args.name[idx] if (args.name and idx < len(args.name)) else path.name
        try:
            if getattr(sys.modules.get("__main__"), "args", None) is not None:
                pass
            df, meta = load_metrics_from_path(path, psnr_col_arg=args.psnr_col)
        except Exception as e:
            print(f"[WARN] skip {r}: {e}", file=sys.stderr); continue
        runs[label] = (df, meta)
        if args.align_ffill:
            i, p, m, v = meta["iter_col"], meta["psnr_col"], meta["mem_col"], meta["vox_col"]
            df = df.sort_values(i).reset_index(drop=True)
            if m: df[m] = df[m].ffill()
            if v: df[v] = df[v].ffill()
            df = df[df[p].notna()]
            runs[label] = (df, meta)

    if not runs:
        print("No valid runs loaded.", file=sys.stderr); sys.exit(1)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    plot_psnr_vs_iter(runs, outdir / "psnr_vs_iter.png")
    ok_mem = plot_scatter(runs, "mem_col", "PSNR vs Memory (MB)", outdir / "psnr_vs_memory.png")
    ok_vox = plot_scatter(runs, "vox_col", "PSNR vs Voxels", outdir / "psnr_vs_voxels.png")

    merged = []
    for name,(df,meta) in runs.items():
        df2 = df.copy(); df2.insert(0, "run", name); merged.append(df2)
    pd.concat(merged, ignore_index=True).to_csv(outdir / "merged_metrics_for_rd.csv", index=False)

    print(outdir)

if __name__ == '__main__':
    main()
