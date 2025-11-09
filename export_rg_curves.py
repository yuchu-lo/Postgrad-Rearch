import sys, pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_metrics(run_dir: Path):
    csv = run_dir/"metrics.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)

    # ---- 欄位歸一化：把 rd/* 與 stats/* 統一成本腳本通用欄位 ----
    # PSNR 類
    rename_pairs = [
        ("rd/psnr_full","psnr_full"),
        ("rd/psnr_fast","psnr_fast"),
        ("rd/psnr_train","train_psnr"),
        ("test/psnr_fast","psnr_fast"),
        ("test/psnr","test_psnr"),
        ("train/PSNR","train_psnr"),
    ]
    for old,new in rename_pairs:
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # 資源類
    if "rd/memory_MB" in df.columns and "total_memory_MB" not in df.columns:
        df["total_memory_MB"] = df["rd/memory_MB"]
    if "stats/total_memory_MB" in df.columns and "total_memory_MB" not in df.columns:
        df["total_memory_MB"] = df["stats/total_memory_MB"]

    if "rd/voxels" in df.columns and "total_voxels" not in df.columns:
        df["total_voxels"] = df["rd/voxels"]
    if "stats/total_voxels" in df.columns and "total_voxels" not in df.columns:
        df["total_voxels"] = df["stats/total_voxels"]

    # 決定要用哪個 PSNR 欄位（依優先順序）
    ps = None
    for k in ["psnr_full","test_psnr","psnr_fast","train_psnr","psnr","PSNR"]:
        if k in df.columns and df[k].notna().any():
            ps = k; break
    if ps is None: 
        return None

    # 基礎欄位檢查
    if "iter" not in df.columns:
        # 有些匯出會用 step/global_step
        for alt in ["step","global_step","iteration"]:
            if alt in df.columns:
                df = df.rename(columns={alt:"iter"})
                break
    need = ["iter", ps, "total_memory_MB", "total_voxels"]
    for k in need:
        if k not in df.columns:
            return None

    # 先按 iter 排序，對 MB/Vox 做 forward-fill，避免掉點
    df = df.sort_values("iter").reset_index(drop=True)
    df["total_memory_MB"] = pd.to_numeric(df["total_memory_MB"], errors="coerce").ffill()
    df["total_voxels"]    = pd.to_numeric(df["total_voxels"], errors="coerce").ffill()

    # 只取 PSNR 有值的行（其餘可為 NaN，被 ffill 補起來）
    df = df.dropna(subset=[ps])
    return df.rename(columns={ps:"psnr"})

def main():
    runs = [Path(p) for p in sys.argv[1:]] or sorted([p for p in Path("./log").iterdir() if p.is_dir() and (p/"metrics.csv").exists()])
    curves = []
    for rd in runs:
        df = load_metrics(rd)
        if df is None: 
            print(f"[skip] {rd}")
            continue
        df = df.dropna(subset=["psnr","total_memory_MB","total_voxels"])
        df["run"] = rd.name
        curves.append(df[["iter","psnr","total_memory_MB","total_voxels","run"]])
    if not curves:
        print("No metrics found"); return
    allc = pd.concat(curves, ignore_index=True)

    # 兩張曲線：PSNR-per-MB、PSNR-per-Voxel
    for xcol, out in [("total_memory_MB","rg_psnr_perMB.png"), ("total_voxels","rg_psnr_perVoxel.png")]:
        plt.figure(figsize=(8,5))
        for run, g in allc.groupby("run"):
            g = g.sort_values(xcol)
            plt.plot(g[xcol], g["psnr"], label=run)
        plt.xlabel(xcol); plt.ylabel("PSNR"); plt.legend()
        plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()
        print("wrote", out)

    # 統整表（每 run 末期點）
    rows = []
    for run, g in allc.groupby("run"):
        g = g.sort_values("iter")
        rows.append({
            "run": run,
            "final_iter": int(g["iter"].iloc[-1]),
            "final_psnr": float(g["psnr"].iloc[-1]),
            "final_MB": float(g["total_memory_MB"].iloc[-1]),
            "final_voxels": float(g["total_voxels"].iloc[-1]),
            "best_psnr": float(g["psnr"].max()),
            "min_MB_for_27dB": float(g[g["psnr"]>=27.0]["total_memory_MB"].min()) if (g["psnr"]>=27.0).any() else None,
        })
    summ = pd.DataFrame(rows).sort_values("run")
    summ.to_csv("rg_summary.csv", index=False)
    print("wrote rg_summary.csv")

if __name__ == "__main__":
    main()
