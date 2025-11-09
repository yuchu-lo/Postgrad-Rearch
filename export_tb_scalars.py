import sys, os, re, glob, math, time
import pandas as pd
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator as ea

TAGS_WANT = [
    "train/PSNR", "test/psnr_fast", "test/psnr",
    "train/loss_kd", "train/loss_seam",
    "stats/total_voxels", "stats/total_memory_MB", "stats/total_patches",
    "rd/psnr_train", "rd/psnr_fast", "rd/psnr_full", "rd/voxels", "rd/memory_MB",
]

def list_event_files(run_dir: Path):
    return sorted(run_dir.rglob("events.out.tfevents.*"))

def load_scalars_from_file(evt_path: Path):
    acc = ea.EventAccumulator(str(evt_path), size_guidance={ea.SCALARS: 200000})
    acc.Reload()
    tags = set(acc.Tags().get('scalars', []))
    data = {}
    for tag in TAGS_WANT:
        if tag in tags:
            scal = acc.Scalars(tag)
            if scal:
                df = pd.DataFrame([{"step": s.step, "value": s.value} for s in scal])
                data[tag] = df
    return data

def merge_by_step_keep_last(dfs):
    if not dfs: 
        return pd.DataFrame(columns=["step","value"])
    out = pd.concat(dfs, ignore_index=True)
    # 同一 step 可能出現多次，保留最後一次
    out = out.sort_values("step").drop_duplicates(subset=["step"], keep="last")
    return out

def load_all_scalars(run_dir: Path):
    files = list_event_files(run_dir)
    if not files:
        return {}
    # 從舊到新讀，讓「較新的檔案」覆蓋舊檔的同 step 值
    tag_to_dfs = {}
    for fp in files:
        rec = load_scalars_from_file(fp)
        for tag, df in rec.items():
            tag_to_dfs.setdefault(tag, []).append(df)
    merged = {tag: merge_by_step_keep_last(dfs) for tag, dfs in tag_to_dfs.items()}
    return merged

def pivot_metrics(data: dict):
    if not data:
        return pd.DataFrame()
    steps = sorted(set().union(*[set(d["step"]) for d in data.values() if not d.empty]))
    out = pd.DataFrame({"iter": steps})
    for tag, df in data.items():
        if df.empty: 
            continue
        out = out.merge(df.rename(columns={"value": tag})[["step", tag]],
                        how="left", left_on="iter", right_on="step").drop(columns=["step"])
    return out

def parse_splits(run_dir: Path):
    iters = []
    for p in [run_dir/"events_log.txt", run_dir/"logs.txt"]:
        if p.exists():
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                if re.search(r"(strict[-_ ]even|even[_-]selective|selective[_-]even|split)", line, re.I):
                    m = re.search(r"iter\D*?(\d+)", line, re.I)
                    if m: iters.append(int(m.group(1)))
    return sorted(set(iters))

def compute_indicators(df: pd.DataFrame, split_iters):
    d = {"first_split": None, "final_psnr": None, "final_test_psnr": None,
         "dip_depth": None, "recovery_steps": None}
    if df.empty: return d
    for col in ["test/psnr", "test/psnr_fast", "train/PSNR"]:
        if col in df and df[col].notna().any():
            d["final_psnr"] = float(df[col].dropna().iloc[-1]); break
    if "test/psnr" in df and df["test/psnr"].notna().any():
        d["final_test_psnr"] = float(df["test/psnr"].dropna().iloc[-1])
    key = "test/psnr_fast" if ("test/psnr_fast" in df and df["test/psnr_fast"].notna().any()) else (
          "train/PSNR" if ("train/PSNR" in df and df["train/PSNR"].notna().any()) else None)
    if key and split_iters:
        s = min(split_iters); d["first_split"] = s
        pre = df[(df["iter"]>=s-200)&(df["iter"]<s)]
        if not pre.empty and pre[key].notna().any():
            baseline = pre[key].median()
            post = df[(df["iter"]>=s)&(df["iter"]<=s+600)]
            if not post.empty and post[key].notna().any():
                min_post = post[key].min()
                d["dip_depth"] = float(baseline - min_post)
                rec = post[post[key] >= baseline - 0.05]
                d["recovery_steps"] = int(rec["iter"].iloc[0] - s) if not rec.empty else None
    return d

def main(run_dirs):
    summaries = []
    for rd in run_dirs:
        run_dir = Path(rd)
        data = load_all_scalars(run_dir)
        if not data:
            print(f"[skip] no event files in {run_dir}")
            continue
        df = pivot_metrics(data)
        if not df.empty:
            (run_dir/"metrics.csv").write_text(df.to_csv(index=False), encoding="utf-8")
        splits = parse_splits(run_dir)
        ind = compute_indicators(df, splits)
        ind["run"] = run_dir.name
        ind["metrics_csv"] = str(run_dir/"metrics.csv")
        summaries.append(ind)
        print(f"[ok] {run_dir.name}: splits={splits}, indicators={ind}")
    if summaries:
        pd.DataFrame(summaries).to_csv("summary.csv", index=False)
        print("=> summary.csv written")
    else:
        print("No runs processed.")

if __name__ == "__main__":
    args = sys.argv[1:] or sorted([str(p) for p in Path("./log").iterdir() if p.is_dir()])
    main(args)
