import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

def load_rd_logs(log_dir):
    """
    Load all rd_curve_log.json files from subdirectories of log_dir.
    """
    all_logs = []
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file == "rd_curve_log.json":
                with open(os.path.join(root, file), "r") as f:
                    data = json.load(f)
                    for entry in data:
                        entry["source"] = os.path.basename(root)
                        all_logs.append(entry)
    return all_logs

def prepare_heatmap_dataframe(logs, metric="PSNR"):
    """
    Create pandas DataFrame from list of logs for heatmap plotting.
    Rows: λ (lambda), Columns: τ (log margin)
    """
    table = defaultdict(dict)
    for log in logs:
        lam = log.get("lambda")
        tau = log.get("tau")
        val = log.get(metric)
        if lam is not None and tau is not None and val is not None:
            lam = float(lam)
            tau = round(float(tau), 5)
            table[lam][tau] = val

    df = pd.DataFrame.from_dict(table, orient="index")
    df.index.name = "λ"
    df.columns.name = "log(τ)"
    return df.sort_index().sort_index(axis=1)

def plot_heatmap(df, metric="PSNR", save_path="logs_critrn/figs/heatmap_psnr.png"):
    if df.empty:
        print("[Error] No data found to plot heatmap.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
    plt.title(f"{metric} Heatmap (λ vs. log(τ))")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Heatmap saved to {save_path}")

if __name__ == "__main__":
    log_root = "logs_critrn"  # path to the folder containing subfolders of experiments
    logs = load_rd_logs(log_root)
    df = prepare_heatmap_dataframe(logs, metric="PSNR")
    plot_heatmap(df, metric="PSNR", save_path=os.path.join(log_root, "figs/heatmap_psnr.png"))

