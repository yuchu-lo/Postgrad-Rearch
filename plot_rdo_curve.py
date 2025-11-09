import json
import os
import matplotlib.pyplot as plt

def load_all_logs(log_dir):
    logs = []
    for file in os.listdir(log_dir):
        if file.endswith(".json"):
            with open(os.path.join(log_dir, file), "r") as f:
                log = json.load(f)
                if 'lambda' in log and 'tau' in log:
                    logs.append(log)
    # Sort by lambda first, then by tau in ascending order if lambda values are equal
    return sorted(logs, key=lambda x: (x['lambda'], x['tau']))

def plot_curves(logs, log_dir):
    labels = [log.get('label', f"λ={log['lambda']}") for log in logs]
    lambdas = [log['lambda'] for log in logs]
    taus = [log['tau'] for log in logs]
    psnrs = [log['final_psnr'] for log in logs]
    voxels = [log['voxel_count'] for log in logs]
    mems = [log['memory_MB'] for log in logs]

    fig_dir = os.path.join(log_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    # PSNR vs voxel count
    plt.figure()
    for label, psnr, voxel in zip(labels, psnrs, voxels):
        plt.scatter(voxel, psnr, label=label)
    plt.plot(voxels, psnrs, linestyle='--')
    plt.xlabel("Total Voxels")
    plt.ylabel("PSNR")
    plt.title("PSNR vs. Voxel Count")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "psnr_vs_voxels.png"))

    # PSNR vs memory usage
    plt.figure()
    for label, psnr, mem in zip(labels, psnrs, mems):
        plt.scatter(mem, psnr, label=label)
    plt.plot(mems, psnrs, linestyle='--')
    plt.xlabel("Total Memory (MB)")
    plt.ylabel("PSNR")
    plt.title("PSNR vs. Memory Usage")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "psnr_vs_memory.png"))

if __name__ == "__main__":
    log_dir = "logs_critrn"
    logs = load_all_logs(log_dir)
    plot_curves(logs, log_dir)
