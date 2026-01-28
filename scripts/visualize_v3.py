"""
增强版可视化脚本 for V2 模型
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.spectral_transformer_v2 import SpectralTemporalTransformerV2
from src.data_utils.dataset import SpectralDataset


def _dtw_distance(a, b):
    n, m = len(a), len(b)
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return dp[n, m] / (n + m)


def compute_metrics(real, pred):
    diff = pred - real
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    if np.std(real) > 1e-8 and np.std(pred) > 1e-8:
        corr = float(np.corrcoef(real, pred)[0, 1])
    else:
        corr = float('nan')
    dtw = float(_dtw_distance(real, pred))
    return {"mae": mae, "rmse": rmse, "corr": corr, "dtw": dtw}


def classify_sample(real_slice, threshold=0.1):
    tail_start = len(real_slice) // 2
    tail_var = np.var(real_slice[tail_start:])
    return "oscillating" if tail_var > threshold else "converging"


def visualize():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(root_dir, "data/processed/spectral_data_v2.npz")
    model_path = os.path.join(root_dir, "models/spectral_transformer_v2_best.pth")

    if not os.path.exists(model_path):
        print(f"❌ 找不到模型: {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SpectralDataset(data_path, max_nodes=20, max_seq_len=40)

    num_samples = 20
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    model = SpectralTemporalTransformerV2(max_nodes=20, d_model=256, nhead=8, num_layers=6, max_seq_len=40).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    converging_metrics = []
    oscillating_metrics = []
    all_metrics = []

    output_dir = os.path.join(root_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    saved_images = []

    with torch.no_grad():
        for i, idx in enumerate(indices):
            data = dataset[idx]
            evals = data['evals'].unsqueeze(0).to(device)
            evecs = data['evecs'].unsqueeze(0).to(device)
            time_idx = data['time_indices'].unsqueeze(0).to(device)
            num_nodes = data['num_nodes'].unsqueeze(0).to(device)

            real_beta = data['betas'].numpy()
            mask = data['mask'].numpy()
            real_beta = real_beta * dataset.beta_std + dataset.beta_mean

            pred_beta = model.generate(evals, evecs, time_idx, num_nodes=num_nodes).cpu().numpy()[0]
            pred_beta = pred_beta * dataset.beta_std + dataset.beta_mean

            valid_len = int(mask.sum())
            real_slice = real_beta[:valid_len]
            pred_slice = pred_beta[:valid_len]

            metrics = compute_metrics(real_slice, pred_slice)
            sample_type = classify_sample(real_slice)

            if sample_type == "converging":
                converging_metrics.append(metrics)
            else:
                oscillating_metrics.append(metrics)
            all_metrics.append((idx, metrics, sample_type))

            if i < 4:
                fig, ax = plt.subplots(1, 1, figsize=(8, 5))
                ax.plot(real_slice, 'k-', label='Ground Truth (FALQON)', linewidth=1.5, alpha=0.7)
                ax.plot(pred_slice, 'r--', label='Prediction (V2)', linewidth=2.5)
                ax.set_title(f"Sample {idx} (N={data['num_nodes'].item()}, {sample_type})")
                ax.set_xlabel("Layer (t)")
                ax.set_ylabel("Beta (Radians)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.text(0.02, 0.02,
                        f"MAE={metrics['mae']:.3f}\nRMSE={metrics['rmse']:.3f}\nCorr={metrics['corr']:.3f}\nDTW={metrics['dtw']:.3f}",
                        transform=ax.transAxes, fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                fig.tight_layout()
                img_path = os.path.join(output_dir, f"vis_v2_{idx}.png")
                fig.savefig(img_path, dpi=150)
                saved_images.append(img_path)
                plt.close(fig)

    print("\n" + "="*60)
    if converging_metrics:
        avg_conv = {"mae": np.mean([m["mae"] for m in converging_metrics]), "corr": np.nanmean([m["corr"] for m in converging_metrics])}
        print(f"收敛型样本 ({len(converging_metrics)}个): MAE={avg_conv['mae']:.3f}, Corr={avg_conv['corr']:.3f}")
    if oscillating_metrics:
        avg_osc = {"mae": np.mean([m["mae"] for m in oscillating_metrics]), "corr": np.nanmean([m["corr"] for m in oscillating_metrics])}
        print(f"振荡型样本 ({len(oscillating_metrics)}个): MAE={avg_osc['mae']:.3f}, Corr={avg_osc['corr']:.3f}")
    avg_all = {"mae": np.mean([m["mae"] for _, m, _ in all_metrics]), "corr": np.nanmean([m["corr"] for _, m, _ in all_metrics])}
    print(f"\n总体平均: MAE={avg_all['mae']:.3f}, Corr={avg_all['corr']:.3f}")
    print(f"\n图片输出目录: {output_dir}")
    if saved_images:
        print("已保存图片:")
        for p in saved_images:
            print(f"- {p}")
    print("="*60)


if __name__ == "__main__":
    visualize()
