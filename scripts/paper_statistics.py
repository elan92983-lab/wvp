"""
论文统计分析脚本
生成 experiments.tex 所需的数据
"""

import torch
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.spectral_transformer import SpectralTemporalTransformer
from src.data_utils.dataset import SpectralDataset


def compute_metrics(real, pred):
    diff = pred - real
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    if np.std(real) > 1e-8 and np.std(pred) > 1e-8:
        corr = float(np.corrcoef(real, pred)[0, 1])
    else:
        corr = float("nan")
    return {"mae": mae, "rmse": rmse, "corr": corr}


def classify_sample(real_slice, threshold=0.1):
    tail_start = len(real_slice) // 2
    tail_var = np.var(real_slice[tail_start:])
    return "converging" if tail_var <= threshold else "oscillating"


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(root_dir, "data/processed/spectral_data_v2.npz")
    model_path = os.path.join(root_dir, "models/spectral_transformer_finetuned.pth")

    if not os.path.exists(model_path):
        print(f"❌ 找不到模型: {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SpectralDataset(data_path, max_nodes=20, max_seq_len=40)

    model = SpectralTemporalTransformer(
        max_nodes=20, d_model=128, nhead=4, num_layers=4, max_seq_len=40
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    # 评估所有样本
    converging_metrics = []
    oscillating_metrics = []
    all_metrics = []

    print("正在评估所有样本...")

    with torch.no_grad():
        for idx in range(len(dataset)):
            data = dataset[idx]
            evals = data["evals"].unsqueeze(0).to(device)
            evecs = data["evecs"].unsqueeze(0).to(device)
            time_idx = data["time_indices"].unsqueeze(0).to(device)
            num_nodes = data["num_nodes"].unsqueeze(0).to(device)

            real_beta = data["betas"].numpy()
            mask = data["mask"].numpy()

            real_beta = real_beta * dataset.beta_std + dataset.beta_mean
            pred_beta = model.generate(evals, evecs, time_idx, num_nodes=num_nodes).cpu().numpy()[0]
            pred_beta = pred_beta * dataset.beta_std + dataset.beta_mean

            valid_len = int(mask.sum())
            real_slice = real_beta[:valid_len]
            pred_slice = pred_beta[:valid_len]

            metrics = compute_metrics(real_slice, pred_slice)
            sample_type = classify_sample(real_slice)

            all_metrics.append(metrics)
            if sample_type == "converging":
                converging_metrics.append(metrics)
            else:
                oscillating_metrics.append(metrics)

            if (idx + 1) % 100 == 0:
                print(f"  已完成 {idx + 1}/{len(dataset)}")

    # 输出统计结果
    print("\n" + "=" * 70)
    print("📊 论文统计数据")
    print("=" * 70)

    def print_stats(name, metrics_list):
        if not metrics_list:
            return
        maes = [m["mae"] for m in metrics_list]
        rmses = [m["rmse"] for m in metrics_list]
        corrs = [m["corr"] for m in metrics_list if not np.isnan(m["corr"])]

        print(f"\n【{name}】(n={len(metrics_list)}, 占比={100*len(metrics_list)/len(all_metrics):.1f}%)")
        print(f"  MAE:  均值={np.mean(maes):.3f}, 标准差={np.std(maes):.3f}")
        print(f"  RMSE: 均值={np.mean(rmses):.3f}, 标准差={np.std(rmses):.3f}")
        if corrs:
            print(f"  Corr: 均值={np.mean(corrs):.3f}, 标准差={np.std(corrs):.3f}")
            print(f"         最佳={np.max(corrs):.3f}, 最差={np.min(corrs):.3f}")
        else:
            print("  Corr: 样本相关系数不可计算")

    print_stats("总体", all_metrics)
    print_stats("收敛型样本", converging_metrics)
    print_stats("振荡型样本", oscillating_metrics)

    print("\n" + "=" * 70)
    print("✅ 统计完成！请将以上数据填入 experiments.tex")
    print("=" * 70)

if __name__ == "__main__":
    main()
