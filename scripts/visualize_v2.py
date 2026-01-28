import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import torch.nn as nn # 需要引入 nn

# 路径黑魔法：确保能找到 src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.spectral_transformer import SpectralTemporalTransformer
from src.data_utils.dataset import SpectralDataset

def _dtw_distance(a, b):
    """简单 DTW 距离（L1 代价），返回归一化距离。"""
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

def visualize():
    # 1. 配置
    # 获取项目根目录 (即 scripts 目录的上一级)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    data_path = os.path.join(root_dir, "data/processed/spectral_data_v2.npz")
    model_dir = os.path.join(root_dir, "models")
    model_path = os.path.join(model_dir, "spectral_transformer_finetuned.pth") 
    
    # 如果找不到 finetuned，尝试找一下目录里存在的模型
    if not os.path.exists(model_path):
        # 自动寻找最新的模型
        if not os.path.exists(model_dir):
            print(f"❌ 找不到模型目录: {model_dir}")
            return
        models = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
        if not models:
            print(f"❌ 找不到模型文件！请检查 {model_dir} 目录")
            return
        models.sort(key=lambda x: int(x.split('ep')[-1].split('.')[0]) if 'ep' in x else 0)
        model_path = os.path.join(model_dir, models[-1])
        print(f"⚠️ 指定模型不存在，自动加载最新模型: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 加载数据
    dataset = SpectralDataset(data_path, max_nodes=20, max_seq_len=40)
    # 随机取 4 个样本
    indices = np.random.choice(len(dataset), 4, replace=False)
    
    # 3. 加载模型
    model = SpectralTemporalTransformer(max_nodes=20, d_model=128, max_seq_len=40).to(device)

    # 现在的模型结构已经和训练时一致了，可以加载权重了
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    
    metrics_all = []
    with torch.no_grad():
        for i, idx in enumerate(indices):
            data = dataset[idx]
            evals = data['evals'].unsqueeze(0).to(device)
            evecs = data['evecs'].unsqueeze(0).to(device)
            time_idx = data['time_indices'].unsqueeze(0).to(device)
            num_nodes = data['num_nodes'].unsqueeze(0).to(device)
            
            real_beta = data['betas'].numpy()
            mask = data['mask'].numpy()
            
            # === Ground Truth 反标准化 ===
            real_beta = real_beta * dataset.beta_std + dataset.beta_mean
            
            # 预测
            pred_beta = model.generate(evals, evecs, time_idx, num_nodes=num_nodes).cpu().numpy()[0]
            
            # === 预测值 反标准化 ===
            # 模型输出的是 Normalized Space，必须还原回物理空间
            pred_beta = pred_beta * dataset.beta_std + dataset.beta_mean
            
            # 只画 mask=1 的部分（真实层数）
            valid_len = int(mask.sum())
            real_slice = real_beta[:valid_len]
            pred_slice = pred_beta[:valid_len]
            metrics = compute_metrics(real_slice, pred_slice)
            metrics_all.append((idx, metrics))

            # 单图绘制
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.plot(real_slice, 'k-', label='Ground Truth (FALQON)', linewidth=1.5, alpha=0.7)
            ax.plot(pred_slice, 'r--', label='Prediction (Phys-Tuned)', linewidth=2.5)

            ax.set_title(f"Sample {idx} (N={data['num_nodes'].item()})")
            ax.set_xlabel("Layer (t)")
            ax.set_ylabel("Beta (Radians)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.text(0.02, 0.02,
                    f"MAE={metrics['mae']:.3f}\nRMSE={metrics['rmse']:.3f}\nCorr={metrics['corr']:.3f}\nDTW={metrics['dtw']:.3f}",
                    transform=ax.transAxes,
                    fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

            fig.tight_layout()
            output_dir = os.path.join(root_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            save_file = os.path.join(output_dir, f"vis_prediction_{idx}.png")
            fig.savefig(save_file, dpi=1200)
            plt.close(fig)
            print(f"✅ 可视化结果已保存至: {save_file}")

    # 保存指标
    if metrics_all:
        avg = {
            "mae": np.mean([m["mae"] for _, m in metrics_all]),
            "rmse": np.mean([m["rmse"] for _, m in metrics_all]),
            "corr": np.nanmean([m["corr"] for _, m in metrics_all]),
            "dtw": np.mean([m["dtw"] for _, m in metrics_all])
        }
        metrics_path = os.path.join(output_dir, "vis_prediction_metrics.txt")
        with open(metrics_path, "w", encoding="utf-8") as f:
            for idx, m in metrics_all:
                f.write(
                    f"Sample {idx}: MAE={m['mae']:.6f}, RMSE={m['rmse']:.6f}, Corr={m['corr']:.6f}, DTW={m['dtw']:.6f}\n"
                )
            f.write(
                f"\nAverage: MAE={avg['mae']:.6f}, RMSE={avg['rmse']:.6f}, Corr={avg['corr']:.6f}, DTW={avg['dtw']:.6f}\n"
            )
        print(f"✅ 指标已保存至: {metrics_path}")

if __name__ == "__main__":
    visualize()