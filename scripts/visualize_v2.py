import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 路径黑魔法：确保能找到 src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.spectral_transformer import SpectralTemporalTransformer
from src.data_utils.dataset import SpectralDataset

def visualize():
    # 1. 配置
    # 获取项目根目录 (即 scripts 目录的上一级)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    data_path = os.path.join(root_dir, "data/processed/spectral_data_v2.npz")
    model_dir = os.path.join(root_dir, "models")
    model_path = os.path.join(model_dir, "spectral_transformer_finetuned.pth") 
    
    # 如果找不到 ep100，尝试找一下目录里存在的模型
    if not os.path.exists(model_path):
        # 自动寻找最新的模型
        if not os.path.exists(model_dir):
            print(f"❌ 找不到模型目录: {model_dir}")
            return
        models = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
        if not models:
            print(f"❌ 找不到模型文件！请检查 {model_dir} 目录")
            return
        models.sort(key=lambda x: int(x.split('ep')[-1].split('.')[0]))
        model_path = os.path.join(model_dir, models[-1])
        print(f"⚠️ 指定模型不存在，自动加载最新模型: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 加载数据
    dataset = SpectralDataset(data_path, max_nodes=20, max_seq_len=40)
    # 随机取 4 个样本
    indices = np.random.choice(len(dataset), 4, replace=False)
    
    # 3. 加载模型
    model = SpectralTemporalTransformer(max_nodes=20, d_model=128, max_seq_len=40).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 4. 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            data = dataset[idx]
            evals = data['evals'].unsqueeze(0).to(device)
            evecs = data['evecs'].unsqueeze(0).to(device)
            time_idx = data['time_indices'].unsqueeze(0).to(device)
            real_beta = data['betas'].numpy()
            mask = data['mask'].numpy()
            # 预测
            pred_beta = model(evals, evecs, time_idx).cpu().numpy()[0]
            # === 反标准化 ===
            pred_beta = pred_beta * dataset.beta_std + dataset.beta_mean
            # 只画 mask=1 的部分（真实层数）
            valid_len = int(mask.sum())
            ax = axes[i]
            ax.plot(real_beta[:valid_len], 'k-', label='Ground Truth (FALQON)', linewidth=1.5)
            ax.plot(pred_beta[:valid_len], 'r--', label='Prediction (Zero-Shot)', linewidth=2)
            ax.set_title(f"Sample {idx}")
            ax.set_xlabel("Layer (t)")
            ax.set_ylabel("Beta")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
    plt.tight_layout()
    output_dir = os.path.join(root_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    save_file = os.path.join(output_dir, "vis_prediction.png")
    plt.savefig(save_file)
    print(f"✅ 可视化结果已保存至: {save_file}")

if __name__ == "__main__":
    visualize()
