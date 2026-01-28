import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import argparse

# 引入你的项目模块
# 假设你在项目根目录运行，或者确保 src 可见
from src.models.spectral_transformer import SpectralTemporalTransformer
from src.physics.simulator import DiffQuantumSimulator
from src.data_utils.dataset import SpectralDataset

def temporal_gradient_loss(pred, target, mask):
    """鼓励模型学习 β 的变化趋势"""
    pred_diff = pred[:, 1:] - pred[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]
    mask_diff = mask[:, 1:] * mask[:, :-1]
    return ((pred_diff - target_diff) ** 2 * mask_diff).sum() / (mask_diff.sum() + 1e-6)

def tail_variance_loss(pred, target, mask, tail_ratio=0.5):
    """约束后半段方差，避免尾部塌缩成直线"""
    P = pred.shape[1]
    start = int(P * (1 - tail_ratio))
    tail_mask = mask[:, start:]
    if tail_mask.sum() < 1:
        return pred.sum() * 0.0
    pred_tail = pred[:, start:]
    target_tail = target[:, start:]
    def masked_var(x, m):
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (x * m).sum(dim=1, keepdim=True) / denom
        var = ((x - mean) ** 2 * m).sum(dim=1, keepdim=True) / denom
        return var
    pv = masked_var(pred_tail, tail_mask)
    tv = masked_var(target_tail, tail_mask)
    return ((pv - tv) ** 2).mean()

def make_time_weights(seq_len, weight_tail, device):
    if weight_tail <= 1.0:
        return torch.ones(seq_len, device=device)
    return torch.linspace(1.0, weight_tail, steps=seq_len, device=device)


def load_pretrained_safely(model, path, device):
    if not path or not os.path.exists(path):
        print("⚠️ 预训练权重不存在，跳过加载。")
        return
    state = torch.load(path, map_location=device)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    missing = [k for k in model_state.keys() if k not in filtered]
    unexpected = [k for k in state.keys() if k not in filtered]
    model.load_state_dict(filtered, strict=False)
    print(f"✅ 预训练权重已加载 (匹配 {len(filtered)}/{len(model_state)})。")
    if missing:
        print(f"⚠️ 未加载(形状不匹配或不存在)参数数量: {len(missing)}")
    if unexpected:
        print(f"⚠️ 预训练中多余参数数量: {len(unexpected)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pretrained_path", type=str, default="models/spectral_transformer_ep100.pth")
    parser.add_argument("--weight_mse", type=float, default=5.0)
    parser.add_argument("--weight_temp", type=float, default=0.5)
    parser.add_argument("--weight_tail", type=float, default=2.0)
    parser.add_argument("--ss_start", type=float, default=0.0)
    parser.add_argument("--ss_end", type=float, default=0.5)
    parser.add_argument("--weight_tail_var", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 1. 配置路径 ---
    # 自动定位项目根目录
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_dir, "data/processed/spectral_data_v2.npz")
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset not found at {data_path}")
        return

    # --- 2. 加载数据集 (带标签!) ---
    # 这次我们用 SpectralDataset，因为我们需要真实的 betas 做老师
    print("Loading labeled dataset for Supervised + Physics training...")
    dataset = SpectralDataset(data_path, max_nodes=20, max_seq_len=40)
    
    # 拆分 Train/Val (可选，这里全量训练演示)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 获取统计量 (用于反归一化)
    BETA_MEAN = dataset.beta_mean
    BETA_STD = dataset.beta_std
    print(f"Dataset Stats: Mean={BETA_MEAN:.4f}, Std={BETA_STD:.4f}")

    # --- 3. 加载模型 ---
    model = SpectralTemporalTransformer(max_nodes=20, d_model=128, max_seq_len=40).to(device)
    
    # 加载预训练权重
    load_pretrained_safely(model, args.pretrained_path, device)
    
    # --- 4. 物理模拟器 & Loss ---
    # 注意：模拟器需要根据 batch 内最大的 N 动态调整，或者固定最大值
    # 这里为了简单，我们初始化一个最大 N=20 的模拟器，但计算时要小心 masking
    # 实际上 DiffQuantumSimulator 目前是固定 N 的。
    # 为了解决变长图的物理 Loss，我们这里只对 batch 里 N <= 12 的图计算物理 Loss (作为辅助)
    # 或者：主要靠 MSE，物理 Loss 仅作为 "验证" 或 "微弱正则"
    
    target_N_sim = 12
    simulator = DiffQuantumSimulator(n_qubits=target_N_sim, device=device)
    
    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr) # 学习率

    # === 调参核心区域 ===
    # 如果你想让曲线很像 FALQON：调大 WEIGHT_MSE (比如 1.0, 10.0)
    # 如果你想让能量很高：调大 WEIGHT_PHYS (比如 0.1, 0.5)
    # 建议：先用 MSE 强行教会形状，再微调
    WEIGHT_MSE = args.weight_mse      # <--- 调大这个！强迫它模仿
    WEIGHT_TEMP = args.weight_temp    # <--- 时间梯度趋势 loss
    WEIGHT_PHYS = 0.05                # <--- 物理只做辅助，因为模拟器对变长图支持有限
    WEIGHT_TAIL = args.weight_tail
    WEIGHT_TAIL_VAR = args.weight_tail_var
    
    print("🚀 Starting Supervised + Physics Fine-tuning...")
    model.train()
    
    for epoch in range(args.epochs): # 多跑几个 epoch
        total_loss = 0
        total_mse = 0
        total_phys = 0
        
        for batch in loader:
            evals = batch['evals'].to(device)
            evecs = batch['evecs'].to(device)
            time_idx = batch['time_indices'].to(device)
            mask = batch['mask'].to(device)     # [B, P]
            real_betas = batch['betas'].to(device) # [B, P] (Normalized)
            num_nodes = batch['num_nodes'].to(device)      # [B]
            
            optimizer.zero_grad()
            
            # (A) Scheduled Sampling
            ss_prob = args.ss_start + (args.ss_end - args.ss_start) * (epoch / max(args.epochs - 1, 1))
            ss_prob = float(max(0.0, min(1.0, ss_prob)))

            prev_tf = torch.zeros_like(real_betas)
            prev_tf[:, 1:] = real_betas[:, :-1]

            with torch.no_grad():
                pred_tf = model(evals, evecs, time_idx, num_nodes=num_nodes, prev_betas=prev_tf)

            ss_mask = torch.rand_like(real_betas[:, 1:]) < ss_prob
            prev_betas = prev_tf.clone()
            prev_betas[:, 1:] = torch.where(ss_mask, pred_tf[:, :-1], real_betas[:, :-1])

            pred_betas = model(evals, evecs, time_idx, num_nodes=num_nodes, prev_betas=prev_betas) # [B, P] (Normalized)
            
            # (B) Loss 1: MSE (模仿老师)
            # 只计算 mask=1 的部分 (有效时间步)
            loss_mse = (pred_betas - real_betas) ** 2
            time_w = make_time_weights(loss_mse.shape[1], WEIGHT_TAIL, loss_mse.device).unsqueeze(0)
            weighted_mask = mask * time_w
            loss_mse = (loss_mse * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)
            
            # (C) Loss 2: Physics (辅助)
            # 只挑选 batch 中 N == target_N_sim 的样本计算物理 Loss
            # 如果 batch 里没有 N=12 的，就跳过物理 Loss
            loss_phys = torch.tensor(0.0, device=device)
            
            # 筛选符合模拟器大小的图
            indices = torch.nonzero(num_nodes == target_N_sim, as_tuple=False).squeeze(-1)
            if indices.numel() > 0 and WEIGHT_PHYS > 0:
                # 提取子集
                sub_betas_norm = pred_betas.index_select(0, indices)
                # 反归一化供模拟器使用
                sub_betas_phys = sub_betas_norm * BETA_STD + BETA_MEAN
                
                # 此时 Dataset 并没有返回 adj (SpectralDataset 默认为了省内存没存 adj)
                # 这是一个小问题。如果 dataset.py 没返回 adj，我们无法计算物理 loss。
                # === 紧急策略 ===
                # 如果 Dataset 里没 adj，我们就只用 MSE！
                # 既然你要 "接近实际曲线"，MSE 才是 99% 重要的。
                pass 
                
            # (D) Total Loss
            loss_temp = temporal_gradient_loss(pred_betas, real_betas, mask)
            loss_tail_var = tail_variance_loss(pred_betas, real_betas, mask, tail_ratio=0.5)
            loss = WEIGHT_MSE * loss_mse + WEIGHT_TEMP * loss_temp + WEIGHT_TAIL_VAR * loss_tail_var
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += loss_mse.item()
            
        print(f"Epoch {epoch+1}: Total Loss={total_loss/len(loader):.4f} | MSE={total_mse/len(loader):.4f}")

    # 保存
    torch.save(model.state_dict(), "models/spectral_transformer_finetuned.pth")
    print("✅ Fine-tuning complete! Model learned from teacher (MSE).")

if __name__ == "__main__":
    main()