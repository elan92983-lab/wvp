def temporal_gradient_loss(pred, target, mask):
    """鼓励模型学习 β 的变化趋势"""
    pred_diff = pred[:, 1:] - pred[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]
    mask_diff = mask[:, 1:] * mask[:, :-1]
    return ((pred_diff - target_diff) ** 2 * mask_diff).sum() / (mask_diff.sum() + 1e-6)

def make_time_weights(seq_len, weight_tail, device):
    if weight_tail <= 1.0:
        return torch.ones(seq_len, device=device)
    return torch.linspace(1.0, weight_tail, steps=seq_len, device=device)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import argparse
from tqdm import tqdm

# 引入新模块
from src.data_utils.dataset import SpectralDataset
from src.models.spectral_transformer import SpectralTemporalTransformer

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, epochs, ss_start, ss_end, weight_tail):
    model.train()
    total_loss = 0
    
    for batch in loader:
        # 数据上云
        evals = batch['evals'].to(device)
        evecs = batch['evecs'].to(device)
        time_idx = batch['time_indices'].to(device)
        targets = batch['betas'].to(device)
        mask = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        # 前向推理
        num_nodes = batch['num_nodes'].to(device)
        ss_prob = ss_start + (ss_end - ss_start) * (epoch / max(epochs - 1, 1))
        ss_prob = float(max(0.0, min(1.0, ss_prob)))

        prev_tf = torch.zeros_like(targets)
        prev_tf[:, 1:] = targets[:, :-1]

        with torch.no_grad():
            pred_tf = model(evals, evecs, time_idx, num_nodes=num_nodes, prev_betas=prev_tf)

        ss_mask = torch.rand_like(targets[:, 1:]) < ss_prob
        prev_betas = prev_tf.clone()
        prev_betas[:, 1:] = torch.where(ss_mask, pred_tf[:, :-1], targets[:, :-1])
        preds = model(evals, evecs, time_idx, num_nodes=num_nodes, prev_betas=prev_betas)
        
        # 计算 Loss (只计算 mask 为 1 的部分)
        time_w = make_time_weights(preds.shape[1], weight_tail, preds.device).unsqueeze(0)
        weighted_mask = mask * time_w
        loss_main = criterion(preds * weighted_mask, targets * weighted_mask)
        loss_main = loss_main / (weighted_mask.sum() + 1e-6) * weighted_mask.numel()
        # temporal gradient loss
        loss_temp = temporal_gradient_loss(preds, targets, mask)
        loss = loss_main + 0.5 * loss_temp
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)  # 增加默认训练轮数
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_path", type=str, default="data/processed/spectral_data_v2.npz")
    parser.add_argument("--weight_tail", type=float, default=2.0)
    parser.add_argument("--ss_start", type=float, default=0.0)
    parser.add_argument("--ss_end", type=float, default=0.5)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 加载数据
    if not os.path.exists(args.data_path):
        print(f"错误: 找不到数据 {args.data_path}，请先运行 generate_dataset_v2.py")
        return

    dataset = SpectralDataset(args.data_path, max_nodes=20, max_seq_len=40)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    
    # 2. 初始化模型
    model = SpectralTemporalTransformer(max_nodes=20, d_model=128, nhead=4, num_layers=4, max_seq_len=40).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)  # 降低学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.HuberLoss(reduction='sum', delta=1.0)
    
    # 3. 训练循环
    print("开始训练 Spectral-Temporal Transformer...")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            args.epochs,
            args.ss_start,
            args.ss_end,
            args.weight_tail
        )
        scheduler.step()
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f} | LR = {scheduler.get_last_lr()[0]:.6e}")
        # 保存中间结果
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"models/spectral_transformer_ep{epoch+1}.pth")

if __name__ == "__main__":
    main()
