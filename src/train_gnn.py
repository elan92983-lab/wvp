import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# 确保从任意路径运行时也能找到 src 包
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset import FALQONDataset
from src.models.gnn import FALQONGNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        adj = batch["adj"].to(device)        # [B, N, N]
        mask = batch["mask"].to(device)      # [B, N]
        targets = batch["betas"].to(device)  # [B, 30]

        optimizer.zero_grad(set_to_none=True)
        outputs = model(adj, mask)           # [B, 30]
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item())

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        adj = batch["adj"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["betas"].to(device)
        outputs = model(adj, mask)
        loss = criterion(outputs, targets)
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser(description="Train FALQON GNN to predict beta sequence")
    default_data_path = os.path.join(PROJECT_ROOT, "data/raw/dataset_v1/train_data_final.npz")

    parser.add_argument("--data_path", type=str, default=default_data_path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--max_nodes", type=int, default=12)
    parser.add_argument("--output_len", type=int, default=30)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--save_dir", type=str, default="models/checkpoints")
    parser.add_argument("--save_name", type=str, default="gnn_model.pth")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 使用设备: {device}")

    if not os.path.exists(args.data_path):
        print(f"Error: data_path not found: {args.data_path}")
        return

    dataset = FALQONDataset(args.data_path, max_nodes=args.max_nodes, max_layers=args.output_len)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = FALQONGNN(
        max_nodes=args.max_nodes,
        output_len=args.output_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.save_name)

    best_val = float("inf")
    start_time = time.time()

    print(f"📊 数据集就绪: 训练集 {len(train_ds)} | 验证集 {len(val_ds)}")
    print(f"💾 最佳模型将保存到: {save_path}")

    for epoch in range(args.epochs):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {tr:.6f} | Val Loss: {va:.6f}")

        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), save_path)
            print(f"    🌟 保存最佳 GNN 权重 (Val Loss: {best_val:.6f})")

    mins = (time.time() - start_time) / 60.0
    print(f"\n✅ 训练结束! 总耗时: {mins:.2f} 分钟")
    print(f"🏆 最佳验证集 Loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
