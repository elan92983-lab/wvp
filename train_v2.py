import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.data_utils.dataset import SpectralDataset
from src.models.spectral_transformer import SpectralTemporalTransformer


def temporal_gradient_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """鼓励模型学习 β 的变化趋势，mask 避免对 padding 计算梯度。"""
    pred_diff = pred[:, 1:] - pred[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]
    mask_diff = mask[:, 1:] * mask[:, :-1]
    return ((pred_diff - target_diff) ** 2 * mask_diff).sum() / (mask_diff.sum() + 1e-6)


def train_one_epoch(
    model: SpectralTemporalTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        evals = batch["evals"].to(device)
        evecs = batch["evecs"].to(device)
        time_idx = batch["time_indices"].to(device)
        targets = batch["betas"].to(device)
        mask = batch["mask"].to(device)
        num_nodes = batch["num_nodes"].to(device)

        optimizer.zero_grad()

        prev_betas = torch.zeros_like(targets)
        prev_betas[:, 1:] = targets[:, :-1]

        preds = model(evals, evecs, time_idx, num_nodes=num_nodes, prev_betas=prev_betas)

        loss_main = criterion(preds * mask, targets * mask)
        loss_main = loss_main / (mask.sum() + 1e-6) * mask.numel()

        loss_temp = temporal_gradient_loss(preds, targets, mask)
        loss = loss_main + 0.5 * loss_temp

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def evaluate(
    model: SpectralTemporalTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []

    with torch.no_grad():
        for batch in loader:
            evals = batch["evals"].to(device)
            evecs = batch["evecs"].to(device)
            time_idx = batch["time_indices"].to(device)
            targets = batch["betas"].to(device)
            mask = batch["mask"].to(device)
            num_nodes = batch["num_nodes"].to(device)

            prev_betas = torch.zeros_like(targets)
            prev_betas[:, 1:] = targets[:, :-1]

            preds = model(evals, evecs, time_idx, num_nodes=num_nodes, prev_betas=prev_betas)
            loss_main = criterion(preds * mask, targets * mask)
            loss_main = loss_main / (mask.sum() + 1e-6) * mask.numel()
            loss = loss_main + 0.5 * temporal_gradient_loss(preds, targets, mask)
            losses.append(loss.item())

    return sum(losses) / max(len(losses), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_path", type=str, default="data/processed/spectral_data_v2.npz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.data_path):
        print(f"错误: 找不到数据 {args.data_path}，请先运行 generate_dataset_v2.py")
        return

    dataset = SpectralDataset(args.data_path, max_nodes=20, max_seq_len=40)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = SpectralTemporalTransformer(max_nodes=20, d_model=128, nhead=4, num_layers=4, max_seq_len=40).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.HuberLoss(reduction="sum", delta=1.0)

    print("开始训练 Spectral-Temporal Transformer...")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        val_loss = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}: Train Loss = {train_loss:.6f} | Val Loss = {val_loss:.6f} "
            f"| LR = {scheduler.get_last_lr()[0]:.6e}"
        )

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"models/spectral_transformer_ep{epoch + 1}.pth")


if __name__ == "__main__":
    main()
