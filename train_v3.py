"""
增强版训练脚本 - 针对研究项目
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

from src.data_utils.dataset import SpectralDataset
from src.models.spectral_transformer_v2 import SpectralTemporalTransformerV2


def temporal_gradient_loss(pred, target, mask):
    pred_diff = pred[:, 1:] - pred[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]
    mask_diff = mask[:, 1:] * mask[:, :-1]
    return ((pred_diff - target_diff) ** 2 * mask_diff).sum() / (mask_diff.sum() + 1e-6)


def multi_scale_loss(pred, target, mask, scales=[1, 2, 4]):
    total_loss = 0
    for s in scales:
        if pred.shape[1] >= s:
            p_down = pred[:, ::s]
            t_down = target[:, ::s]
            m_down = mask[:, ::s]
            mse = ((p_down - t_down) ** 2 * m_down).sum() / m_down.sum().clamp_min(1.0)
            total_loss += mse / len(scales)
    return total_loss


def hard_sample_weight(target, mask, tail_ratio=0.5, scale=2.0):
    P = target.shape[1]
    tail_start = int(P * (1 - tail_ratio))
    tail = target[:, tail_start:]
    tail_mask = mask[:, tail_start:]
    denom = tail_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (tail * tail_mask).sum(dim=1, keepdim=True) / denom
    var = ((tail - mean) ** 2 * tail_mask).sum(dim=1) / denom.squeeze(-1)
    std = var.sqrt()
    weight = 1.0 + std * scale
    return weight.unsqueeze(1)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs,
                    ss_start, ss_end, weight_tail, use_hard_mining=True):
    model.train()
    total_loss = 0
    ss_prob = ss_start + (ss_end - ss_start) * (epoch / max(total_epochs - 1, 1))

    for batch in loader:
        evals = batch['evals'].to(device)
        evecs = batch['evecs'].to(device)
        time_idx = batch['time_indices'].to(device)
        targets = batch['betas'].to(device)
        mask = batch['mask'].to(device)
        num_nodes = batch['num_nodes'].to(device)

        optimizer.zero_grad()

        prev_betas = torch.zeros_like(targets)
        prev_betas[:, 1:] = targets[:, :-1]

        if ss_prob > 0 and epoch > 10:
            with torch.no_grad():
                pred_for_ss = model(evals, evecs, time_idx, num_nodes=num_nodes, prev_betas=prev_betas)
            ss_mask = torch.rand_like(targets[:, 1:]) < ss_prob
            prev_betas_mixed = prev_betas.clone()
            prev_betas_mixed[:, 1:] = torch.where(ss_mask, pred_for_ss[:, :-1], targets[:, :-1])
            prev_betas = prev_betas_mixed

        preds = model(evals, evecs, time_idx, num_nodes=num_nodes, prev_betas=prev_betas)

        P = preds.shape[1]
        time_w = torch.linspace(1.0, weight_tail, steps=P, device=device).unsqueeze(0)

        if use_hard_mining:
            sample_w = hard_sample_weight(targets, mask)
        else:
            sample_w = 1.0

        weighted_mask = mask * time_w * sample_w

        loss_main = ((preds - targets) ** 2 * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)
        loss_temp = temporal_gradient_loss(preds, targets, mask)
        loss_multi = multi_scale_loss(preds, targets, mask)
        loss = loss_main + 0.3 * loss_temp + 0.2 * loss_multi

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    total_corr = 0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            evals = batch['evals'].to(device)
            evecs = batch['evecs'].to(device)
            time_idx = batch['time_indices'].to(device)
            targets = batch['betas'].to(device)
            mask = batch['mask'].to(device)
            num_nodes = batch['num_nodes'].to(device)

            preds = model.generate(evals, evecs, time_idx, num_nodes=num_nodes)
            for i in range(preds.shape[0]):
                valid_len = int(mask[i].sum().item())
                if valid_len > 2:
                    p = preds[i, :valid_len].cpu().numpy()
                    t = targets[i, :valid_len].cpu().numpy()
                    if np.std(p) > 1e-6 and np.std(t) > 1e-6:
                        corr = np.corrcoef(p, t)[0, 1]
                        if not np.isnan(corr):
                            total_corr += corr
                            total_samples += 1
    return total_corr / max(total_samples, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_path", type=str, default="data/processed/spectral_data_v2.npz")
    parser.add_argument("--weight_tail", type=float, default=3.0)
    parser.add_argument("--ss_start", type=float, default=0.0)
    parser.add_argument("--ss_end", type=float, default=0.4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.dirname(__file__))
    models_dir = os.path.join(root_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "spectral_transformer_v2_best.pth")
    if os.path.islink(best_model_path):
        os.unlink(best_model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.data_path):
        print(f"错误: 找不到数据 {args.data_path}")
        return

    dataset = SpectralDataset(args.data_path, max_nodes=20, max_seq_len=40)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)

    model = SpectralTemporalTransformerV2(
        max_nodes=20,
        d_model=args.d_model,
        nhead=8,
        num_layers=args.num_layers,
        max_seq_len=40,
        dropout=0.1
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    criterion = nn.MSELoss()

    best_corr = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, args.epochs, args.ss_start, args.ss_end, args.weight_tail
        )
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            val_corr = validate(model, val_loader, device)
            print(f"Epoch {epoch+1}: Loss={train_loss:.6f}, Val Corr={val_corr:.4f}, LR={scheduler.get_last_lr()[0]:.2e}")
            if val_corr > best_corr:
                best_corr = val_corr
                torch.save(model.state_dict(), best_model_path)
                print(f"  🌟 新最佳模型! Corr={best_corr:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= args.patience // 10:
                print(f"早停: {args.patience} epochs 无改进")
                break
        else:
            print(f"Epoch {epoch+1}: Loss={train_loss:.6f}, LR={scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(models_dir, f"spectral_transformer_v2_ep{epoch+1}.pth"))

    print(f"\n训练完成! 最佳验证 Corr: {best_corr:.4f}")


if __name__ == "__main__":
    main()
