import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import time
import argparse

# 引入我们自己写的模块
from src.data.dataset import FALQONDataset
from src.models.transformer import FALQONTransformer

def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in loader:
        # 1. 搬运数据到 GPU
        adj = batch['adj'].to(device)   # [B, N, N]
        mask = batch['mask'].to(device) # [B, N]
        targets = batch['betas'].to(device) # [B, 30]
        
        # 2. 前向传播
        optimizer.zero_grad()
        outputs = model(adj, mask) # [B, 30]
        
        # 3. 计算损失 & 反向传播
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            adj = batch['adj'].to(device)
            mask = batch['mask'].to(device)
            targets = batch['betas'].to(device)
            
            outputs = model(adj, mask)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--data_path", type=str, default="data/raw/dataset_v1/train_data_final.npz")
    parser.add_argument("--save_dir", type=str, default="models/checkpoints")
    args = parser.parse_args()

    # 1. 设置设备 (优先用 GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 使用设备: {device}")

    # 2. 准备数据
    full_dataset = FALQONDataset(args.data_path)
    
    # 划分 80% 训练, 20% 验证
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"📊 数据集就绪: 训练集 {len(train_dataset)} | 验证集 {len(val_dataset)}")

    # 3. 初始化模型
    model = FALQONTransformer().to(device)
    
    # 4. 定义损失函数和优化器
    criterion = nn.MSELoss() # 均方误差 (Mean Squared Error)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 5. 训练循环
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    print("\n🚀 开始训练...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        # 打印进度
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"    🌟 新的最佳模型已保存 (Loss: {best_val_loss:.6f})")

    total_time = (time.time() - start_time) / 60
    print(f"\n✅ 训练结束! 总耗时: {total_time:.2f} 分钟")
    print(f"🏆 最佳验证集 Loss: {best_val_loss:.6f}")
    print(f"💾 模型保存在: {args.save_dir}")

if __name__ == "__main__":
    main()
