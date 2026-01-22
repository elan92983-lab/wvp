import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os

# 1. 核心修改：导入您重构后的 FALQONTransformer
from src.data_utils.dataset import QuantumFALQONDataset
from src.models.transformer import FALQONTransformer  # 修改导入路径

def train():
    # 硬件配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 超参数设置 (需与模型定义匹配)
    batch_size = 64
    learning_rate = 1e-4
    epochs = 100
    max_nodes = 12   # 与模型中的 max_nodes 一致 [cite: 1]
    output_len = 30  # FALQON 演化步数 [cite: 1]
    d_model = 64     # 嵌入维度 [cite: 1]

    # 3. 数据准备
    data_path = 'data/raw/dataset_v1/train_data.npz'
    if not os.path.exists(data_path):
        print(f"错误: 找不到数据集 {data_path}。请先运行生成脚本。")
        return

    dataset = QuantumFALQONDataset(data_path)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4. 模型初始化 (传入您重构后的参数)
    model = FALQONTransformer(
        max_nodes=max_nodes, 
        output_len=output_len,
        d_model=d_model,
        nhead=8,
        num_layers=4
    ).to(device)

    # 5. 损失函数与优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("开始训练基于“谱位置编码”的 Transformer 模型...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # 假设您的 dataset 返回 adj 和 betas
        # 如果 dataset 还返回 mask，请对应修改：for adj, betas, mask in train_loader:
        for data in train_loader:
            # 根据您 QuantumFALQONDataset 的具体实现取值
            adj, betas = data[0], data[1]
            
            # 6. 核心修改：禁止展平 (Remove .view())
            # 谱位置编码模块需要 [Batch, N, N] 矩阵进行特征值分解 [cite: 1]
            x = adj.to(device).float() 
            y = betas.to(device).float()

            # 如果有 mask (用于处理变长图)，也需传入
            # mask = data[2].to(device) if len(data) > 2 else None
            mask = None # 暂时设为 None，除非您的 DataLoader 提供了有效节点掩码

            # 前向传播
            optimizer.zero_grad()
            outputs = model(x, src_key_padding_mask=mask)
            
            # 计算损失
            loss = criterion(outputs, y)
            loss.backward()
            
            # 梯度裁剪（防止特征值分解导致的不稳定）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    # 7. 保存模型
    os.makedirs('models/checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'models/checkpoints/spectral_transformer.pth')
    print("训练完成，谱编码模型已保存。")

if __name__ == "__main__":
    train()