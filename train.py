import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os

# 导入你之前定义的组件
from src.data_utils.dataset import QuantumFALQONDataset
from src.models.quantum_transformer import GadgetTokenizedTransformer

def train():
    # 1. 硬件配置：优先使用 GPU (CUDA) 加速训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 超参数设置 (基于 2026 创新协议)
    batch_size = 64
    learning_rate = 1e-4
    epochs = 100
    input_dim = 100  # 假设邻接矩阵展平后的最大长度 (10x10)
    model_dim = 128
    nhead = 8
    num_layers = 4
    seq_len = 30     # 对应 FALQON 的演化层数

    # 3. 数据准备
    data_path = 'data/raw/dataset_v1/train_data.npz'
    if not os.path.exists(data_path):
        print(f"错误: 找不到数据集 {data_path}。请等待服务器生成完成。")
        return

    dataset = QuantumFALQONDataset(data_path)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4. 模型初始化
    model = GadgetTokenizedTransformer(
        input_dim=input_dim, 
        model_dim=model_dim, 
        nhead=nhead, 
        num_layers=num_layers, 
        seq_len=seq_len
    ).to(device)

    # 5. 损失函数与优化器
    # 使用均方误差 (MSE) 来衡量参数预测的准确性: $MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 6. 训练循环
    print("开始训练学生模型...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for adj, betas in train_loader:
            # 数据搬运到 GPU
            # 展平邻接矩阵作为输入特征
            x = adj.view(adj.size(0), -1).to(device) 
            y = betas.to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(x)
            
            # 计算损失并反向传播
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    # 7. 保存训练好的“零样本”预测模型
    os.makedirs('models/checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'models/checkpoints/student_transformer.pth')
    print("训练完成，模型已保存。")

if __name__ == "__main__":

