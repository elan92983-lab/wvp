import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys

# 确保从任意路径运行时也能找到 src 包
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 导入数据集和模型实现
from src.data_utils.dataset import QuantumFALQONDataset
from src.models.quantum_transformer import GadgetTokenizedTransformer


class FALQONTransformer(nn.Module):
    def __init__(self, max_nodes=12, output_len=30, d_model=64, nhead=8, num_layers=4):
        super().__init__()
        self.max_nodes = max_nodes
        self.output_len = output_len
        self.model = GadgetTokenizedTransformer(
            input_dim=max_nodes * max_nodes,
            model_dim=d_model,
            nhead=nhead,
            num_layers=num_layers,
            seq_len=output_len,
        )

    def forward(self, adj, mask=None):
        # adj: [B, N, N] (N 可能已 padding 至 max_nodes)
        x = adj.view(adj.size(0), -1)
        return self.model(x)

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
    # 优先使用生成脚本产出的 train_data_final.npz，回退到 train_data.npz
    data_candidates = [
        os.path.join(PROJECT_ROOT, 'data/raw/dataset_v1/train_data_final.npz'),
        os.path.join(PROJECT_ROOT, 'data/raw/dataset_v1/train_data.npz'),
    ]

    data_path = None
    for p in data_candidates:
        if os.path.exists(p):
            data_path = p
            break

    if data_path is None:
        print(f"错误: 找不到数据集。请先运行生成脚本。预期路径例如: {data_candidates[0]}")
        return

    dataset = QuantumFALQONDataset(data_path)

    # 根据数据自动确定最大节点数，确保 padding 足够
    found_max_nodes = 0
    for i in range(len(dataset)):
        adj_shape = dataset[i][0].shape
        if len(adj_shape) >= 2:
            n = adj_shape[0]
        else:
            n = int(np.sqrt(adj_shape[0])) if adj_shape[0] > 1 else 1
        if n > found_max_nodes:
            found_max_nodes = n

    if found_max_nodes > max_nodes:
        print(f"注意: 数据中检测到最大节点数 {found_max_nodes} > 设置的 max_nodes {max_nodes}，将自动提升。")
        max_nodes = found_max_nodes

    # 自定义 collate_fn：对邻接矩阵进行 padding，使每批数据具有相同尺寸
    def pad_collate(batch):
        adjs, betas = zip(*batch)
        B = len(adjs)
        padded_adjs = torch.zeros((B, max_nodes, max_nodes), dtype=torch.float32)
        masks = torch.zeros((B, max_nodes), dtype=torch.bool)
        for i, adj in enumerate(adjs):
            n = adj.shape[0]
            padded_adjs[i, :n, :n] = adj
            masks[i, :n] = 1
        betas = torch.stack(betas, dim=0)
        return padded_adjs, betas, masks

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

    # 4. 模型初始化
    model = FALQONTransformer(
        max_nodes=max_nodes,
        output_len=output_len,
        d_model=d_model,
        nhead=8,
        num_layers=4,
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
            adj, betas, mask = data
            
            # 6. 取出并搬运到设备
            adj = adj.to(device).float()
            y = betas.to(device).float()

            # 前向传播
            optimizer.zero_grad()
            outputs = model(adj, mask=mask)
            
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