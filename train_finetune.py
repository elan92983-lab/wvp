import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import networkx as nx
import os
import scipy.linalg

from src.models.spectral_transformer import SpectralTemporalTransformer
from src.physics.simulator import DiffQuantumSimulator

# --- 1. 简易的数据生成器 (只生成图，不需要标签) ---
class UnlabeledGraphDataset(Dataset):
    def __init__(self, num_samples=1000, min_n=10, max_n=14):
        self.data = []
        print(f"Generating {num_samples} unlabeled graphs for physics fine-tuning...")
        for _ in range(num_samples):
            n = np.random.randint(min_n, max_n + 1)
            # 随机正则图或 ER 图
            if np.random.rand() > 0.5 and (n*3)%2==0:
                g = nx.random_regular_graph(3, n)
            else:
                g = nx.erdos_renyi_graph(n, 0.6)
                
            adj = nx.to_numpy_array(g)
            evals, evecs = self.get_spectral(adj)
            
            # Padding
            evals_pad = np.zeros(20, dtype=np.float32)
            evals_pad[:n] = evals
            evecs_pad = np.zeros((20, 20), dtype=np.float32)
            evecs_pad[:n, :n] = evecs
            adj_pad = np.zeros((20, 20), dtype=np.float32)
            adj_pad[:n, :n] = adj
            
            self.data.append({
                'evals': torch.from_numpy(evals_pad),
                'evecs': torch.from_numpy(evecs_pad),
                'adj': torch.from_numpy(adj_pad),
                'n': n
            })
            
    def get_spectral(self, adj):
        deg = np.sum(adj, axis=1)
        d_inv_sqrt = np.power(deg, -0.5, where=deg!=0)
        d_inv_sqrt[deg==0] = 0.0
        D_inv = np.diag(d_inv_sqrt)
        L = np.eye(len(adj)) - D_inv @ adj @ D_inv
        evals, evecs = scipy.linalg.eigh(L)
        return evals.astype(np.float32), evecs.astype(np.float32)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def main():
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 加载预训练模型
    model = SpectralTemporalTransformer(max_nodes=20, d_model=128, max_seq_len=40).to(device)
    pretrained_path = "models/spectral_transformer_ep100.pth" # 确保这里路径对
    
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained model: {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
    else:
        print("⚠️ Warning: No pretrained model found. Training from scratch (Hard!)")
        
    # 2. 准备物理模拟器
    # 注意: MaxCut 模拟器的 qubit 数必须等于图节点数
    # 为了 Batch 并行，我们按节点数分组训练，或者简单的：
    # 我们的模拟器支持 batch，但要求 batch 内的 n_qubits 相同吗？
    # 上面的 simulator 代码中，N 是固定的 self.n。
    # 所以我们需要为不同的 N 实例化不同的模拟器，或者在一个 batch 里只放相同 N 的图。
    # 策略：我们只微调 N=12 的图作为演示 (Scale up test)
    
    target_N = 12
    simulator = DiffQuantumSimulator(n_qubits=target_N, device=device)
    
    # 3. 准备数据 (只生成 N=12 的图)
    dataset = UnlabeledGraphDataset(num_samples=500, min_n=target_N, max_n=target_N)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # 微调学习率要小
    
    # 4. 训练循环
    print("🚀 Starting Physics-Informed Fine-tuning (PALQO Strategy)...")
    for epoch in range(20):
        total_energy = 0
        for batch in loader:
            evals = batch['evals'].to(device)
            evecs = batch['evecs'].to(device)
            adj = batch['adj'].to(device) # [B, 20, 20]
            
            # 截取有效的 adj 部分传入模拟器 (因为模拟器是 N=12)
            adj_eff = adj[:, :target_N, :target_N]
            
            # 生成时间索引
            time_idx = torch.arange(40, device=device).unsqueeze(0).expand(evals.shape[0], -1)
            
            optimizer.zero_grad()
            
            # (A) 学生模型预测 Beta
            pred_betas = model(evals, evecs, time_idx) # [B, 40]
            
            # (B) 物理模拟器计算能量 (Physics Loss)
            # 注意: 这里的模拟器是可微的！
            energy = simulator.compute_maxcut_energy(pred_betas, adj_eff)
            
            # Loss = Average Energy (我们希望能量越低越好)
            loss = energy.mean()
            
            loss.backward()
            optimizer.step()
            
            total_energy += loss.item()
            
        avg_energy = total_energy / len(loader)
        print(f"Epoch {epoch+1}: Physics Loss (Energy) = {avg_energy:.4f}")
        
    # 保存微调后的模型
    torch.save(model.state_dict(), "models/spectral_transformer_finetuned.pth")
    print("✅ Fine-tuning complete!")

if __name__ == "__main__":
    main()
