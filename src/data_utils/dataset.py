import torch
from torch.utils.data import Dataset
import numpy as np

class QuantumFALQONDataset(Dataset):
    def __init__(self, npz_path):
        # 加载服务器生成的压缩数据
        loaded = np.load(npz_path, allow_pickle=True)
        self.data = loaded['data']
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 输入特征：邻接矩阵 (Adjacency Matrix)
        # 将其展平或作为图输入，这里先演示基础的展平处理
        adj = torch.tensor(item['adj'], dtype=torch.float32)
        
        # 目标标签：FALQON 参数曲线 (Betas)
        betas = torch.tensor(item['betas'], dtype=torch.float32)
        
        return adj, betas
