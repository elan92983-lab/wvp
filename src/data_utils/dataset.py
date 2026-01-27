import torch
from torch.utils.data import Dataset
import numpy as np

class SpectralDataset(Dataset):
    def __init__(self, data_path, max_nodes=20, max_seq_len=40):
        """
        加载 generate_dataset_v2.py 生成的数据
        """
        self.data = np.load(data_path, allow_pickle=True)['data']
        self.max_nodes = max_nodes
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. 获取原始数据
        evals = item['evals']*10.0 # [N]
        evecs = item['evecs'] # [N, N]
        betas = item['betas'] # [P_real]
        
        N = evals.shape[0]
        P = betas.shape[0]
        
        # 2. 补零对齐 (Padding) 到 max_nodes
        # 特征值 padding
        evals_pad = np.zeros(self.max_nodes, dtype=np.float32)
        evals_pad[:N] = evals
        
        # 特征向量 padding
        evecs_pad = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
        evecs_pad[:N, :N] = evecs
        
        # Beta padding (如果生成的层数少于 max_seq_len)
        betas_pad = np.zeros(self.max_seq_len, dtype=np.float32)
        real_len = min(P, self.max_seq_len)
        betas_pad[:real_len] = betas[:real_len]
        
        # 生成时间索引 [0, 1, 2, ..., max_seq_len-1]
        time_indices = np.arange(self.max_seq_len)
        
        # 3. 构建 Mask (用于 Loss 计算时忽略 pad 部分)
        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        mask[:real_len] = 1.0
        
        return {
            'evals': torch.from_numpy(evals_pad),
            'evecs': torch.from_numpy(evecs_pad),
            'time_indices': torch.from_numpy(time_indices).long(),
            'betas': torch.from_numpy(betas_pad),
            'mask': torch.from_numpy(mask)
        }
