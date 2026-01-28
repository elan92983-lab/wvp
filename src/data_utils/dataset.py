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
        # === 新增：计算全局 beta 的均值和标准差 ===
        all_betas = np.concatenate([item['betas'] for item in self.data])
        self.beta_mean = all_betas.mean()
        self.beta_std = all_betas.std() + 1e-6

    @staticmethod
    def _align_evecs_sign(evecs: np.ndarray) -> np.ndarray:
        """
        对特征向量进行符号对齐：每个特征向量列中，绝对值最大的分量强制为正。
        evecs: [N, N] (列为特征向量)
        """
        if evecs.ndim != 2:
            return evecs
        aligned = evecs.copy()
        # 每一列对应一个特征向量
        max_idx = np.argmax(np.abs(aligned), axis=0)
        col_idx = np.arange(aligned.shape[1])
        signs = np.sign(aligned[max_idx, col_idx])
        signs[signs == 0] = 1.0
        aligned *= signs
        return aligned
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # 1. 获取原始数据
        evals = item['evals'] # [N]
        evecs = item['evecs'] # [N, N]
        betas = item['betas'] # [P_real]
        # === 新增：特征向量符号对齐 ===
        evecs = self._align_evecs_sign(evecs)
        # === 新增：标准化 betas ===
        betas = (betas - self.beta_mean) / self.beta_std
        N = evals.shape[0]
        P = betas.shape[0]
        # 2. 补零对齐 (Padding) 到 max_nodes
        evals_pad = np.zeros(self.max_nodes, dtype=np.float32)
        evals_pad[:N] = evals
        evecs_pad = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
        evecs_pad[:N, :N] = evecs
        betas_pad = np.zeros(self.max_seq_len, dtype=np.float32)
        real_len = min(P, self.max_seq_len)
        betas_pad[:real_len] = betas[:real_len]
        time_indices = np.arange(self.max_seq_len)
        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        mask[:real_len] = 1.0
        return {
            'evals': torch.from_numpy(evals_pad),
            'evecs': torch.from_numpy(evecs_pad),
            'time_indices': torch.from_numpy(time_indices).long(),
            'betas': torch.from_numpy(betas_pad),
            'mask': torch.from_numpy(mask),
            'num_nodes': torch.tensor(N, dtype=torch.int64)
        }
