import torch
import torch.nn as nn
import numpy as np

class DiffQuantumSimulator(nn.Module):
    """
    一个基于 PyTorch 的轻量级可微分量子模拟器。
    专门用于 MaxCut 问题和 QAOA/FALQON 线路。
    支持 Batch 并行计算，无需安装 Qiskit/PennyLane 即可进行反向传播。
    """
    def __init__(self, n_qubits, device):
        super().__init__()
        self.n = n_qubits
        self.device = device
        self.dim = 2 ** n_qubits
        
        # 1. 预计算 Pauli Z 算符的对角元
        # 形状: [2^N]
        # 比如 N=2, Z0=[1, 1, -1, -1], Z1=[1, -1, 1, -1]
        self._z_ops = []
        for i in range(n_qubits):
            # 利用克罗内克积生成对角线
            # I (x) ... (x) Z (x) ... (x) I
            # Z = diag([1, -1])
            op = torch.tensor([1., -1.], device=device)
            # 前面的 I
            if i > 0:
                op = torch.kron(torch.ones(2**i, device=device), op)
            # 后面的 I
            if i < n_qubits - 1:
                op = torch.kron(op, torch.ones(2**(n_qubits - 1 - i), device=device))
            self._z_ops.append(op)
            
        self._z_ops = torch.stack(self._z_ops) # [N, 2^N]

        # 2. 预计算 X 驱动项 (Hd = sum X)
        # 由于 X 导致状态混合，不能仅用对角矩阵。
        # 但 e^{-i beta X} = cos(beta) I - i sin(beta) X
        # 对于全连接层，我们可以用爱因斯坦求和或者简单的矩阵乘法
        # 这里为了效率，我们针对每一层单独做 Rx 旋转
        pass

    def compute_maxcut_energy(self, batch_betas, adj_matrices):
        """
        Args:
            batch_betas: [Batch, P] 控制参数序列
            adj_matrices: [Batch, N, N] 图邻接矩阵
        Returns:
            energies: [Batch] 最终能量 <H_P>
        """
        B, P = batch_betas.shape
        N = self.n
        
        # --- 1. 构建问题哈密顿量 H_P (对角线) ---
        # Hp = 0.5 * sum_{ij} w_{ij} (I - Z_i Z_j)
        # 我们只需要计算 -0.5 * sum w_{ij} Z_i Z_j 的期望，常数项最后加
        # Z_products shape: [N, N, 2^N] 表示 Z_i * Z_j 的对角元
        # 为了省显存，我们在循环里算
        
        hp_diag = torch.zeros(B, self.dim, device=self.device)
        
        # 这是一个稀疏操作的简单实现
        # adj: [B, N, N]
        # z_ops: [N, 2^N]
        for b in range(B):
            # 获取该图的所有边
            adj = adj_matrices[b] # [N, N]
            rows, cols = torch.where(torch.triu(adj, diagonal=1) > 0.5)
            
            # Hp_diag = 0.5 * sum (1 - ZiZj)
            #         = 0.5 * |E| - 0.5 * sum(ZiZj)
            for u, v in zip(rows, cols):
                z_u = self._z_ops[u]
                z_v = self._z_ops[v]
                hp_diag[b] += 0.5 * (1 - z_u * z_v)
                
        # --- 2. 初始状态 |+> ---
        # vector: [1/sqrt(D), ..., 1/sqrt(D)]
        state = torch.ones(B, self.dim, dtype=torch.complex64, device=self.device)
        state = state / torch.sqrt(torch.tensor(self.dim, dtype=torch.float32, device=self.device))
        
        # --- 3. 逐层演化 ---
        # 这是一个动力系统模拟
        for t in range(P):
            beta = batch_betas[:, t] # [B]
            
            # (A) 问题层 U_P = exp(-i * dt * H_P)
            # FALQON 中通常 dt=1 (常数) 或者也是参数。这里假设 dt=1.0 固定
            # exp(-i H_P) 是对角矩阵乘法
            # 注意: 这里的 H_P 是每个图特有的
            evolve_p = torch.exp(-1j * hp_diag * 1.0) # [B, 2^N]
            state = state * evolve_p
            
            # (B) 驱动层 U_D = exp(-i * beta * H_D)
            # H_D = sum X_i.  U_D = prod exp(-i beta X_i)
            # Rx(theta) = [[cos, -isin], [-isin, cos]]
            # 这是一个全局旋转。为了速度，我们利用张量积结构。
            # 或者，更简单的：既然是 symmetric 的，我们可以用简单的矩阵操作?
            # 不，最快的方法是逐个 qubit 应用 Rx。
            
            # Rx gate matrix:
            # [[c, -is], [-is, c]]
            c = torch.cos(beta).unsqueeze(1) # [B, 1]
            s = torch.sin(beta).unsqueeze(1) # [B, 1]
            
            # 这是一个瓶颈，对于 N=14 比较慢，但对于 N=8~12 很快
            # 我们将 state reshape 为 [B, 2, 2, ..., 2]
            state = state.view([B] + [2]*N)
            
            for i in range(N):
                # 对第 i 个维度应用 Rx
                # state: [B, ..., 2_i, ...]
                # 我们要把第 i 维移到最后，做矩阵乘法，再移回来
                
                # 构造 2x2 变换矩阵 [B, 2, 2]
                mat = torch.zeros(B, 2, 2, dtype=torch.complex64, device=self.device)
                mat[:, 0, 0] = c[:, 0]
                mat[:, 0, 1] = -1j * s[:, 0]
                mat[:, 1, 0] = -1j * s[:, 0]
                mat[:, 1, 1] = c[:, 0]
                
                # Permute: 把 qubit i 放到最后一个维度
                dims = list(range(1, N+1)) # 0 is batch
                dims.remove(i+1)
                dims = [0] + dims + [i+1] # [batch, other_qubits, qubit_i]
                
                state = state.permute(dims)
                
                # 现在 state 是 [B, ..., 2]
                # mat 是 [B, 2, 2]
                # 我们希望做 result[..., k] = sum_j state[..., j] * mat[B, j, k]
                # 展平前部
                shape_back = state.shape
                state = state.reshape(-1, 2) # [B*..., 2]
                
                # 扩展 mat 以匹配 batch
                # 这里比较 tricky，因为 B 维在外层。
                # 简单做法：利用 einsum
                # state 恢复为 [B, dim/2, 2]
                state = state.view(B, -1, 2)
                # mat: [B, 2, 2]
                # result: [B, dim/2, 2]
                state = torch.einsum('bdi,bij->bdj', state, mat)
                
                # 还原形状
                state = state.view(shape_back)
                
                # 还原维度顺序 (inverse permute)
                # dims 是当前的顺序，我们需要 argsort 来还原
                inv_dims = np.argsort(dims)
                state = state.permute(tuple(inv_dims))
                
            # 展平回 [B, 2^N]
            state = state.reshape(B, self.dim)
            
        # --- 4. 计算最终期望值 <psi | Hp | psi> ---
        # Hp 是对角阵，所以 = sum |psi_k|^2 * E_k
        probs = torch.abs(state) ** 2
        energy = torch.sum(probs * hp_diag, dim=1)
        
        return energy
