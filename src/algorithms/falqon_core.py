import numpy as np
import networkx as nx
import os
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.linalg import expm

class FALQON:
    def __init__(self, graph, alpha=0.1):
        """
        初始化 FALQON 算法。
        基于量子李雅普诺夫控制理论，通过反馈律自动确定参数。
        """
        self.graph = graph
        self.n_qubits = len(graph.nodes)
        self.alpha = alpha
        
        # 1. 构建问题哈密顿量 Hp (MaxCut)
        # Hp = 0.5 * sum(I - ZiZj) -> 我们只需要 ZZ 项来驱动相位演化
        pauli_list = []
        for u, v in graph.edges:
            # 这里的系数 0.5 来源于 MaxCut 的定义
            pauli_list.append(("ZZ", [u, v], 0.5))
        self.Hp = SparsePauliOp.from_sparse_list(pauli_list, num_qubits=self.n_qubits)
        
        # 2. 构建驱动哈密顿量 Hd (横场 X)
        self.Hd = SparsePauliOp.from_sparse_list([("X", [i], 1.0) for i in range(self.n_qubits)], num_qubits=self.n_qubits)
        
        # 3. 预计算对易子 A = i[Hd, Hp]，用于反馈测量
        # 注意：SparsePauliOp 支持矩阵乘法 @
        hp_mat = self.Hp.to_matrix()
        hd_mat = self.Hd.to_matrix()
        self.A = 1j * (hd_mat @ hp_mat - hp_mat @ hd_mat)

    def compute_expectation(self, state_vec, operator_mat):
        """计算算符在当前状态下的期望值: <psi|O|psi>"""
        # 使用 Statevector 的 expectation_value，传入矩阵形式
        return state_vec.expectation_value(operator_mat).real

    def train(self, max_layers=10):
        """
        执行 FALQON 迭代，生成参数曲线 (Betas) 和能量曲线 (Energies)。
        """
        betas = []
        energies = []
        
        # 初始态: |+> 态 (所有比特在 X 轴正方向)
        current_state = Statevector.from_label('+' * self.n_qubits)
        
        # 预计算固定步长的 Hp 演化矩阵 U_p = exp(-i * Hp * dt), dt=1.0
        hp_mat = self.Hp.to_matrix()
        u_p = expm(-1j * hp_mat * 1.0)
        
        # 缓存 Hd 矩阵用于每一层的反馈计算
        hd_mat = self.Hd.to_matrix()
        
        print(f"开始 FALQON 演化 (n_qubits={self.n_qubits}, alpha={self.alpha})...")
        
        for p in range(max_layers):
            # 记录当前能量
            energy = self.compute_expectation(current_state, hp_mat)
            energies.append(energy)
            
            # 根据反馈律计算下一层的 beta: beta = -alpha * <psi|i[Hd, Hp]|psi>
            # 这里的 self.A 已经是 i[Hd, Hp]
            beta_val = -self.alpha * self.compute_expectation(current_state, self.A)
            betas.append(beta_val)
            
            # 演化状态: psi_{p+1} = exp(-i * beta * Hd) * exp(-i * Hp) * psi_p
            # 1. 应用 Hp
            current_state = current_state.evolve(u_p)
            # 2. 计算并应用自适应的 Hd
            u_d = expm(-1j * hd_mat * beta_val)
            current_state = current_state.evolve(u_d)
            
            if p % 5 == 0 or p == max_layers - 1:
                print(f"  层数 {p:2d}: Beta = {beta_val:7.4f}, Energy = {energy:7.4f}")
            
        return betas, energies

if __name__ == "__main__":
    # 创建保存目录
    os.makedirs('data/raw', exist_ok=True)
    
    # 测试案例：3节点的路径图 (0-1-2)
    test_graph = nx.path_graph(3)
    
    # 正确实例化类
    try:
        falqon_solver = FALQON(test_graph, alpha=0.5)
        betas, energies = falqon_solver.train(max_layers=20)
        
        # 保存结果
        save_path = 'data/raw/betas_example.npy'
        np.save(save_path, np.array(betas))
        print(f"\n测试运行成功！")
        print(f"参数曲线已保存至: {save_path}")
    except Exception as e:
        print(f"\n运行失败: {e}")