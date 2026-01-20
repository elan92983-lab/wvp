import torch
import numpy as np
import networkx as nx
import os
import tqdm
from scipy.linalg import expm
from qiskit.quantum_info import Statevector

# 导入你现有的模型和算法类
from src.models.transformer import FALQONTransformer
from src.algorithms.falqon_core import FALQON

def run_fixed_evolution(g, betas):
    """
    输入：图 g 和 预定义的 beta 序列
    输出：按照这组 beta 演化后得到的最后一步能量值
    逻辑：同步自 falqon_core.py 的演化逻辑，但使用固定参数
    """
    n_qubits = len(g.nodes)
    # 初始化 FALQON 实例以获取 Hp 矩阵
    falqon_instance = FALQON(g, alpha=0.5)
    hp_mat = falqon_instance.Hp.to_matrix()
    hd_mat = falqon_instance.Hd.to_matrix()
    
    # 初始态: |+> 态
    current_state = Statevector.from_label('+' * n_qubits)
    
    # 预计算固定步长的 Hp 演化矩阵
    u_p = expm(-1j * hp_mat * 1.0)
    
    # 按预测的 betas 序列进行演化
    for beta_val in betas:
        # 1. 应用 Hp
        current_state = current_state.evolve(u_p)
        # 2. 计算并应用输入的 Hd (来自 AI 预测)
        u_d = expm(-1j * hd_mat * beta_val)
        current_state = current_state.evolve(u_d)
    
    # 计算最终能量: <psi|Hp|psi>
    final_energy = current_state.expectation_value(hp_mat).real
    return final_energy

def main():
    # 1. 配置环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/checkpoints/best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型文件: {model_path}，请确认训练已完成并保存。")
        return

    # 2. 加载训练好的 Transformer 模型
    # 这里的参数必须和训练时完全一致
    model = FALQONTransformer(max_nodes=12, output_len=30).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ 已加载模型权重: {model_path}")

    # 3. 开始大规模评估
    num_tests = 100  # 建议测试 100 个新样本以获得稳定的统计数据
    ar_list = []
    
    print(f"🧪 正在对 {num_tests} 个未知随机图进行性能评估...")
    
    for i in tqdm.tqdm(range(num_tests)):
        # 随机生成 4-10 个节点的图（模拟训练集的分布）
        num_nodes = 12
        g = nx.erdos_renyi_graph(num_nodes, p=0.5)
        while not nx.is_connected(g):
            g = nx.erdos_renyi_graph(num_nodes, p=0.5)

        # --- 第一步：运行 Teacher (经典 FALQON) 获取基准能量 ---
        teacher = FALQON(g, alpha=0.5)
        _, teacher_energies = teacher.train(max_layers=30)
        e_true = teacher_energies[-1]

        # --- 第二步：运行 Student (Transformer) 预测参数并计算能量 ---
        adj = nx.to_numpy_array(g)
        padded_adj = np.zeros((12, 12), dtype=np.float32)
        padded_adj[:num_nodes, :num_nodes] = adj
        mask = np.zeros(12, dtype=np.float32)
        mask[:num_nodes] = 1.0

        adj_tensor = torch.tensor(padded_adj).unsqueeze(0).to(device).float()
        mask_tensor = torch.tensor(mask).unsqueeze(0).to(device).float()

        with torch.no_grad():
            # 预测 30 层的 betas
            pred_betas = model(adj_tensor, mask_tensor).cpu().numpy().flatten()
        
        # 将 AI 预测的参数带入演化过程
        e_ai = run_fixed_evolution(g, pred_betas)

         # --- 第三步：计算 MaxCut 切割数而非原始能量比 ---
        num_edges = len(g.edges)
        
        # 计算 Cut = 0.5 * (边数 - 2 * 能量)
        cut_true = 0.5 * (num_edges - 2 * e_true)
        cut_ai = 0.5 * (num_edges - 2 * e_ai)
        
        # 初始态 (|+> 态) 的 Cut 值作为基准
        # 初始能量为 0，所以初始 Cut = 0.5 * num_edges
        cut_initial = 0.5 * num_edges

        # 论文常用的改进比 (Improvement Ratio) 或 归一化近似比
        if cut_true > 0:
            ratio = cut_ai / cut_true
            ar_list.append(ratio)
    # 4. 输出统计结果
    avg_ar = np.mean(ar_list)
    std_ar = np.std(ar_list)
    
    print("\n" + "="*40)
    print(f"📊 【最终评估报告】")
    print(f"测试样本总数: {num_tests}")
    print(f"平均近似比 (Avg AR): {avg_ar:.4f}")
    print(f"近似比标准差 (Std):   {std_ar:.4f}")
    print(f"最差表现 (Min AR):    {min(ar_list):.4f}")
    print(f"最佳表现 (Max AR):    {max(ar_list):.4f}")
    print("="*40)
    
    if avg_ar >= 0.95:
        print("🚀 结果卓越！AI 几乎完美替代了量子测量反馈。")
    elif avg_ar >= 0.85:
        print("👍 结果良好！模型具备很强的泛化能力。")
    else:
        print("⚠️ 结果一般，建议增加训练数据或调整模型超参数。")

if __name__ == "__main__":
    main()
