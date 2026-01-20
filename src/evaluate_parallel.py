import torch
import numpy as np
import networkx as nx
import multiprocessing
from functools import partial
import tqdm
import sys
import os
from scipy.linalg import expm
from qiskit.quantum_info import Statevector

from src.models.transformer import FALQONTransformer
from src.algorithms.falqon_core import FALQON

# ---------------------------------------------------------
# 核心计算函数：会被多个 CPU 核心并行调用
# ---------------------------------------------------------
def evaluate_single_instance(instance_id, model_state_dict, device_str, n_nodes=12):
    # 每个进程内部重新初始化环境，避免资源竞争
    device = torch.device(device_str)
    # 确保 d_model 与你方案二训练时一致 (64)
    model = FALQONTransformer(max_nodes=12, output_len=30, d_model=64).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    # 1. 生成随机图 (12 节点)
    g = nx.erdos_renyi_graph(n_nodes, p=0.5)
    while not nx.is_connected(g):
        g = nx.erdos_renyi_graph(n_nodes, p=0.5)

    # 2. Teacher (经典 FALQON)
    teacher = FALQON(g, alpha=0.5)
    _, teacher_energies = teacher.train(max_layers=30)
    e_true = teacher_energies[-1]

    # 3. Student (AI 预测)
    adj = nx.to_numpy_array(g)
    padded_adj = np.zeros((12, 12), dtype=np.float32)
    padded_adj[:n_nodes, :n_nodes] = adj
    mask = np.zeros(12, dtype=np.float32)
    mask[:n_nodes] = 1.0

    adj_t = torch.tensor(padded_adj).unsqueeze(0).to(device).float()
    mask_t = torch.tensor(mask).unsqueeze(0).to(device).float()

    with torch.no_grad():
        pred_betas = model(adj_t, mask_t).cpu().numpy().flatten()

    # 4. 执行预测参数的演化
    hp_mat = teacher.Hp.to_matrix()
    hd_mat = teacher.Hd.to_matrix()
    current_state = Statevector.from_label('+' * n_nodes)
    u_p = expm(-1j * hp_mat * 1.0)
    
    for b in pred_betas:
        current_state = current_state.evolve(u_p)
        # 演化矩阵应用
        u_d = expm(-1j * hd_mat * b)
        current_state = current_state.evolve(u_d)
        
    e_ai = current_state.expectation_value(hp_mat).real

    # 5. 计算 Cut Value 比值 (AR)
    num_edges = len(g.edges)
    cut_true = 0.5 * (num_edges - 2 * e_true)
    cut_ai = 0.5 * (num_edges - 2 * e_ai)
    
    return cut_ai / cut_true if cut_true > 1e-6 else None

def main():
    # 接收命令行参数：python src/evaluate_parallel.py [start_idx] [num_per_job]
    start_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_per_job = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    model_path = "models/checkpoints/best_model.pth"
    n_nodes = 12  
    device_str = "cpu"
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found")
        return

    state_dict = torch.load(model_path, map_location=device_str)
    num_cores = int(multiprocessing.cpu_count())
    
    print(f"🚀 子任务启动 | 范围: {start_idx} 到 {start_idx + num_per_job} | 核心数: {num_cores}")

    # 使用进程池并行计算当前范围内的样本
    with multiprocessing.Pool(processes=num_cores) as pool:
        func = partial(evaluate_single_instance, model_state_dict=state_dict, device_str=device_str, n_nodes=n_nodes)
        results = list(tqdm.tqdm(pool.imap(func, range(num_per_job)), total=num_per_job))

    # 过滤 None 并保存本片段结果
    ar_list = [r for r in results if r is not None]
    
    os.makedirs("output/ar_parts", exist_ok=True)
    save_path = f"output/ar_parts/part_{start_idx}.npy"
    np.save(save_path, np.array(ar_list))
    
    print(f"💾 片段计算完成，结果保存至: {save_path}")

if __name__ == "__main__":
    main()