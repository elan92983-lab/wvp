import torch
import numpy as np
import networkx as nx
import multiprocessing
from functools import partial
import tqdm
import sys
import os
import argparse
from scipy.linalg import expm
from qiskit.quantum_info import Statevector

from src.models.transformer import FALQONTransformer
from src.models.gnn import FALQONGNN
from src.algorithms.falqon_core import FALQON

# ---------------------------------------------------------
# 核心计算函数：会被多个 CPU 核心并行调用
# ---------------------------------------------------------
def evaluate_single_instance(
    sample,
    model_state_dict,
    device_str,
    max_nodes=12,
    gnn_state_dict=None,
):
    # 每个进程内部重新初始化环境，避免资源竞争
    device = torch.device(device_str)
    # 确保 d_model 与你方案二训练时一致 (64)
    model = FALQONTransformer(max_nodes=max_nodes, output_len=30, d_model=64).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    gnn_model = None
    if gnn_state_dict is not None:
        gnn_model = FALQONGNN(max_nodes=max_nodes, output_len=30).to(device)
        gnn_model.load_state_dict(gnn_state_dict)
        gnn_model.eval()

    # 1. 读取样本中的图
    adj = sample["adj"]
    n_nodes = adj.shape[0]
    if n_nodes > max_nodes:
        raise ValueError(f"样本节点数 {n_nodes} 超过 max_nodes={max_nodes}，请调整 --nodes")
    g = nx.from_numpy_array(adj)

    # 2. Teacher (经典 FALQON) - 仅用于构建 Hp/Hd 和读取能量
    teacher = FALQON(g, alpha=0.5)
    if "energies" in sample:
        e_true = sample["energies"][-1]
    else:
        _, teacher_energies = teacher.train(max_layers=30)
        e_true = teacher_energies[-1]

    # 3. Student (AI 预测)
    padded_adj = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    padded_adj[:n_nodes, :n_nodes] = adj
    mask = np.zeros(max_nodes, dtype=np.float32)
    mask[:n_nodes] = 1.0

    adj_t = torch.tensor(padded_adj).unsqueeze(0).to(device).float()
    mask_t = torch.tensor(mask).unsqueeze(0).to(device).float()

    with torch.no_grad():
        pred_betas = model(adj_t, mask_t).cpu().numpy().flatten()

        pred_betas_gnn = None
        if gnn_model is not None:
            pred_betas_gnn = gnn_model(adj_t, mask_t).cpu().numpy().flatten()

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

    e_gnn = None
    if pred_betas_gnn is not None:
        current_state_gnn = Statevector.from_label('+' * n_nodes)
        for b in pred_betas_gnn:
            current_state_gnn = current_state_gnn.evolve(u_p)
            u_d = expm(-1j * hd_mat * b)
            current_state_gnn = current_state_gnn.evolve(u_d)
        e_gnn = current_state_gnn.expectation_value(hp_mat).real

    # 5. 计算 Cut Value 比值 (AR)
    num_edges = len(g.edges)
    cut_true = 0.5 * (num_edges - 2 * e_true)
    cut_ai = 0.5 * (num_edges - 2 * e_ai)
    
    if cut_true <= 1e-6:
        return None

    ar_transformer = cut_ai / cut_true
    ar_gnn = (0.5 * (num_edges - 2 * e_gnn) / cut_true) if e_gnn is not None else None
    ar_classical = 1.0

    return ar_transformer, ar_gnn, ar_classical

def main():
    parser = argparse.ArgumentParser(description="Evaluate Avg AR and Std on dataset")
    parser.add_argument("--model_path", type=str, default="models/checkpoints/best_model.pth")
    parser.add_argument("--gnn_model_path", type=str, default=None, help="可选：GNN 权重 .pth（state_dict）")
    parser.add_argument("--test_data", type=str, default="data/raw/dataset_v1/train_data_final.npz")
    parser.add_argument("--nodes", type=int, default=12, help="模型的 max_nodes（与训练一致）")
    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--start", type=int, default=None, help="可选：起始样本索引（包含）")
    parser.add_argument("--end", type=int, default=None, help="可选：结束样本索引（不包含）")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: {args.model_path} not found")
        return
    if args.gnn_model_path is not None and not os.path.exists(args.gnn_model_path):
        print(f"Error: {args.gnn_model_path} not found")
        return
    if not os.path.exists(args.test_data):
        print(f"Error: {args.test_data} not found")
        return

    device_str = "cpu"
    state_dict = torch.load(args.model_path, map_location=device_str)

    gnn_state_dict = None
    if args.gnn_model_path is not None:
        gnn_state_dict = torch.load(args.gnn_model_path, map_location=device_str)

    loaded = np.load(args.test_data, allow_pickle=True)
    samples = list(loaded["data"])

    total = len(samples)
    s = args.start if args.start is not None else 0
    e = args.end if args.end is not None else total
    if s < 0 or e < 0 or s >= total or s >= e:
        print(f"Error: invalid start/end range: start={s}, end={e}, total={total}")
        return

    subset = samples[s:e]
    print(f"🚀 开始评估 | 处理样本索引: {s} 到 {e} (共 {len(subset)}) | 核心数: {args.num_workers}")

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        func = partial(
            evaluate_single_instance,
            model_state_dict=state_dict,
            device_str=device_str,
            max_nodes=args.nodes,
            gnn_state_dict=gnn_state_dict,
        )
        results = list(tqdm.tqdm(pool.imap(func, subset), total=len(subset)))

    results = [r for r in results if r is not None]
    if len(results) == 0:
        print("❌ 没有有效的 AR 结果")
        return

    ar_transformer_list = [r[0] for r in results if r[0] is not None]
    ar_gnn_list = [r[1] for r in results if r[1] is not None]
    ar_classical_list = [r[2] for r in results if r[2] is not None]

    # 保存分片结果，供后续合并
    os.makedirs("output/ar_parts", exist_ok=True)
    save_path = f"output/ar_parts/part_{s}.npy"
    np.save(save_path, np.array(ar_transformer_list, dtype=np.float64))
    print(f"💾 Transformer 片段保存至: {save_path} (有效 {len(ar_transformer_list)} 条)")

    if args.gnn_model_path is not None:
        save_path_gnn = f"output/ar_parts/part_{s}_gnn.npy"
        np.save(save_path_gnn, np.array(ar_gnn_list, dtype=np.float64))
        print(f"💾 GNN 片段保存至: {save_path_gnn} (有效 {len(ar_gnn_list)} 条)")

    avg_t = float(np.mean(ar_transformer_list))
    std_t = float(np.std(ar_transformer_list))
    print(f"✅ Avg AR (Transformer, this part): {avg_t:.6f}")
    print(f"✅ Std  (Transformer, this part): {std_t:.6f}")

    if args.gnn_model_path is not None:
        avg_g = float(np.mean(ar_gnn_list)) if len(ar_gnn_list) else float('nan')
        std_g = float(np.std(ar_gnn_list)) if len(ar_gnn_list) else float('nan')
        print(f"✅ Avg AR (GNN, this part): {avg_g:.6f}")
        print(f"✅ Std  (GNN, this part): {std_g:.6f}")

    # Classical (FALQON) 是基线，按定义 AR=1（相对于自身）
    if len(ar_classical_list):
        print(f"✅ Avg AR (Classical FALQON, this part): {float(np.mean(ar_classical_list)):.6f}")

if __name__ == "__main__":
    main()