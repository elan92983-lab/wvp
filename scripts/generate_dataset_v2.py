import numpy as np
import networkx as nx
import os
import multiprocessing
import scipy.linalg
from tqdm import tqdm
import sys
import argparse

# 确保能找到 src 包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.algorithms.falqon_core import FALQON

def get_spectral_decomposition(adj):
    """
    计算归一化拉普拉斯矩阵的特征值和特征向量。
    L = I - D^{-1/2} A D^{-1/2}
    """
    N = adj.shape[0]
    deg = np.sum(adj, axis=1)
    # 处理孤立节点防止除零
    d_inv_sqrt = np.power(deg, -0.5, where=deg!=0)
    d_inv_sqrt[deg==0] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    
    # 归一化拉普拉斯矩阵
    L = np.eye(N) - D_inv_sqrt @ adj @ D_inv_sqrt
    
    # 特征分解 (eigh 用于对称矩阵，更稳健)
    evals, evecs = scipy.linalg.eigh(L)
    
    # 排序特征值 (从小到大)
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    return evals, evecs

def generate_single_instance(instance_id):
    try:
        # --- 修改点 1: 混合生成策略 (支持正则图以验证 Kesten-McKay) ---
        # 50% 概率生成 Erdős-Rényi, 50% 生成随机正则图
        num_nodes = np.random.randint(6, 14) #稍微增大一点规模
        
        if np.random.rand() > 0.5:
            # 随机正则图 (d=3)
            # 注意: n * d 必须是偶数
            if (num_nodes * 3) % 2 != 0: num_nodes += 1
            g = nx.random_regular_graph(3, num_nodes)
        else:
            # 经典 ER 图
            g = nx.erdos_renyi_graph(num_nodes, p=0.6)

        if not nx.is_connected(g):
            return None

        # --- 修改点 2: 运行 FALQON 获取标签 ---
        falqon = FALQON(g, alpha=1.0) # 增大一点 alpha 加快收敛
        betas, energies = falqon.train(max_layers=40) # 增加到 40 层用于学习长程依赖

        # --- 修改点 3: 提取谱信息 (新架构的核心输入) ---
        adj = nx.to_numpy_array(g)
        evals, evecs = get_spectral_decomposition(adj)

        return {
            "node_count": num_nodes,
            "adj": adj,
            "evals": evals.astype(np.float32), # 特征值 [N]
            "evecs": evecs.astype(np.float32), # 特征向量 [N, N]
            "betas": np.array(betas).astype(np.float32),
            "energies": np.array(energies).astype(np.float32)
        }
    except Exception as e:
        # print(f"Error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate spectral dataset for FALQON.")
    parser.add_argument("--start", type=int, default=0, help="Start index of samples.")
    parser.add_argument("--end", type=int, default=20000, help="End index of samples.")
    parser.add_argument("--part_id", type=int, default=0, help="Part ID for file naming.")
    parser.add_argument("--output_dir", type=str, default="data/raw/dataset_v2", help="Output directory.")
    parser.add_argument("--cores", type=int, default=32, help="Number of CPU cores to use.")
    args = parser.parse_args()

    num_samples = args.end - args.start
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    num_cores = args.cores
    print(f"🚀 开始生成谱-时序数据集 (Tasks: {args.start}-{args.end}, Part: {args.part_id})...")
    
    results = []
    with multiprocessing.Pool(processes=num_cores) as pool:
        for res in tqdm(pool.imap_unordered(generate_single_instance, range(num_samples)), total=num_samples):
            if res is not None:
                results.append(res)
                
    save_path = os.path.join(output_dir, f"part_{args.part_id}.npz")
    np.savez_compressed(save_path, data=results)
    print(f"✅ 数据生成完毕: {save_path}")

if __name__ == "__main__":
    main()
