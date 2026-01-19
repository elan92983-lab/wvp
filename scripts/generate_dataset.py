import numpy as np
import networkx as nx
import os
import multiprocessing
from tqdm import tqdm
from src.algorithms.falqon_core import FALQON

def generate_single_instance(instance_id):
    """
    单个样本生成任务：生成随机图并运行 FALQON 获取教师经验数据。
    """
    try:
        # 1. 随机生成节点数 (针对 2026 年初期 FTQC 研究，建议先从 4-10 个比特开始)
        num_nodes = np.random.randint(4, 11)
        
        # 2. 生成 Erdős-Rényi 随机图，概率 p 为 0.5 确保复杂度
        g = nx.erdos_renyi_graph(num_nodes, p=0.5)
        
        # 确保图是连通的，否则 MaxCut 问题会退化
        if not nx.is_connected(g):
            return None

        # 3. 实例化教师模型 (FALQON)
        # alpha=0.5 是文献 [3] 推荐的步长，有助于捕捉“峰-尾”结构
        falqon = FALQON(g, alpha=0.5)
        
        # 4. 运行演化，获取 30 层的参数序列
        # 增加层数有助于 Transformer 学习长序列规律
        betas, energies = falqon.train(max_layers=30)
        
        # 5. 返回结果字典
        return {
            "node_count": num_nodes,
            "adj": nx.to_numpy_array(g), # 图的邻接矩阵（Transformer 的输入）
            "betas": np.array(betas),    # 最优参数曲线（Transformer 的标签）
            "energies": np.array(energies)
        }
    except Exception as e:
        return None

def main():
    # 设定生成规模
    num_samples = 10000  # 对应 2026 开发协议中的数据规模需求
    output_dir = "data/raw/dataset_v1"
    os.makedirs(output_dir, exist_ok=True)

    print(f"🚀 启动并行数据工厂...")
    print(f"目标：生成 {num_samples} 个 FALQON 教师样本")
    
    # 获取 CPU 核心数，保留 1-2 个核心以防服务器死机
    num_cores = 32
    print(f"使用核心数: {num_cores}")

    results = []
    # 使用进程池进行大规模并行计算
    with multiprocessing.Pool(processes=num_cores) as pool:
        # 使用 tqdm 显示进度条
        for res in tqdm(pool.imap_unordered(generate_single_instance, range(num_samples)), total=num_samples):
            if res is not None:
                results.append(res)
    
    # 6. 保存为压缩格式，节省服务器磁盘空间
    save_path = f"{output_dir}/train_data.npz"
    np.savez_compressed(save_path, data=results)
    
    print(f"\n✅ 数据集生成完成！")
    print(f"有效样本数: {len(results)}")
    print(f"保存路径: {save_path}")

if __name__ == "__main__":
    main()
