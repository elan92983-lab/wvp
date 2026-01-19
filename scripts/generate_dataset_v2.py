import numpy as np
import networkx as nx
import os
import multiprocessing
import argparse
from tqdm import tqdm
from src.algorithms.falqon_core import FALQON

def generate_single_instance(instance_id):
    """
    单个样本生成任务：生成随机图并运行 FALQON 获取教师经验数据。
    (逻辑保持不变)
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
        # 在大规模并发时，单个错误不应中断整个进程，返回 None 即可
        return None

def main():
    # --- 修改 1: 引入参数解析，适配 Job Array ---
    parser = argparse.ArgumentParser(description="FALQON Dataset Generator (Parallel)")
    parser.add_argument("--start", type=int, default=0, help="当前任务的起始索引")
    parser.add_argument("--end", type=int, default=100, help="当前任务的结束索引")
    parser.add_argument("--part_id", type=int, default=0, help="当前分片文件的编号 (对应 Slurm Array ID)")
    args = parser.parse_args()

    # --- 修改 2: 动态获取核心数 ---
    # 优先读取 Slurm 分配的核心数，如果没读到（比如本地测试），则默认使用较少的核
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        num_cores = int(slurm_cpus)
    else:
        # 本地运行时保留 2 个核给系统
        num_cores = max(1, multiprocessing.cpu_count() - 2)

    # 计算本任务需要生成的数量
    target_range = range(args.start, args.end)
    num_samples_this_job = len(target_range)

    # --- 修改 3: 输出路径改为子文件夹 ---
    # 数据将保存在 data/raw/dataset_v1/parts/ 目录下
    output_dir = "data/raw/dataset_v1/parts"
    os.makedirs(output_dir, exist_ok=True)

    print(f"🚀 [任务 ID {args.part_id}] 启动: 处理范围 {args.start} -> {args.end} (共 {num_samples_this_job} 个)")
    print(f"🖥️  运行节点: {os.environ.get('SLURMD_NODENAME', 'Localhost')}")
    print(f"🔥 使用核心数: {num_cores}")

    results = []
    
    # 使用进程池进行并行计算
    # 注意：这里只并行处理 target_range 里的这一小部分数据
    with multiprocessing.Pool(processes=num_cores) as pool:
        # 使用 tqdm 显示进度 (如果是 Job Array，日志里的进度条可能会比较多，但不影响运行)
        for res in tqdm(pool.imap_unordered(generate_single_instance, target_range), total=num_samples_this_job):
            if res is not None:
                results.append(res)
    
    # --- 修改 4: 保存为独立的分片文件 ---
    save_path = f"{output_dir}/part_{args.part_id}.npz"
    np.savez_compressed(save_path, data=results)
    
    print(f"\n✅ [任务 ID {args.part_id}] 完成！")
    print(f"📊 有效样本数: {len(results)}")
    print(f"💾 已保存: {save_path}")

if __name__ == "__main__":
    main()