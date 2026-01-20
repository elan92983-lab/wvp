import numpy as np
import networkx as nx
import os
import multiprocessing
from tqdm import tqdm
# 假设你的 src 文件夹已经在正确位置
from src.algorithms.falqon_core import FALQON 

def generate_single_instance_test(instance_id):
    """
    测试用的轻量级生成函数
    """
    try:
        # 测试：只用 4-6 个节点，计算快
        num_nodes = np.random.randint(4, 7)
        g = nx.erdos_renyi_graph(num_nodes, p=0.5)
        
        if not nx.is_connected(g):
            return None

        # 测试：层数 max_layers=5，只为验证代码逻辑，不求优化效果
        falqon = FALQON(g, alpha=0.5)
        betas, energies = falqon.train(max_layers=5)
        
        return {
            "node_count": num_nodes,
            "adj": nx.to_numpy_array(g),
            "betas": np.array(betas),
            "energies": np.array(energies)
        }
    except Exception as e:
        # 测试阶段打印错误信息以便调试
        print(f"Error in instance {instance_id}: {e}")
        return None

def main():
    # --- 修改配置区 ---
    num_samples = 20           # 测试只跑 20 个
    num_cores = 2              # 测试只用 2 个核
    output_dir = "data/test_mini" # 存放在单独的测试文件夹
    # ------------------

    os.makedirs(output_dir, exist_ok=True)
    print(f"🚀 开始测试运行: 目标 {num_samples} 个样本, 存入 {output_dir}")

    results = []
    
    # 既然是测试，允许打印更多信息
    print(f"使用 CPU 核心数: {num_cores}")
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        # 使用 tqdm 显示进度
        for res in tqdm(pool.imap_unordered(generate_single_instance_test, range(num_samples)), total=num_samples):
            if res is not None:
                results.append(res)
    
    # 保存测试数据
    save_path = f"{output_dir}/test_data.npz"
    np.savez_compressed(save_path, data=results)
    
    print(f"\n✅ 测试运行成功！")
    print(f"有效样本数: {len(results)}")
    print(f"文件已保存: {save_path}")

if __name__ == "__main__":
    main()
