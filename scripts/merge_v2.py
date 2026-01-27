import numpy as np
import glob
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Merge dataset_v2 parts.")
    parser.add_argument("--input_dir", type=str, default="data/raw/dataset_v2", help="Directory containing part_*.npz files.")
    parser.add_argument("--output_file", type=str, default="data/processed/spectral_data_v2.npz", help="Path to save the merged file.")
    args = parser.parse_args()

    parts_dir = args.input_dir
    output_file = args.output_file
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 找到所有 part_*.npz 文件
    files = sorted(glob.glob(os.path.join(parts_dir, "part_*.npz")), 
                   key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
    
    print(f"找到 {len(files)} 个分片文件，开始合并...")
    
    if len(files) == 0:
        print(f"❌ 在 {parts_dir} 没找到数据文件，请检查路径！")
        return

    all_results = []
    
    for f in files:
        try:
            # 检查文件大小，避免合并崩溃或不完整的文件
            if os.path.getsize(f) < 1024: # 小于 1KB 可能是空的
                print(f"⚠️ 跳过疑似损坏/过小的文件: {f}")
                continue
                
            with np.load(f, allow_pickle=True) as data:
                if 'data' not in data:
                    print(f"⚠️ 文件 {f} 中缺少 'data' 键，跳过。")
                    continue
                chunk = data['data']
                # 处理 numpy array/list 转换
                if isinstance(chunk, np.ndarray) and chunk.ndim == 0:
                    # 如果是 0 维 array (通常是保存了 list 后的对象类型)
                    chunk = chunk.item()
                
                all_results.extend(chunk)
                print(f"已合并: {f} (当前总数: {len(all_results)})")
        except Exception as e:
            print(f"⚠️ 读取 {f} 失败: {e}")

    if len(all_results) == 0:
        print("❌ 没有合并到任何有效数据。")
        return

    print(f"\n✅ 合并完成！")
    print(f"最终样本总数: {len(all_results)}")
    
    # 保存最终大文件
    np.savez_compressed(output_file, data=all_results)
    print(f"文件已保存至: {output_file}")

if __name__ == "__main__":
    main()
