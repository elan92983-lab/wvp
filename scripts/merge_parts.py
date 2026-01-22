import numpy as np
import glob
import os

def main():
    parts_dir = "data/raw/dataset_v1/parts"
    output_file = "data/raw/dataset_v1/train_data_final.npz"
    
    # 找到所有 part_*.npz 文件
    files = sorted(glob.glob(os.path.join(parts_dir, "part_*.npz")))
    files = [f for f in files if 0 <= int(os.path.basename(f).split("_")[1].split(".")[0]) <= 19]
    print(f"找到 {len(files)} 个分片文件(0-19)，开始合并...")
    
    if len(files) == 0:
        print("❌ 没找到数据文件，请检查任务是否运行成功！")
        return

    all_results = []
    
    for f in files:
        try:
            with np.load(f, allow_pickle=True) as data:
                # 读取 'data' 键对应的内容
                chunk = data['data']
                all_results.extend(chunk)
                print(f"已合并: {f} (当前总数: {len(all_results)})")
        except Exception as e:
            print(f"⚠️ 读取 {f} 失败: {e}")

    print(f"\n✅ 合并完成！")
    print(f"最终样本总数: {len(all_results)}")
    
    # 保存最终大文件
    np.savez_compressed(output_file, data=all_results)
    print(f"文件已保存至: {output_file}")

if __name__ == "__main__":
    main()
