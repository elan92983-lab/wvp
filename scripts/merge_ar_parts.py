import argparse
import glob
import os
import re

import numpy as np


def _extract_start_idx(path: str) -> int:
    """Extract start index from filename like part_0.npy or part_0_gnn.npy."""
    name = os.path.basename(path)
    m = re.match(r"part_(\d+)(?:_gnn)?\.npy$", name)
    return int(m.group(1)) if m else 10**18


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge AR parts and compute global Avg AR / Std")
    parser.add_argument(
        "--parts_dir",
        type=str,
        default="output/ar_parts",
        help="目录，包含 part_*.npy",
    )
    parser.add_argument(
        "--kind",
        type=str,
        choices=["transformer", "gnn"],
        default="transformer",
        help="选择合并哪一类结果：transformer=part_数字.npy；gnn=part_数字_gnn.npy",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="可选：自定义文件匹配模式（相对 parts_dir）。提供后将覆盖 --kind 逻辑。",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="可选：保存合并后的 npy 路径，例如 output/ar_parts/merged.npy",
    )
    args = parser.parse_args()

    if args.pattern is not None:
        pattern = args.pattern
    else:
        # NOTE: glob pattern part_[0-9]*.npy matches part_0_gnn.npy too.
        # We will filter strictly in the loop.
        pattern = "part_[0-9]*.npy"

    pattern_path = os.path.join(args.parts_dir, pattern)
    files = sorted(glob.glob(pattern_path), key=_extract_start_idx)
    
    # Strict filtering based on kind
    if args.kind == "transformer":
        # must NOT contain _gnn
        files = [f for f in files if "_gnn" not in os.path.basename(f)]
    elif args.kind == "gnn":
        # must contain _gnn (glob might already handle this if we used *gnn, but safe to enforce)
        files = [f for f in files if "_gnn" in os.path.basename(f)]

    if not files:
        print(f"❌ 没找到分片文件: {pattern_path} (kind={args.kind})")
        return

    all_values: list[np.ndarray] = []
    total = 0

    print(f"找到 {len(files)} 个分片文件，开始合并...")
    for f in files:
        try:
            arr = np.load(f)
            arr = np.asarray(arr, dtype=np.float64).reshape(-1)
            if arr.size == 0:
                print(f"⚠️ 跳过空分片: {f}")
                continue
            all_values.append(arr)
            total += arr.size
            print(f"已读取: {f} (累计 {total})")
        except Exception as e:
            print(f"⚠️ 读取失败: {f}: {e}")

    if total == 0:
        print("❌ 所有分片都为空或读取失败，无法统计")
        return

    merged = np.concatenate(all_values, axis=0)
    avg_ar = float(np.mean(merged))
    std_ar = float(np.std(merged))

    print("\n✅ 全局统计完成")
    print(f"样本数: {merged.size}")
    print(f"Avg AR: {avg_ar:.6f}")
    print(f"Std:   {std_ar:.6f}")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        np.save(args.save, merged)
        print(f"已保存合并结果到: {args.save}")


if __name__ == "__main__":
    main()
