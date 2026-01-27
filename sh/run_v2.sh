#!/bin/bash

# 环境准备
mkdir -p ./output/gen
mkdir -p ./data/raw/dataset_v2

# 激活环境 (根据 run_massive_v2.sh 中的路径)
source /home/ztc2025/miniconda3/bin/activate /home/ztc2025/miniconda3/envs/quantum_lab
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "🚀 Starting local data generation..."

# 运行 Python 生成脚本 (默认生成一小部分数据用于检查)
python -u scripts/generate_dataset_v2.py \
    --start 0 \
    --end 10 \
    --part_id 999 \
    --output_dir ./data/raw/dataset_v2 \
    --cores 4

echo "✅ Local generation task finished."
