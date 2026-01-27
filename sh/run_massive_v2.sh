#!/bin/bash
#SBATCH --job-name=FALQON_Gen_N20
#SBATCH --partition=normal
#SBATCH --array=2,9,11             ### 只运行未完成的任务 9 和 11
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32       ### 关键：提高并行度，单任务申请更多 CPU
#SBATCH --mem=32GB               ### 进一步增加内存至 32GB
#SBATCH --time=24:00:00          ### 增加到 24 小时以确保完成
#SBATCH -o ./output/gen/part_%a.out 
#SBATCH -e ./output/gen/part_%a.err

# 1. 环境准备
# 确保目录存在
mkdir -p ./output/gen
mkdir -p ./data/raw/dataset_v2

source /home/ztc2025/miniconda3/bin/activate /home/ztc2025/miniconda3/envs/quantum_lab
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 2. 计算当前任务的数据范围
# 每个子任务生成 50 条轨迹，总计 20 * 50 = 1000 条样本
SAMPLES_PER_TASK=50
START_IDX=$(( SLURM_ARRAY_TASK_ID * SAMPLES_PER_TASK ))
END_IDX=$(( START_IDX + SAMPLES_PER_TASK ))

echo "Task ID: $SLURM_ARRAY_TASK_ID generating graphs from index $START_IDX to $END_IDX"

# 3. 运行 Python 生成脚本
# generate_dataset_v2.py 现在支持 --start / --end / --part_id / --output_dir / --cores
python -u scripts/generate_dataset_v2.py \
    --start $START_IDX \
    --end $END_IDX \
    --part_id $SLURM_ARRAY_TASK_ID \
    --output_dir ./data/raw/dataset_v2 \
    --cores 32