#!/bin/bash
#SBATCH --job-name=FALQON_Gen_N20
#SBATCH --partition=normal
#SBATCH --array=0-19             ### 启动 20 个子任务
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32       ### 关键：提高并行度，单任务申请更多 CPU
#SBATCH --mem=8GB                ### 增加内存以应对大矩阵运算
#SBATCH --time=12:00:00          ### N=20 生成较慢，限时设为 12 小时
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
# generate_dataset_v2.py 支持的参数为 --start / --end / --part_id
python -u scripts/generate_dataset_v2.py \
    --start $START_IDX \
    --end $END_IDX \
    --part_id $SLURM_ARRAY_TASK_ID