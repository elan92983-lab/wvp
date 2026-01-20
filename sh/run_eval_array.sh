#!/bin/bash
#SBATCH --job-name=FALQON_Array
#SBATCH --partition=normal
#SBATCH --array=0-9             ### 启动 10 个子任务 (Index 0-9)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32      ### 每个任务分配 32 核
#SBATCH --mem=32GB
#SBATCH --time=23:30:00
#SBATCH -o ./output/array-%A_%a.out
#SBATCH -e ./output/array-%A_%a.err

# 1. 环境激活
source /home/ztc2025/miniconda3/bin/activate /home/ztc2025/miniconda3/envs/quantum_lab
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 2. 计算当前任务负责的样本起始索引
# SLURM_ARRAY_TASK_ID 会从 0 变到 9
NUM_PER_JOB=10
START_IDX=$((SLURM_ARRAY_TASK_ID * NUM_PER_JOB))

echo "Running task array ID: $SLURM_ARRAY_TASK_ID, starting from index: $START_IDX"

# 3. 运行脚本 (每个任务跑 10 个样本)
python -u src/evaluate_parallel.py $START_IDX $NUM_PER_JOB