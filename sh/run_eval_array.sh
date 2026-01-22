#!/bin/bash
##SBATCH --job-name=FALQON_Eval
##SBATCH --partition=normal
##SBATCH --array=0-19                ### 修改为需要的数组范围
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=32
##SBATCH --mem=8GB
##SBATCH --time=02:00:00
##SBATCH -o ./output/ar_parts/part_%a.out
##SBATCH -e ./output/ar_parts/part_%a.err

# Simple Slurm array runner for evaluate_parallel.py
# Adjust SAMPLES_PER_TASK and ARRAY range as needed.

set -euo pipefail

# Ensure output dir
mkdir -p ./output/ar_parts

# Activate environment (modify to your env)
source /home/ztc2025/miniconda3/bin/activate /home/ztc2025/miniconda3/envs/quantum_lab
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# 每个子任务处理样本数量（根据数据集大小调整）
SAMPLES_PER_TASK=${SAMPLES_PER_TASK:-50}

START_IDX=$(( SLURM_ARRAY_TASK_ID * SAMPLES_PER_TASK ))
END_IDX=$(( START_IDX + SAMPLES_PER_TASK ))

echo "Task ID: $SLURM_ARRAY_TASK_ID processing samples $START_IDX to $END_IDX"

python3 -u -m src.evaluate_parallel \
  --model_path models/checkpoints/best_model.pth \
  --test_data data/raw/dataset_v1/train_data_final.npz \
  --nodes 12 \
  --start ${START_IDX} --end ${END_IDX} \
  --num_workers ${SLURM_CPUS_ON_NODE:-32}

echo "Done: $SLURM_ARRAY_TASK_ID"
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