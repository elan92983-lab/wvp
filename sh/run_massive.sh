#!/bin/bash
#SBATCH --job-name=FALQON_Array
#SBATCH --partition=normal
#SBATCH --array=0-99            ### 启动 100 个子任务，编号 0 到 99
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2       ### 每个子任务只占 2 个核 (刚好填满缝隙)
#SBATCH --mem=4GB               ### 每个任务给 4G 内存
#SBATCH --time=04:00:00         ### 限时 4 小时 (跑100个数据很快的)
#SBATCH -o ./output/part_%a.out ### 日志分开存，%a 代表任务编号
#SBATCH -e ./output/part_%a.err

# 1. 环境准备
source /home/ztc2025/miniconda3/bin/activate /home/ztc2025/miniconda3/envs/quantum_lab
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 2. 计算当前任务负责的数据范围
# 每个任务跑 100 个数据
BATCH_SIZE=100
START_IDX=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
END_IDX=$(( START_IDX + BATCH_SIZE ))

echo "Task ID: $SLURM_ARRAY_TASK_ID running range: $START_IDX to $END_IDX on node $(hostname)"

# 3. 运行 Python (传入参数)
python -u scripts/generate_dataset_v2.py \
    --start $START_IDX \
    --end $END_IDX \
    --part_id $SLURM_ARRAY_TASK_ID