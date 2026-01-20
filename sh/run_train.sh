#!/bin/bash
#SBATCH --job-name=FALQON_Train
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8       ### 给数据加载留够 CPU
#SBATCH --gres=gpu:1            ### 关键：申请 1 块 GPU
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH -o ./output/train-%j.out
#SBATCH -e ./output/train-%j.err

# 1. 环境准备
echo "Job Start: $(date)"
echo "Node: $(hostname)"
source /home/ztc2025/miniconda3/bin/activate /home/ztc2025/miniconda3/envs/quantum_lab
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 2. 检查 GPU 是否可见
nvidia-smi

# 3. 开始训练
# 你可以调整参数，比如 batch_size 或 epochs
python -u src/train.py \
    --epochs 200 \
    --batch_size 128 \
    --lr 0.001

echo "Job End: $(date)"
