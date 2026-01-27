#!/bin/bash
#SBATCH --job-name=FALQON_Finetune
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH -o ./output/finetune-%j.out
#SBATCH -e ./output/finetune-%j.err

# 环境准备
echo "Job Start: $(date)"
echo "Node: $(hostname)"

source /home/ztc2025/miniconda3/bin/activate /home/ztc2025/miniconda3/envs/quantum_lab
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 确保目录存在
mkdir -p models
mkdir -p output

# 运行微调脚本
# train_finetune.py 会加载 models/spectral_transformer_ep100.pth
python -u train_finetune.py

echo "Job End: $(date)"
