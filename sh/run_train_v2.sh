#!/bin/bash
#SBATCH --job-name=FALQON_Spectral_Train
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH -o ./output/train_v2-%j.out
#SBATCH -e ./output/train_v2-%j.err

# 环境准备
echo "Job Start: $(date)"
echo "Node: $(hostname)"

source /home/ztc2025/miniconda3/bin/activate /home/ztc2025/miniconda3/envs/quantum_lab
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 创建目录
mkdir -p models
mkdir -p output

# 运行训练脚本
# 数据路径默认是 data/processed/spectral_data_v2.npz
python -u train_v2.py \
    --epochs 200 \
    --batch_size 128 \
    --data_path data/processed/spectral_data_v2.npz

echo "Job End: $(date)"
