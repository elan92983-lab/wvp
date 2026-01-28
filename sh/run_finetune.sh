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

# Train finetune script
python -u train_finetune.py \
	--epochs 80 \
	--lr 3e-4 \
	--pretrained_path models/spectral_transformer_ep200.pth \
	--weight_mse 5.0 \
	--weight_temp 0.5 \
	--weight_tail 2.0 \
	--ss_start 0.0 \
	--ss_end 0.5 \
	--weight_tail_var 0.2

echo "Job End: $(date)"
