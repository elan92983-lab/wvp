#!/bin/bash
#SBATCH --job-name=FALQON_Train_GNN
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1               # 在 cn3(GPU) 上训练更快；如果你想用 CPU 训练可注释这一行
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH -o ./output/train-gnn-%j.out
#SBATCH -e ./output/train-gnn-%j.err

set -euo pipefail

echo "Job Start: $(date)"
echo "Node: $(hostname)"

source /home/ztc2025/miniconda3/bin/activate /home/ztc2025/miniconda3/envs/quantum_lab
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# 可选：检查 GPU
nvidia-smi || true

# 训练并保存到 models/checkpoints/gnn_model.pth
python3 -u -m src.train_gnn \
  --epochs 200 \
  --batch_size 128 \
  --lr 0.001 \
  --num_workers 4 \
  --max_nodes 12 \
  --output_len 30 \
  --hidden_dim 128 \
  --num_layers 3 \
  --dropout 0.1 \
  --save_dir models/checkpoints \
  --save_name gnn_model.pth

echo "Job End: $(date)"
