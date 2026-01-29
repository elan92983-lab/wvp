#!/bin/bash
#SBATCH --job-name=FALQON_Finetune_Best
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH -o ./output/finetune_best-%j.out
#SBATCH -e ./output/finetune_best-%j.err

echo "Job Start: $(date)"
source /home/ztc2025/miniconda3/bin/activate /home/ztc2025/miniconda3/envs/quantum_lab
export PYTHONPATH=$PYTHONPATH:$(pwd)

mkdir -p models output

# 使用之前效果好的配置
python -u train_finetune.py \
    --epochs 80 \
    --lr 3e-4 \
    --pretrained_path models/spectral_transformer_ep200.pth \
    --weight_mse 5.0 \
    --weight_temp 0.5 \
    --weight_tail 2.0 \
    --ss_start 0.0 \
    --ss_end 0.3 \
    --weight_tail_var 0.2

echo "Job End: $(date)"
