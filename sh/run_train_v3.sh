#!/bin/bash
#SBATCH --job-name=FALQON_V3
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH -o ./output/train_v3-%j.out
#SBATCH -e ./output/train_v3-%j.err

echo "Job Start: $(date)"
source /home/ztc2025/miniconda3/bin/activate /home/ztc2025/miniconda3/envs/quantum_lab
export PYTHONPATH=$PYTHONPATH:$(pwd)

cd /home/ztc2025/plyproj/wvp

mkdir -p models output

python -u train_v3.py \
    --epochs 300 \
    --batch_size 32 \
    --lr 1e-4 \
    --d_model 256 \
    --num_layers 6 \
    --weight_tail 3.0 \
    --ss_start 0.0 \
    --ss_end 0.4 \
    --patience 50

echo "Job End: $(date)"