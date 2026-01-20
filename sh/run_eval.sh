#!/bin/bash
#SBATCH --job-name=FALQON_Eval
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32      ### 申请 32 个核全力加速
#SBATCH --mem=32GB              ### 矩阵运算比较吃内存，给 32G
#SBATCH --time=01:00:00         ### 32核并行跑 100 个 12 节点样本，1小时绝对够了
#SBATCH -o ./output/eval-%j.out
#SBATCH -e ./output/eval-%j.err

source /home/ztc2025/miniconda3/bin/activate /home/ztc2025/miniconda3/envs/quantum_lab
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 运行并行评估脚本
python -u src/evaluate_parallel.py
