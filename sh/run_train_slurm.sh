#!/bin/bash
#SBATCH --job-name=FALQON_Spectral_Train
#SBATCH --partition=normal          ### 请确保该分区名称在您的集群中存在
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           ### 充足的 CPU 用于数据加载 (collate_fn) [cite: 1, 7]
#SBATCH --gres=gpu:1                ### 申请 1 块 GPU 加速训练 [cite: 1, 78]
#SBATCH --mem=16GB                  ### 12 节点规模任务建议 16GB 以上内存 [cite: 1, 19]
#SBATCH --time=24:00:00             ### 预留 24 小时训练时长
#SBATCH -o ./output/train-%j.out    ### 标准输出日志，%j 会替换为作业 ID
#SBATCH -e ./output/train-%j.err    ### 错误日志

# --- 1. 环境准备 ---
echo "Job Start: $(date)"
echo "Node: $(hostname)"
echo "User: $(whoami)"

# 激活您的指定 Conda 环境
source /home/ztc2025/miniconda3/bin/activate /home/ztc2025/miniconda3/envs/quantum_lab

# 将项目根目录加入 PYTHONPATH，确保能正确导入 src 模块 [cite: 1, 25]
export PYTHONPATH=$PYTHONPATH:$(pwd)

# --- 2. 硬件检查 ---
nvidia-smi

# --- 3. 执行训练 ---
# 注意：如果您的 src/train.py 内部没有使用 argparse 接收参数，
# 下面的命令行参数将不会生效，脚本会运行 train() 函数中的默认值。
python -u src/train.py

echo "Job End: $(date)"