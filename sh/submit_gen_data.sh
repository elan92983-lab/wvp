#!/bin/bash
#SBATCH --job-name=FALQON_Gen
#SBATCH --partition=normal           ### 使用你确认的 normal 分区
#SBATCH -t 20-00:00                  ### 运行时间限制
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32           ### 申请 32 核并行加速
#SBATCH --mem=64GB                   ### 申请 64GB 内存防止 OOM
#SBATCH -o ./output/%x-%j.out
#SBATCH -e ./output/%x-%j.err

# ==========================================
# 环境激活部分 (针对你的私有 Miniconda 优化)
# ==========================================

# 1. 使用绝对路径初始化 conda 并在脚本中激活环境
# 这样可以确保跳过系统全局的 anaconda 环境，直接进入你的 quantum_lab
source /home/ztc2025/miniconda3/bin/activate /home/ztc2025/miniconda3/envs/quantum_lab

# 2. 验证 Python 路径和 Qiskit 库 (结果会记录在 .err 日志中)
echo "当前 Python 路径: $(which python)"
python -c "import qiskit; print('✅ Qiskit 已成功加载，版本:', qiskit.__version__)" || { echo "❌ 仍找不到 Qiskit，请检查环境安装"; exit 1; }

# ==========================================
# 任务运行部分
# ==========================================

# 1. 确保必要的目录存在
mkdir -p ./output
mkdir -p ./data/raw/dataset_v1

# 2. 设置 PYTHONPATH 确保 Python 能找到 src 文件夹
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 3. 执行数据生成脚本
# 提示：请确保 scripts/generate_dataset.py 里的 num_cores 已改为 32
python -u scripts/generate_dataset.py
