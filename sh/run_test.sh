#!/bin/bash
#SBATCH --job-name=FALQON_TEST      ### 任务名改为 TEST
#SBATCH --partition=normal
#SBATCH --time=00:20:00             ### 测试只给10分钟，防止死循环浪费钱/资源
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4           ### 测试给4个核足够了
#SBATCH --mem=8GB                   ### 内存给8G足够
#SBATCH -o ./output/test-%j.out     ### 日志输出
#SBATCH -e ./output/test-%j.err

# ==========================================
# 1. 环境准备
# ==========================================
echo "JOB START: $(date)"
echo "Running on node: $(hostname)"

# 激活环境 (根据你的实际路径)
source /home/ztc2025/miniconda3/bin/activate /home/ztc2025/miniconda3/envs/quantum_lab

# 检查 Python 环境
which python
python -c "import qiskit; print('Qiskit version:', qiskit.__version__)"

# ==========================================
# 2. 运行测试
# ==========================================
# 确保目录存在
mkdir -p ./output
mkdir -p ./data/test_mini

# 添加当前目录到 Python 路径，防止找不到 src 模块
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 运行刚才写的测试脚本
# 注意：这里路径是 scripts/test_generate_mini.py
python -u scripts/test_generate_mini.py

echo "JOB END: $(date)"
