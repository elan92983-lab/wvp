#!/bin/bash
#SBATCH --job-name=FALQON_Eval_AR
#SBATCH --partition=normal
#SBATCH --array=0-16                  # 830 样本、每任务 50 时：0-16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8GB
#SBATCH --time=12:00:00
#SBATCH --exclude=cn3                 # 避免 GPU 节点（cn3），自动落到 cn1/cn2
#SBATCH -o ./output/ar_parts/eval_%A_%a.out
#SBATCH -e ./output/ar_parts/eval_%A_%a.err

set -euo pipefail

mkdir -p ./output/ar_parts

echo "Job Start: $(date)"
echo "Node: $(hostname)"

source /home/ztc2025/miniconda3/bin/activate /home/ztc2025/miniconda3/envs/quantum_lab
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# 防止每个进程内部再开多线程（否则会严重超卖 CPU）
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 每个数组任务处理多少样本（可在 sbatch 前 export 覆盖）
SAMPLES_PER_TASK=${SAMPLES_PER_TASK:-50}
START_IDX=$(( SLURM_ARRAY_TASK_ID * SAMPLES_PER_TASK ))
END_IDX=$(( START_IDX + SAMPLES_PER_TASK ))

echo "Task ID: $SLURM_ARRAY_TASK_ID processing samples [$START_IDX, $END_IDX)"

# 可选：自动启用 GNN
# 约定默认权重路径：models/checkpoints/gnn_model.pth
if [[ -z "${GNN_MODEL_PATH:-}" ]] && [[ -f "models/checkpoints/gnn_model.pth" ]]; then
  export GNN_MODEL_PATH="models/checkpoints/gnn_model.pth"
fi

EXTRA_ARGS=()
if [[ -n "${GNN_MODEL_PATH:-}" ]]; then
  EXTRA_ARGS+=(--gnn_model_path "${GNN_MODEL_PATH}")
fi

if [[ -n "${GNN_MODEL_PATH:-}" ]]; then
  echo "GNN enabled: ${GNN_MODEL_PATH}"
else
  echo "GNN disabled: no GNN_MODEL_PATH and no models/checkpoints/gnn_model.pth"
fi

python3 -u -m src.evaluate_parallel \
  --model_path models/checkpoints/best_model.pth \
  --test_data data/raw/dataset_v1/train_data_final.npz \
  --nodes 12 \
  --start ${START_IDX} --end ${END_IDX} \
  --num_workers ${SLURM_CPUS_PER_TASK} \
  "${EXTRA_ARGS[@]}"

echo "Job End: $(date)"
