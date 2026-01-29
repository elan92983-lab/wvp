#!/bin/bash
#SBATCH --job-name=FALQON_Scalability
#SBATCH --partition=normal
#SBATCH --array=0-3            ### Four configs: 0=in_domain,1=mild_extrap,2=strong_extrap,3=extreme_extrap
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16     ### parallelism for graph generation
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH -o ./output/gen/scal_%a.out
#SBATCH -e ./output/gen/scal_%a.err

# 1. Prepare environment
mkdir -p ./output/gen
mkdir -p ./data/scalability_test

source /home/ztc2025/miniconda3/bin/activate /home/ztc2025/miniconda3/envs/quantum_lab
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 2. Map SLURM array task to config name
configs=(in_domain mild_extrap strong_extrap extreme_extrap)
CFG_NAME=${configs[$SLURM_ARRAY_TASK_ID]}

# Optional override via environment variables
SAMPLES_PER_CONFIG=${SAMPLES_PER_CONFIG:-}
MAX_LAYERS=${MAX_LAYERS:-40}
SEED=${SEED:-$SLURM_ARRAY_TASK_ID}
MAX_SIM_QUBITS=${MAX_SIM_QUBITS:-12}  # do not attempt dense simulation above this N

echo "Task $SLURM_ARRAY_TASK_ID running config: $CFG_NAME"

echo "Running: python scripts/generate_scalability_data.py --config $CFG_NAME --output_dir data/scalability_test --max_layers $MAX_LAYERS --seed $SEED ${SAMPLES_PER_CONFIG:+--samples $SAMPLES_PER_CONFIG}"

python -u scripts/generate_scalability_data.py \
    --config $CFG_NAME \
    --output_dir data/scalability_test \
    --max_layers $MAX_LAYERS \
    ${SAMPLES_PER_CONFIG:+--samples $SAMPLES_PER_CONFIG} \
    --seed $SEED \
    --max_sim_qubits $MAX_SIM_QUBITS
