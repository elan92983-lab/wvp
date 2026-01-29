#!/bin/bash
#SBATCH --job-name=FALQON_Noise
#SBATCH --partition=normal
#SBATCH --array=0-4            ### Five noise configs: 0=no_noise,1=low_noise,2=medium_noise,3=high_noise,4=extreme_noise
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH -o ./output/gen/noise_%a.out
#SBATCH -e ./output/gen/noise_%a.err

mkdir -p ./output/gen
mkdir -p ./data/noise_test

source /home/ztc2025/miniconda3/bin/activate /home/ztc2025/miniconda3/envs/quantum_lab
export PYTHONPATH=$PYTHONPATH:$(pwd)

configs=(no_noise low_noise medium_noise high_noise extreme_noise)
CFG_NAME=${configs[$SLURM_ARRAY_TASK_ID]}

SAMPLES_PER_CONFIG=${SAMPLES_PER_CONFIG:-50}
N_MIN=${N_MIN:-6}
N_MAX=${N_MAX:-10}
MAX_LAYERS=${MAX_LAYERS:-40}
SEED=${SEED:-$SLURM_ARRAY_TASK_ID}

echo "Task $SLURM_ARRAY_TASK_ID running noise config: $CFG_NAME"

python -u scripts/generate_noisy_data.py \
    --noise $CFG_NAME \
    --samples-per-config $SAMPLES_PER_CONFIG \
    --n-min $N_MIN --n-max $N_MAX \
    --output-dir data/noise_test \
    --max-layers $MAX_LAYERS \
    --seed $SEED
