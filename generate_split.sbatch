#!/bin/bash
#SBATCH --job-name=seg-split
#SBATCH --partition=a100
#SBATCH --qos=normal
#SBATCH --account=peirong
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.bootstrap.out
#SBATCH --error=logs/%x-%j.bootstrap.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/kxu56/seg}"
mkdir -p logs output1

LOG_OUT="logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
LOG_ERR="logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"

exec >"${LOG_OUT}" 2>"${LOG_ERR}"

module load anaconda3
conda activate monai

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Working dir: $(pwd)"
echo "Python: $(which python)"
echo "Dataset: atlas"
echo "Real root: /scratch/peirong/kxu56/atlas"
echo "Gen root: USB/assets/uncond_gen"
echo "Split file: splits/atlas_train_val.csv"
echo "Split seed: 42"
echo "Train ratio: 0.8"

python generate_split.py \
  --dataset atlas \
  --real-root /scratch/peirong/kxu56/atlas \
  --gen-root USB/assets/uncond_gen \
  --split-file outputs1/atlas_train_val.csv \
  --seed 42 \
  --train-ratio 0.8 \
  --overwrite
