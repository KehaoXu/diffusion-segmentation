#!/bin/bash
#SBATCH --job-name=seg-split
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --account=peirong
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/kxu56/seg}"
mkdir -p logs outputs1

SPLIT_FILE="${SPLIT_FILE:-atlas_train_val2.csv}"

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
echo "Split file: ${SPLIT_FILE}"
echo "Split seed: 42"
echo "Train ratio: 0.8"

python split_train_val.py \
  --real-root /scratch/peirong/kxu56/atlas \
  --split-file "${SPLIT_FILE}" \
  --split-seed 42 \
  --train-ratio 0.8
