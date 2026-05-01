#!/bin/bash
#SBATCH --job-name=seg
#SBATCH --partition=a100
#SBATCH --qos=normal
#SBATCH --account=peirong
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/kxu56/seg}"
mkdir -p logs

GEN_SEED="${GEN_SEED:-42}"

GEN_RATIOS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
GEN_RATIO="${GEN_RATIOS[SLURM_ARRAY_TASK_ID]}"
GEN_RATIO_SUFFIX="${GEN_RATIO#0.}"
GEN_RATIO_TAG="${GEN_RATIO//./p}"

OUT_DIR="${OUT_DIR:-outputs/gen_seed_${GEN_SEED}/atlas+uncond${GEN_RATIO_SUFFIX}}"
SPLIT_FILE="${SPLIT_FILE:-atlas_train_val.csv}"

module load anaconda3
conda activate monai

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "OUT_DIR: ${OUT_DIR}"
echo "SPLIT_FILE: ${SPLIT_FILE}"
echo "GEN_RATIO: ${GEN_RATIO}"
echo "GEN_SEED: ${GEN_SEED}"
echo "Host: $(hostname)"
echo "Working dir: $(pwd)"
echo "Python: $(which python)"

python train.py \
  --out-dir "${OUT_DIR}" \
  --split-file "${SPLIT_FILE}" \
  --gen-root /scratch/peirong/kxu56/USB/assets/uncond_gen \
  --gen-ratio "${GEN_RATIO}" \
  --gen-seed "${GEN_SEED}" \
  --cache-workers "${SLURM_CPUS_PER_TASK}" \
  --loader-workers "${SLURM_CPUS_PER_TASK}" \
  --use-wandb \
  --wandb-project atlas-seg2 \
  --wandb-run-name "seg-aug-${GEN_SEED}-${GEN_RATIO_TAG}-${SLURM_JOB_ID}" \
  --wandb-mode offline
