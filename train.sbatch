#!/bin/bash
#SBATCH --job-name=seg
#SBATCH --partition=a100
#SBATCH --qos=normal
#SBATCH --account=peirong
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.bootstrap.out
#SBATCH --error=logs/%x-%j.bootstrap.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/kxu56/seg}"
mkdir -p logs outputs

GEN_RATIO="${GEN_RATIO:-0.3}"
GEN_RATIO_SUFFIX="${GEN_RATIO#0.}"
OUT_DIR="${OUT_DIR:-outputs/atlas1+uncond${GEN_RATIO_SUFFIX}}"
GEN_RATIO_TAG="${GEN_RATIO//./p}"
LOG_OUT="logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-${GEN_RATIO_TAG}.out"
LOG_ERR="logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-${GEN_RATIO_TAG}.err"

exec >"${LOG_OUT}" 2>"${LOG_ERR}"

# Load your Python environment here.

module load anaconda3
conda activate monai

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Working dir: $(pwd)"
echo "Visible GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "Python: $(which python)"

# train.py is single-process, single-GPU code.
# Keep 1 task and 1 GPU unless you later add DDP/torchrun support.
# For first-time W&B usage, start with offline mode.

python train.py \
  --out-dir "${OUT_DIR}" \
  --real-root atlas \
  --gen-root USB/assets/uncond_gen \
  --gen-ratio "${GEN_RATIO}" \
  --max-epochs 100 \
  --patch-size 128,128,128 \
  --roi-size 128,128,128 \
  --num-samples-per-vol 8 \
  --sw-batch-size 8 \
  --train-batch-size 2 \
  --val-batch-size 1 \
  --cache-workers "${SLURM_CPUS_PER_TASK}" \
  --loader-workers "${SLURM_CPUS_PER_TASK}" \
  --device cuda:0 \
  --use-wandb \
  --wandb-project atlas-seg \
  --wandb-run-name "seg-aug-${SLURM_JOB_ID}" \
  --wandb-mode offline

# To switch to online sync later:
# 1. install wandb in your env
# 2. run `wandb login` once on the cluster, or set WANDB_API_KEY
# 3. change `--wandb-mode offline` to `--wandb-mode online`
