#!/bin/bash
#SBATCH --job-name=atlas-seg
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --account=peirong_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/kxu56/seg}"
mkdir -p logs outputs

# Load your Python environment here.
# Pick one setup that matches your cluster.
#
# If your cluster uses conda directly:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate seg
#
# If your cluster uses environment modules:
# module load miniconda3
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate seg
#
# Optional: for online W&B runs, export your API key before `python train.py`.
# export WANDB_API_KEY=your_api_key

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
  --out-dir outputs/atlas1 \
  --real-root atlas \
  --gen-root uncond_gen \
  --gen-ratio 0 \
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
  --wandb-run-name "atlas-${SLURM_JOB_ID}" \
  --wandb-mode offline

# To switch to online sync later:
# 1. install wandb in your env
# 2. run `wandb login` once on the cluster, or set WANDB_API_KEY
# 3. change `--wandb-mode offline` to `--wandb-mode online`
