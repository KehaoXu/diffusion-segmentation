#!/bin/bash
#SBATCH --job-name=seg-eval
#SBATCH --partition=a100
#SBATCH --qos=normal
#SBATCH --account=peirong
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.bootstrap.out
#SBATCH --error=logs/%x-%j.bootstrap.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/kxu56/seg}"
mkdir -p logs outputs vis_results

LOG_OUT="logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
LOG_ERR="logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"

exec >"${LOG_OUT}" 2>"${LOG_ERR}"

SPLIT_FILE="${SPLIT_FILE:-splits/atlas_train_val.csv}"

module load anaconda3
conda activate monai

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Working dir: $(pwd)"
echo "Checkpoint source: outputs"
echo "Visualization dir: vis_results"
echo "Metrics json: outputs/eval_results_${SLURM_JOB_ID}.json"
echo "Visible GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "Python: $(which python)"

python eval.py \
  --checkpoint outputs \
  --real-root /scratch/peirong/kxu56/atlas \
  --gen-root USB/assets/uncond_gen \
  --split-file "${SPLIT_FILE}" \
  --vis-count 5 \
  --vis-dir vis_results1 \
  --metrics-json "vis_results1/eval_results_${SLURM_JOB_ID}.json" \
  --cache-workers "${SLURM_CPUS_PER_TASK}" \
  --loader-workers "${SLURM_CPUS_PER_TASK}" \
  --device cuda:0
