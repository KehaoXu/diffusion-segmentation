#!/bin/bash
#SBATCH --job-name=inspect-val
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --account=peirong
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.bootstrap.out
#SBATCH --error=logs/%x-%j.bootstrap.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/kxu56/seg}"
mkdir -p logs

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
echo "Inspect script: inspect_val_dataset.py"
echo "Config: outputs/atlas1+uncond9/run_config.json"
echo "Real root: /scratch/peirong/kxu56/atlas"
echo "Gen root: USB/assets/uncond_gen"
echo "Count: 10"

python inspect_val_dataset.py \
  --config outputs/atlas1+uncond9/run_config.json \
  --real-root /scratch/peirong/kxu56/atlas \
  --gen-root USB/assets/uncond_gen \
  --count 10 \
  --seed 42 \
  --cache-rate 0 \
  --cache-workers "${SLURM_CPUS_PER_TASK}" \
  --loader-workers 0
