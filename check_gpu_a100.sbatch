#!/bin/bash
#SBATCH --job-name=check-gpu
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --account=pliu53_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/kxu56/seg}"
mkdir -p logs

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Working dir: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo
echo "nvidia-smi:"
nvidia-smi
echo
echo "GPU summary:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
