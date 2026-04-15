#!/bin/bash
set -euo pipefail

# SPLIT_FILE="${SPLIT_FILE:-splits/atlas_train_val.csv}"

# sbatch --export=ALL,GEN_RATIO="0.0",OUT_DIR="outputs/atlas11",SPLIT_FILE="${SPLIT_FILE}" train.sbatch

# sbatch --export=ALL,GEN_RATIO="0.1",OUT_DIR="outputs/atlas1+uncond11",SPLIT_FILE="${SPLIT_FILE}" train.sbatch

GEN_RATIOS=(
  0.0
  0.1
  0.2
  0.3
  0.4
  0.5
  0.6
  0.7
  0.8
  0.9
)

for gr in "${GEN_RATIOS[@]}"; do
  echo "Submitting train.sbatch with GEN_RATIO=${gr}"
  sbatch --export=ALL,GEN_RATIO="${gr}" train.sbatch
done
