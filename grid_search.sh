#!/bin/bash
set -euo pipefail

# sbatch --export=ALL,GEN_RATIO="0.2",OUT_DIR="outputs/atlas1+uncond22" train.sbatch

GEN_RATIOS=(
  0.2
)

for gr in "${GEN_RATIOS[@]}"; do
  echo "Submitting train.sbatch with GEN_RATIO=${gr}"
  sbatch --export=ALL,GEN_RATIO="${gr}" train.sbatch
done
