#!/bin/bash
set -euo pipefail

GEN_RATIOS=(
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
