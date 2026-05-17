# Evaluating Synthetic MRI Augmentation for 3D Lesion Segmentation


Training and evaluation code for 3D binary lesion segmentation with a MONAI 3D UNet. The root workflow uses NIfTI data and can optionally add synthetic samples produced by the `USB/` code.

## Repository

```text
.
+-- atlas_train_val.csv       # Example split CSV
+-- split_train_val.py        # Create a fixed train/validation split
+-- train.py                  # Train the segmentation model
+-- evaluate.py               # Evaluate trained checkpoints
+-- seg_training/             # Model, data pipeline, training loop, run logging
+-- USB/                      # Synthetic-data generation support code
```

## Environment

Recommended: Linux/HPC with CUDA.

```bash
conda create -n <env-name> python=3.10
conda activate <env-name>
pip install torch monai numpy scipy tqdm matplotlib wandb
```

`wandb` is only required when training with `--use-wandb`. USB-specific dependencies are listed in `USB/requirements.txt`.

## Data

Real data:

```text
<real-root>/
+-- T1/
|   +-- sub-*.nii.gz
+-- pathology_maps_segmentation/
    +-- sub-*.nii.gz
```

Synthetic data:

```text
<gen-root>/
+-- y0_<case_id>.nii.gz       # image
+-- x0_<case_id>.nii.gz       # label
```

## Split

```bash
python split_train_val.py \
  --real-root <real-root> \
  --split-file atlas_train_val.csv \
  --split-seed 42 \
  --train-ratio 0.8
```

## Train

```bash
python train.py \
  --out-dir <output-dir> \
  --split-file atlas_train_val.csv \
  --gen-root <gen-root> \
  --gen-ratio 0.3 \
  --gen-seed 42 \
  --cache-workers 8 \
  --loader-workers 8 \
  --show-progress
```

`--gen-ratio` is relative to the number of real training samples. For example, `0.3` adds up to 30 synthetic samples for 100 real training samples.

Default training settings: 3D UNet, `128 x 128 x 128` patches, batch size `2`, `100` epochs, `1e-4` learning rate, `DiceCELoss`, and cosine LR scheduling.

## Evaluate

```bash
python evaluate.py \
  --checkpoint "<output-dir>/unet3d_best_*.pt" \
  --split-file atlas_train_val.csv \
  --path-prefix <prefix for relative paths in the split CSV>
```

`evaluate.py` reports Dice, IoU, precision, and recall. If brain masks are available through `--brain-mask-dir`, it also records lesion-to-brain volume ratios.
