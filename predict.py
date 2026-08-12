"""Minimal single-volume inference for the trained 3D lesion segmenter."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)

from seg_training.engine import ROI_SIZE, SW_BATCH_SIZE
from seg_training.model import build_model


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict a lesion mask for one MRI NIfTI volume.")
    parser.add_argument("--input", required=True, help="Input MRI .nii or .nii.gz file.")
    parser.add_argument("--output", required=True, help="Output binary mask .nii or .nii.gz file.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser


def choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_weights(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = build_model().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_preprocess() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image"], image_only=False),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            CropForegroundd(keys=["image"], source_key="image", margin=10),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image"]),
        ]
    )


def threshold_prediction(logits: torch.Tensor) -> torch.Tensor:
    """Convert one batched binary prediction to channel-first 3D output."""
    return (torch.sigmoid(logits[0]) > 0.5).float()


def run_prediction(
    input_path: Path,
    output_path: Path,
    model: torch.nn.Module,
    preprocess: Compose,
) -> None:
    item = preprocess({"image": str(input_path)})
    image = item["image"].unsqueeze(0).to(next(model.parameters()).device)

    with torch.inference_mode():
        logits = sliding_window_inference(image, ROI_SIZE, SW_BATCH_SIZE, model)
        item["pred"] = threshold_prediction(logits)

    item = Invertd(
        keys=["pred"],
        transform=preprocess,
        orig_keys=["image"],
        nearest_interp=True,
        to_tensor=True,
    )(item)

    pred = item["pred"].detach().cpu().numpy()
    while pred.ndim > 3:
        pred = pred[0]
    affine = np.asarray(item["pred"].meta.get("affine", np.eye(4)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(pred.astype(np.uint8), affine), str(output_path))


def predict(input_path: Path, checkpoint_path: Path, output_path: Path, device_name: str) -> None:
    device = choose_device(device_name)
    model = load_weights(checkpoint_path, device)
    run_prediction(input_path, output_path, model, build_preprocess())
    print(f"Saved mask to {output_path}")


def main() -> None:
    args = build_argparser().parse_args()
    predict(
        Path(args.input),
        Path(args.checkpoint),
        Path(args.output),
        args.device,
    )


if __name__ == "__main__":
    main()
