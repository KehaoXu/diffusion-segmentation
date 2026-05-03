import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from monai.data import CacheDataset, DataLoader
from monai.inferers import sliding_window_inference
from tqdm import tqdm

from seg_training.data import CACHE_RATE, VAL_BATCH_SIZE, build_transforms, load_split
from seg_training.engine import ROI_SIZE, SW_BATCH_SIZE
from seg_training.model import build_model


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one trained 3D UNet checkpoint on the validation split.")
    parser.add_argument("--checkpoint", required=True, help="Path to one checkpoint file.")
    parser.add_argument("--split-file", required=True, help="Path to the saved train/val split CSV.")
    parser.add_argument(
        "--path-prefix",
        default=None,
        help="Optional prefix prepended to relative image/label paths from the split CSV.",
    )
    parser.add_argument("--metrics-json", default="eval_results/metrics.json", help="Path to save metrics JSON.")
    parser.add_argument("--vis-dir", default="eval_results/visualizations", help="Directory for visualization PNGs.")
    parser.add_argument("--vis-count", type=int, default=5, help="Number of validation samples to visualize.")
    return parser


def resolve_checkpoint(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.exists():
        return path.resolve()

    matches = sorted(Path().glob(path_text))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise ValueError(f"--checkpoint matched multiple files; pass one exact path: {path_text}")
    raise FileNotFoundError(f"Checkpoint not found: {path_text}")


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def auto_workers() -> int:
    return min(4, os.cpu_count() or 1)


def create_val_loader(split_file: Path, workers: int, path_prefix: Optional[Path] = None):
    _, val_data = load_split(split_file, path_prefix=path_prefix)
    _, _, val_tf, post_tf = build_transforms()
    print(f"Preparing validation cache for {len(val_data)} samples...")
    val_ds = CacheDataset(
        data=val_data,
        transform=val_tf,
        cache_rate=CACHE_RATE,
        num_workers=workers,
        progress=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
    )
    return val_loader, post_tf


def extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Could not find model weights in checkpoint.")


def adapt_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
    model_keys: Iterable[str],
) -> Dict[str, torch.Tensor]:
    model_keys = set(model_keys)
    if any(key in model_keys for key in state_dict):
        return state_dict

    for prefix in ("module.", "model.", "net.", "unet.", "backbone."):
        stripped = {
            key[len(prefix) :] if key.startswith(prefix) else key: value
            for key, value in state_dict.items()
        }
        if any(key in model_keys for key in stripped):
            return stripped
    return state_dict


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = build_model()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = adapt_state_dict_keys(extract_state_dict(checkpoint), model.state_dict().keys())
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys when loading checkpoint: {unexpected}")
    model.to(device)
    model.eval()
    return model


def sample_name_from_item(item: Dict[str, object], fallback: str) -> str:
    image_path = str(item.get("image", fallback))
    name = Path(image_path).name
    for suffix in (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    name = "".join(char if char in allowed else "_" for char in name).strip("._")
    return name or fallback


def select_visual_indices(num_samples: int, vis_count: int) -> List[int]:
    if vis_count <= 0 or num_samples <= 0:
        return []
    count = min(vis_count, num_samples)
    return sorted(random.SystemRandom().sample(range(num_samples), count))


def compute_binary_metrics(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> Tuple[float, float]:
    pred = pred.float().reshape(pred.shape[0], -1)
    target = target.float().reshape(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    pred_sum = pred.sum(dim=1)
    target_sum = target.sum(dim=1)
    union = pred_sum + target_sum - intersection
    empty = (pred_sum + target_sum) == 0

    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    dice = torch.where(empty, torch.ones_like(dice), dice)
    iou = torch.where(empty, torch.ones_like(iou), iou)
    return float(dice.mean().item()), float(iou.mean().item())


def mean_and_sample_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        raise RuntimeError("No validation samples were evaluated.")
    tensor = torch.tensor(values, dtype=torch.float64)
    mean = float(tensor.mean().item())
    std = float(tensor.std(unbiased=True).item()) if len(values) > 1 else 0.0
    return mean, std


def format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f}±{std:.4f}"


def select_volume_for_display(tensor: torch.Tensor, is_image: bool) -> torch.Tensor:
    sample = tensor.detach().cpu()[0]
    if sample.ndim == 4:
        if is_image or sample.shape[0] == 1:
            return sample[0]
        return sample[1]
    return sample


def central_slice(volume: torch.Tensor) -> torch.Tensor:
    return volume[:, :, volume.shape[-1] // 2]


def normalize_for_display(image_slice: torch.Tensor) -> torch.Tensor:
    image_slice = image_slice.float()
    min_value = image_slice.min()
    max_value = image_slice.max()
    if float(max_value - min_value) < 1e-8:
        return torch.zeros_like(image_slice)
    return (image_slice - min_value) / (max_value - min_value)


def save_visualization(
    image: torch.Tensor,
    label: torch.Tensor,
    pred: torch.Tensor,
    sample_name: str,
    vis_dir: Path,
) -> str:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required when --vis-count is greater than 0.") from exc

    vis_dir.mkdir(parents=True, exist_ok=True)
    image_slice = normalize_for_display(central_slice(select_volume_for_display(image, is_image=True))).numpy()
    label_slice = central_slice(select_volume_for_display(label, is_image=False)).numpy()
    pred_slice = central_slice(select_volume_for_display(pred, is_image=False)).numpy()

    fig, axis = plt.subplots(1, 1, figsize=(6, 6))
    axis.imshow(image_slice, cmap="gray")
    axis.imshow(label_slice, cmap="Greens", alpha=0.35, vmin=0.0, vmax=1.0)
    axis.imshow(pred_slice, cmap="Reds", alpha=0.35, vmin=0.0, vmax=1.0)
    axis.set_title("Image + GT(green) + Pred(red)")
    axis.axis("off")
    fig.tight_layout()

    output_path = vis_dir / f"{sample_name}.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def evaluate(
    model: torch.nn.Module,
    val_loader,
    post_tf,
    device: torch.device,
    vis_dir: Path,
    vis_count: int,
) -> Dict[str, object]:
    visual_indices = set(select_visual_indices(len(val_loader.dataset), vis_count))
    sample_metrics: List[Dict[str, object]] = []
    visualizations: List[str] = []

    with torch.no_grad():
        progress = tqdm(
            val_loader,
            desc="Evaluating validation set",
            total=len(val_loader),
            unit="volume",
            dynamic_ncols=True,
        )
        for sample_index, batch in enumerate(progress):
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            logits = sliding_window_inference(image, ROI_SIZE, SW_BATCH_SIZE, model)
            pred = post_tf(logits)
            dice, iou = compute_binary_metrics(pred, label)

            sample_name = sample_name_from_item(val_loader.dataset.data[sample_index], f"sample_{sample_index:04d}")
            sample_metrics.append(
                {
                    "index": sample_index,
                    "sample": sample_name,
                    "image": str(val_loader.dataset.data[sample_index].get("image", "")),
                    "label": str(val_loader.dataset.data[sample_index].get("label", "")),
                    "dice": dice,
                    "iou": iou,
                }
            )
            running_dice = sum(item["dice"] for item in sample_metrics) / len(sample_metrics)
            running_iou = sum(item["iou"] for item in sample_metrics) / len(sample_metrics)
            progress.set_postfix(
                {
                    "sample": sample_name,
                    "dice": f"{dice:.4f}",
                    "iou": f"{iou:.4f}",
                    "mean_dice": f"{running_dice:.4f}",
                    "mean_iou": f"{running_iou:.4f}",
                }
            )

            if sample_index in visual_indices:
                visualizations.append(
                    save_visualization(
                        image=image,
                        label=label,
                        pred=pred,
                        sample_name=sample_name,
                        vis_dir=vis_dir,
                    )
                )

    dice_mean, dice_std = mean_and_sample_std([item["dice"] for item in sample_metrics])
    iou_mean, iou_std = mean_and_sample_std([item["iou"] for item in sample_metrics])
    return {
        "dice": format_mean_std(dice_mean, dice_std),
        "iou": format_mean_std(iou_mean, iou_std),
        "dice_mean": dice_mean,
        "dice_std": dice_std,
        "iou_mean": iou_mean,
        "iou_std": iou_std,
        "num_samples": len(sample_metrics),
        "samples": sample_metrics,
        "visualizations": visualizations,
    }


def main() -> None:
    args = build_argparser().parse_args()
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    split_file = Path(args.split_file).expanduser().resolve()
    path_prefix = Path(args.path_prefix).expanduser() if args.path_prefix else None
    metrics_path = Path(args.metrics_json).expanduser().resolve()
    vis_dir = Path(args.vis_dir).expanduser().resolve()
    device = select_device()
    workers = auto_workers()

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split file: {split_file}")
    if path_prefix is not None:
        print(f"Path prefix: {path_prefix}")
    if device.type == "cuda":
        print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("Device: cpu")
    print(f"Workers: {workers}")

    val_loader, post_tf = create_val_loader(split_file, workers, path_prefix=path_prefix)
    model = load_model(checkpoint_path, device)
    results = evaluate(
        model=model,
        val_loader=val_loader,
        post_tf=post_tf,
        device=device,
        vis_dir=vis_dir,
        vis_count=args.vis_count,
    )

    payload = {
        "checkpoint": str(checkpoint_path),
        "split_file": str(split_file),
        "path_prefix": str(path_prefix) if path_prefix is not None else None,
        "device": str(device),
        "workers": workers,
        **results,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Num samples: {results['num_samples']}")
    print(f"Dice: {results['dice']}")
    print(f"IoU: {results['iou']}")
    print(f"Metrics JSON: {metrics_path}")
    if args.vis_count > 0:
        print(f"Visualizations: {vis_dir}")


if __name__ == "__main__":
    main()
