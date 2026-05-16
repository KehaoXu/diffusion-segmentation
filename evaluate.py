import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from monai.data import CacheDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)
from tqdm import tqdm

from seg_training.data import CACHE_RATE, VAL_BATCH_SIZE, load_split
from seg_training.engine import ROI_SIZE, SW_BATCH_SIZE
from seg_training.model import build_model


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one or more trained 3D UNet checkpoints on the validation split."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Exact .pt file, directory, or glob pattern "
            "(e.g. outputs/gen_seed_40/atlas+uncond*/unet3d_best_*.pt)."
        ),
    )
    parser.add_argument("--split-file", required=True, help="Path to the saved train/val split CSV.")
    parser.add_argument(
        "--path-prefix",
        default=None,
        help="Optional prefix prepended to relative image/label paths from the split CSV.",
    )
    parser.add_argument(
        "--brain-mask-dir",
        default="atlas/strip_brain_mask",
        help=(
            "Directory of brain mask NIfTI files matched by image basename. "
            "Relative paths are resolved under --path-prefix when given. "
            "Default: atlas/strip_brain_mask."
        ),
    )
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="(Single-checkpoint only) Override output metrics JSON path.",
    )
    parser.add_argument(
        "--vis-dir",
        default=None,
        help="(Single-checkpoint only) Override output visualization directory.",
    )
    parser.add_argument("--vis-count", type=int, default=5, help="Number of validation samples to visualize per Dice bin.")
    parser.add_argument(
        "--dist-plot",
        default=None,
        help='Override distribution plot path. Pass empty string "" to disable.',
    )
    return parser


def resolve_checkpoints(path_text: str) -> List[Path]:
    path = Path(path_text).expanduser()
    if path.exists():
        if path.is_dir():
            matches = sorted(path.glob("*.pt"))
            if not matches:
                raise FileNotFoundError(f"No .pt files found in directory: {path}")
            return [p.resolve() for p in matches]
        return [path.resolve()]
    matches = sorted(Path().glob(path_text))
    if matches:
        return [p.resolve() for p in matches]
    raise FileNotFoundError(f"Checkpoint not found: {path_text}")


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def auto_workers() -> int:
    return min(4, os.cpu_count() or 1)


def _build_val_tf_with_brain_mask() -> Compose:
    all_keys = ["image", "label", "brain_mask"]
    return Compose([
        LoadImaged(keys=all_keys, image_only=False),
        EnsureChannelFirstd(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        Spacingd(keys=all_keys, pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest", "nearest")),
        CropForegroundd(keys=all_keys, source_key="image", margin=10),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=all_keys),
    ])


def create_val_loader(
    split_file: Path,
    workers: int,
    path_prefix: Optional[Path] = None,
    brain_mask_dir: Optional[Path] = None,
):
    _, val_data = load_split(split_file, path_prefix=path_prefix)

    if brain_mask_dir is not None:
        for item in val_data:
            image_name = Path(item["image"]).name
            brain_mask_path = brain_mask_dir / image_name
            if not brain_mask_path.exists():
                raise FileNotFoundError(
                    f"Brain mask not found for sample '{image_name}': {brain_mask_path}"
                )
            item["brain_mask"] = str(brain_mask_path)
        val_tf = _build_val_tf_with_brain_mask()
    else:
        from seg_training.data import build_transforms
        _, _, val_tf, _ = build_transforms()

    post_tf = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

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
            key[len(prefix):] if key.startswith(prefix) else key: value
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


def dice_bin_name(dice: float) -> str:
    bin_index = min(int(dice / 0.2), 4)
    lower = bin_index * 0.2
    upper = lower + 0.2
    return f"dice_{lower:.1f}_{upper:.1f}"


def select_dice_binned_visuals(
    sample_metrics: Sequence[Dict[str, object]],
    vis_count: int,
) -> Dict[int, str]:
    if vis_count <= 0:
        return {}

    bins: Dict[str, List[int]] = {f"dice_{index / 5:.1f}_{(index + 1) / 5:.1f}": [] for index in range(5)}
    for item in sample_metrics:
        dice = float(item["dice"])
        bins[dice_bin_name(dice)].append(int(item["index"]))

    rng = random.SystemRandom()
    selected: Dict[int, str] = {}
    for bin_name, indices in bins.items():
        if not indices:
            continue
        count = min(vis_count, len(indices))
        for sample_index in rng.sample(indices, count):
            selected[sample_index] = bin_name
    return selected


def compute_binary_metric_components(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = pred.float().reshape(pred.shape[0], -1)
    target = target.float().reshape(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    pred_sum = pred.sum(dim=1)
    target_sum = target.sum(dim=1)
    return intersection, pred_sum, target_sum


def compute_binary_metrics(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> Tuple[float, float, float, float]:
    intersection, pred_sum, target_sum = compute_binary_metric_components(pred, target)
    union = pred_sum + target_sum - intersection
    empty = (pred_sum + target_sum) == 0
    pred_empty = pred_sum == 0
    target_empty = target_sum == 0

    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    precision = (intersection + eps) / (pred_sum + eps)
    recall = (intersection + eps) / (target_sum + eps)
    dice = torch.where(empty, torch.ones_like(dice), dice)
    iou = torch.where(empty, torch.ones_like(iou), iou)
    precision = torch.where(empty, torch.ones_like(precision), precision)
    precision = torch.where(pred_empty & ~empty, torch.zeros_like(precision), precision)
    recall = torch.where(empty, torch.ones_like(recall), recall)
    recall = torch.where(target_empty & ~empty, torch.zeros_like(recall), recall)
    return float(dice.mean().item()), float(iou.mean().item()), float(precision.mean().item()), float(recall.mean().item())


def mean_and_sample_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        raise RuntimeError("No validation samples were evaluated.")
    tensor = torch.tensor(values, dtype=torch.float64)
    mean = float(tensor.mean().item())
    std = float(tensor.std(unbiased=True).item()) if len(values) > 1 else 0.0
    return mean, std


def compute_global_binary_metrics(
    intersection: float,
    pred_sum: float,
    target_sum: float,
    eps: float = 1e-8,
) -> Tuple[float, float, float, float]:
    union = pred_sum + target_sum - intersection
    if pred_sum + target_sum == 0:
        return 1.0, 1.0, 1.0, 1.0
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    precision = 0.0 if pred_sum == 0 else (intersection + eps) / (pred_sum + eps)
    recall = 0.0 if target_sum == 0 else (intersection + eps) / (target_sum + eps)
    return float(dice), float(iou), float(precision), float(recall)


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
    dice: float,
    lesion_brain_ratio: Optional[float] = None,
) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
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
    title = f"Image + GT(green) + Pred(red) | Dice={dice:.4f}"
    if lesion_brain_ratio is not None:
        title += f"\nLesion/Brain Ratio={lesion_brain_ratio:.6f}"
    axis.set_title(title)
    axis.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output_path = vis_dir / f"{sample_name}.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def save_metric_distributions(
    sample_metrics: Sequence[Dict[str, object]],
    output_path: Path,
    global_dice: float,
    global_iou: float,
) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to save metric distribution plots.") from exc

    if not sample_metrics:
        raise RuntimeError("No sample metrics are available for plotting.")

    dice_values = [float(item["dice"]) for item in sample_metrics]
    iou_values = [float(item["iou"]) for item in sample_metrics]
    mean_dice = sum(dice_values) / len(dice_values)
    mean_iou = sum(iou_values) / len(iou_values)
    case_indices = list(range(len(sample_metrics)))
    sorted_dice = sorted(dice_values)
    sorted_iou = sorted(iou_values)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    bins = [index / 20 for index in range(21)]
    axes[0, 0].hist(dice_values, bins=bins, color="#4C78A8", alpha=0.8, edgecolor="white")
    axes[0, 0].axvline(global_dice, color="#C43C39", linestyle="--", label=f"global={global_dice:.4f}")
    axes[0, 0].axvline(mean_dice, color="#F28E2B", linestyle="--", label=f"mean={mean_dice:.4f}")
    axes[0, 0].set_title("Dice Distribution")
    axes[0, 0].set_xlabel("Dice")
    axes[0, 0].set_ylabel("Cases")
    axes[0, 0].set_xlim(0.0, 1.0)
    axes[0, 0].legend()

    axes[0, 1].hist(iou_values, bins=bins, color="#59A14F", alpha=0.8, edgecolor="white")
    axes[0, 1].axvline(global_iou, color="#C43C39", linestyle="--", label=f"global={global_iou:.4f}")
    axes[0, 1].axvline(mean_iou, color="#F28E2B", linestyle="--", label=f"mean={mean_iou:.4f}")
    axes[0, 1].set_title("IoU Distribution")
    axes[0, 1].set_xlabel("IoU")
    axes[0, 1].set_ylabel("Cases")
    axes[0, 1].set_xlim(0.0, 1.0)
    axes[0, 1].legend()

    axes[1, 0].boxplot(
        [dice_values, iou_values],
        tick_labels=["Dice", "IoU"],
        showmeans=True,
        patch_artist=True,
        boxprops={"facecolor": "#E6EEF8"},
        medianprops={"color": "#222222"},
        meanprops={"marker": "o", "markerfacecolor": "#C43C39", "markeredgecolor": "#C43C39"},
    )
    axes[1, 0].set_title("Metric Spread")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_ylim(0.0, 1.0)

    axes[1, 1].plot(case_indices, sorted_dice, label="Dice", color="#4C78A8")
    axes[1, 1].plot(case_indices, sorted_iou, label="IoU", color="#59A14F")
    axes[1, 1].set_title("Sorted Per-Case Scores")
    axes[1, 1].set_xlabel("Case rank")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def save_score_vs_ratio_plot(
    sample_metrics: Sequence[Dict[str, object]],
    output_path: Path,
    checkpoint_label: str,
    global_dice: float,
    global_iou: float,
) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to save score vs ratio plots.") from exc

    ratios = [float(item["lesion_brain_ratio"]) for item in sample_metrics]
    dice_values = [float(item["dice"]) for item in sample_metrics]
    iou_values = [float(item["iou"]) for item in sample_metrics]
    mean_dice = sum(dice_values) / len(dice_values)
    mean_iou = sum(iou_values) / len(iou_values)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(ratios, dice_values, alpha=0.75, color="#4C78A8", edgecolors="none")
    axes[0].axhline(global_dice, color="#C43C39", linestyle="--", label=f"global={global_dice:.4f}")
    axes[0].axhline(mean_dice, color="#F28E2B", linestyle="--", label=f"mean={mean_dice:.4f}")
    axes[0].set_xlabel("Lesion / Brain Volume Ratio")
    axes[0].set_ylabel("Dice")
    axes[0].set_title(f"Dice vs Lesion-Brain Ratio\n{checkpoint_label}")
    axes[0].set_xlim(0.0, 0.02)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend()

    axes[1].scatter(ratios, iou_values, alpha=0.75, color="#59A14F", edgecolors="none")
    axes[1].axhline(global_iou, color="#C43C39", linestyle="--", label=f"global={global_iou:.4f}")
    axes[1].axhline(mean_iou, color="#F28E2B", linestyle="--", label=f"mean={mean_iou:.4f}")
    axes[1].set_xlabel("Lesion / Brain Volume Ratio")
    axes[1].set_ylabel("IoU")
    axes[1].set_title(f"IoU vs Lesion-Brain Ratio\n{checkpoint_label}")
    axes[1].set_xlim(0.0, 0.02)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def save_dice_binned_visualizations(
    model: torch.nn.Module,
    val_loader,
    post_tf,
    device: torch.device,
    vis_dir: Path,
    sample_metrics: Sequence[Dict[str, object]],
    vis_count: int,
) -> List[str]:
    selected_bins = select_dice_binned_visuals(sample_metrics, vis_count)
    if not selected_bins:
        return []

    metrics_by_index = {int(item["index"]): item for item in sample_metrics}
    visualizations: List[str] = []

    with torch.no_grad():
        progress = tqdm(
            val_loader,
            desc="Saving Dice-binned visuals",
            total=len(val_loader),
            unit="volume",
            dynamic_ncols=True,
        )
        for sample_index, batch in enumerate(progress):
            bin_name = selected_bins.get(sample_index)
            if bin_name is None:
                continue

            image = batch["image"].to(device)
            label = batch["label"].to(device)
            logits = sliding_window_inference(image, ROI_SIZE, SW_BATCH_SIZE, model)
            pred = post_tf(logits)
            metric = metrics_by_index[sample_index]
            sample_name = str(metric["sample"])
            dice = float(metric["dice"])
            lesion_brain_ratio = metric.get("lesion_brain_ratio")
            visualizations.append(
                save_visualization(
                    image=image,
                    label=label,
                    pred=pred,
                    sample_name=sample_name,
                    vis_dir=vis_dir / bin_name,
                    dice=dice,
                    lesion_brain_ratio=(
                        float(lesion_brain_ratio)
                        if lesion_brain_ratio is not None
                        else None
                    ),
                )
            )
            del selected_bins[sample_index]
            if not selected_bins:
                break

    return visualizations


def evaluate(
    model: torch.nn.Module,
    val_loader,
    post_tf,
    device: torch.device,
    vis_dir: Path,
    vis_count: int,
    dist_plot_path: Optional[Path],
    ratio_plot_path: Optional[Path],
    checkpoint_label: str,
) -> Dict[str, object]:
    has_brain_mask = "brain_mask" in val_loader.dataset.data[0]
    sample_metrics: List[Dict[str, object]] = []
    global_intersection = 0.0
    global_pred_sum = 0.0
    global_target_sum = 0.0

    with torch.no_grad():
        progress = tqdm(
            val_loader,
            desc="Evaluating",
            total=len(val_loader),
            unit="volume",
            dynamic_ncols=True,
        )
        for sample_index, batch in enumerate(progress):
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            logits = sliding_window_inference(image, ROI_SIZE, SW_BATCH_SIZE, model)
            pred = post_tf(logits)
            dice, iou, precision, recall = compute_binary_metrics(pred, label)
            intersections, pred_sums, target_sums = compute_binary_metric_components(pred, label)
            global_intersection += float(intersections.sum().item())
            global_pred_sum += float(pred_sums.sum().item())
            global_target_sum += float(target_sums.sum().item())

            sample_name = sample_name_from_item(val_loader.dataset.data[sample_index], f"sample_{sample_index:04d}")

            lesion_voxels = float((label > 0.5).sum().item())
            if has_brain_mask:
                brain_mask = batch["brain_mask"].to(device)
                brain_mask_voxels = float((brain_mask > 0.5).sum().item())
                if brain_mask_voxels == 0:
                    bm_path = val_loader.dataset.data[sample_index].get("brain_mask", "unknown")
                    raise RuntimeError(f"Empty brain mask for sample '{sample_name}': {bm_path}")
                lesion_brain_ratio = lesion_voxels / brain_mask_voxels
                brain_mask_path = str(val_loader.dataset.data[sample_index].get("brain_mask", ""))
            else:
                brain_mask_voxels = None
                lesion_brain_ratio = None
                brain_mask_path = None

            sample_metrics.append(
                {
                    "index": sample_index,
                    "sample": sample_name,
                    "image": str(val_loader.dataset.data[sample_index].get("image", "")),
                    "label": str(val_loader.dataset.data[sample_index].get("label", "")),
                    "dice": dice,
                    "iou": iou,
                    "precision": precision,
                    "recall": recall,
                    "lesion_voxels": lesion_voxels,
                    "brain_mask_voxels": brain_mask_voxels,
                    "lesion_brain_ratio": lesion_brain_ratio,
                    "brain_mask_path": brain_mask_path,
                }
            )
            running_dice = sum(item["dice"] for item in sample_metrics) / len(sample_metrics)
            running_precision = sum(item["precision"] for item in sample_metrics) / len(sample_metrics)
            running_recall = sum(item["recall"] for item in sample_metrics) / len(sample_metrics)
            running_global_dice, *_ = compute_global_binary_metrics(
                global_intersection,
                global_pred_sum,
                global_target_sum,
            )
            progress.set_postfix(
                {
                    "sample": sample_name,
                    "dice": f"{dice:.4f}",
                    "prec": f"{precision:.4f}",
                    "rec": f"{recall:.4f}",
                    "mean_dice": f"{running_dice:.4f}",
                    "global_dice": f"{running_global_dice:.4f}",
                    "mean_prec": f"{running_precision:.4f}",
                    "mean_rec": f"{running_recall:.4f}",
                }
            )

    dice_mean, dice_std = mean_and_sample_std([item["dice"] for item in sample_metrics])
    iou_mean, iou_std = mean_and_sample_std([item["iou"] for item in sample_metrics])
    precision_mean, precision_std = mean_and_sample_std([item["precision"] for item in sample_metrics])
    recall_mean, recall_std = mean_and_sample_std([item["recall"] for item in sample_metrics])
    global_dice, global_iou, global_precision, global_recall = compute_global_binary_metrics(
        global_intersection,
        global_pred_sum,
        global_target_sum,
    )
    visualizations = save_dice_binned_visualizations(
        model=model,
        val_loader=val_loader,
        post_tf=post_tf,
        device=device,
        vis_dir=vis_dir,
        sample_metrics=sample_metrics,
        vis_count=vis_count,
    )
    metric_distribution = (
        save_metric_distributions(sample_metrics, dist_plot_path, global_dice, global_iou)
        if dist_plot_path is not None
        else None
    )
    ratio_plot = (
        save_score_vs_ratio_plot(sample_metrics, ratio_plot_path, checkpoint_label, global_dice, global_iou)
        if ratio_plot_path is not None and has_brain_mask
        else None
    )
    return {
        "mean_dice": dice_mean,
        "mean_dice_std": dice_std,
        "mean_dice_text": format_mean_std(dice_mean, dice_std),
        "mean_iou": iou_mean,
        "mean_iou_std": iou_std,
        "mean_iou_text": format_mean_std(iou_mean, iou_std),
        "mean_precision": precision_mean,
        "mean_precision_std": precision_std,
        "mean_precision_text": format_mean_std(precision_mean, precision_std),
        "mean_recall": recall_mean,
        "mean_recall_std": recall_std,
        "mean_recall_text": format_mean_std(recall_mean, recall_std),
        "global_dice": global_dice,
        "global_iou": global_iou,
        "global_precision": global_precision,
        "global_recall": global_recall,
        "global_intersection": global_intersection,
        "global_pred_voxels": global_pred_sum,
        "global_target_voxels": global_target_sum,
        "dice": format_mean_std(dice_mean, dice_std),
        "iou": format_mean_std(iou_mean, iou_std),
        "precision": format_mean_std(precision_mean, precision_std),
        "recall": format_mean_std(recall_mean, recall_std),
        "dice_mean": dice_mean,
        "dice_std": dice_std,
        "iou_mean": iou_mean,
        "iou_std": iou_std,
        "precision_mean": precision_mean,
        "precision_std": precision_std,
        "recall_mean": recall_mean,
        "recall_std": recall_std,
        "num_samples": len(sample_metrics),
        "samples": sample_metrics,
        "visualizations": visualizations,
        "metric_distribution": metric_distribution,
        "ratio_plot": ratio_plot,
    }


def main() -> None:
    args = build_argparser().parse_args()
    checkpoints = resolve_checkpoints(args.checkpoint)
    split_file = Path(args.split_file).expanduser().resolve()
    path_prefix = Path(args.path_prefix).expanduser() if args.path_prefix else None
    device = select_device()
    workers = auto_workers()

    brain_mask_dir_arg = Path(args.brain_mask_dir)
    if not brain_mask_dir_arg.is_absolute() and path_prefix is not None:
        brain_mask_dir = (path_prefix / brain_mask_dir_arg).resolve()
    else:
        brain_mask_dir = brain_mask_dir_arg.expanduser().resolve()

    val_loader, post_tf = create_val_loader(
        split_file, workers, path_prefix=path_prefix, brain_mask_dir=brain_mask_dir
    )

    for checkpoint_path in checkpoints:
        print(f"Checkpoint: {checkpoint_path}")
        ckpt_dir = checkpoint_path.parent
        eval_dir = ckpt_dir / "eval"

        metrics_path = eval_dir / "eval_metrics.json"
        vis_dir = eval_dir / "visualizations"
        ratio_plot_path: Optional[Path] = eval_dir / "score_vs_lesion_brain_ratio.png"

        if args.dist_plot == "":
            dist_plot_path: Optional[Path] = None
        else:
            dist_plot_path = eval_dir / "metric_distributions.png"

        # single-checkpoint path overrides for backward compat
        if len(checkpoints) == 1:
            if args.metrics_json is not None:
                metrics_path = Path(args.metrics_json).expanduser().resolve()
            if args.vis_dir is not None:
                vis_dir = Path(args.vis_dir).expanduser().resolve()
            if args.dist_plot not in (None, ""):
                dist_plot_path = Path(args.dist_plot).expanduser().resolve()

        model = load_model(checkpoint_path, device)
        results = evaluate(
            model=model,
            val_loader=val_loader,
            post_tf=post_tf,
            device=device,
            vis_dir=vis_dir,
            vis_count=args.vis_count,
            dist_plot_path=dist_plot_path,
            ratio_plot_path=ratio_plot_path,
            checkpoint_label=ckpt_dir.name,
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


if __name__ == "__main__":
    main()
