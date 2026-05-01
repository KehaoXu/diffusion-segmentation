import argparse
import json
import random
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from monai.data import CacheDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from tqdm import tqdm

from seg_training import TrainConfig, build_model
from seg_training.data import (
    CACHE_RATE,
    VAL_BATCH_SIZE,
    build_transforms,
    load_split,
)
from seg_training.engine import ROI_SIZE, SW_BATCH_SIZE, validate_one_epoch


DEFAULT_VIS_DIR = Path("vis_results")
ALLOWED_CHECKPOINT_DIRS = [
    Path("outputs/atlas1"),
    Path("outputs/atlas1+uncond1"),
    Path("outputs/atlas1+uncond2"),
    Path("outputs/atlas1+uncond3"),
    Path("outputs/atlas1+uncond4"),
    Path("outputs/atlas1+uncond5"),
    Path("outputs/atlas1+uncond6"),
    Path("outputs/atlas1+uncond7"),
    Path("outputs/atlas1+uncond8"),
    Path("outputs/atlas1+uncond9"),
]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Kept for compatibility; evaluation only scans the fixed allowed checkpoint directories.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to run_config.json. If omitted, the script will try to infer it.",
    )
    parser.add_argument("--gen-root", default=None, help="Override synthetic dataset root from config.")
    parser.add_argument("--split-file", default=None, help="Path to a previously saved train/val split.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--loader-workers", type=int, default=None, help="Validation dataloader workers.")
    parser.add_argument("--cache-workers", type=int, default=None, help="CacheDataset workers.")
    parser.add_argument(
        "--vis-count",
        type=int,
        default=5,
        help="Number of validation samples to visualize.",
    )
    parser.add_argument(
        "--vis-dir",
        default=str(DEFAULT_VIS_DIR),
        help="Directory to save visualization images.",
    )
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="Optional path to save metrics as JSON.",
    )
    parser.add_argument(
        "--ckpt_type",
        choices=("best", "last"),
        default="best",
        help="Checkpoint selection rule for each experiment directory.",
    )
    return parser


def infer_run_config_path(checkpoint_path: Path) -> Optional[Path]:
    candidates = [
        checkpoint_path.parent / "run_config.json",
        checkpoint_path.with_name("run_config.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _extract_epoch_number(path: Path) -> int:
    matches = re.findall(r"(\d+)", path.stem)
    if not matches:
        return -1
    return int(matches[-1])


def _select_checkpoint(candidates: Sequence[Path], ckpt_type: str) -> Optional[Path]:
    if not candidates:
        return None

    if ckpt_type == "best":
        filtered = [candidate for candidate in candidates if "best" in candidate.name.lower()]
        if not filtered:
            return None
        return sorted(filtered, key=lambda candidate: (_extract_epoch_number(candidate), candidate.name))[-1]

    return sorted(candidates, key=lambda candidate: (_extract_epoch_number(candidate), candidate.name))[-1]


def list_checkpoints(path: Path, ckpt_type: str) -> List[Path]:
    del path

    checkpoints: List[Path] = []
    for directory in ALLOWED_CHECKPOINT_DIRS:
        resolved_dir = directory.expanduser().resolve()
        if not resolved_dir.is_dir():
            print(f"Warning: checkpoint directory not found, skipping: {resolved_dir}")
            continue

        all_candidates: List[Path] = []
        for pattern in ("*.pt", "*.pth", "*.ckpt"):
            all_candidates.extend(resolved_dir.glob(pattern))

        all_candidates = sorted(set(all_candidates))
        selected = _select_checkpoint(all_candidates, ckpt_type)
        if selected is None:
            print(f"Warning: no {ckpt_type} checkpoint found in {resolved_dir}, skipping.")
            continue

        checkpoints.append(selected)

    if not checkpoints:
        raise FileNotFoundError(f"No valid {ckpt_type} checkpoints found in the allowed directories.")
    return checkpoints


def sanitize_filename(name: str) -> str:
    sanitized = name
    for suffix in (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd"):
        if sanitized.endswith(suffix):
            sanitized = sanitized[: -len(suffix)]
            break

    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    sanitized = "".join(char if char in allowed else "_" for char in sanitized)
    sanitized = sanitized.strip("._")
    return sanitized or "sample"


def sample_name_from_path(path: str) -> str:
    return sanitize_filename(Path(path).name)


def extract_sample_names(dataset, sampled_indices: Sequence[int]) -> Dict[int, str]:
    sample_names: Dict[int, str] = {}
    for sample_index in sampled_indices:
        item = dataset.data[sample_index]
        image_path = item["image"]
        sample_names[sample_index] = sample_name_from_path(str(image_path))
    return sample_names


def create_run_output_dir(base_dir: str) -> Path:
    root = Path(base_dir).expanduser().resolve()
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_config_from_args(
    args: argparse.Namespace,
    checkpoint_path: Optional[Path] = None,
) -> TrainConfig:
    checkpoint_path = (
        checkpoint_path.expanduser().resolve()
        if checkpoint_path is not None
        else Path(args.checkpoint).expanduser().resolve()
    )
    config_path = Path(args.config).expanduser().resolve() if args.config else infer_run_config_path(checkpoint_path)

    config_kwargs: Dict[str, object] = {}
    if config_path is not None and config_path.exists():
        with config_path.open("r") as f:
            payload = json.load(f)
        config_kwargs.update(payload.get("config", {}))
    if args.gen_root is not None:
        config_kwargs["gen_root"] = args.gen_root
    if args.split_file is not None:
        config_kwargs["split_file"] = args.split_file
    if args.device is not None:
        config_kwargs["device"] = args.device
    if args.loader_workers is not None:
        config_kwargs["loader_workers"] = args.loader_workers
    if args.cache_workers is not None:
        config_kwargs["cache_workers"] = args.cache_workers

    valid_keys = set(TrainConfig.__init__.__code__.co_varnames[1:TrainConfig.__init__.__code__.co_argcount])
    config_kwargs = {key: value for key, value in config_kwargs.items() if key in valid_keys}

    config = TrainConfig(**config_kwargs)
    if args.device is None:
        config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return config


def create_val_loader(config: TrainConfig):
    if config.split_file is None:
        raise ValueError("config.split_file must be set. Provide the saved split file for evaluation.")
    _, val_data = load_split(config.split_file)
    _, _, val_tf, post_tf = build_transforms()
    val_ds = CacheDataset(
        data=val_data,
        transform=val_tf,
        cache_rate=CACHE_RATE,
        num_workers=config.cache_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.loader_workers,
    )
    return val_loader, post_tf


def ensure_shared_eval_config(base_config: TrainConfig, config: TrainConfig, checkpoint_path: Path) -> None:
    comparable_fields = (
        "cache_workers",
        "loader_workers",
    )
    mismatches = [
        field
        for field in comparable_fields
        if getattr(base_config, field) != getattr(config, field)
    ]
    if mismatches:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has incompatible validation config fields: {mismatches}"
        )


def _extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Could not find a valid state_dict in the checkpoint.")


def _adapt_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
    model_keys: Iterable[str],
) -> Dict[str, torch.Tensor]:
    model_keys = set(model_keys)
    if set(state_dict.keys()) == model_keys or any(key in model_keys for key in state_dict):
        return state_dict

    prefixes = ("model.", "module.", "net.", "unet.", "backbone.")
    for prefix in prefixes:
        stripped = {}
        changed = False
        for key, value in state_dict.items():
            if key.startswith(prefix):
                stripped[key[len(prefix) :]] = value
                changed = True
            else:
                stripped[key] = value
        if changed and any(key in model_keys for key in stripped):
            return stripped

    return state_dict


def load_model(config: TrainConfig, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = build_model()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    state_dict = _adapt_state_dict_keys(state_dict, model.state_dict().keys())
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys when loading checkpoint: {unexpected}")
    model.to(device)
    model.eval()
    return model


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    eps: float = 1e-8,
) -> Tuple[float, float]:
    pred = pred.float()
    target = target.float()

    if num_classes > 1:
        pred = pred[:, 1:]
        target = target[:, 1:]

    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)

    intersection = (pred * target).sum(dim=1)
    pred_sum = pred.sum(dim=1)
    target_sum = target.sum(dim=1)
    union = pred_sum + target_sum - intersection

    empty_mask = (pred_sum + target_sum) == 0

    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)

    dice = torch.where(empty_mask, torch.ones_like(dice), dice)
    iou = torch.where(empty_mask, torch.ones_like(iou), iou)
    return float(dice.mean().item()), float(iou.mean().item())


def _select_volume_for_display(tensor: torch.Tensor, sample_index: int, is_image: bool) -> torch.Tensor:
    sample = tensor[sample_index].detach().cpu()
    if sample.ndim == 4:
        if is_image:
            return sample[0]
        if sample.shape[0] == 1:
            return sample[0]
        return sample[1]
    return sample


def _central_slice(volume: torch.Tensor) -> torch.Tensor:
    depth_index = volume.shape[-1] // 2
    return volume[:, :, depth_index]


def normalize_for_display(image_slice: torch.Tensor) -> torch.Tensor:
    image_slice = image_slice.float()
    min_value = image_slice.min()
    max_value = image_slice.max()
    if float(max_value - min_value) < 1e-8:
        return torch.zeros_like(image_slice)
    return (image_slice - min_value) / (max_value - min_value)


def visualize(
    image: torch.Tensor,
    label: torch.Tensor,
    pred: torch.Tensor,
    sample_name: str,
    vis_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualization. Install it or run with --vis-count 0."
        ) from exc

    vis_dir.mkdir(parents=True, exist_ok=True)

    image_volume = _select_volume_for_display(image, 0, is_image=True)
    label_volume = _select_volume_for_display(label, 0, is_image=False)
    pred_volume = _select_volume_for_display(pred, 0, is_image=False)

    image_slice = normalize_for_display(_central_slice(image_volume)).numpy()
    label_slice = _central_slice(label_volume).numpy()
    pred_slice = _central_slice(pred_volume).numpy()

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


def sample_visualization_indices(total_samples: int, vis_count: int) -> List[int]:
    if vis_count <= 0 or total_samples <= 0:
        return []
    count = min(vis_count, total_samples)
    rng = random.SystemRandom()
    return sorted(rng.sample(range(total_samples), count))


def collect_visualization_samples(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    roi_size,
    sw_batch_size: int,
    post_trans,
    sampled_indices: Sequence[int],
    sample_names: Dict[int, str],
    vis_dir: Path,
) -> None:
    if not sampled_indices:
        return

    sampled_set = set(sampled_indices)
    global_index = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Sampling visuals"):
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            batch_size = image.shape[0]

            needed_local_indices = [
                local_index
                for local_index in range(batch_size)
                if (global_index + local_index) in sampled_set
            ]

            if needed_local_indices:
                logits = run_model_inference(model, image, device, roi_size, sw_batch_size)
                pred = post_trans(logits)
                target = label

                for local_index in needed_local_indices:
                    sample_index = global_index + local_index
                    sample_name = sample_names[sample_index]
                    image_sample = image[local_index : local_index + 1]
                    label_sample = target[local_index : local_index + 1]
                    pred_sample = pred[local_index : local_index + 1]
                    visualize(
                        image=image_sample,
                        label=label_sample,
                        pred=pred_sample,
                        sample_name=sample_name,
                        vis_dir=vis_dir,
                    )

            global_index += batch_size
            if global_index > max(sampled_indices):
                break


def create_summary_figure(
    sample_name: str,
    model_image_paths: Dict[str, Path],
    summary_dir: Path,
) -> None:
    if not model_image_paths:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualization. Install it or run with --vis-count 0."
        ) from exc

    summary_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(model_image_paths.keys())
    num_models = len(model_names)
    num_cols = min(3, num_models)
    num_rows = (num_models + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

    if hasattr(axes, "ravel"):
        axes_list = list(axes.ravel())
    else:
        axes_list = [axes]

    for axis, model_name in zip(axes_list, model_names):
        image_path = model_image_paths[model_name]
        axis.imshow(plt.imread(image_path))
        axis.set_title(model_name)
        axis.axis("off")

    for axis in axes_list[num_models:]:
        axis.axis("off")

    fig.suptitle(f"{sample_name}: GT(green) vs Pred(red)", fontsize=14)
    fig.tight_layout()
    fig.savefig(summary_dir / f"{sample_name}.png", bbox_inches="tight")
    plt.close(fig)


def run_model_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    roi_size,
    sw_batch_size: int,
):
    image = image.to(device)
    return sliding_window_inference(image, roi_size, sw_batch_size, model)


def compute_dataset_iou(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    roi_size,
    sw_batch_size: int,
    post_trans,
) -> float:
    total_iou = 0.0
    total_samples = 0
    progress = tqdm(val_loader, desc="Computing IoU")

    with torch.no_grad():
        for batch in progress:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            logits = run_model_inference(model, image, device, roi_size, sw_batch_size)
            pred = post_trans(logits)
            batch_iou = compute_metrics(pred, label, num_classes=pred.shape[1])[1]
            batch_size = image.shape[0]
            total_iou += batch_iou * batch_size
            total_samples += batch_size
            progress.set_postfix({"iou": f"{batch_iou:.4f}"})

    if total_samples == 0:
        raise RuntimeError("Validation loader is empty; no samples were evaluated.")
    return total_iou / total_samples


def evaluate(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    roi_size,
    sw_batch_size: int,
    post_trans,
) -> Dict[str, object]:
    loss_fn = DiceCELoss(to_onehot_y=False, sigmoid=True)
    metric = DiceMetric(include_background=False, reduction="mean")
    _, dice = validate_one_epoch(
        model=model,
        loader=val_loader,
        loss_fn=loss_fn,
        metric=metric,
        post_trans=post_trans,
        device=device,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
    )
    iou = compute_dataset_iou(
        model=model,
        val_loader=val_loader,
        device=device,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        post_trans=post_trans,
    )
    return {
        "dice": dice,
        "iou": iou,
        "num_batches": len(val_loader),
        "num_samples": len(val_loader.dataset),
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    model_config: TrainConfig,
    val_loader,
    post_trans,
    args: argparse.Namespace,
    sampled_indices: Sequence[int],
    sample_names: Dict[int, str],
) -> Dict[str, object]:
    device = torch.device(model_config.device if torch.cuda.is_available() else "cpu")
    model = load_model(model_config, checkpoint_path, device)
    results = evaluate(
        model=model,
        val_loader=val_loader,
        device=device,
        roi_size=ROI_SIZE,
        sw_batch_size=SW_BATCH_SIZE,
        post_trans=post_trans,
    )

    model_vis_dir = Path(args.vis_dir).expanduser().resolve() / checkpoint_path.parent.name
    collect_visualization_samples(
        model=model,
        val_loader=val_loader,
        device=device,
        roi_size=ROI_SIZE,
        sw_batch_size=SW_BATCH_SIZE,
        post_trans=post_trans,
        sampled_indices=sampled_indices,
        sample_names=sample_names,
        vis_dir=model_vis_dir,
    )

    results["checkpoint"] = str(checkpoint_path)
    results["vis_dir"] = str(model_vis_dir)
    results["sampled_indices"] = list(sampled_indices)
    return results


def save_model_results(results: Dict[str, object], model_vis_dir: Path) -> None:
    model_vis_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = model_vis_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(results, f, indent=2)


def append_metrics_record(metrics_path: Path, result: Dict[str, object]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if metrics_path.exists():
        with metrics_path.open("r") as f:
            payload = json.load(f)
    else:
        payload = {"meta": {}, "results": {}}

    if "results" not in payload or not isinstance(payload["results"], dict):
        payload = {"meta": payload.get("meta", {}), "results": {}}

    experiment_name = Path(result["checkpoint"]).parent.name
    per_model_result = {
        "checkpoint": result["checkpoint"],
        "vis_dir": result["vis_dir"],
        "dice": result["dice"],
        "iou": result["iou"],
    }
    payload["results"][experiment_name] = per_model_result
    with metrics_path.open("w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = build_argparser().parse_args()
    checkpoint_root = Path(args.checkpoint).expanduser().resolve()
    checkpoint_paths = list_checkpoints(checkpoint_root, args.ckpt_type)
    run_vis_dir = create_run_output_dir(args.vis_dir)
    args.vis_dir = str(run_vis_dir)

    base_config = build_config_from_args(args, checkpoint_path=checkpoint_paths[0])
    val_loader, post_trans = create_val_loader(base_config)
    total_samples = len(val_loader.dataset)
    shared_sampled_indices = sample_visualization_indices(
        total_samples=total_samples,
        vis_count=args.vis_count,
    )
    shared_sample_names = extract_sample_names(val_loader.dataset, shared_sampled_indices)
    processed_models: List[Tuple[str, Path]] = []
    summary_dir = Path(args.vis_dir).expanduser().resolve() / "summary"
    metrics_filename = Path(args.metrics_json).name if args.metrics_json else "eval_results.json"
    metrics_path = summary_dir / metrics_filename

    summary_dir.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as f:
        json.dump(
            {
                "meta": {
                    "num_batches": len(val_loader),
                    "num_samples": total_samples,
                    "sampled_indices": shared_sampled_indices,
                },
                "results": {},
            },
            f,
            indent=2,
        )

    for checkpoint_path in checkpoint_paths:
        print(f"Evaluating checkpoint: {checkpoint_path}")
        model_config = build_config_from_args(args, checkpoint_path=checkpoint_path)
        ensure_shared_eval_config(base_config, model_config, checkpoint_path)
        results = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            model_config=model_config,
            val_loader=val_loader,
            post_trans=post_trans,
            args=args,
            sampled_indices=shared_sampled_indices,
            sample_names=shared_sample_names,
        )
        model_vis_dir = Path(results["vis_dir"])
        save_model_results(results, model_vis_dir)
        append_metrics_record(metrics_path, results)
        processed_models.append((checkpoint_path.parent.name, model_vis_dir))
        print("Validation Results:")
        print(f"Dice: {results['dice']:.4f}")
        print(f"IoU: {results['iou']:.4f}")
        print(f"Visualizations: {results['vis_dir']}")

        if results["num_samples"] != total_samples:
            raise RuntimeError(
                f"Validation sample count mismatch for {checkpoint_path}: "
                f"{results['num_samples']} vs {total_samples}"
            )

    if shared_sampled_indices:
        for sample_index in shared_sampled_indices:
            sample_name = shared_sample_names[sample_index]
            model_image_paths: Dict[str, Path] = {}
            for model_name, model_vis_dir in processed_models:
                image_path = model_vis_dir / f"{sample_name}.png"
                if image_path.exists():
                    model_image_paths[model_name] = image_path
            create_summary_figure(
                sample_name=sample_name,
                model_image_paths=model_image_paths,
                summary_dir=summary_dir,
            )


if __name__ == "__main__":
    main()
