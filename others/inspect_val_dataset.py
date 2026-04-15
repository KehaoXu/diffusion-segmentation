import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from seg_training import TrainConfig, create_dataloaders


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print the first N validation samples and check whether they are consistent."
    )
    parser.add_argument("--config", default=None, help="Optional path to run_config.json.")
    parser.add_argument("--real-root", default=None, help="Override real dataset root.")
    parser.add_argument("--gen-root", default=None, help="Override synthetic dataset root.")
    parser.add_argument("--split-file", default=None, help="Path to a previously generated split file.")
    parser.add_argument("--seed", type=int, default=None, help="Override dataset split seed.")
    parser.add_argument("--count", type=int, default=10, help="How many validation samples to inspect.")
    parser.add_argument("--cache-rate", type=float, default=0.0, help="CacheDataset cache rate.")
    parser.add_argument("--cache-workers", type=int, default=0, help="CacheDataset worker count.")
    parser.add_argument("--loader-workers", type=int, default=0, help="DataLoader worker count.")
    return parser


def load_config(args: argparse.Namespace) -> TrainConfig:
    config_kwargs: Dict[str, Any] = {}
    if args.config is not None:
        with Path(args.config).expanduser().open("r") as f:
            payload = json.load(f)
        config_kwargs.update(payload.get("config", {}))

    tuple_keys = {"patch_size", "roi_size", "channels", "strides", "spacing"}
    for key in tuple_keys:
        if key in config_kwargs and isinstance(config_kwargs[key], list):
            config_kwargs[key] = tuple(config_kwargs[key])

    if args.real_root is not None:
        config_kwargs["real_root"] = args.real_root
    if args.gen_root is not None:
        config_kwargs["gen_root"] = args.gen_root
    if args.split_file is not None:
        config_kwargs["split_file"] = args.split_file
    if args.seed is not None:
        config_kwargs["seed"] = args.seed

    config_kwargs["cache_rate"] = args.cache_rate
    config_kwargs["cache_workers"] = args.cache_workers
    config_kwargs["loader_workers"] = args.loader_workers
    config_kwargs["train_batch_size"] = 1
    config_kwargs["val_batch_size"] = 1
    config_kwargs["device"] = "cpu"
    return TrainConfig(**config_kwargs)


def tensor_digest(value: Any) -> Optional[str]:
    if not torch.is_tensor(value):
        return None
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha1(tensor.numpy().tobytes()).hexdigest()[:12]


def summarize_sample(dataset, index: int) -> Dict[str, Any]:
    raw_item = dataset.data[index]
    sample = dataset[index]

    image = sample["image"]
    label = sample["label"]
    image_sum = float(image.sum().item()) if torch.is_tensor(image) else None
    label_sum = float(label.sum().item()) if torch.is_tensor(label) else None

    return {
        "index": index,
        "image_path": str(raw_item["image"]),
        "label_path": str(raw_item["label"]),
        "image_shape": tuple(image.shape) if torch.is_tensor(image) else None,
        "label_shape": tuple(label.shape) if torch.is_tensor(label) else None,
        "image_sum": image_sum,
        "label_sum": label_sum,
        "image_digest": tensor_digest(image),
        "label_digest": tensor_digest(label),
    }


def compare_summary(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    keys = (
        "image_path",
        "label_path",
        "image_shape",
        "label_shape",
        "image_sum",
        "label_sum",
        "image_digest",
        "label_digest",
    )
    return all(a[key] == b[key] for key in keys)


def main() -> None:
    args = build_argparser().parse_args()
    config = load_config(args)

    _, val_loader_a, _ = create_dataloaders(config)
    _, val_loader_b, _ = create_dataloaders(config)
    val_dataset_a = val_loader_a.dataset
    val_dataset_b = val_loader_b.dataset

    total = min(args.count, len(val_dataset_a), len(val_dataset_b))
    print(f"val_dataset size: {len(val_dataset_a)}")
    print(f"inspecting first {total} samples")
    print()

    all_consistent = True
    for index in range(total):
        summary_a = summarize_sample(val_dataset_a, index)
        summary_b = summarize_sample(val_dataset_b, index)
        same = compare_summary(summary_a, summary_b)
        all_consistent = all_consistent and same

        print(f"[{index}] same={same}")
        print(f"  image: {summary_a['image_path']}")
        print(f"  label: {summary_a['label_path']}")
        print(
            "  image_shape={} image_sum={:.6f} image_digest={}".format(
                summary_a["image_shape"], summary_a["image_sum"], summary_a["image_digest"]
            )
        )
        print(
            "  label_shape={} label_sum={:.6f} label_digest={}".format(
                summary_a["label_shape"], summary_a["label_sum"], summary_a["label_digest"]
            )
        )
        if not same:
            print(f"  second_build_image: {summary_b['image_path']}")
            print(f"  second_build_label: {summary_b['label_path']}")
            print(
                "  second_build_image_shape={} image_sum={:.6f} image_digest={}".format(
                    summary_b["image_shape"], summary_b["image_sum"], summary_b["image_digest"]
                )
            )
            print(
                "  second_build_label_shape={} label_sum={:.6f} label_digest={}".format(
                    summary_b["label_shape"], summary_b["label_sum"], summary_b["label_digest"]
                )
            )
        print()

    print(f"overall_consistent={all_consistent}")


if __name__ == "__main__":
    main()
