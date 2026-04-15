import csv
import glob
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_dilation, gaussian_filter
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Activations,
    AsDiscrete,
    AsDiscreted,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    Spacingd,
)

from .config import TrainConfig


def _resolve_path(value: Path) -> Path:
    return value.expanduser().resolve()


def _normalize_item(item: Dict[str, str]) -> Dict[str, str]:
    return {
        "image": str(_resolve_path(Path(item["image"]))),
        "label": str(_resolve_path(Path(item["label"]))),
    }


def shuffle_data_list(
    data_list: List[Dict[str, str]],
    seed: int,
) -> List[Dict[str, str]]:
    shuffled = list(data_list)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled


def save_split_file(
    split_path: Path,
    *,
    dataset_name: str,
    real_root: Path,
    train_ratio: float,
    filter_empty: bool,
    split_seed: int,
    train_data: List[Dict[str, str]],
    val_data: List[Dict[str, str]],
) -> Path:
    split_path = split_path.expanduser().resolve()
    split_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_train = [_normalize_item(item) for item in train_data]
    normalized_val = [_normalize_item(item) for item in val_data]
    fieldnames = [
        "split",
        "image",
        "label",
        "dataset_name",
        "real_root",
        "train_ratio",
        "filter_empty",
        "split_seed",
    ]
    resolved_real_root = str(_resolve_path(real_root))
    with split_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, items in (("train", normalized_train), ("val", normalized_val)):
            for item in items:
                writer.writerow(
                    {
                        "split": split_name,
                        "image": item["image"],
                        "label": item["label"],
                        "dataset_name": dataset_name,
                        "real_root": resolved_real_root,
                        "train_ratio": train_ratio,
                        "filter_empty": filter_empty,
                        "split_seed": split_seed,
                    }
                )
    return split_path


def load_split_file(split_path: Path) -> Dict[str, object]:
    split_path = split_path.expanduser().resolve()
    train_data: List[Dict[str, str]] = []
    val_data: List[Dict[str, str]] = []
    dataset_name = None
    real_root = None
    train_ratio = None
    filter_empty = None
    split_seed = None

    with split_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split_name = row["split"].strip()
            item = _normalize_item({"image": row["image"], "label": row["label"]})

            if dataset_name is None:
                dataset_name = row.get("dataset_name") or None
            if real_root is None:
                raw_root = row.get("real_root") or None
                real_root = str(_resolve_path(Path(raw_root))) if raw_root else None
            if train_ratio is None:
                raw_train_ratio = row.get("train_ratio")
                train_ratio = float(raw_train_ratio) if raw_train_ratio not in (None, "") else None
            if filter_empty is None:
                raw_filter_empty = row.get("filter_empty")
                if raw_filter_empty not in (None, ""):
                    filter_empty = raw_filter_empty.strip().lower() == "true"
            if split_seed is None:
                raw_split_seed = row.get("split_seed")
                split_seed = int(raw_split_seed) if raw_split_seed not in (None, "") else None

            if split_name == "train":
                train_data.append(item)
            elif split_name == "val":
                val_data.append(item)
            else:
                raise ValueError(f"Unknown split name '{split_name}' in {split_path}")

    payload = {
        "dataset_name": dataset_name,
        "real_root": real_root,
        "train_ratio": train_ratio,
        "filter_empty": filter_empty,
        "split_seed": split_seed,
        "train": train_data,
        "val": val_data,
    }
    return payload


def validate_split_file(
    split_payload: Dict[str, object],
    *,
    real_root: Path,
    filter_empty: bool,
) -> None:
    expected_root = str(_resolve_path(real_root))
    payload_root = str(split_payload.get("real_root"))
    if payload_root != expected_root:
        raise ValueError(
            f"Split file real_root mismatch: expected {expected_root}, found {payload_root}"
        )

    payload_filter_empty = bool(split_payload.get("filter_empty"))
    if payload_filter_empty != filter_empty:
        raise ValueError(
            "Split file filter_empty mismatch: "
            f"expected {filter_empty}, found {payload_filter_empty}"
        )

    train_data = split_payload["train"]
    val_data = split_payload["val"]
    train_keys = {(item["image"], item["label"]) for item in train_data}
    val_keys = {(item["image"], item["label"]) for item in val_data}
    overlap = train_keys & val_keys
    if overlap:
        raise ValueError(f"Split file contains {len(overlap)} overlapping train/val samples")


class GaussianThresholdBackgroundd(MapTransform):
    def __init__(
        self,
        keys: Sequence[str],
        label_key: str = "label",
        sigma: float = 0.5,
        thr_ratio: float = 0.02,
        use_abs: bool = False,
        margin: int = 1,
    ) -> None:
        super().__init__(keys)
        self.label_key = label_key
        self.sigma = sigma
        self.thr_ratio = thr_ratio
        self.use_abs = use_abs
        self.margin = margin

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        d = dict(data)

        image = d[self.keys[0]]
        label = d[self.label_key]
        is_tensor = torch.is_tensor(image)

        if is_tensor:
            img_np = image.detach().cpu().numpy().copy()
            label_np = label.detach().cpu().numpy()
        else:
            img_np = np.asarray(image).copy()
            label_np = np.asarray(label)

        label_mask = label_np > 0.5
        image_mask = np.zeros_like(img_np, dtype=bool)

        for channel in range(img_np.shape[0]):
            image_channel = img_np[channel]
            image_proc = np.abs(image_channel) if self.use_abs else image_channel
            smoothed = gaussian_filter(image_proc, sigma=self.sigma)
            threshold = float(smoothed.max()) * self.thr_ratio
            image_mask[channel] = smoothed >= threshold

        final_mask = image_mask | label_mask

        if self.margin > 0:
            structure = np.ones((3, 3, 3), dtype=bool)
            for channel in range(final_mask.shape[0]):
                final_mask[channel] = binary_dilation(
                    final_mask[channel],
                    structure=structure,
                    iterations=self.margin,
                )

        img_np = np.where(final_mask, img_np, 0.0)

        if is_tensor:
            d[self.keys[0]] = torch.as_tensor(img_np, dtype=image.dtype, device=image.device)
        else:
            d[self.keys[0]] = img_np

        return d


class AlignAxesd(MapTransform):
    def __init__(
        self,
        keys: Sequence[str] = ("image", "label"),
        transpose_order: Tuple[int, int, int] = (0, 2, 1),
        flip_axes: Sequence[int] = (2,),
    ) -> None:
        super().__init__(keys)
        self.transpose_order = transpose_order
        self.flip_axes = tuple(flip_axes)

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            image = d[key]

            if image.ndim == 4:
                if self.transpose_order is not None:
                    image = np.transpose(image, (0,) + tuple(i + 1 for i in self.transpose_order))
                for axis in self.flip_axes:
                    image = np.flip(image, axis=axis + 1)
            elif image.ndim == 3:
                if self.transpose_order is not None:
                    image = np.transpose(image, self.transpose_order)
                for axis in self.flip_axes:
                    image = np.flip(image, axis=axis)
            else:
                raise ValueError(f"{key} shape {image.shape} is not supported")

            d[key] = image.copy()

        return d


class DatasetBuilder:
    def __init__(
        self,
        real_root: Path,
        gen_root: Path,
        seed: int = 42,
        train_ratio: float = 0.8,
        filter_empty: bool = True,
        gen_ratio: float = 0.0,
    ) -> None:
        self.real_root = Path(real_root)
        self.gen_root = Path(gen_root)
        self.seed = seed
        self.train_ratio = train_ratio
        self.filter_empty = filter_empty
        self.gen_ratio = gen_ratio

    def build_real_list(self) -> List[Dict[str, str]]:
        data_list: List[Dict[str, str]] = []
        img_paths = glob.glob(
            os.path.join(self.real_root.as_posix(), "sub-strokecase*/ses-0001/dwi/*_dwi.nii.gz")
        )

        for image_path in img_paths:
            case_id = os.path.basename(image_path).split("_")[0]
            label_path = os.path.join(
                self.real_root.as_posix(),
                "derivatives",
                case_id,
                "ses-0001",
                f"{case_id}_ses-0001_msk.nii.gz",
            )
            if os.path.exists(label_path):
                data_list.append({"image": image_path, "label": label_path})

        print(f"[ISLES] Found {len(data_list)} cases")
        return data_list

    def filter_foreground(self, data_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not self.filter_empty:
            return data_list

        filtered: List[Dict[str, str]] = []
        skipped = 0
        for item in tqdm(data_list, desc="Filtering foreground"):
            label = nib.load(item["label"]).get_fdata()
            if np.any(label > 0):
                filtered.append(item)
            else:
                skipped += 1

        print(f"[Filter] keep {len(filtered)} cases, skip {skipped}")
        return filtered

    def split_train_val(
        self, data_list: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        data_list = shuffle_data_list(data_list, seed=self.seed)
        split_idx = int(self.train_ratio * len(data_list))
        train_data = data_list[:split_idx]
        val_data = data_list[split_idx:]
        print(f"[Split] train={len(train_data)}, val={len(val_data)}")
        return train_data, val_data

    def generate_split(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        real_list = self.build_real_list()
        real_list = self.filter_foreground(real_list)
        return self.split_train_val(real_list)

    def load_split(
        self,
        split_path: Path,
        *,
        shuffle_seed: Optional[int] = None,
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        payload = load_split_file(split_path)
        validate_split_file(
            payload,
            real_root=self.real_root,
            filter_empty=self.filter_empty,
        )
        train_data = list(payload["train"])
        val_data = list(payload["val"])
        if shuffle_seed is not None:
            train_data = shuffle_data_list(train_data, seed=shuffle_seed)
            val_data = shuffle_data_list(val_data, seed=shuffle_seed + 1)
        print(f"[Split] loaded train={len(train_data)}, val={len(val_data)} from {split_path}")
        return train_data, val_data

    def save_split(
        self,
        split_path: Path,
    ) -> Path:
        train_data, val_data = self.generate_split()
        return save_split_file(
            split_path,
            dataset_name=self.__class__.__name__,
            real_root=self.real_root,
            train_ratio=self.train_ratio,
            filter_empty=self.filter_empty,
            split_seed=self.seed,
            train_data=train_data,
            val_data=val_data,
        )

    def add_gen(self, train_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        len_train_data = len(train_data)
        img_paths = glob.glob(os.path.join(self.gen_root.as_posix(), "y0_*.nii.gz"))

        count = 0
        for image_path in img_paths:
            if count >= int(self.gen_ratio * len_train_data):
                break
            case_id = os.path.basename(image_path).split("_")[1]
            label_path = os.path.join(self.gen_root.as_posix(), f"x0_{case_id}")
            if os.path.exists(label_path):
                train_data.append({"image": image_path, "label": label_path})
                count += 1

        print(f"[Uncond] Added {count} synthetic samples")
        return train_data

    def build(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        train_data, val_data = self.generate_split()
        train_data = self.add_gen(train_data)
        return train_data, val_data


class AtlasBuilder(DatasetBuilder):
    def generate_split(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        real_data = self.build_real_list()
        return self.split_train_val(real_data)

    def build_real_list(self) -> List[Dict[str, str]]:
        data_list: List[Dict[str, str]] = []
        img_paths = glob.glob(os.path.join(self.real_root.as_posix(), "T1/sub-*.nii.gz"))

        for image_path in tqdm(img_paths, desc="Building real list"):
            case_id = os.path.basename(image_path)
            label_path = os.path.join(
                self.real_root.as_posix(),
                "pathology_maps_segmentation",
                case_id,
            )
            if os.path.exists(label_path):
                data_list.append({"image": image_path, "label": label_path})

        print(f"Found {len(data_list)} real cases")
        return data_list

    def build_gen_list(self, train_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        data_list: List[Dict[str, str]] = []
        count = int(self.gen_ratio * len(train_data))
        img_paths = glob.glob(os.path.join(self.gen_root.as_posix(), "y0_*.nii.gz"))

        for index, image_path in enumerate(tqdm(img_paths, desc="Building synthetic list")):
            if index >= count:
                break
            case_id = os.path.basename(image_path).split(".")[0].split("_")[1]
            label_path = os.path.join(self.gen_root.as_posix(), f"x0_{case_id}.nii.gz")
            if os.path.exists(label_path):
                data_list.append({"image": image_path, "label": label_path})

        print(f"Found {len(data_list)} synthetic cases")
        return data_list

    def build(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
        train_data, val_data = self.generate_split()
        gen_data = self.build_gen_list(train_data)
        return train_data, val_data, gen_data


def build_transforms(config: TrainConfig) -> Tuple[Compose, Compose, Compose, Compose]:
    common = [
        LoadImaged(keys=config.common_keys, image_only=False),
        EnsureChannelFirstd(keys=config.common_keys),
        Orientationd(keys=config.common_keys, axcodes="RAS"),
        Spacingd(keys=config.common_keys, pixdim=config.spacing, mode=("bilinear", "nearest")),
    ]

    crop_resize_norm = [
        CropForegroundd(keys=config.common_keys, source_key="image", margin=config.crop_margin),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ]

    rand_crop = [
        RandCropByPosNegLabeld(
            keys=config.common_keys,
            label_key="label",
            spatial_size=config.patch_size,
            pos=1,
            neg=1,
            num_samples=config.num_samples_per_vol,
            image_key="image",
            allow_smaller=True,
        )
    ]

    augmentation = [
        RandFlipd(keys=config.common_keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=config.common_keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=config.common_keys, prob=0.5, spatial_axis=2),
        RandAffined(
            keys=config.common_keys,
            prob=0.3,
            rotate_range=(0.3, 0.3, 0.3),
            scale_range=(0.1, 0.1, 0.1),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
    ]

    gen_specific = []
    if config.use_gen_axes_align:
        gen_specific.append(
            AlignAxesd(keys=config.common_keys, transpose_order=(0, 2, 1), flip_axes=(2,))
        )
    gen_specific.extend(
        [
            AsDiscreted(keys=["label"], threshold=0.5, to_onehot=None, dtype=np.float32),
            GaussianThresholdBackgroundd(keys=["image"], sigma=0.5, thr_ratio=0.02, margin=1),
        ]
    )

    gen_tf = Compose(common + gen_specific + crop_resize_norm + rand_crop + augmentation + [EnsureTyped(keys=config.common_keys)])
    train_tf = Compose(common + crop_resize_norm + rand_crop + augmentation + [EnsureTyped(keys=config.common_keys)])
    val_tf = Compose(common + crop_resize_norm + [EnsureTyped(keys=config.common_keys)])
    post_tf = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    return gen_tf, train_tf, val_tf, post_tf


def create_dataloaders(config: TrainConfig):
    builder = AtlasBuilder(
        real_root=config.real_root,
        gen_root=config.gen_root,
        seed=config.seed,
        train_ratio=config.train_ratio,
        filter_empty=config.filter_empty,
        gen_ratio=config.gen_ratio,
    )
    if config.split_file is None:
        raise ValueError("config.split_file must be set. Generate a split file before training.")
    train_data, val_data = builder.load_split(config.split_file, shuffle_seed=config.seed)
    gen_data = builder.build_gen_list(train_data)

    gen_tf, train_tf, val_tf, post_tf = build_transforms(config)

    gen_ds = CacheDataset(
        data=gen_data,
        transform=gen_tf,
        cache_rate=config.cache_rate,
        num_workers=config.cache_workers,
    )
    real_ds = CacheDataset(
        data=train_data,
        transform=train_tf,
        cache_rate=config.cache_rate,
        num_workers=config.cache_workers,
    )
    val_ds = CacheDataset(
        data=val_data,
        transform=val_tf,
        cache_rate=config.cache_rate,
        num_workers=config.cache_workers,
    )

    train_ds = ConcatDataset([real_ds, gen_ds])
    train_loader = DataLoader(
        train_ds,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.loader_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.loader_workers,
    )
    return train_loader, val_loader, post_tf
