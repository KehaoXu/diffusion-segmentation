import csv
import glob
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


CACHE_RATE = 1.0
TRAIN_BATCH_SIZE = 2
VAL_BATCH_SIZE = 1


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
        real_root: Path = Path("/scratch/peirong/kxu56/atlas"),
        gen_root: Path = Path("/scratch/peirong/kxu56/USB/assets/uncond_gen"),
        split_seed: int = 42,
        gen_seed: int = 42,
        train_ratio: float = 0.8,
        gen_ratio: float = 0.0,
        show_progress: bool = False,
    ) -> None:
        self.real_root = Path(real_root)
        self.gen_root = Path(gen_root)
        self.split_seed = split_seed
        self.gen_seed = gen_seed
        self.train_ratio = train_ratio
        self.gen_ratio = gen_ratio
        self.show_progress = show_progress

    def build_real_list(self) -> List[Dict[str, str]]:
        data_list: List[Dict[str, str]] = []
        img_paths = sorted(glob.glob(os.path.join(self.real_root.as_posix(), "T1/sub-*.nii.gz")))

        for image_path in tqdm(img_paths, desc="Building real list", disable=not self.show_progress):
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

    def build_gen_list(self, len_train_data: int) -> List[Dict[str, str]]:
        data_list: List[Dict[str, str]] = []
        count = int(self.gen_ratio * len_train_data)
        img_paths = sorted(glob.glob(os.path.join(self.gen_root.as_posix(), "y0_*.nii.gz")))
        rng = random.Random(self.gen_seed)
        rng.shuffle(img_paths)

        for image_path in tqdm(img_paths, desc="Building synthetic list", disable=not self.show_progress):
            if len(data_list) >= count:
                break
            case_id = os.path.basename(image_path).split(".")[0].split("_")[1]
            label_path = os.path.join(self.gen_root.as_posix(), f"x0_{case_id}.nii.gz")
            if os.path.exists(label_path):
                data_list.append({"image": image_path, "label": label_path})

        print(f"Found {len(data_list)} synthetic cases")
        return data_list

    def split_train_val(
        self, data_list: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        
        shuffled = list(data_list)
        rng = random.Random(self.split_seed)
        rng.shuffle(shuffled)

        split_idx = int(self.train_ratio * len(shuffled))
        train_data = shuffled[:split_idx]
        val_data = shuffled[split_idx:]
        print(f"[Split] train={len(train_data)}, val={len(val_data)}")
        return train_data, val_data

    def save_split(
        self,
        split_path: Path,
    ) -> Path:
        data_list = self.build_real_list()
        train_data, val_data = self.split_train_val(data_list)
        
        train_ratio=self.train_ratio
        split_seed=self.split_seed

        split_path = split_path.expanduser().resolve()
        split_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "split",
            "image",
            "label",
            "train_ratio",
            "split_seed",
        ]
        with split_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for split_name, items in (("train", train_data), ("val", val_data)):
                for item in items:
                    writer.writerow(
                        {
                            "split": split_name,
                            "image": item["image"],
                            "label": item["label"],
                            "train_ratio": train_ratio,
                            "split_seed": split_seed,
                        }
                    )
        print(
            f"\nSaved split file: {split_path}"
            f"\ntrain={len(train_data)}, val={len(val_data)}"
            f"\nsplit_seed={split_seed}"
        )


def load_split(
    split_path: Path,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    split_path = split_path.expanduser().resolve()

    train_data: List[Dict[str, str]] = []
    val_data: List[Dict[str, str]] = []

    with split_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split_name = row["split"].strip()
            item = {"image": row["image"], "label": row["label"]}

            if split_name == "train":
                train_data.append(item)
            elif split_name == "val":
                val_data.append(item)
            else:
                raise ValueError(f"Unknown split name '{split_name}' in {split_path}")

    train_data = list(train_data)
    val_data = list(val_data)

    print(f"[Split] loaded train={len(train_data)}, val={len(val_data)} from {split_path}")

    return train_data, val_data


def build_transforms() -> Tuple[Compose, Compose, Compose, Compose]:
    keys = ["image", "label"]
    common = [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ]

    crop_norm = [
        CropForegroundd(keys=keys, source_key="image", margin=10),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ]

    rand_crop = [
        RandCropByPosNegLabeld(
            keys=keys,
            label_key="label",
            spatial_size=(128, 128, 128),
            pos=1,
            neg=1,
            num_samples=8,
            image_key="image",
            allow_smaller=True,
        )
    ]

    augmentation = [
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        RandAffined(
            keys=keys,
            prob=0.3,
            rotate_range=(0.3, 0.3, 0.3),
            scale_range=(0.1, 0.1, 0.1),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
    ]

    gen_specific = [
        AlignAxesd(keys=keys, transpose_order=(0, 2, 1), flip_axes=(2,)),
        AsDiscreted(keys=["label"], threshold=0.5, to_onehot=None, dtype=np.float32),
        GaussianThresholdBackgroundd(keys=["image"], sigma=0.5, thr_ratio=0.02, margin=1),
    ]

    gen_tf = Compose(common + gen_specific + crop_norm + rand_crop + augmentation + [EnsureTyped(keys=keys)])
    train_tf = Compose(common + crop_norm + rand_crop + augmentation + [EnsureTyped(keys=keys)])
    val_tf = Compose(common + crop_norm + [EnsureTyped(keys=keys)])
    post_tf = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    return gen_tf, train_tf, val_tf, post_tf


def create_dataloaders(config: TrainConfig):
    builder = DatasetBuilder(
        gen_root=config.gen_root,
        gen_seed=config.gen_seed,
        gen_ratio=config.gen_ratio,
        show_progress=config.show_progress,
    )
    if config.split_file is None:
        raise ValueError("config.split_file must be set. Generate a split file before training.")
    
    train_data, val_data = load_split(config.split_file)
    gen_data = builder.build_gen_list(len(train_data))

    gen_tf, train_tf, val_tf, post_tf = build_transforms()

    gen_ds = CacheDataset(
        data=gen_data,
        transform=gen_tf,
        cache_rate=CACHE_RATE,
        num_workers=config.cache_workers,
        progress=config.show_progress,
    )
    real_ds = CacheDataset(
        data=train_data,
        transform=train_tf,
        cache_rate=CACHE_RATE,
        num_workers=config.cache_workers,
        progress=config.show_progress,
    )
    val_ds = CacheDataset(
        data=val_data,
        transform=val_tf,
        cache_rate=CACHE_RATE,
        num_workers=config.cache_workers,
        progress=config.show_progress,
    )

    train_ds = ConcatDataset([real_ds, gen_ds])
    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.loader_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.loader_workers,
    )
    return train_loader, val_loader, post_tf
