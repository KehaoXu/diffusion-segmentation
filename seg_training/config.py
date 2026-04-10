from pathlib import Path


class TrainConfig(object):
    def __init__(
        self,
        out_dir="outputs/atlas1",
        real_root="atlas",
        gen_root="uncond_gen",
        seed=42,
        train_ratio=0.8,
        filter_empty=True,
        gen_ratio=0.0,
        max_epochs=100,
        patch_size=(128, 128, 128),
        num_samples_per_vol=8,
        lr=1e-4,
        roi_size=(128, 128, 128),
        sw_batch_size=8,
        in_channels=1,
        out_channels=1,
        cache_rate=1.0,
        cache_workers=16,
        loader_workers=16,
        train_batch_size=2,
        val_batch_size=1,
        load_best=False,
        device="cuda:0",
        save_every=5,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        use_gen_axes_align=True,
        spacing=(1.0, 1.0, 1.0),
        crop_margin=10,
        use_wandb=False,
        wandb_project=None,
        wandb_entity=None,
        wandb_run_name=None,
        wandb_mode="online",
    ):
        self.out_dir = Path(out_dir)
        self.real_root = Path(real_root)
        self.gen_root = Path(gen_root)
        self.seed = seed
        self.train_ratio = train_ratio
        self.filter_empty = filter_empty
        self.gen_ratio = gen_ratio
        self.max_epochs = max_epochs
        self.patch_size = patch_size
        self.num_samples_per_vol = num_samples_per_vol
        self.lr = lr
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size if sw_batch_size > 0 else num_samples_per_vol
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cache_rate = cache_rate
        self.cache_workers = cache_workers
        self.loader_workers = loader_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.load_best = load_best
        self.device = device
        self.save_every = save_every
        self.channels = channels
        self.strides = strides
        self.num_res_units = num_res_units
        self.use_gen_axes_align = use_gen_axes_align
        self.spacing = spacing
        self.crop_margin = crop_margin
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name
        self.wandb_mode = wandb_mode

    @property
    def common_keys(self):
        return ("image", "label")

    def ensure_output_dir(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self):
        return {
            "out_dir": str(self.out_dir),
            "real_root": str(self.real_root),
            "gen_root": str(self.gen_root),
            "seed": self.seed,
            "train_ratio": self.train_ratio,
            "filter_empty": self.filter_empty,
            "gen_ratio": self.gen_ratio,
            "max_epochs": self.max_epochs,
            "patch_size": self.patch_size,
            "num_samples_per_vol": self.num_samples_per_vol,
            "lr": self.lr,
            "roi_size": self.roi_size,
            "sw_batch_size": self.sw_batch_size,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "cache_rate": self.cache_rate,
            "cache_workers": self.cache_workers,
            "loader_workers": self.loader_workers,
            "train_batch_size": self.train_batch_size,
            "val_batch_size": self.val_batch_size,
            "load_best": self.load_best,
            "device": self.device,
            "save_every": self.save_every,
            "channels": self.channels,
            "strides": self.strides,
            "num_res_units": self.num_res_units,
            "use_gen_axes_align": self.use_gen_axes_align,
            "spacing": self.spacing,
            "crop_margin": self.crop_margin,
            "use_wandb": self.use_wandb,
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
            "wandb_run_name": self.wandb_run_name,
            "wandb_mode": self.wandb_mode,
        }


def parse_tuple(raw):
    parts = [int(part.strip()) for part in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("Expected three comma-separated integers, got: {}".format(raw))
    return tuple(parts)
