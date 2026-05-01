from pathlib import Path


class TrainConfig(object):
    def __init__(
        self,
        out_dir="outputs/atlas1",
        gen_root="uncond_gen",
        split_file=None,
        gen_seed=42,
        gen_ratio=0.0,
        cache_workers=16,
        loader_workers=16,
        load_best=False,
        show_progress=False,
        device="cuda:0",
        save_every=5,
        use_wandb=False,
        wandb_project=None,
        wandb_run_name=None,
        wandb_mode="online",
    ):
        self.out_dir = Path(out_dir)
        self.gen_root = Path(gen_root)
        self.split_file = Path(split_file) if split_file is not None else None
        self.gen_seed = gen_seed
        self.gen_ratio = gen_ratio
        self.cache_workers = cache_workers
        self.loader_workers = loader_workers
        self.load_best = load_best
        self.show_progress = show_progress
        self.device = device
        self.save_every = save_every
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_mode = wandb_mode

    def _has_existing_outputs(self, path: Path) -> bool:
        if not path.exists():
            return False
        return any(path.iterdir())

    def reserve_output_dir(self) -> Path:
        requested = self.out_dir
        if not self._has_existing_outputs(requested):
            requested.mkdir(parents=True, exist_ok=True)
            return requested

        parent = requested.parent
        stem = requested.name
        suffix = 1
        while True:
            candidate = parent / f"{stem}{suffix}"
            if not candidate.exists():
                candidate.mkdir(parents=True, exist_ok=False)
                self.out_dir = candidate
                return candidate
            if not self._has_existing_outputs(candidate):
                candidate.mkdir(parents=True, exist_ok=True)
                self.out_dir = candidate
                return candidate
            suffix += 1

    def ensure_output_dir(self):
        self.reserve_output_dir()

    def to_dict(self):
        return {
            "out_dir": str(self.out_dir),
            "gen_root": str(self.gen_root),
            "split_file": str(self.split_file) if self.split_file is not None else None,
            "gen_seed": self.gen_seed,
            "gen_ratio": self.gen_ratio,
            "cache_workers": self.cache_workers,
            "loader_workers": self.loader_workers,
            "load_best": self.load_best,
            "show_progress": self.show_progress,
            "device": self.device,
            "save_every": self.save_every,
            "use_wandb": self.use_wandb,
            "wandb_project": self.wandb_project,
            "wandb_run_name": self.wandb_run_name,
            "wandb_mode": self.wandb_mode,
        }

    def to_serializable_dict(self):
        payload = {}
        for key, value in self.to_dict().items():
            if isinstance(value, Path):
                payload[key] = str(value)
            elif isinstance(value, tuple):
                payload[key] = list(value)
            else:
                payload[key] = value
        return payload
