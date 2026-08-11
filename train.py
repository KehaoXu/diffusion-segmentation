import argparse

from seg_training import (
    TrainConfig,
    build_model,
    collect_runtime_context,
    create_dataloaders,
    train,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train 3D UNet segmentation model.")
    parser.add_argument("--out-dir", default="outputs/atlas")
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--gen-root", default="USB/assets/uncond")
    parser.add_argument("--gen-ratio", type=float, default=0.0)
    parser.add_argument("--gen-seed", type=int, default=42)
    parser.add_argument("--cache-workers", type=int, default=16)
    parser.add_argument("--loader-workers", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--load-best", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="atlas-seg")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = TrainConfig(
        out_dir=args.out_dir,
        split_file=args.split_file,
        gen_root=args.gen_root,
        gen_ratio=args.gen_ratio,
        gen_seed=args.gen_seed,
        cache_workers=args.cache_workers,
        loader_workers=args.loader_workers,
        load_best=args.load_best,
        show_progress=args.show_progress,
        device=args.device,
        save_every=args.save_every,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
    )
    requested_out_dir = config.out_dir
    config.ensure_output_dir()
    if config.out_dir != requested_out_dir:
        print(f"Requested output dir exists and is not empty: {requested_out_dir}")
        print(f"Using a new output dir instead: {config.out_dir}")
    else:
        print(f"Using output dir: {config.out_dir}")
    runtime = collect_runtime_context()

    train_loader, val_loader, post_trans = create_dataloaders(config)
    model = build_model()
    train(config, model, train_loader, val_loader, post_trans, runtime=runtime)


if __name__ == "__main__":
    main()
