import argparse

from seg_training import (
    TrainConfig,
    build_model,
    collect_runtime_context,
    create_dataloaders,
    train,
)
from seg_training.config import parse_tuple


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train 3D UNet segmentation model.")
    parser.add_argument("--out-dir", default="outputs/atlas1")
    parser.add_argument("--real-root", default="atlas")
    parser.add_argument("--gen-root", default="uncond_gen")
    parser.add_argument("--split-file", default="output1/atlas_train_val.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--gen-ratio", type=float, default=0.0)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patch-size", type=parse_tuple, default=parse_tuple("128,128,128"))
    parser.add_argument("--roi-size", type=parse_tuple, default=parse_tuple("128,128,128"))
    parser.add_argument("--num-samples-per-vol", type=int, default=8)
    parser.add_argument("--sw-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cache-rate", type=float, default=1.0)
    parser.add_argument("--cache-workers", type=int, default=16)
    parser.add_argument("--loader-workers", type=int, default=16)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--load-best", action="store_true")
    parser.add_argument("--no-filter-empty", action="store_true")
    parser.add_argument("--no-gen-axes-align", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="atlas-seg")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = TrainConfig(
        out_dir=args.out_dir,
        real_root=args.real_root,
        gen_root=args.gen_root,
        split_file=args.split_file,
        seed=args.seed,
        train_ratio=args.train_ratio,
        filter_empty=not args.no_filter_empty,
        gen_ratio=args.gen_ratio,
        max_epochs=args.max_epochs,
        patch_size=args.patch_size,
        num_samples_per_vol=args.num_samples_per_vol,
        lr=args.lr,
        roi_size=args.roi_size,
        sw_batch_size=args.sw_batch_size,
        cache_rate=args.cache_rate,
        cache_workers=args.cache_workers,
        loader_workers=args.loader_workers,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        load_best=args.load_best,
        device=args.device,
        save_every=args.save_every,
        use_gen_axes_align=not args.no_gen_axes_align,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
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
    model = build_model(config)
    train(config, model, train_loader, val_loader, post_trans, runtime=runtime)


if __name__ == "__main__":
    main()
