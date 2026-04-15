import argparse
from pathlib import Path

from seg_training.data import AtlasBuilder, DatasetBuilder, load_split_file


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and save a fixed train/validation split for segmentation training."
    )
    parser.add_argument("--dataset", choices=("atlas", "isles"), default="atlas")
    parser.add_argument("--real-root", required=True)
    parser.add_argument("--gen-root", default="uncond_gen")
    parser.add_argument("--split-file", required=True, help="Output CSV file for the saved split.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used only for split generation.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--no-filter-empty", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing split file. Otherwise the script will fail if the file exists.",
    )
    return parser


def build_builder(args: argparse.Namespace):
    builder_cls = AtlasBuilder if args.dataset == "atlas" else DatasetBuilder
    return builder_cls(
        real_root=Path(args.real_root),
        gen_root=Path(args.gen_root),
        seed=args.seed,
        train_ratio=args.train_ratio,
        filter_empty=not args.no_filter_empty,
        gen_ratio=0.0,
    )


def main() -> None:
    args = build_argparser().parse_args()
    split_path = Path(args.split_file).expanduser().resolve()
    if split_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Split file already exists: {split_path}. Use --overwrite to replace it."
        )

    builder = build_builder(args)
    saved_path = builder.save_split(split_path)
    payload = load_split_file(saved_path)
    print(f"Saved split file: {saved_path}")
    print(f"train={len(payload['train'])}, val={len(payload['val'])}")
    print(f"dataset={payload['dataset_name']}")
    print(f"real_root={payload['real_root']}")
    print(f"split_seed={payload['split_seed']}")


if __name__ == "__main__":
    main()
