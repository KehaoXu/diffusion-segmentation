import argparse
from pathlib import Path

from seg_training.data import DatasetBuilder
from seg_training.splits import split_filename


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and save a fixed train/validation split for segmentation training."
    )
    parser.add_argument("--real-root", default="atlas")
    parser.add_argument("--split-file", default="atlas_train_val.csv", help="Output CSV file for the saved split.")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed used only for split generation.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--show-progress", action="store_true")

    return parser


def main() -> None:
    args = build_argparser().parse_args()
    split_path = split_filename(
        Path(args.split_file),
        train_ratio=args.train_ratio,
        split_seed=args.split_seed,
    ).expanduser().resolve()
    if split_path.exists():
        print(f"Split file already exists: {split_path}, skipping split.")
        return
    
    builder = DatasetBuilder(
        real_root=Path(args.real_root),
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        show_progress=args.show_progress,
    )

    builder.save_split(split_path)

if __name__ == "__main__":
    main()
