import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


PATH_HEADER_PATTERN = re.compile(r"^(?P<name>[^()]+)(?:\((?P<base>.*)\))?$")


def split_filename(split_path: Path, train_ratio: float, split_seed: int) -> Path:
    split_path = Path(split_path)
    ratio_text = f"{train_ratio:g}"
    suffix = f"_{ratio_text}_{split_seed}"
    if split_path.stem.endswith(suffix):
        return split_path
    return split_path.with_name(f"{split_path.stem}{suffix}{split_path.suffix}")


def _split_path_header(header: str) -> Tuple[str, Optional[str]]:
    match = PATH_HEADER_PATTERN.match(header.strip())
    if match is None:
        return header.strip(), None
    base = match.group("base")
    return match.group("name").strip(), base.strip() if base else None


def _find_path_column(fieldnames: Sequence[str], name: str) -> Tuple[str, Optional[str]]:
    for fieldname in fieldnames:
        column_name, base_path = _split_path_header(fieldname)
        if column_name == name:
            return fieldname, base_path
    raise ValueError(f"Split CSV is missing required '{name}' column")


def _common_parent(paths: Sequence[str]) -> Path:
    parents = [Path(path).parent for path in paths]
    if not parents:
        return Path(".")
    common = Path(*Path(*parents[0].parts).parts)
    for parent in parents[1:]:
        while common != common.parent and not _is_relative_to(parent, common):
            common = common.parent
    return common


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _atlas_relative_path(path: Path) -> str:
    parts = path.parts
    for index, part in enumerate(parts):
        if part == "atlas":
            return Path(*parts[index:]).as_posix()
    return path.as_posix()


def _path_header(name: str, paths: Sequence[str]) -> str:
    base = _atlas_relative_path(_common_parent(paths))
    return f"{name}({base})"


def _join_header_base(base_path: Optional[str], value: str) -> str:
    value = value.strip()
    if base_path is None or not value:
        return value
    path = Path(value)
    if path.is_absolute():
        return value
    return str(Path(base_path) / path)


def _apply_path_prefix(path_text: str, path_prefix: Optional[Path]) -> str:
    path = Path(path_text)
    if path_prefix is None or path.is_absolute():
        return path_text
    return str(path_prefix / path)


def save_split_csv(
    split_path: Path,
    train_data: Sequence[Dict[str, str]],
    val_data: Sequence[Dict[str, str]],
) -> Path:
    split_path = Path(split_path).expanduser().resolve()
    split_path.parent.mkdir(parents=True, exist_ok=True)

    all_items = list(train_data) + list(val_data)
    image_column = _path_header("image", [item["image"] for item in all_items])
    label_column = _path_header("label", [item["label"] for item in all_items])
    fieldnames = ["split", image_column, label_column]

    with split_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, items in (("train", train_data), ("val", val_data)):
            for item in items:
                writer.writerow(
                    {
                        "split": split_name,
                        image_column: Path(item["image"]).name,
                        label_column: Path(item["label"]).name,
                    }
                )

    return split_path


def load_split_csv(
    split_path: Path,
    path_prefix: Optional[Path] = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    split_path = Path(split_path).expanduser().resolve()
    prefix = Path(path_prefix).expanduser() if path_prefix is not None else None

    train_data: List[Dict[str, str]] = []
    val_data: List[Dict[str, str]] = []

    with split_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Split CSV has no header: {split_path}")
        image_column, image_base = _find_path_column(reader.fieldnames, "image")
        label_column, label_base = _find_path_column(reader.fieldnames, "label")

        for row in reader:
            split_name = row["split"].strip()
            image_path = _join_header_base(image_base, row[image_column])
            label_path = _join_header_base(label_base, row[label_column])
            item = {
                "image": _apply_path_prefix(image_path, prefix),
                "label": _apply_path_prefix(label_path, prefix),
            }

            if split_name == "train":
                train_data.append(item)
            elif split_name == "val":
                val_data.append(item)
            else:
                raise ValueError(f"Unknown split name '{split_name}' in {split_path}")

    return train_data, val_data
