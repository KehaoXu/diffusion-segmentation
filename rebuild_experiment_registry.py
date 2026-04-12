import argparse
import csv
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


OUTPUT_OVERRIDES = {
    "atlas1+uncond1": {
        "wandb_out_dir": "outputs/atlas1+uncond",
        "gen_ratio": 0.1,
        "host": "c008",
        "slurm_job_id": "1273817",
        "slurm_job_name": "seg",
        "wandb_run_name": "seg-aug-1273817",
        "status": "completed",
        "error_message": "",
    },
}


def format_value(value):
    if isinstance(value, list):
        return "x".join(str(item) for item in value)
    if isinstance(value, tuple):
        return "x".join(str(item) for item in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return value


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def format_duration(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.1f}"


def iso_from_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def parse_iso(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    normalized = raw.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def load_default_config() -> Dict[str, Any]:
    spec = importlib.util.spec_from_file_location("seg_training_config", "seg_training/config.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.TrainConfig().to_dict()


def parse_wandb_args(args: List[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if not arg.startswith("--"):
            i += 1
            continue
        key = arg[2:].replace("-", "_")
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            parsed[key] = args[i + 1]
            i += 2
        else:
            parsed[key] = True
            i += 1
    return parsed


def normalize_wandb_config(parsed: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    config = dict(defaults)
    int_keys = {
        "seed",
        "max_epochs",
        "num_samples_per_vol",
        "sw_batch_size",
        "cache_workers",
        "loader_workers",
        "train_batch_size",
        "val_batch_size",
        "save_every",
    }
    float_keys = {"train_ratio", "gen_ratio", "lr", "cache_rate"}
    tuple_keys = {"patch_size", "roi_size"}
    bool_keys = {"use_wandb", "load_best"}

    for key, raw in parsed.items():
        if key in int_keys:
            config[key] = int(raw)
        elif key in float_keys:
            config[key] = float(raw)
        elif key in tuple_keys:
            config[key] = tuple(int(part) for part in str(raw).split(","))
        elif key in bool_keys:
            config[key] = bool(raw)
        else:
            config[key] = raw
    return config


def build_wandb_index(wandb_dir: Path, defaults: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for metadata_path in sorted(wandb_dir.glob("offline-run-*/files/wandb-metadata.json")):
        metadata = load_json(metadata_path)
        parsed = parse_wandb_args(metadata.get("args", []))
        out_dir = parsed.get("out_dir")
        if not out_dir:
            continue
        config = normalize_wandb_config(parsed, defaults)
        slurm = metadata.get("slurm", {}) or {}
        debug_log = metadata_path.parents[1] / "logs" / "debug.log"
        finished_at = parse_finished_at_from_debug_log(debug_log)
        index[out_dir] = {
            "config": config,
            "runtime": {
                "command": " ".join(metadata.get("args", [])),
                "host": metadata.get("host", ""),
                "cwd": metadata.get("root", ""),
                "pid": "",
                "slurm_job_id": slurm.get("jobid") or slurm.get("job_id") or "",
                "slurm_job_name": slurm.get("job_name", ""),
            },
            "started_at": normalize_iso(metadata.get("started_at", "")),
            "finished_at": finished_at,
            "run_id": infer_run_id(metadata.get("started_at", ""), slurm.get("jobid") or slurm.get("job_id") or ""),
        }
    return index


def normalize_iso(raw: str) -> str:
    dt = parse_iso(raw)
    return dt.isoformat() if dt else ""


def infer_run_id(started_at: str, slurm_job_id: str) -> str:
    dt = parse_iso(started_at)
    if dt is None:
        return f"recovered-{slurm_job_id}" if slurm_job_id else "recovered-run"
    return f"{dt.strftime('%Y%m%d-%H%M%S')}-{slurm_job_id or 'recovered'}"


def parse_finished_at_from_debug_log(path: Path) -> str:
    if not path.exists():
        return ""
    last_timestamp = ""
    with path.open("r") as f:
        for line in f:
            if len(line) >= 19 and line[4] == "-" and line[7] == "-":
                ts = line[:19]
                try:
                    dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                last_timestamp = dt.isoformat()
    return last_timestamp


def discover_output_dirs(outputs_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in outputs_dir.iterdir()
        if path.is_dir() and (path / "training_log.csv").exists()
    )


def resolve_checkpoint(output_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(output_dir.glob(pattern), key=lambda path: int(path.stem.split("_")[-1]))
    return matches[-1] if matches else None


def infer_summary(output_dir: Path, max_epochs: int) -> Dict[str, Any]:
    log_path = output_dir / "training_log.csv"
    rows = load_csv_rows(log_path)
    best_row = max(rows, key=lambda row: float(row["val_dice"]))
    final_row = rows[-1]
    best_checkpoint = resolve_checkpoint(output_dir, "unet3d_best_*.pt")
    last_checkpoint = resolve_checkpoint(output_dir, "unet3d_last_*.pt")

    finished_candidates = [log_path]
    if best_checkpoint is not None:
        finished_candidates.append(best_checkpoint)
    if last_checkpoint is not None:
        finished_candidates.append(last_checkpoint)
    finished_at = iso_from_timestamp(max(path.stat().st_mtime for path in finished_candidates))

    final_epoch = int(final_row["epoch"])
    completed = final_epoch >= max_epochs and last_checkpoint is not None
    return {
        "status": "completed" if completed else "partial",
        "finished_at": finished_at,
        "best_val_dice": float(best_row["val_dice"]),
        "best_epoch": int(best_row["epoch"]),
        "final_val_dice": float(final_row["val_dice"]),
        "final_epoch": final_epoch,
        "best_checkpoint": str(best_checkpoint.relative_to(output_dir.parent.parent)) if best_checkpoint else "",
        "last_checkpoint": str(last_checkpoint.relative_to(output_dir.parent.parent)) if last_checkpoint else "",
        "log_path": str(log_path.relative_to(output_dir.parent.parent)),
        "error_message": "",
    }


def build_row(
    output_dir: Path,
    defaults: Dict[str, Any],
    wandb_index: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    override = OUTPUT_OVERRIDES.get(output_dir.name, {})
    local_run_config_path = output_dir / "run_config.json"
    local_run_config = load_json(local_run_config_path) if local_run_config_path.exists() else {}
    local_summary_path = output_dir / "run_summary.json"
    local_summary = load_json(local_summary_path) if local_summary_path.exists() else {}

    source_out_dir = override.get("wandb_out_dir", f"outputs/{output_dir.name}")
    wandb_info = wandb_index.get(source_out_dir, {})

    config = dict(defaults)
    config.update(wandb_info.get("config", {}))
    config.update(local_run_config.get("config", {}))
    config.update({k: v for k, v in override.items() if k not in {"wandb_out_dir", "status", "error_message"}})
    config["out_dir"] = f"outputs/{output_dir.name}"

    runtime = {}
    runtime.update(wandb_info.get("runtime", {}))
    runtime.update(local_run_config.get("runtime", {}))
    runtime["host"] = override.get("host", runtime.get("host", ""))
    runtime["slurm_job_id"] = override.get("slurm_job_id", runtime.get("slurm_job_id", ""))
    runtime["slurm_job_name"] = override.get("slurm_job_name", runtime.get("slurm_job_name", "seg"))

    inferred = infer_summary(output_dir, int(config.get("max_epochs", 0)))
    status = override.get("status") or inferred["status"]
    error_message = override.get("error_message", "")
    started_at = override.get("started_at") or wandb_info.get("started_at") or local_run_config.get("started_at", "")
    finished_at = override.get("finished_at") or wandb_info.get("finished_at") or inferred["finished_at"]

    duration_sec = None
    started_dt = parse_iso(started_at)
    finished_dt = parse_iso(finished_at)
    if started_dt is not None and finished_dt is not None:
        duration_sec = (finished_dt - started_dt).total_seconds()

    row: Dict[str, Any] = {
        "gen_ratio": config.get("gen_ratio"),
        "best_val_dice": format_metric(inferred["best_val_dice"]),
        "final_val_dice": format_metric(inferred["final_val_dice"]),
        "best_epoch": inferred["best_epoch"],
        "final_epoch": inferred["final_epoch"],
        "run_id": override.get("run_id")
        or wandb_info.get("run_id")
        or local_run_config.get("run_id", ""),
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": format_duration(duration_sec),
        "host": runtime.get("host", ""),
        "slurm_job_id": runtime.get("slurm_job_id", ""),
        "slurm_job_name": runtime.get("slurm_job_name", ""),
        "best_checkpoint": inferred["best_checkpoint"],
        "last_checkpoint": inferred["last_checkpoint"],
        "log_path": inferred["log_path"],
        "error_message": error_message,
    }

    for key, value in config.items():
        row[key] = format_value(value)
    return {key: format_value(value) for key, value in row.items()}


def ordered_fieldnames(rows: Iterable[Dict[str, str]]) -> List[str]:
    base_fieldnames = [
        "gen_ratio",
        "best_val_dice",
        "final_val_dice",
        "best_epoch",
        "final_epoch",
        "lr",
        "train_batch_size",
        "val_batch_size",
        "patch_size",
        "roi_size",
        "num_samples_per_vol",
        "sw_batch_size",
        "in_channels",
        "out_channels",
        "channels",
        "strides",
        "num_res_units",
        "spacing",
        "crop_margin",
        "max_epochs",
        "cache_rate",
        "cache_workers",
        "loader_workers",
        "train_ratio",
        "filter_empty",
        "load_best",
        "use_gen_axes_align",
        "run_id",
        "status",
        "out_dir",
        "real_root",
        "gen_root",
        "wandb_run_name",
        "wandb_project",
        "wandb_entity",
        "wandb_mode",
        "use_wandb",
        "started_at",
        "finished_at",
        "duration_sec",
        "seed",
        "device",
        "host",
        "slurm_job_id",
        "slurm_job_name",
        "save_every",
        "best_checkpoint",
        "last_checkpoint",
        "log_path",
        "error_message",
    ]
    seen = set(base_fieldnames)
    fieldnames = list(base_fieldnames)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild outputs/experiment_registry.csv from real output directories."
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Directory containing experiment output subdirectories.",
    )
    parser.add_argument(
        "--wandb-dir",
        default="wandb",
        help="Directory containing offline wandb runs.",
    )
    parser.add_argument(
        "--registry-path",
        default=None,
        help="Optional explicit path for experiment_registry.csv. Defaults to <outputs-dir>/experiment_registry.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir).expanduser().resolve()
    wandb_dir = Path(args.wandb_dir).expanduser().resolve()
    registry_path = (
        Path(args.registry_path).expanduser().resolve()
        if args.registry_path
        else outputs_dir / "experiment_registry.csv"
    )

    defaults = load_default_config()
    wandb_index = build_wandb_index(wandb_dir, defaults)
    output_dirs = discover_output_dirs(outputs_dir)
    rows = [build_row(output_dir, defaults, wandb_index) for output_dir in output_dirs]
    rows.sort(key=lambda row: row.get("out_dir", ""))

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ordered_fieldnames(rows)

    with registry_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Rebuilt {registry_path} with {len(rows)} output run(s).")


if __name__ == "__main__":
    main()
