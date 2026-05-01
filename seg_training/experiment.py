import csv
import fcntl
import json
import os
import socket
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import TrainConfig


REGISTRY_FIELDNAMES = [
    "gen_ratio",
    "best_val_dice",
    "final_val_dice",
    "best_epoch",
    "final_epoch",
    "cache_workers",
    "loader_workers",
    "load_best",
    "run_id",
    "status",
    "out_dir",
    "gen_root",
    "wandb_run_name",
    "wandb_project",
    "wandb_mode",
    "use_wandb",
    "started_at",
    "finished_at",
    "duration_sec",
    "gen_seed",
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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def format_duration(value: float) -> str:
    return f"{float(value):.1f}"


def normalize_path(path: Optional[Path]) -> str:
    if path is None:
        return ""
    return str(path)


def format_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return "x".join(str(item) for item in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return value


@dataclass
class RuntimeContext:
    run_id: str
    started_at: str
    command: str
    host: str
    cwd: str
    pid: int
    slurm_job_id: str
    slurm_job_name: str


def collect_runtime_context() -> RuntimeContext:
    started_at = utc_now_iso()
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    suffix = slurm_job_id or uuid.uuid4().hex[:8]
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{suffix}"
    return RuntimeContext(
        run_id=run_id,
        started_at=started_at,
        command=" ".join(sys.argv),
        host=socket.gethostname(),
        cwd=os.getcwd(),
        pid=os.getpid(),
        slurm_job_id=slurm_job_id,
        slurm_job_name=os.environ.get("SLURM_JOB_NAME", ""),
    )


class ExperimentRecorder(object):
    def __init__(self, config: TrainConfig, runtime: RuntimeContext):
        self.config = config
        self.runtime = runtime
        self.registry_path = config.out_dir.parent / "experiment_registry.csv"
        self.config_path = config.out_dir / "run_config.json"
        self.summary_path = config.out_dir / "run_summary.json"
        self.started_at = runtime.started_at
        self._write_run_config()

    def _write_run_config(self) -> None:
        payload = {
            "run_id": self.runtime.run_id,
            "started_at": self.runtime.started_at,
            "runtime": {
                "command": self.runtime.command,
                "host": self.runtime.host,
                "cwd": self.runtime.cwd,
                "pid": self.runtime.pid,
                "slurm_job_id": self.runtime.slurm_job_id,
                "slurm_job_name": self.runtime.slurm_job_name,
            },
            "config": self.config.to_serializable_dict(),
        }
        with self.config_path.open("w") as f:
            json.dump(payload, f, indent=2)

    def finalize(
        self,
        *,
        status: str,
        best_val_dice: float,
        best_epoch: int,
        final_val_dice: Optional[float],
        final_epoch: int,
        best_checkpoint: Optional[Path],
        last_checkpoint: Optional[Path],
        log_path: Path,
        error_message: str = "",
    ) -> None:
        finished_at = utc_now_iso()
        started = datetime.fromisoformat(self.started_at)
        finished = datetime.fromisoformat(finished_at)
        duration_sec = (finished - started).total_seconds()

        summary = {
            "run_id": self.runtime.run_id,
            "status": status,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "duration_sec": duration_sec,
            "best_val_dice": best_val_dice,
            "best_epoch": best_epoch,
            "final_val_dice": final_val_dice,
            "final_epoch": final_epoch,
            "best_checkpoint": normalize_path(best_checkpoint) or None,
            "last_checkpoint": normalize_path(last_checkpoint) or None,
            "log_path": normalize_path(log_path),
            "error_message": error_message,
        }
        with self.summary_path.open("w") as f:
            json.dump(summary, f, indent=2)

        self._append_registry_row(
            finished_at=finished_at,
            duration_sec=duration_sec,
            status=status,
            best_val_dice=best_val_dice,
            best_epoch=best_epoch,
            final_val_dice=final_val_dice,
            final_epoch=final_epoch,
            best_checkpoint=best_checkpoint,
            last_checkpoint=last_checkpoint,
            log_path=log_path,
            error_message=error_message,
        )

    def _append_registry_row(
        self,
        *,
        finished_at: str,
        duration_sec: float,
        status: str,
        best_val_dice: float,
        best_epoch: int,
        final_val_dice: Optional[float],
        final_epoch: int,
        best_checkpoint: Optional[Path],
        last_checkpoint: Optional[Path],
        log_path: Path,
        error_message: str,
    ) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "run_id": self.runtime.run_id,
            "status": status,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "duration_sec": format_duration(duration_sec),
            "host": self.runtime.host,
            "slurm_job_id": self.runtime.slurm_job_id,
            "slurm_job_name": self.runtime.slurm_job_name,
            "best_val_dice": format_metric(best_val_dice),
            "best_epoch": best_epoch,
            "final_val_dice": format_metric(final_val_dice),
            "final_epoch": final_epoch,
            "best_checkpoint": normalize_path(best_checkpoint),
            "last_checkpoint": normalize_path(last_checkpoint),
            "log_path": normalize_path(log_path),
            "error_message": error_message,
        }
        for key, value in self.config.to_dict().items():
            row[key] = format_value(value)

        fieldnames = REGISTRY_FIELDNAMES + [key for key in row.keys() if key not in REGISTRY_FIELDNAMES]

        with self.registry_path.open("a", newline="") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            try:
                f.seek(0, os.SEEK_END)
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
