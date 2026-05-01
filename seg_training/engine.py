import csv
import warnings
from pathlib import Path
from typing import Optional, Tuple

import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm

from .config import TrainConfig
from .experiment import ExperimentRecorder, RuntimeContext


MAX_EPOCHS = 100
LEARNING_RATE = 1e-4
ROI_SIZE = (128, 128, 128)
SW_BATCH_SIZE = 8

try:
    import wandb
except ImportError:
    wandb = None


class CheckpointState(object):
    def __init__(self, best_path=None, last_path=None, best_dice=-1.0, best_epoch=0, start_epoch=0):
        self.best_path = best_path
        self.last_path = last_path
        self.best_dice = best_dice
        self.best_epoch = best_epoch
        self.start_epoch = start_epoch


def configure_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message="Using a non-tuple sequence for multidimensional indexing is deprecated",
    )
    warnings.filterwarnings("ignore", message=".*Num foregrounds.*")


def ensure_log_file(log_path: Path) -> None:
    if log_path.exists():
        return
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_dice", "val_dice", "lr"])


def resolve_checkpoint(out_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(out_dir.glob(pattern), key=lambda path: int(path.stem.split("_")[-1]))
    if not matches:
        return None
    return matches[-1]


def load_training_state(
    config: TrainConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: lr_scheduler._LRScheduler,
    device: torch.device,
    log_path: Path,
) -> CheckpointState:
    best_path = resolve_checkpoint(config.out_dir, "unet3d_best_*.pt")
    last_path = resolve_checkpoint(config.out_dir, "unet3d_last_*.pt")

    if best_path is None or last_path is None:
        print("No model found, training from scratch.")
        return CheckpointState(
            best_path=best_path,
            last_path=last_path,
            best_dice=-1.0,
            best_epoch=0,
            start_epoch=0,
        )

    best_epoch = int(best_path.stem.split("_")[-1])
    last_epoch = int(last_path.stem.split("_")[-1])
    best_dice = -1.0

    if log_path.exists():
        with log_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows and best_epoch - 1 < len(rows):
            best_dice = float(rows[best_epoch - 1]["val_dice"])

    checkpoint_path = best_path if config.load_best else last_path
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["lr_scheduler"])
    start_epoch = int(ckpt["epoch"])

    if config.load_best:
        print(f"Loaded best model: {best_path} from epoch {best_epoch}")
    else:
        print(f"Loaded last model: {last_path} from epoch {last_epoch}")

    return CheckpointState(
        best_path=best_path,
        last_path=last_path,
        best_dice=best_dice,
        best_epoch=best_epoch,
        start_epoch=start_epoch,
    )


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: lr_scheduler._LRScheduler,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
        },
        path,
    )


def append_log(
    log_path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    train_dice: float,
    val_dice: float,
    lr: float,
) -> None:
    with log_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                epoch,
                f"{train_loss:.4f}",
                f"{val_loss:.4f}",
                f"{train_dice:.4f}",
                f"{val_dice:.4f}",
                f"{lr:.6e}",
            ]
        )


def init_wandb(config, model):
    if not config.use_wandb or config.wandb_mode == "disabled":
        return None
    if wandb is None:
        raise ImportError(
            "Weights & Biases is not installed. Install `wandb` or run without --use-wandb."
        )

    project = config.wandb_project or "seg-training"
    run = wandb.init(
        project=project,
        name=config.wandb_run_name,
        mode=config.wandb_mode,
        config=config.to_dict(),
    )
    run.watch(model, log="gradients", log_freq=100)
    return run


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    metric: DiceMetric,
    post_trans,
    device: torch.device,
    epoch: int,
    max_epochs: int,
    show_progress: bool,
) -> Tuple[float, float]:
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{max_epochs} [Train]", disable=not show_progress)
    current_lr = optimizer.param_groups[0]["lr"]

    for batch_idx, batch in enumerate(pbar, start=1):
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        assert image.shape == label.shape

        logits = model(image)
        loss = loss_fn(logits, label)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        avg_loss = epoch_loss / batch_idx

        pred = post_trans(logits)
        metric(pred, label)
        pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.6e}"})

    avg_dice = metric.aggregate().item()
    metric.reset()
    return epoch_loss / max(len(loader), 1), avg_dice


def validate_one_epoch(
    model: torch.nn.Module,
    loader,
    loss_fn,
    metric: DiceMetric,
    post_trans,
    device: torch.device,
    roi_size,
    sw_batch_size: int,
    show_progress: bool,
) -> Tuple[float, float]:
    model.eval()
    epoch_loss = 0.0
    pbar = tqdm(loader, desc="[Val]", disable=not show_progress)

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar, start=1):
            image = batch["image"].to(device)
            label = batch["label"].to(device)

            logits = sliding_window_inference(image, roi_size, sw_batch_size, model)
            loss = loss_fn(logits, label)
            epoch_loss += loss.item()
            avg_loss = epoch_loss / batch_idx

            pred = post_trans(logits)
            metric(pred, label)
            pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})

    avg_dice = metric.aggregate().item()
    metric.reset()
    return epoch_loss / max(len(loader), 1), avg_dice


def train(
    config: TrainConfig,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    post_trans,
    runtime: RuntimeContext,
) -> None:
    configure_warnings()
    config.ensure_output_dir()
    recorder = ExperimentRecorder(config, runtime)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("device:", device)
    print("Params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    log_path = config.out_dir / "training_log.csv"
    ensure_log_file(log_path)

    loss_fn = DiceCELoss(to_onehot_y=False, sigmoid=True)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)
    metric = DiceMetric(include_background=False, reduction="mean")
    wandb_run = init_wandb(config, model)

    state = load_training_state(config, model, optimizer, scheduler, device, log_path)
    final_val_dice = None
    final_epoch = state.start_epoch

    try:
        for epoch_idx in range(state.start_epoch, MAX_EPOCHS):
            epoch = epoch_idx + 1
            train_loss, train_dice = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                metric=metric,
                post_trans=post_trans,
                device=device,
                epoch=epoch,
                max_epochs=MAX_EPOCHS,
                show_progress=config.show_progress,
            )
            val_loss, val_dice = validate_one_epoch(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                metric=metric,
                post_trans=post_trans,
                device=device,
                roi_size=ROI_SIZE,
                sw_batch_size=SW_BATCH_SIZE,
                show_progress=config.show_progress,
            )
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch} train dice: {train_dice:.4f}, val dice: {val_dice:.4f}")
            append_log(log_path, epoch, train_loss, val_loss, train_dice, val_dice, current_lr)
            scheduler.step()

            is_best = val_dice > state.best_dice
            if is_best:
                state.best_dice = val_dice
                state.best_epoch = epoch
                new_best_path = config.out_dir / f"unet3d_best_{epoch}.pt"
                save_checkpoint(new_best_path, epoch, model, optimizer, scheduler)
                if state.best_path is not None and state.best_path.exists():
                    state.best_path.unlink()
                state.best_path = new_best_path
                print(f"saved best -> {state.best_path}")

            if wandb_run is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "train/dice": train_dice,
                        "val/loss": val_loss,
                        "val/dice": val_dice,
                        "lr": current_lr,
                        "best_val_dice": state.best_dice,
                        "checkpoint/is_best": int(is_best),
                    },
                    step=epoch,
                )

            if epoch % config.save_every == 0:
                new_last_path = config.out_dir / f"unet3d_last_{epoch}.pt"
                save_checkpoint(new_last_path, epoch, model, optimizer, scheduler)
                if state.last_path is not None and state.last_path.exists():
                    state.last_path.unlink()
                state.last_path = new_last_path
                print(f"saved last -> {state.last_path}")

            final_val_dice = val_dice
            final_epoch = epoch

        final_last_path = config.out_dir / f"unet3d_last_{MAX_EPOCHS}.pt"
        save_checkpoint(final_last_path, MAX_EPOCHS, model, optimizer, scheduler)
        state.last_path = final_last_path
        print("Training done. best dice:", state.best_dice)
        recorder.finalize(
            status="completed",
            best_val_dice=state.best_dice,
            best_epoch=state.best_epoch,
            final_val_dice=final_val_dice,
            final_epoch=final_epoch,
            best_checkpoint=state.best_path,
            last_checkpoint=state.last_path,
            log_path=log_path,
        )
    except Exception as exc:
        recorder.finalize(
            status="failed",
            best_val_dice=state.best_dice,
            best_epoch=state.best_epoch,
            final_val_dice=final_val_dice,
            final_epoch=final_epoch,
            best_checkpoint=state.best_path,
            last_checkpoint=state.last_path,
            log_path=log_path,
            error_message=str(exc),
        )
        raise
    finally:
        if wandb_run is not None:
            wandb.finish()
