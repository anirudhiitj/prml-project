from __future__ import annotations

import argparse
import json
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Suppress common warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from src.dataset import build_dataloader
from src.losses import pit_loss
from src.model import DPRNNTasNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DPRNN-TasNet")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path to resume")
    parser.add_argument("--status-file", type=str, default="", help="JSON file for progress tracking")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory for results")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    def _load_recursive(path: Path):
        cfg_obj = OmegaConf.load(path)
        cfg_obj_dict = OmegaConf.to_container(cfg_obj, resolve=False)
        defaults = cfg_obj_dict.get("defaults", []) if isinstance(cfg_obj_dict, dict) else []

        merged = OmegaConf.create()
        for entry in defaults:
            if isinstance(entry, str):
                dep_path = (path.parent / f"{entry}.yaml").resolve()
                merged = OmegaConf.merge(merged, _load_recursive(dep_path))

        return OmegaConf.merge(merged, cfg_obj)

    final_cfg = _load_recursive(Path(config_path).resolve())
    return OmegaConf.to_container(final_cfg, resolve=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed() -> Tuple[int, int, int]:
    if not is_distributed():
        return 0, 0, 1

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def maybe_init_wandb(cfg: dict, rank: int) -> None:
    use_wandb = cfg["logging"]["use_wandb"]
    if rank != 0 or not use_wandb:
        return
    if wandb is None:
        raise RuntimeError("wandb is enabled in config but not installed")

    wandb.init(
        project=cfg["logging"]["project"],
        name=cfg["experiment_name"],
        config=cfg,
    )


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    val_sisnr: float,
    cfg: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()

    ckpt = {
        "epoch": epoch,
        "model_state": model_state,
        "optim_state": optimizer.state_dict(),
        "sched_state": scheduler.state_dict(),
        "val_sisnr": val_sisnr,
        "config": cfg,
    }
    torch.save(ckpt, path)


def update_status_file(
    status_file: str,
    epoch: int,
    train_loss: float,
    train_sisnr: float,
    val_loss: float,
    val_sisnr: float,
    learning_rate: float,
    epoch_time: float,
) -> None:
    """Update training status JSON file with current epoch metrics."""
    if not status_file:
        return

    status_path = Path(status_file)
    try:
        if status_path.exists():
            with open(status_path, 'r') as f:
                status = json.load(f)
        else:
            status = {
                "phase": 1,
                "num_speakers": 2,
                "start_time": None,
                "end_time": None,
                "status": "RUNNING",
                "gpus": [5, 6],
                "total_epochs": None,
                "current_epoch": 0,
                "best_val_sisnr": -float('inf'),
                "best_val_sisnr_epoch": -1,
                "training_history": {
                    "epoch": [],
                    "train_loss": [],
                    "train_sisnr": [],
                    "val_loss": [],
                    "val_sisnr": [],
                    "learning_rate": [],
                    "grad_norm": [],
                    "epoch_time": []
                },
                "time_per_epoch_estimates": [],
                "eta_hours": None
            }

        # Update current values
        status["current_epoch"] = epoch
        status["training_history"]["epoch"].append(epoch)
        status["training_history"]["train_loss"].append(float(train_loss))
        status["training_history"]["train_sisnr"].append(float(train_sisnr))
        status["training_history"]["val_loss"].append(float(val_loss))
        status["training_history"]["val_sisnr"].append(float(val_sisnr))
        status["training_history"]["learning_rate"].append(float(learning_rate))
        status["training_history"]["epoch_time"].append(float(epoch_time))

        # Track best validation SI-SNR
        if val_sisnr > status["best_val_sisnr"]:
            status["best_val_sisnr"] = val_sisnr
            status["best_val_sisnr_epoch"] = epoch

        # Write updated status
        status_path.parent.mkdir(parents=True, exist_ok=True)
        with open(status_path, 'w') as f:
            json.dump(status, f, indent=4)
    except Exception as e:
        print(f"[WARNING] Failed to update status file {status_file}: {e}")



def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_clip_norm: float,
    snr_weight: float,
    train: bool,
    log_interval: int,
    rank: int,
) -> Tuple[float, float]:
    mode = "train" if train else "eval"
    model.train(train)

    running_loss = 0.0
    running_si = 0.0
    n_steps = 0

    pbar = tqdm(loader, disable=(rank != 0), desc=mode)

    for step, (mixture, sources) in enumerate(pbar):
        mixture = mixture.to(device, non_blocking=True)
        sources = sources.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                estimates = model(mixture)
                min_len = min(estimates.shape[-1], sources.shape[-1])
                estimates = estimates[..., :min_len]
                sources = sources[..., :min_len]
                loss, mean_sisnr = pit_loss(estimates, sources, snr_weight=snr_weight)

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

        running_loss += float(loss.item())
        running_si += float(mean_sisnr.item())
        n_steps += 1

        if rank == 0 and step % log_interval == 0:
            pbar.set_postfix(loss=running_loss / n_steps, sisnr=running_si / n_steps)

    return running_loss / max(1, n_steps), running_si / max(1, n_steps)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    set_seed(int(cfg["seed"]) + rank)
    maybe_init_wandb(cfg, rank)

    model = DPRNNTasNet(**cfg["model"]).to(device)
    if is_distributed():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = Adam(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        betas=tuple(cfg["train"]["betas"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=cfg["scheduler"]["mode"],
        factor=float(cfg["scheduler"]["factor"]),
        patience=int(cfg["scheduler"]["patience"]),
        min_lr=float(cfg["scheduler"]["min_lr"]),
    )

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda" and bool(cfg["train"]["amp"])))

    data_root = Path(cfg["data"]["root"])
    train_loader = build_dataloader(
        split_root=data_root / "train",
        num_speakers=int(cfg["model"]["num_speakers"]),
        batch_size=int(cfg["data"]["train_batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        distributed=is_distributed(),
        shuffle=True,
        clip_samples=int(cfg["data"]["clip_samples"]),
        sample_rate=int(cfg["data"]["sample_rate"]),
    )
    val_loader = build_dataloader(
        split_root=data_root / "val",
        num_speakers=int(cfg["model"]["num_speakers"]),
        batch_size=int(cfg["data"]["val_batch_size"]),
        num_workers=max(2, int(cfg["data"]["num_workers"]) // 2),
        distributed=is_distributed(),
        shuffle=False,
        clip_samples=int(cfg["data"]["clip_samples"]),
        sample_rate=int(cfg["data"]["sample_rate"]),
    )

    start_epoch = 0
    best_val_si = -1e9
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        if isinstance(model, DDP):
            model.module.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["sched_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_si = float(ckpt["val_sisnr"])

    ckpt_dir = Path(cfg["checkpoints"]["dir"])
    num_epochs = int(cfg["train"]["epochs"])

    for epoch in range(start_epoch, num_epochs):
        if is_distributed() and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_si = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip_norm=float(cfg["train"]["grad_clip_norm"]),
            snr_weight=float(cfg["train"]["snr_weight"]),
            train=True,
            log_interval=int(cfg["logging"]["log_interval"]),
            rank=rank,
        )

        with torch.no_grad():
            val_loss, val_si = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                grad_clip_norm=float(cfg["train"]["grad_clip_norm"]),
                snr_weight=float(cfg["train"]["snr_weight"]),
                train=False,
                log_interval=int(cfg["logging"]["log_interval"]),
                rank=rank,
            )

        scheduler.step(val_si)

        if rank == 0:
            log_payload = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/si_snr": train_si,
                "val/loss": val_loss,
                "val/si_snr": val_si,
                "lr": optimizer.param_groups[0]["lr"],
            }
            if cfg["logging"]["use_wandb"] and wandb is not None:
                wandb.log(log_payload)

            # Update status file with current epoch metrics
            update_status_file(
                status_file=args.status_file,
                epoch=epoch,
                train_loss=train_loss,
                train_sisnr=train_si,
                val_loss=val_loss,
                val_sisnr=val_si,
                learning_rate=optimizer.param_groups[0]["lr"],
                epoch_time=0.0,  # We don't track time in train.py currently
            )

            save_checkpoint(
                path=ckpt_dir / "latest.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                val_sisnr=val_si,
                cfg=cfg,
            )

            if val_si > best_val_si:
                best_val_si = val_si
                save_checkpoint(
                    path=ckpt_dir / "best.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    val_sisnr=val_si,
                    cfg=cfg,
                )

            if (epoch + 1) % int(cfg["checkpoints"]["save_every"]) == 0:
                save_checkpoint(
                    path=ckpt_dir / f"epoch_{epoch + 1}.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    val_sisnr=val_si,
                    cfg=cfg,
                )

    if rank == 0 and cfg["logging"]["use_wandb"] and wandb is not None:
        wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
