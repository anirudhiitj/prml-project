#!/usr/bin/env python3
"""
Train DPRNN-TasNet on full LibriSpeech mixtures using DataParallel on GPUs 4 & 5.
Usage:
    CUDA_VISIBLE_DEVICES=4,5 dprnn2/bin/python train_full.py --config configs/3spk_full.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from src.dataset import build_dataloader
from src.losses import pit_loss
from src.model import DPRNNTasNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train DPRNN-TasNet (multi-GPU)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default="")
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scaler,
    device: torch.device,
    grad_clip_norm: float,
    snr_weight: float,
    train: bool,
    log_interval: int,
):
    mode = "train" if train else "val"
    model.train(train)

    running_loss = 0.0
    running_si = 0.0
    n_steps = 0

    pbar = tqdm(loader, desc=mode, leave=False)

    for step, (mixture, sources) in enumerate(pbar):
        mixture = mixture.to(device, non_blocking=True)
        sources = sources.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type="cuda", enabled=True):
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

        if step % log_interval == 0:
            pbar.set_postfix(
                loss=f"{running_loss / n_steps:.4f}",
                sisnr=f"{running_si / n_steps:.2f}",
            )

    avg_loss = running_loss / max(1, n_steps)
    avg_si = running_si / max(1, n_steps)
    return avg_loss, avg_si


def save_checkpoint(path: Path, epoch, model, optimizer, scheduler, val_sisnr, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unwrap DataParallel
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(
        {
            "epoch": epoch,
            "model_state": model_state,
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict(),
            "val_sisnr": val_sisnr,
            "config": cfg,
        },
        path,
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    # Detect available GPUs
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda:0")
    print(f"\n{'='*80}")
    print(f"DPRNN-TasNet Training — Full LibriSpeech")
    print(f"{'='*80}")
    print(f"GPUs available: {num_gpus}")
    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name} ({mem:.0f} GB)")
    print()

    # Build model
    model = DPRNNTasNet(**cfg["model"]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params / 1e6:.1f}M params ({trainable_params / 1e6:.1f}M trainable)")

    # Wrap in DataParallel if multi-GPU
    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel across {num_gpus} GPUs")
    print()

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
    scaler = torch.amp.GradScaler(enabled=True)

    # Data
    data_root = Path(cfg["data"]["root"])
    train_loader = build_dataloader(
        split_root=data_root / "train",
        num_speakers=int(cfg["model"]["num_speakers"]),
        batch_size=int(cfg["data"]["train_batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        distributed=False,
        shuffle=True,
        clip_samples=int(cfg["data"]["clip_samples"]),
        sample_rate=int(cfg["data"]["sample_rate"]),
    )
    val_loader = build_dataloader(
        split_root=data_root / "val",
        num_speakers=int(cfg["model"]["num_speakers"]),
        batch_size=int(cfg["data"]["val_batch_size"]),
        num_workers=max(4, int(cfg["data"]["num_workers"]) // 2),
        distributed=False,
        shuffle=False,
        clip_samples=int(cfg["data"]["clip_samples"]),
        sample_rate=int(cfg["data"]["sample_rate"]),
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Train batch size: {cfg['data']['train_batch_size']}")
    print(f"Steps/epoch: {len(train_loader)}")
    print()

    # Resume
    start_epoch = 0
    best_val_si = -1e9
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        raw_model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["sched_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_si = float(ckpt["val_sisnr"])
        print(f"Resumed from epoch {start_epoch}, best val SI-SNR: {best_val_si:.2f} dB")

    ckpt_dir = Path(cfg["checkpoints"]["dir"])
    num_epochs = int(cfg["train"]["epochs"])

    # Training history for JSON logging
    history_path = ckpt_dir / "training_history.json"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    epoch_times = []

    print(f"Starting training for {num_epochs} epochs...")
    print(f"{'='*80}\n")

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

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
            )

        scheduler.step(val_si)
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        lr = optimizer.param_groups[0]["lr"]
        is_best = val_si > best_val_si

        if is_best:
            best_val_si = val_si
            save_checkpoint(ckpt_dir / "best.pt", epoch, model, optimizer, scheduler, val_si, cfg)

        save_checkpoint(ckpt_dir / "latest.pt", epoch, model, optimizer, scheduler, val_si, cfg)

        if (epoch + 1) % int(cfg["checkpoints"]["save_every"]) == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch + 1}.pt", epoch, model, optimizer, scheduler, val_si, cfg)

        # ETA
        avg_epoch_time = np.mean(epoch_times[-10:])
        remaining = num_epochs - epoch - 1
        eta_hours = remaining * avg_epoch_time / 3600

        best_marker = " ⭐ BEST" if is_best else ""
        print(
            f"Epoch {epoch + 1:3d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train SI-SNR: {train_si:+.2f} dB | "
            f"Val Loss: {val_loss:.4f} | Val SI-SNR: {val_si:+.2f} dB | "
            f"LR: {lr:.2e} | Time: {epoch_time:.0f}s | "
            f"ETA: {eta_hours:.1f}h{best_marker}"
        )

        # Save history
        history = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_sisnr": float(train_si),
            "val_loss": float(val_loss),
            "val_sisnr": float(val_si),
            "lr": float(lr),
            "epoch_time": float(epoch_time),
            "best_val_sisnr": float(best_val_si),
            "eta_hours": float(eta_hours),
        }
        # Append to history file
        with open(history_path, "a") as f:
            f.write(json.dumps(history) + "\n")

    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"Best Val SI-SNR: {best_val_si:.2f} dB")
    print(f"Total time: {sum(epoch_times)/3600:.1f} hours")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
