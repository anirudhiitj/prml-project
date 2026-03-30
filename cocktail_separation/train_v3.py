#!/usr/bin/env python3
"""
Train DPRNN-TasNet 3spk_v3: Fine-tune from 3spk_v2 best checkpoint with:
  - Dynamic remixing (infinite effective data)
  - Random gain ±5dB + speed perturbation
  - LR warmup + cosine annealing
  - Tighter gradient clipping (1.0)
  - Configurable dropout
  - Early stopping
  - Rich live progress display

Usage:
    CUDA_VISIBLE_DEVICES=4,6 dprnn2/bin/python train_v3.py --config configs/3spk_v3.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from src.dataset_augmented import build_augmented_dataloaders
from src.losses import pit_loss
from src.model import DPRNNTasNet


# ─── Utilities ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train DPRNN-TasNet v3")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint")
    parser.add_argument("--finetune", type=str, default="", help="Fine-tune from checkpoint (model weights only)")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    def _load_recursive(path: Path):
        cfg_obj = OmegaConf.load(path)
        cfg_dict = OmegaConf.to_container(cfg_obj, resolve=False)
        defaults = cfg_dict.get("defaults", []) if isinstance(cfg_dict, dict) else []
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


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def get_lr_schedule(optimizer, warmup_epochs: int, total_epochs: int, min_lr_ratio: float = 0.01):
    """Warmup + cosine annealing schedule."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)


# ─── Training Loop ────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scaler,
    device: torch.device,
    grad_clip_norm: float,
    snr_weight: float,
    train: bool,
    epoch: int,
    total_epochs: int,
):
    mode = "TRAIN" if train else "VAL  "
    model.train(train)

    running_loss = 0.0
    running_si = 0.0
    n_steps = 0
    total_steps = len(loader)
    epoch_start = time.time()

    for step, (mixture, sources) in enumerate(loader):
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
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                
                # Skip step if gradients are too large (stability guard)
                if torch.isfinite(grad_norm) and grad_norm < grad_clip_norm * 10:
                    scaler.step(optimizer)
                scaler.update()

        running_loss += float(loss.item())
        running_si += float(mean_sisnr.item())
        n_steps += 1

        # Live progress bar
        if step % 5 == 0 or step == total_steps - 1:
            elapsed = time.time() - epoch_start
            avg_step_time = elapsed / (step + 1)
            eta_epoch = avg_step_time * (total_steps - step - 1)
            avg_loss = running_loss / n_steps
            avg_si = running_si / n_steps
            pct = 100.0 * (step + 1) / total_steps
            bar_len = 30
            filled = int(bar_len * (step + 1) / total_steps)
            bar = "█" * filled + "░" * (bar_len - filled)

            lr_now = optimizer.param_groups[0]["lr"]
            sys.stdout.write(
                f"\r  {mode} Epoch {epoch+1:3d}/{total_epochs} |{bar}| "
                f"{step+1}/{total_steps} ({pct:5.1f}%) "
                f"loss={avg_loss:+.4f} SI-SNR={avg_si:+.2f}dB "
                f"lr={lr_now:.2e} ETA={format_time(eta_epoch)}"
            )
            sys.stdout.flush()

    sys.stdout.write("\n")
    avg_loss = running_loss / max(1, n_steps)
    avg_si = running_si / max(1, n_steps)
    return avg_loss, avg_si


def save_checkpoint(path: Path, epoch, model, optimizer, scheduler, scaler, val_sisnr, best_val_sisnr, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save({
        "epoch": epoch,
        "model_state": model_state,
        "optim_state": optimizer.state_dict(),
        "sched_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "val_sisnr": val_sisnr,
        "best_val_sisnr": best_val_sisnr,
        "config": cfg,
    }, path)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda:0")

    print(f"\n{'='*90}")
    print(f"  DPRNN-TasNet 3spk_v3 — Fine-tune with Dynamic Mixing + Augmentation")
    print(f"{'='*90}")
    print(f"  GPUs: {num_gpus}")
    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"    GPU {i}: {name} ({mem:.0f} GB)")
    print()

    # ── Build model ───────────────────────────────────────────────────────────
    model_cfg = cfg["model"]
    dropout = float(cfg.get("model", {}).get("dropout", 0.2))

    model = DPRNNTasNet(**{k: v for k, v in model_cfg.items() if k != "dropout"}).to(device)

    # Patch dropout in DPRNN blocks if configurable
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_params / 1e6:.1f}M params, dropout={dropout}")

    # ── Fine-tune from checkpoint ─────────────────────────────────────────────
    finetune_path = args.finetune or cfg.get("finetune_from", "")
    if finetune_path:
        print(f"  Loading weights from: {finetune_path}")
        ckpt = torch.load(finetune_path, map_location="cpu")
        raw_model = model
        raw_model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded model from epoch {ckpt['epoch']+1}, val SI-SNR={ckpt['val_sisnr']:.2f} dB")
        print(f"  (Optimizer/scheduler reset for fine-tuning)")

    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"  Using DataParallel across {num_gpus} GPUs")

    # ── Optimizer + Schedule ──────────────────────────────────────────────────
    train_cfg = cfg["train"]
    num_epochs = int(train_cfg["epochs"])
    warmup_epochs = int(train_cfg.get("warmup_epochs", 5))

    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        betas=tuple(train_cfg["betas"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = get_lr_schedule(optimizer, warmup_epochs, num_epochs, min_lr_ratio=0.01)
    scaler = torch.amp.GradScaler(enabled=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    aug_cfg = cfg.get("augmentation", {})

    train_loader, val_loader = build_augmented_dataloaders(
        data_root=data_cfg["root"],
        num_speakers=int(model_cfg["num_speakers"]),
        train_batch_size=int(data_cfg["train_batch_size"]),
        val_batch_size=int(data_cfg["val_batch_size"]),
        num_workers=int(data_cfg["num_workers"]),
        clip_samples=int(data_cfg["clip_samples"]),
        sample_rate=int(data_cfg["sample_rate"]),
        gain_range_db=float(aug_cfg.get("gain_range_db", 5.0)),
        speed_perturb=bool(aug_cfg.get("speed_perturb", True)),
        epoch_size=int(aug_cfg.get("epoch_size", 0)),
    )

    print(f"\n  Train: {len(train_loader.dataset)} mixtures/epoch (dynamic remix from source pool)")
    print(f"  Val:   {len(val_loader.dataset)} mixtures (static)")
    print(f"  Batch: {data_cfg['train_batch_size']} train, {data_cfg['val_batch_size']} val")
    print(f"  Steps/epoch: {len(train_loader)}")

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_si = -1e9
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        raw = model.module if isinstance(model, nn.DataParallel) else model
        raw.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["sched_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_si = float(ckpt.get("best_val_sisnr", ckpt["val_sisnr"]))
        print(f"\n  Resumed from epoch {start_epoch}, best val SI-SNR: {best_val_si:.2f} dB")

    # ── Checkpoints & History ─────────────────────────────────────────────────
    ckpt_dir = Path(cfg["checkpoints"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history_path = ckpt_dir / "training_history.json"

    # Early stopping
    early_stop_cfg = cfg.get("early_stopping", {})
    patience = int(early_stop_cfg.get("patience", 30))
    min_delta = float(early_stop_cfg.get("min_delta", 0.01))
    epochs_no_improve = 0

    epoch_times = []
    grad_clip_norm = float(train_cfg["grad_clip_norm"])
    snr_weight = float(train_cfg["snr_weight"])
    save_every = int(cfg["checkpoints"]["save_every"])

    print(f"\n  Epochs: {num_epochs}, Warmup: {warmup_epochs}")
    print(f"  LR: {train_cfg['lr']} → cosine decay, Weight decay: {train_cfg['weight_decay']}")
    print(f"  Grad clip: {grad_clip_norm}, Early stop patience: {patience}")
    print(f"{'='*90}\n")

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        train_loss, train_si = run_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            scaler=scaler, device=device, grad_clip_norm=grad_clip_norm,
            snr_weight=snr_weight, train=True, epoch=epoch, total_epochs=num_epochs,
        )

        with torch.no_grad():
            val_loss, val_si = run_epoch(
                model=model, loader=val_loader, optimizer=optimizer,
                scaler=scaler, device=device, grad_clip_norm=grad_clip_norm,
                snr_weight=snr_weight, train=False, epoch=epoch, total_epochs=num_epochs,
            )

        scheduler.step()
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        lr = optimizer.param_groups[0]["lr"]
        is_best = val_si > best_val_si + min_delta

        if val_si > best_val_si:
            best_val_si = val_si
            epochs_no_improve = 0
            save_checkpoint(ckpt_dir / "best.pt", epoch, model, optimizer, scheduler, scaler, val_si, best_val_si, cfg)
        else:
            epochs_no_improve += 1

        save_checkpoint(ckpt_dir / "latest.pt", epoch, model, optimizer, scheduler, scaler, val_si, best_val_si, cfg)

        if (epoch + 1) % save_every == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch+1}.pt", epoch, model, optimizer, scheduler, scaler, val_si, best_val_si, cfg)

        # ETA calculation
        avg_time = np.mean(epoch_times[-10:])
        remaining = num_epochs - epoch - 1
        eta_total = remaining * avg_time
        total_elapsed = sum(epoch_times)

        gap = train_si - val_si
        best_marker = " ⭐ NEW BEST" if is_best else ""

        print(
            f"  ╔══ Epoch {epoch+1:3d}/{num_epochs} ══════════════════════════════════════════════════════════╗"
        )
        print(
            f"  ║  Train  loss={train_loss:+.4f}  SI-SNR={train_si:+.2f} dB"
        )
        print(
            f"  ║  Val    loss={val_loss:+.4f}  SI-SNR={val_si:+.2f} dB  "
            f"(best={best_val_si:+.2f} dB){best_marker}"
        )
        print(
            f"  ║  Gap={gap:.2f}dB  LR={lr:.2e}  "
            f"Time={format_time(epoch_time)}  Elapsed={format_time(total_elapsed)}  "
            f"ETA={format_time(eta_total)}  NoImprove={epochs_no_improve}/{patience}"
        )
        print(
            f"  ╚{'═'*76}╝\n"
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
            "train_val_gap": float(gap),
            "eta_hours": float(eta_total / 3600),
            "epochs_no_improve": epochs_no_improve,
        }
        with open(history_path, "a") as f:
            f.write(json.dumps(history) + "\n")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\n  ⚠️  Early stopping triggered after {patience} epochs without improvement.")
            print(f"  Best Val SI-SNR: {best_val_si:.2f} dB at epoch {epoch + 1 - patience}")
            break

    print(f"\n{'='*90}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best Val SI-SNR: {best_val_si:.2f} dB")
    print(f"  Total time: {format_time(sum(epoch_times))}")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
