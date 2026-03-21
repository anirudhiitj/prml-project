"""
Training script for DPRNN-TasNet.

Usage:
    python train.py                         # Train with default config
    python train.py --config configs/custom.yaml
    python train.py --overfit_one_batch --epochs 50   # Sanity check
"""

import argparse
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import PreMixedDataset
from losses.pit_loss import PITLoss
from models.dprnn_tasnet import DPRNNTasNet


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> DPRNNTasNet:
    """Instantiate DPRNNTasNet from config."""
    return DPRNNTasNet(
        N=cfg["encoder_dim"],
        L=cfg["encoder_kernel"],
        H=cfg["hidden_size"],
        K=cfg["chunk_size"],
        P=cfg.get("hop_size", None),
        B=cfg["num_dprnn_blocks"],
        C=cfg["num_sources"],
        rnn_type=cfg.get("rnn_type", "lstm"),
        num_layers=cfg.get("rnn_num_layers", 1),
        bidirectional=cfg.get("bidirectional", True),
        dropout=cfg.get("dropout", 0.0),
        encoder_stride=cfg.get("encoder_stride", None),
        mask_activation=cfg.get("mask_activation", "relu"),
    )


def train_one_epoch(model, dataloader, criterion, optimizer, max_grad_norm, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for mixture, sources in tqdm(dataloader, desc="  Train", leave=False):
        mixture = mixture.to(device)   # (B, 1, T)
        sources = sources.to(device)   # (B, C, T)

        # Forward pass
        estimates = model(mixture)     # (B, C, T)

        # Compute PIT loss
        loss = criterion(estimates, sources)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate. Returns average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for mixture, sources in tqdm(dataloader, desc="  Val  ", leave=False):
        mixture = mixture.to(device)
        sources = sources.to(device)

        estimates = model(mixture)
        loss = criterion(estimates, sources)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train DPRNN-TasNet")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--overfit_one_batch", action="store_true",
                        help="Overfit on a single batch (sanity check)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Build model ----
    model = build_model(cfg).to(device)
    num_params = model.num_parameters()
    print(f"Model parameters: {num_params:,}")

    # ---- Loss & optimizer ----
    criterion = PITLoss(num_sources=cfg["num_sources"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=cfg.get("scheduler_patience", 5),
        factor=cfg.get("scheduler_factor", 0.5),
    )

    # ---- Datasets ----
    train_dataset = PreMixedDataset(
        data_dir=cfg["train_dir"],
        sample_rate=cfg["sample_rate"],
        max_len=cfg["max_audio_len"],
        num_sources=cfg["num_sources"],
    )
    val_dataset = PreMixedDataset(
        data_dir=cfg["val_dir"],
        sample_rate=cfg["sample_rate"],
        max_len=cfg["max_audio_len"],
        num_sources=cfg["num_sources"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=True,
    )

    # If overfitting on one batch, replace loaders
    if args.overfit_one_batch:
        print(">>> OVERFITTING ON ONE BATCH (sanity check) <<<")
        single_batch = next(iter(train_loader))
        single_batch = (single_batch[0].to(device), single_batch[1].to(device))

        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            mixture, sources = single_batch
            estimates = model(mixture)
            loss = criterion(estimates, sources)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("max_grad_norm", 5.0))
            optimizer.step()
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:4d} | Loss: {loss.item():.6f}")
        print("Done. If loss decreased significantly, the model can learn!")
        return

    # ---- Checkpoint directory ----
    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Training loop ----
    best_val_loss = float("inf")
    max_grad_norm = cfg.get("max_grad_norm", 5.0)

    for epoch in range(1, cfg["epochs"] + 1):
        print(f"\nEpoch [{epoch}/{cfg['epochs']}]")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     max_grad_norm, device)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": cfg,
            }, ckpt_path)
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "config": cfg,
        }, os.path.join(ckpt_dir, "latest_model.pth"))

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
