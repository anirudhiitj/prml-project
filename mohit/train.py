"""
Training script for the RCNN Cocktail Party Audio Separation model.

Supports:
- Pre-generated dataset (from prepare_data.py) or on-the-fly generation
- SI-SNR + PIT loss in both waveform and spectrogram domains
- Mixed-precision training (AMP)
- Gradient clipping
- Learning rate scheduling (ReduceLROnPlateau)
- TensorBoard logging
- Checkpoint saving (best + periodic)

Usage:
    # Using pre-generated data:
    CUDA_VISIBLE_DEVICES=3 python train.py --data_dir ./data/generated --epochs 50

    # Using on-the-fly generation (downloads LibriSpeech automatically):
    CUDA_VISIBLE_DEVICES=3 python train.py --librispeech_root ./data --epochs 50

    # Quick smoke test:
    CUDA_VISIBLE_DEVICES=3 python train.py --epochs 1 --num_train 100 --num_val 20 --batch_size 4
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("WARNING: tensorboard not installed. Logging disabled.")
    class SummaryWriter:
        """Dummy SummaryWriter when tensorboard is not available."""
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass
from tqdm import tqdm

from model import RCNNSeparator
from dataset import PreGeneratedDataset, LibriMixDataset, WavFolderDataset
from utils import STFTHelper, si_snr, pit_loss, negative_si_snr, pit_mse_loss


def collate_fn(batch):
    """Custom collate function to stack dict samples into batched tensors."""
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        collated[key] = torch.stack([sample[key] for sample in batch], dim=0)
    return collated


def train_one_epoch(model, dataloader, optimizer, scaler, stft_helper, device,
                    epoch, writer, use_waveform_loss=True, grad_clip=5.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_si_snr = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        mixture_mag = batch['mixture_mag'].to(device)   # (B, F, T)
        source1_mag = batch['source1_mag'].to(device)
        source2_mag = batch['source2_mag'].to(device)
        mixture_phase = batch['mixture_phase'].to(device)
        mixture_wav = batch['mixture_wav'].to(device)    # (B, samples)
        source1_wav = batch['source1_wav'].to(device)
        source2_wav = batch['source2_wav'].to(device)

        # Prepare input: (B, 1, F, T)
        mix_input = mixture_mag.unsqueeze(1)

        optimizer.zero_grad()

        with autocast(device_type='cuda'):
            # Forward pass → masks (B, 2, F, T)
            masks = model(mix_input)

            if use_waveform_loss:
                # Reconstruct waveforms from masked spectrograms
                est_wavs = []
                for s in range(masks.shape[1]):
                    est_mag = masks[:, s] * mixture_mag
                    est_wav = stft_helper.istft(est_mag, mixture_phase,
                                                length=mixture_wav.shape[-1])
                    est_wavs.append(est_wav)

                est_sources = torch.stack(est_wavs, dim=1)  # (B, 2, samples)
                tgt_sources = torch.stack([source1_wav, source2_wav], dim=1)  # (B, 2, samples)

                # PIT loss with SI-SNR
                loss = pit_loss(est_sources, tgt_sources, loss_fn=negative_si_snr)
            else:
                # Spectrogram domain MSE loss with PIT
                target_mags = torch.stack([source1_mag, source2_mag], dim=1)
                loss = pit_mse_loss(masks, target_mags, mixture_mag.unsqueeze(1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        # Compute SI-SNR for logging (with waveform reconstruction)
        if use_waveform_loss:
            with torch.no_grad():
                # Use the best permutation's SI-SNR (match PIT logic)
                si_snr_p1 = (si_snr(est_sources[:, 0], source1_wav) +
                             si_snr(est_sources[:, 1], source2_wav)).mean().item() / 2
                si_snr_p2 = (si_snr(est_sources[:, 0], source2_wav) +
                             si_snr(est_sources[:, 1], source1_wav)).mean().item() / 2
                avg_si_snr = max(si_snr_p1, si_snr_p2)
                total_si_snr += avg_si_snr

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(num_batches, 1)
    avg_si_snr_val = total_si_snr / max(num_batches, 1) if use_waveform_loss else 0

    global_step = epoch * len(dataloader)
    writer.add_scalar('Train/Loss', avg_loss, epoch)
    writer.add_scalar('Train/SI-SNR', avg_si_snr_val, epoch)

    return avg_loss, avg_si_snr_val


@torch.no_grad()
def validate(model, dataloader, stft_helper, device, epoch, writer,
             use_waveform_loss=True):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_si_snr = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}", leave=False)
    for batch in pbar:
        mixture_mag = batch['mixture_mag'].to(device)
        source1_mag = batch['source1_mag'].to(device)
        source2_mag = batch['source2_mag'].to(device)
        mixture_phase = batch['mixture_phase'].to(device)
        mixture_wav = batch['mixture_wav'].to(device)
        source1_wav = batch['source1_wav'].to(device)
        source2_wav = batch['source2_wav'].to(device)

        mix_input = mixture_mag.unsqueeze(1)
        masks = model(mix_input)

        if use_waveform_loss:
            est_wavs = []
            for s in range(masks.shape[1]):
                est_mag = masks[:, s] * mixture_mag
                est_wav = stft_helper.istft(est_mag, mixture_phase,
                                            length=mixture_wav.shape[-1])
                est_wavs.append(est_wav)

            est_sources = torch.stack(est_wavs, dim=1)
            tgt_sources = torch.stack([source1_wav, source2_wav], dim=1)

            loss = pit_loss(est_sources, tgt_sources, loss_fn=negative_si_snr)
            # Use the best permutation's SI-SNR (match PIT logic)
            si_snr_p1 = (si_snr(est_sources[:, 0], source1_wav) +
                         si_snr(est_sources[:, 1], source2_wav)).mean().item() / 2
            si_snr_p2 = (si_snr(est_sources[:, 0], source2_wav) +
                         si_snr(est_sources[:, 1], source1_wav)).mean().item() / 2
            avg_si_snr = max(si_snr_p1, si_snr_p2)
            total_si_snr += avg_si_snr
        else:
            target_mags = torch.stack([source1_mag, source2_mag], dim=1)
            loss = pit_mse_loss(masks, target_mags, mixture_mag.unsqueeze(1))

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(num_batches, 1)
    avg_si_snr_val = total_si_snr / max(num_batches, 1) if use_waveform_loss else 0

    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/SI-SNR', avg_si_snr_val, epoch)

    return avg_loss, avg_si_snr_val


def main():
    parser = argparse.ArgumentParser(description="Train RCNN Speech Separator")

    # Data
    parser.add_argument('--wav_data_dir', type=str, default=None,
                        help='Directory with wav-folder mixtures (train/ and val/ subdirs with mixture.wav, s1.wav, s2.wav)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory with pre-generated .pt files (from prepare_data.py)')
    parser.add_argument('--librispeech_root', type=str, default='./data',
                        help='Root dir for LibriSpeech (used if --data_dir not set)')
    parser.add_argument('--num_train', type=int, default=5000,
                        help='Number of training samples (for on-the-fly mode)')
    parser.add_argument('--num_val', type=int, default=500,
                        help='Number of validation samples (for on-the-fly mode)')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Max training samples to use from wav_data_dir (None=all)')
    parser.add_argument('--max_val_samples', type=int, default=None,
                        help='Max validation samples to use from wav_data_dir (None=all)')

    # Model
    parser.add_argument('--n_fft', type=int, default=512)
    parser.add_argument('--hop_length', type=int, default=128)
    parser.add_argument('--n_sources', type=int, default=2)
    parser.add_argument('--lstm_hidden', type=int, default=256)
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--loss_domain', type=str, default='waveform',
                        choices=['waveform', 'spectrogram'],
                        help='Loss domain: waveform (SI-SNR+PIT) or spectrogram (MSE+PIT)')
    parser.add_argument('--num_workers', type=int, default=4)

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Logging
    parser.add_argument('--log_dir', type=str, default='./runs')

    args = parser.parse_args()

    # ─── Device ───
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ─── STFT Helper ───
    stft_helper = STFTHelper(n_fft=args.n_fft, hop_length=args.hop_length)

    # ─── Dataset ───
    if args.wav_data_dir and os.path.exists(os.path.join(args.wav_data_dir, 'train')):
        print("Using wav-folder dataset from:", args.wav_data_dir)
        train_dataset = WavFolderDataset(
            os.path.join(args.wav_data_dir, 'train'),
            n_fft=args.n_fft, hop_length=args.hop_length,
            max_samples=args.max_train_samples,
        )
        val_dataset = WavFolderDataset(
            os.path.join(args.wav_data_dir, 'val'),
            n_fft=args.n_fft, hop_length=args.hop_length,
            max_samples=args.max_val_samples,
        )
    elif args.data_dir and os.path.exists(os.path.join(args.data_dir, 'train')):
        print("Using pre-generated dataset from:", args.data_dir)
        train_dataset = PreGeneratedDataset(os.path.join(args.data_dir, 'train'))
        val_dataset = PreGeneratedDataset(os.path.join(args.data_dir, 'val'))
    else:
        print("Using on-the-fly LibriSpeech mixture generation")
        train_dataset = LibriMixDataset(
            root_dir=args.librispeech_root,
            subset="train-clean-100",
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            num_samples=args.num_train,
        )
        val_dataset = LibriMixDataset(
            root_dir=args.librispeech_root,
            subset="dev-clean",
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            num_samples=args.num_val,
        )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    # ─── Model ───
    model = RCNNSeparator(
        n_fft=args.n_fft,
        n_sources=args.n_sources,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ─── Optimizer & Scheduler ───
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    scaler = GradScaler()

    # ─── Resume ───
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # ─── Logging ───
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    use_waveform_loss = args.loss_domain == 'waveform'
    print(f"Loss domain: {args.loss_domain}")
    print(f"Training for {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    print("=" * 60)

    # ─── Training Loop ───
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_si_snr = train_one_epoch(
            model, train_loader, optimizer, scaler, stft_helper,
            device, epoch, writer, use_waveform_loss, args.grad_clip,
        )

        val_loss, val_si_snr = validate(
            model, val_loader, stft_helper, device, epoch, writer,
            use_waveform_loss,
        )

        scheduler.step(val_loss)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Train SI-SNR: {train_si_snr:.2f} dB | "
              f"Val SI-SNR: {val_si_snr:.2f} dB | "
              f"LR: {lr:.2e} | "
              f"Time: {elapsed:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'args': vars(args),
            }, best_path)
            print(f"  → Saved best model (val_loss={best_val_loss:.4f})")

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch:03d}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'args': vars(args),
            }, ckpt_path)
            print(f"  → Saved checkpoint: {ckpt_path}")

    writer.close()
    print("=" * 60)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {os.path.join(args.checkpoint_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()
