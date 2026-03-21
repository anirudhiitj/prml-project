"""
Evaluation script for DPRNN-TasNet.

Loads a trained checkpoint, runs on the test set, and reports
SI-SNRi and SDRi metrics.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth
    python evaluate.py --checkpoint checkpoints/best_model.pth --test_dir data/test
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import PreMixedDataset
from models.dprnn_tasnet import DPRNNTasNet
from losses.pit_loss import si_snr
from utils.metrics import si_snri, sdri
from utils.audio_utils import save_audio
from train import build_model


def find_best_permutation(estimates, targets, num_sources):
    """
    Find the best permutation of estimates to targets using SI-SNR.
    Returns reordered estimates.
    """
    from itertools import permutations

    B, C, T = estimates.shape
    best_estimates = estimates.clone()

    for b in range(B):
        best_snr = float("-inf")
        best_perm = None
        for perm in permutations(range(num_sources)):
            total = sum(
                si_snr(estimates[b, perm[c]].unsqueeze(0), targets[b, c].unsqueeze(0)).item()
                for c in range(num_sources)
            )
            if total > best_snr:
                best_snr = total
                best_perm = perm

        for c in range(num_sources):
            best_estimates[b, c] = estimates[b, best_perm[c]]

    return best_estimates


@torch.no_grad()
def evaluate(model, dataloader, device, num_sources, save_dir=None):
    """Evaluate model on a dataset. Returns mean SI-SNRi and SDRi."""
    model.eval()

    all_si_snri = []
    all_sdri = []
    sample_idx = 0

    for mixture, sources in tqdm(dataloader, desc="Evaluating"):
        mixture = mixture.to(device)  # (B, 1, T)
        sources = sources.to(device)  # (B, C, T)

        estimates = model(mixture)    # (B, C, T)

        # Reorder estimates to match targets (best permutation)
        estimates = find_best_permutation(estimates, sources, num_sources)

        B = mixture.shape[0]

        for b in range(B):
            for c in range(num_sources):
                est = estimates[b, c]     # (T,)
                tgt = sources[b, c]       # (T,)
                mix = mixture[b, 0]       # (T,)

                si_val = si_snri(est.unsqueeze(0), tgt.unsqueeze(0), mix.unsqueeze(0)).item()
                sd_val = sdri(est.unsqueeze(0), tgt.unsqueeze(0), mix.unsqueeze(0)).item()

                all_si_snri.append(si_val)
                all_sdri.append(sd_val)

            # Optionally save separated audio
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                save_audio(mixture[b], os.path.join(save_dir, f"sample{sample_idx}_mix.wav"))
                for c in range(num_sources):
                    save_audio(estimates[b, c],
                               os.path.join(save_dir, f"sample{sample_idx}_est_s{c+1}.wav"))
                    save_audio(sources[b, c],
                               os.path.join(save_dir, f"sample{sample_idx}_ref_s{c+1}.wav"))

            sample_idx += 1

    mean_si_snri = sum(all_si_snri) / len(all_si_snri)
    mean_sdri = sum(all_sdri) / len(all_sdri)

    return mean_si_snri, mean_sdri


def main():
    parser = argparse.ArgumentParser(description="Evaluate DPRNN-TasNet")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Test data directory (overrides config)")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Directory to save separated audio files")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    # Build and load model
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")

    # Dataset
    test_dir = args.test_dir or cfg.get("test_dir", "data/test")
    test_dataset = PreMixedDataset(
        data_dir=test_dir,
        sample_rate=cfg["sample_rate"],
        max_len=cfg["max_audio_len"],
        num_sources=cfg["num_sources"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Evaluate
    mean_si_snri, mean_sdri = evaluate(
        model, test_loader, device, cfg["num_sources"], args.save_dir
    )

    print(f"\n{'='*40}")
    print(f"  Test Results")
    print(f"{'='*40}")
    print(f"  SI-SNRi: {mean_si_snri:.2f} dB")
    print(f"  SDRi:    {mean_sdri:.2f} dB")
    print(f"{'='*40}")

    if args.save_dir:
        print(f"\nSeparated audio saved to: {args.save_dir}/")


if __name__ == "__main__":
    main()
