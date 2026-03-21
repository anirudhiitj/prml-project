"""
Inference script — separate a single mixed audio file.

Usage:
    python inference.py --input mix.wav --checkpoint checkpoints/best_model.pth
    python inference.py --input mix.wav --checkpoint checkpoints/best_model.pth --output_dir results/
"""

import argparse
import os

import torch

from models.dprnn_tasnet import DPRNNTasNet
from utils.audio_utils import load_audio, save_audio
from train import build_model


def separate(model, mixture_path: str, output_dir: str,
             sample_rate: int, device: torch.device):
    """
    Separate a single mixture audio file into individual sources.

    Args:
        model: Trained DPRNNTasNet model.
        mixture_path: Path to the input mixture WAV file.
        output_dir: Directory to save output WAV files.
        sample_rate: Audio sample rate.
        device: Torch device.
    """
    # Load audio
    mixture = load_audio(mixture_path, sample_rate)  # (1, T)
    mixture = mixture.unsqueeze(0).to(device)         # (1, 1, T)

    # Forward pass
    model.eval()
    with torch.no_grad():
        estimates = model(mixture)                     # (1, C, T)

    # Save each separated source
    os.makedirs(output_dir, exist_ok=True)
    num_sources = estimates.shape[1]
    basename = os.path.splitext(os.path.basename(mixture_path))[0]

    for c in range(num_sources):
        out_path = os.path.join(output_dir, f"{basename}_source{c+1}.wav")
        save_audio(estimates[0, c], out_path, sample_rate)
        print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="DPRNN-TasNet Inference")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input mixture WAV file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for separated files")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    # Build and load model
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Model loaded (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

    # Run separation
    print(f"\nSeparating: {args.input}")
    separate(model, args.input, args.output_dir, cfg["sample_rate"], device)
    print("\nDone!")


if __name__ == "__main__":
    main()
