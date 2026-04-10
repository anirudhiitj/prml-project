"""
Inference script for the RCNN Cocktail Party Audio Separation model.

Takes a mixed audio file and separates it into individual speaker sources.

Usage:
    CUDA_VISIBLE_DEVICES=3 python inference.py \
        --input mixed_audio.wav \
        --checkpoint ./checkpoints/best_model.pt \
        --output_dir ./separated_output
"""

import os
import argparse
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from model import RCNNSeparator, RCNNSeparatorWaveform
from utils import STFTHelper, load_audio, save_audio, normalize_waveform


def ensure_checkpoint(checkpoint_path: str) -> str:
    """If checkpoint doesn't exist but .part* files do, merge them first."""
    target = Path(checkpoint_path)
    if target.exists():
        return str(target)

    parts = sorted(target.parent.glob(target.name + ".part*"),
                   key=lambda p: int(p.suffix.lstrip(".part")))
    if not parts:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Merging {len(parts)} parts -> {target.name} ...")
    with open(target, "wb") as out:
        for part in parts:
            out.write(part.read_bytes())
    print(f"  Merged: {target.name} ({target.stat().st_size / 1024**2:.1f} MB)")
    return str(target)


def separate_audio(model, stft_helper, input_path, output_dir, device,
                   target_sr=8000, segment_length=32000):
    """
    Separate a mixed audio file into individual sources.

    Handles long audio by processing in overlapping segments
    and stitching the results together.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load audio
    print(f"Loading audio: {input_path}")
    waveform, sr = load_audio(input_path, target_sr=target_sr)
    original_length = waveform.shape[0]
    print(f"  Duration: {original_length / target_sr:.2f}s, {original_length} samples")

    # If audio is short enough, process in one go
    if original_length <= segment_length:
        # Pad if too short
        if original_length < segment_length:
            waveform = torch.nn.functional.pad(waveform, (0, segment_length - original_length))

        waveform = waveform.to(device)
        mag, phase = stft_helper.stft(waveform)
        mag_input = mag.unsqueeze(0).unsqueeze(0)  # (1, 1, F, T)

        with torch.no_grad():
            masks = model(mag_input)  # (1, n_sources, F, T)

        separated = []
        for s in range(masks.shape[1]):
            est_mag = masks[0, s] * mag
            est_wav = stft_helper.istft(est_mag, phase, length=original_length)
            separated.append(normalize_waveform(est_wav.cpu()))
    else:
        # Process in overlapping segments
        hop = segment_length // 2  # 50% overlap
        n_sources = 2
        output_wavs = [torch.zeros(original_length) for _ in range(n_sources)]
        overlap_count = torch.zeros(original_length)

        for start in range(0, original_length, hop):
            end = min(start + segment_length, original_length)
            segment = waveform[start:end]

            # Pad if last segment is too short
            if segment.shape[0] < segment_length:
                segment = torch.nn.functional.pad(
                    segment, (0, segment_length - segment.shape[0])
                )

            segment = segment.to(device)
            mag, phase = stft_helper.stft(segment)
            mag_input = mag.unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                masks = model(mag_input)

            actual_len = min(segment_length, end - start)
            for s in range(n_sources):
                est_mag = masks[0, s] * mag
                est_wav = stft_helper.istft(est_mag, phase, length=segment_length)
                output_wavs[s][start:start + actual_len] += est_wav[:actual_len].cpu()

            overlap_count[start:start + actual_len] += 1

        # Average overlapping regions
        separated = []
        for s in range(n_sources):
            output_wavs[s] = output_wavs[s] / overlap_count.clamp(min=1)
            separated.append(normalize_waveform(output_wavs[s]))

    # Save separated sources
    for i, wav in enumerate(separated):
        out_path = os.path.join(output_dir, f"separated_source_{i + 1}.wav")
        save_audio(out_path, wav, sr=target_sr)
        print(f"  Saved: {out_path}")

    # Save the original (for comparison)
    orig_path = os.path.join(output_dir, "original_mixture.wav")
    save_audio(orig_path, waveform[:original_length].cpu(), sr=target_sr)
    print(f"  Saved: {orig_path}")

    # Generate visualization
    visualize_separation(waveform[:original_length].cpu(), separated,
                         target_sr, output_dir)

    return separated


def visualize_separation(mixture, separated_sources, sr, output_dir):
    """Create a visualization of the separation results."""
    n_sources = len(separated_sources)
    fig, axes = plt.subplots(n_sources + 1, 2, figsize=(16, 4 * (n_sources + 1)))

    # Mixture waveform and spectrogram
    t = np.arange(len(mixture)) / sr

    axes[0, 0].plot(t, mixture.numpy(), color='#3498db', linewidth=0.5)
    axes[0, 0].set_title("Mixture — Waveform", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].set_xlim(0, t[-1])

    axes[0, 1].specgram(mixture.numpy(), Fs=sr, cmap='magma')
    axes[0, 1].set_title("Mixture — Spectrogram", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Frequency (Hz)")

    colors = ['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    for i, src in enumerate(separated_sources):
        t_src = np.arange(len(src)) / sr
        color = colors[i % len(colors)]

        axes[i + 1, 0].plot(t_src, src.numpy(), color=color, linewidth=0.5)
        axes[i + 1, 0].set_title(f"Source {i + 1} — Waveform", fontsize=12, fontweight='bold')
        axes[i + 1, 0].set_xlabel("Time (s)")
        axes[i + 1, 0].set_ylabel("Amplitude")
        axes[i + 1, 0].set_xlim(0, t_src[-1])

        axes[i + 1, 1].specgram(src.numpy(), Fs=sr, cmap='magma')
        axes[i + 1, 1].set_title(f"Source {i + 1} — Spectrogram", fontsize=12, fontweight='bold')
        axes[i + 1, 1].set_xlabel("Time (s)")
        axes[i + 1, 1].set_ylabel("Frequency (Hz)")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "separation_visualization.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="RCNN Audio Separation — Inference")

    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input mixed audio file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./separated_output',
                        help='Directory to save separated audio files')
    parser.add_argument('--target_sr', type=int, default=8000,
                        help='Target sample rate')
    parser.add_argument('--segment_length', type=int, default=32000,
                        help='Segment length for processing')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint (auto-merge from parts if needed)
    args.checkpoint = ensure_checkpoint(args.checkpoint)
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = ckpt.get('args', {})

    # Reconstruct model
    n_fft = model_args.get('n_fft', 512)
    hop_length = model_args.get('hop_length', 128)

    model = RCNNSeparator(
        n_fft=n_fft,
        n_sources=model_args.get('n_sources', 2),
        lstm_hidden=model_args.get('lstm_hidden', 256),
        lstm_layers=model_args.get('lstm_layers', 2),
        dropout=0.0,  # No dropout at inference
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Loaded model from epoch {ckpt.get('epoch', '?')}")

    stft_helper = STFTHelper(n_fft=n_fft, hop_length=hop_length)

    # Run separation
    separate_audio(
        model, stft_helper, args.input, args.output_dir, device,
        target_sr=args.target_sr, segment_length=args.segment_length,
    )

    print("\nDone! Check the output directory for separated audio files.")


if __name__ == "__main__":
    main()
