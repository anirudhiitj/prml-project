#!/usr/bin/env python3
"""
DPRNN-TasNet Inference Script
Separates multi-speaker audio into individual speaker sources.
Generates spectrograms and metrics after separation.
"""

import argparse
import torch
import torchaudio
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from src.model import DPRNNTasNet
from src.losses import si_snr, snr


def load_audio(audio_path: str, sample_rate: int = 16000, max_duration: float = 10.0) -> torch.Tensor:
    import librosa

    print(f"📂 Loading audio: {audio_path}")

    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception:
        waveform_np, sr = librosa.load(audio_path, sr=None, mono=True)
        waveform = torch.from_numpy(waveform_np).unsqueeze(0).float()

    if not isinstance(waveform, torch.Tensor):
        waveform = torch.from_numpy(waveform).float()

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    max_samples = int(sample_rate * max_duration)
    if waveform.shape[1] > max_samples:
        print(f"⚠️  Audio longer than {max_duration}s, truncating to {max_duration}s")
        waveform = waveform[:, :max_samples]

    duration = waveform.shape[1] / sample_rate
    print(f"✅ Loaded: {waveform.shape[1]:,} samples ({duration:.2f}s) @ {sample_rate}Hz")

    return waveform


def load_model(checkpoint_path: str, num_speakers: int, device: str = "cuda") -> torch.nn.Module:
    print(f"🔄 Loading model from: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]

    model = DPRNNTasNet(
        num_speakers=int(cfg["model"]["num_speakers"]),
        encoder_dim=int(cfg["model"]["encoder_dim"]),
        encoder_kernel=int(cfg["model"]["encoder_kernel"]),
        encoder_stride=int(cfg["model"]["encoder_stride"]),
        bottleneck_dim=int(cfg["model"]["bottleneck_dim"]),
        chunk_size=int(cfg["model"]["chunk_size"]),
        num_dprnn_blocks=int(cfg["model"]["num_dprnn_blocks"]),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"✅ Model loaded (epoch {ckpt['epoch']}, val_sisnr={float(ckpt['val_sisnr']):.2f} dB, {int(cfg['model']['num_speakers'])} speakers)")
    return model, cfg


@torch.no_grad()
def separate_speakers(model: torch.nn.Module, mixture: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    mixture = mixture.to(device)
    print(f"🔊 Separating {mixture.shape[1]:,} samples...")
    estimates = model(mixture)  # [1, num_speakers, num_samples]
    estimates = estimates.squeeze(0)  # [num_speakers, num_samples]
    return estimates.cpu()


def save_separated_audio(estimates: torch.Tensor, output_dir: str, sample_rate: int = 16000) -> None:
    import soundfile as sf

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving separated audio to: {output_dir}")

    for i, estimate in enumerate(estimates):
        output_file = output_path / f"speaker_{i+1}.wav"
        audio_np = estimate.numpy()
        max_val = np.abs(audio_np).max()
        if max_val > 1.0:
            audio_np = audio_np / max_val
        sf.write(str(output_file), audio_np, sample_rate, subtype='PCM_16')
        print(f"   ✅ Speaker {i+1}: {output_file}")


def compute_metrics(estimates: torch.Tensor, mixture: torch.Tensor, sources: torch.Tensor = None):
    """Compute separation quality metrics.

    Args:
        estimates: (num_speakers, T) separated signals
        mixture: (1, T) input mixture
        sources: (num_speakers, T) ground-truth sources (optional)

    Returns:
        dict of metric name -> value
    """
    num_spk = estimates.shape[0]
    metrics = {}

    # --- Energy / basic stats ---
    mix_energy = (mixture ** 2).mean().sqrt().item()
    metrics["mixture_rms"] = mix_energy
    for i in range(num_spk):
        est = estimates[i]
        metrics[f"speaker_{i+1}_rms"] = (est ** 2).mean().sqrt().item()
        metrics[f"speaker_{i+1}_peak"] = est.abs().max().item()

    # --- If ground-truth sources are provided, compute SI-SNR and SNR ---
    if sources is not None:
        min_len = min(estimates.shape[1], sources.shape[1])
        est_trimmed = estimates[:, :min_len]
        src_trimmed = sources[:, :min_len]

        from src.losses import _pairwise_metric, _hungarian_best_assignment

        pw = _pairwise_metric(est_trimmed.unsqueeze(0), src_trimmed.unsqueeze(0), si_snr)
        rows, cols = _hungarian_best_assignment(pw[0])

        for idx, (r, c) in enumerate(zip(rows, cols)):
            est_i = est_trimmed[r].unsqueeze(0)
            src_j = src_trimmed[c].unsqueeze(0)
            si = si_snr(est_i, src_j).item()
            sn = snr(est_i, src_j).item()
            metrics[f"speaker_{r.item()+1}_si_snr_dB"] = si
            metrics[f"speaker_{r.item()+1}_snr_dB"] = sn

        metrics["avg_si_snr_dB"] = np.mean([v for k, v in metrics.items() if k.endswith("_si_snr_dB")])
        metrics["avg_snr_dB"] = np.mean([v for k, v in metrics.items() if k.endswith("_snr_dB")])

    return metrics


def print_metrics(metrics: dict) -> None:
    print(f"\n{'='*60}")
    print(f"{'SEPARATION METRICS':^60}")
    print(f"{'='*60}")
    print(f"  {'Mixture RMS':.<40} {metrics['mixture_rms']:.6f}")
    print(f"  {'-'*56}")
    spk_keys = sorted(set(int(k.split('_')[1]) for k in metrics if k.startswith("speaker_") and "_rms" in k))
    for i in spk_keys:
        print(f"  Speaker {i}:")
        print(f"    {'RMS Energy':.<38} {metrics[f'speaker_{i}_rms']:.6f}")
        print(f"    {'Peak Amplitude':.<38} {metrics[f'speaker_{i}_peak']:.6f}")
        if f"speaker_{i}_si_snr_dB" in metrics:
            print(f"    {'SI-SNR':.<38} {metrics[f'speaker_{i}_si_snr_dB']:+.2f} dB")
            print(f"    {'SNR':.<38} {metrics[f'speaker_{i}_snr_dB']:+.2f} dB")
    if "avg_si_snr_dB" in metrics:
        print(f"  {'-'*56}")
        print(f"  {'Avg SI-SNR':.<40} {metrics['avg_si_snr_dB']:+.2f} dB")
        print(f"  {'Avg SNR':.<40} {metrics['avg_snr_dB']:+.2f} dB")
    print(f"{'='*60}")


def generate_spectrogram_image(
    mixture: torch.Tensor,
    estimates: torch.Tensor,
    metrics: dict,
    output_path: str,
    sample_rate: int = 16000,
    sources: torch.Tensor = None,
) -> None:
    """Generate a single image with waveforms + spectrograms of mixture + separated speakers + metrics.

    Layout per audio track: [waveform (left) | spectrogram (right)]
    Args:
        mixture: (1, T)
        estimates: (num_speakers, T)
        metrics: dict of computed metrics
        output_path: path to save the PNG
        sample_rate: audio sample rate
        sources: (num_speakers, T) optional ground-truth sources
    """
    num_spk = estimates.shape[0]
    has_gt = sources is not None

    # Each audio track gets one row with 2 columns: waveform + spectrogram
    n_audio_rows = 1 + num_spk + (num_spk if has_gt else 0)
    n_rows = n_audio_rows + 1  # +1 for metrics text panel

    fig = plt.figure(figsize=(18, 3.2 * n_rows))
    # 2 columns: waveform (width 1) | spectrogram (width 2.5)
    outer_gs = gridspec.GridSpec(
        n_rows, 1,
        height_ratios=[1] * n_audio_rows + [0.8],
        hspace=0.55,
    )

    n_fft = 1024
    hop_length = 256

    def plot_waveform(ax, waveform_1d, title, sr, color):
        t = np.arange(len(waveform_1d)) / sr
        ax.plot(t, waveform_1d, color=color, linewidth=0.6, alpha=0.85)
        ax.set_xlim(0, t[-1])
        peak = max(np.abs(waveform_1d).max(), 1e-6)
        ax.set_ylim(-peak * 1.1, peak * 1.1)
        ax.set_ylabel("Amplitude")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
        # Show actual peak value so scale is clear
        ax.annotate(f"peak={peak:.4f}", xy=(0.98, 0.92), xycoords="axes fraction",
                    fontsize=7, ha="right", color="gray")
        ax.tick_params(axis="x", labelsize=8)

    def plot_spectrogram(ax, waveform_1d, sr):
        S = np.abs(np.fft.rfft(
            np.lib.stride_tricks.sliding_window_view(
                np.pad(waveform_1d, (n_fft // 2, n_fft // 2)),
                n_fft
            )[::hop_length] * np.hanning(n_fft)
        ))
        S_db = 20 * np.log10(np.maximum(S, 1e-10))
        time_axis = np.arange(S_db.shape[0]) * hop_length / sr
        freq_axis = np.arange(S_db.shape[1]) * sr / n_fft
        im = ax.pcolormesh(time_axis, freq_axis, S_db.T, shading="gouraud", cmap="magma")
        ax.set_ylabel("Freq (Hz)")
        ax.set_ylim(0, sr // 2)
        ax.tick_params(axis="x", labelsize=8)
        return im

    def add_audio_row(row_idx, waveform_1d, label, color):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer_gs[row_idx], width_ratios=[1, 2.5], wspace=0.25
        )
        ax_w = fig.add_subplot(inner[0])
        ax_s = fig.add_subplot(inner[1])
        plot_waveform(ax_w, waveform_1d, label, sample_rate, color)
        plot_spectrogram(ax_s, waveform_1d, sample_rate)
        ax_s.set_title(f"{label} — Spectrogram", fontsize=11, fontweight="bold")
        return ax_w, ax_s

    COLORS = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]

    row = 0

    # --- Mixture ---
    mix_np = mixture.squeeze().numpy()
    add_audio_row(row, mix_np, "Input Mixture", "#607D8B")
    row += 1

    # --- Ground-truth sources (optional) ---
    if has_gt:
        for i in range(num_spk):
            src_np = sources[i].numpy()
            add_audio_row(row, src_np, f"Ground Truth — Speaker {i+1}", COLORS[i % len(COLORS)])
            row += 1

    # --- Separated estimates ---
    for i in range(num_spk):
        est_np = estimates[i].numpy()
        ax_w, ax_s = add_audio_row(row, est_np, f"Separated — Speaker {i+1}", COLORS[i % len(COLORS)])
        if i == num_spk - 1:
            ax_w.set_xlabel("Time (s)")
            ax_s.set_xlabel("Time (s)")
        row += 1

    # --- Metrics text panel ---
    ax_text = fig.add_subplot(outer_gs[row])
    ax_text.axis("off")

    lines = ["METRICS"]
    for i in sorted(set(int(k.split('_')[1]) for k in metrics if k.startswith("speaker_") and "_rms" in k)):
        parts = [f"Speaker {i}:  RMS={metrics[f'speaker_{i}_rms']:.4f}  Peak={metrics[f'speaker_{i}_peak']:.4f}"]
        if f"speaker_{i}_si_snr_dB" in metrics:
            parts.append(f"SI-SNR={metrics[f'speaker_{i}_si_snr_dB']:+.2f} dB  SNR={metrics[f'speaker_{i}_snr_dB']:+.2f} dB")
        lines.append("  ".join(parts))
    if "avg_si_snr_dB" in metrics:
        lines.append(f"Average:  SI-SNR={metrics['avg_si_snr_dB']:+.2f} dB   SNR={metrics['avg_snr_dB']:+.2f} dB")

    ax_text.text(
        0.5, 0.5, "\n".join(lines),
        transform=ax_text.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="center",
        horizontalalignment="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="gray"),
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n🖼️  Spectrogram image saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Separate multi-speaker audio using DPRNN-TasNet")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num-speakers", type=int, required=True, help="Number of speakers (2-5)")
    parser.add_argument("--output-dir", type=str, default="separated_output", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--sources", type=str, nargs="+", default=None,
        help="Paths to ground-truth source wav files (one per speaker) for metric computation",
    )

    args = parser.parse_args()

    # Validate
    if not Path(args.audio).exists():
        print(f"❌ Audio file not found: {args.audio}")
        return

    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return

    if args.num_speakers < 2 or args.num_speakers > 5:
        print(f"❌ Invalid num_speakers: {args.num_speakers}. Must be 2-5.")
        return

    print(f"\n{'='*70}")
    print(f"DPRNN-TasNet Speaker Separation")
    print(f"{'='*70}")
    print(f"Audio:         {args.audio}")
    print(f"Checkpoint:    {args.checkpoint}")
    print(f"Num Speakers:  {args.num_speakers}")
    print(f"Device:        {args.device}")
    print(f"Output Dir:    {args.output_dir}")
    if args.sources:
        print(f"GT Sources:    {args.sources}")
    print(f"{'='*70}\n")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load audio and model
    mixture = load_audio(args.audio, max_duration=10.0)
    model, cfg = load_model(args.checkpoint, args.num_speakers, device=str(device))

    # Load ground-truth sources if provided
    gt_sources = None
    if args.sources:
        gt_list = []
        for sp in args.sources:
            gt_list.append(load_audio(sp, max_duration=10.0).squeeze(0))
        gt_sources = torch.stack(gt_list, dim=0)  # (num_speakers, T)

    # Separate speakers
    estimates = separate_speakers(model, mixture, device=str(device))

    # Save output
    save_separated_audio(estimates, args.output_dir)

    # Compute & print metrics
    metrics = compute_metrics(estimates, mixture, sources=gt_sources)
    print_metrics(metrics)

    # Generate combined spectrogram image
    img_path = str(Path(args.output_dir) / "separation_results.png")
    generate_spectrogram_image(
        mixture, estimates, metrics, img_path,
        sample_rate=16000, sources=gt_sources,
    )

    print(f"\n{'='*70}")
    print(f"✅ Separation complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
