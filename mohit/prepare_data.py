"""
Data preparation script for the RCNN Cocktail Party Audio Separation project.

Downloads LibriSpeech and pre-generates fixed sets of 2-speaker mixtures
saved as .pt files for deterministic, fast training.

Usage:
    python prepare_data.py --data_root ./data --num_train 5000 --num_val 500
"""

import os
import argparse
import torch
import torchaudio
import random
from collections import defaultdict
from tqdm import tqdm

from utils import STFTHelper


def load_and_preprocess(dataset, idx, target_sr=8000, segment_length=32000):
    """Load, resample, and trim/pad a single utterance from LibriSpeech."""
    waveform, sr, _, _, _, _ = dataset[idx]
    waveform = waveform.squeeze(0)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    if waveform.shape[0] >= segment_length:
        start = random.randint(0, waveform.shape[0] - segment_length)
        waveform = waveform[start:start + segment_length]
    else:
        pad_len = segment_length - waveform.shape[0]
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))

    return waveform


def generate_mixtures(dataset, speaker_to_indices, speaker_ids,
                      output_dir, num_samples, stft_helper,
                      target_sr=8000, segment_length=32000, seed=42):
    """Generate and save mixture samples."""
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)

    for i in tqdm(range(num_samples), desc=f"Generating → {output_dir}"):
        # Pick 2 different speakers
        spk1, spk2 = rng.sample(speaker_ids, 2)
        idx1 = rng.choice(speaker_to_indices[spk1])
        idx2 = rng.choice(speaker_to_indices[spk2])
        snr_offset = rng.uniform(-5, 5)

        # Load & preprocess
        source1 = load_and_preprocess(dataset, idx1, target_sr, segment_length)
        source2 = load_and_preprocess(dataset, idx2, target_sr, segment_length)

        # Normalize
        source1 = source1 / (source1.abs().max() + 1e-8)
        source2 = source2 / (source2.abs().max() + 1e-8)

        # Apply SNR offset
        snr_linear = 10 ** (snr_offset / 20.0)
        source2 = source2 * snr_linear

        # Mix
        mixture = source1 + source2

        # Compute STFTs
        mix_mag, mix_phase = stft_helper.stft(mixture)
        s1_mag, _ = stft_helper.stft(source1)
        s2_mag, _ = stft_helper.stft(source2)

        # Save
        sample = {
            'mixture_mag': mix_mag,
            'source1_mag': s1_mag,
            'source2_mag': s2_mag,
            'mixture_phase': mix_phase,
            'mixture_wav': mixture,
            'source1_wav': source1,
            'source2_wav': source2,
        }
        torch.save(sample, os.path.join(output_dir, f"sample_{i:06d}.pt"))

    print(f"Saved {num_samples} samples to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare LibriSpeech mixture dataset")
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for LibriSpeech download')
    parser.add_argument('--output_root', type=str, default='./data/generated',
                        help='Output directory for generated mixtures')
    parser.add_argument('--num_train', type=int, default=5000,
                        help='Number of training mixtures')
    parser.add_argument('--num_val', type=int, default=500,
                        help='Number of validation mixtures')
    parser.add_argument('--target_sr', type=int, default=8000,
                        help='Target sample rate')
    parser.add_argument('--segment_length', type=int, default=32000,
                        help='Segment length in samples (4s @ 8kHz = 32000)')
    parser.add_argument('--n_fft', type=int, default=512)
    parser.add_argument('--hop_length', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    stft_helper = STFTHelper(n_fft=args.n_fft, hop_length=args.hop_length)

    # ─── Download LibriSpeech ───
    print("=" * 60)
    print("Downloading LibriSpeech train-clean-100 ...")
    print("=" * 60)
    train_dataset = torchaudio.datasets.LIBRISPEECH(
        root=args.data_root, url="train-clean-100", download=True
    )

    print("=" * 60)
    print("Downloading LibriSpeech dev-clean ...")
    print("=" * 60)
    val_dataset = torchaudio.datasets.LIBRISPEECH(
        root=args.data_root, url="dev-clean", download=True
    )

    # ─── Group by speaker ───
    def group_by_speaker(dataset):
        speaker_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            _, _, _, speaker_id, _, _ = dataset[idx]
            speaker_to_indices[speaker_id].append(idx)
        return speaker_to_indices

    train_speakers = group_by_speaker(train_dataset)
    val_speakers = group_by_speaker(val_dataset)

    print(f"Train: {len(train_speakers)} speakers, {len(train_dataset)} utterances")
    print(f"Val:   {len(val_speakers)} speakers, {len(val_dataset)} utterances")

    # ─── Generate Mixtures ───
    train_dir = os.path.join(args.output_root, "train")
    val_dir = os.path.join(args.output_root, "val")

    generate_mixtures(
        train_dataset, train_speakers, list(train_speakers.keys()),
        train_dir, args.num_train, stft_helper,
        args.target_sr, args.segment_length, args.seed,
    )

    generate_mixtures(
        val_dataset, val_speakers, list(val_speakers.keys()),
        val_dir, args.num_val, stft_helper,
        args.target_sr, args.segment_length, args.seed + 1,
    )

    print("=" * 60)
    print("Data preparation complete!")
    print(f"  Train: {train_dir} ({args.num_train} samples)")
    print(f"  Val:   {val_dir} ({args.num_val} samples)")
    print("=" * 60)


if __name__ == "__main__":
    main()
