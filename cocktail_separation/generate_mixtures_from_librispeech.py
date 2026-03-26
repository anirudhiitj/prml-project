#!/usr/bin/env python3
"""
Generate 3-speaker mixtures from LibriSpeech FLAC files.
Expects data in: data/librispeech_kaggle/LibriSpeech/{train-clean-100,train-clean-360}/
Each subfolder: speaker_id/chapter_id/*.flac
"""

import os
import sys
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm


def discover_speakers(librispeech_root: Path) -> dict:
    """Discover all speakers and their FLAC files from LibriSpeech directory structure."""
    speakers = defaultdict(list)

    for subset_dir in sorted(librispeech_root.iterdir()):
        if not subset_dir.is_dir():
            continue
        if not subset_dir.name.startswith("train-clean") and not subset_dir.name.startswith("dev-clean") and not subset_dir.name.startswith("test-clean"):
            # Also check for the LibriSpeech subdirectory structure
            sub_libri = subset_dir / "LibriSpeech"
            if sub_libri.exists():
                for inner_subset in sorted(sub_libri.iterdir()):
                    if inner_subset.is_dir():
                        for spk_dir in sorted(inner_subset.iterdir()):
                            if spk_dir.is_dir():
                                for chapter_dir in sorted(spk_dir.iterdir()):
                                    if chapter_dir.is_dir():
                                        flacs = sorted(chapter_dir.glob("*.flac"))
                                        if flacs:
                                            speakers[spk_dir.name].extend(flacs)
                continue

        for spk_dir in sorted(subset_dir.iterdir()):
            if spk_dir.is_dir():
                for chapter_dir in sorted(spk_dir.iterdir()):
                    if chapter_dir.is_dir():
                        flacs = sorted(chapter_dir.glob("*.flac"))
                        if flacs:
                            speakers[spk_dir.name].extend(flacs)

    return dict(speakers)


def load_audio_16k(path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load audio and resample to 16kHz mono."""
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = resample_poly(audio, target_sr, sr).astype(np.float32)
    return audio


def extract_clips(audio: np.ndarray, clip_samples: int, sr: int = 16000) -> list:
    """Extract non-overlapping clips from audio, skip if too short."""
    clips = []
    # Minimum clip length: 1 second
    if len(audio) < sr:
        return clips

    if len(audio) >= clip_samples:
        # Extract non-overlapping clips
        for start in range(0, len(audio) - clip_samples + 1, clip_samples):
            clip = audio[start:start + clip_samples]
            # Skip near-silent clips
            if np.std(clip) > 1e-4:
                clips.append(clip)
    else:
        # Pad short audio to clip length
        padded = np.zeros(clip_samples, dtype=np.float32)
        padded[:len(audio)] = audio
        if np.std(audio) > 1e-4:
            clips.append(padded)

    return clips


def generate_mixtures(
    speakers: dict,
    output_dir: Path,
    num_speakers: int = 3,
    num_mixtures_train: int = 20000,
    num_mixtures_val: int = 2000,
    clip_samples: int = 64000,
    sr: int = 16000,
):
    """Generate speaker mixtures with speaker-disjoint train/val splits."""
    print(f"\n{'='*80}")
    print(f"GENERATING {num_speakers}-SPEAKER MIXTURES")
    print(f"{'='*80}")
    print(f"Total speakers available: {len(speakers)}")
    print(f"Train mixtures: {num_mixtures_train}, Val mixtures: {num_mixtures_val}")
    print(f"Clip length: {clip_samples/sr:.1f}s ({clip_samples} samples)")
    print()

    # Load and extract clips for all speakers
    print("Step 1: Loading and extracting clips from all speakers...")
    speaker_clips = {}
    speaker_ids = sorted(speakers.keys())

    for spk_id in tqdm(speaker_ids, desc="Loading speakers"):
        all_clips = []
        for flac_path in speakers[spk_id]:
            try:
                audio = load_audio_16k(flac_path, sr)
                clips = extract_clips(audio, clip_samples, sr)
                all_clips.extend(clips)
            except Exception as e:
                continue

        if len(all_clips) >= 2:  # Need at least 2 clips per speaker
            speaker_clips[spk_id] = all_clips

    print(f"\nUsable speakers: {len(speaker_clips)}")
    total_clips = sum(len(c) for c in speaker_clips.values())
    print(f"Total clips: {total_clips}")
    print(f"Average clips/speaker: {total_clips/len(speaker_clips):.1f}")

    # Speaker-disjoint split: 80% train, 20% val
    all_speaker_ids = sorted(speaker_clips.keys())
    random.shuffle(all_speaker_ids)
    split_idx = int(0.8 * len(all_speaker_ids))
    train_speakers = all_speaker_ids[:split_idx]
    val_speakers = all_speaker_ids[split_idx:]

    print(f"\nTrain speakers: {len(train_speakers)}")
    print(f"Val speakers: {len(val_speakers)}")

    for split, spk_subset, num_mixtures in [
        ("train", train_speakers, num_mixtures_train),
        ("val", val_speakers, num_mixtures_val),
    ]:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating {num_mixtures} {split} mixtures from {len(spk_subset)} speakers...")

        generated = 0
        for mix_idx in tqdm(range(num_mixtures), desc=f"Generating {split}"):
            try:
                # Randomly select speakers (no same speaker twice)
                selected = random.sample(spk_subset, num_speakers)

                raw_sources = []
                for spk_id in selected:
                    clip = random.choice(speaker_clips[spk_id])
                    raw_sources.append(clip.copy())

                # Random gain for each speaker: uniform in [-5, +5] dB
                gains_db = [random.uniform(-5, 5) for _ in range(num_speakers)]
                gains = [10 ** (g / 20) for g in gains_db]

                # Time-shifted mixing: each speaker starts at a random offset
                # This creates realistic partial overlaps (30-100% overlap)
                # First speaker always starts at 0, others have random offsets
                max_offset = int(clip_samples * 0.5)  # up to 50% of clip length
                offsets = [0] + [random.randint(0, max_offset) for _ in range(num_speakers - 1)]

                # Output length is the clip_samples (fixed)
                out_len = clip_samples
                mixture = np.zeros(out_len, dtype=np.float32)
                scaled_sources = []

                for src, gain, offset in zip(raw_sources, gains, offsets):
                    # Place source at offset within the output window
                    padded = np.zeros(out_len, dtype=np.float32)
                    available = min(len(src), out_len - offset)
                    if available > 0:
                        padded[offset:offset + available] = src[:available] * gain
                    mixture += padded
                    scaled_sources.append(padded)

                # Normalize mixture to prevent clipping
                peak = np.abs(mixture).max()
                if peak > 0:
                    scale_factor = 0.9 / peak
                    mixture *= scale_factor
                    scaled_sources = [s * scale_factor for s in scaled_sources]

                # Save
                mix_dir = split_dir / f"{mix_idx:06d}"
                mix_dir.mkdir(parents=True, exist_ok=True)

                sf.write(str(mix_dir / "mixture.wav"), mixture, sr)
                for spk_idx, src in enumerate(scaled_sources):
                    sf.write(str(mix_dir / f"s{spk_idx + 1}.wav"), src, sr)

                generated += 1

            except Exception as e:
                print(f"\nError generating mixture {mix_idx}: {e}")
                continue

        print(f"✅ Generated {generated} {split} mixtures")

    print(f"\n{'='*80}")
    print(f"MIXTURE GENERATION COMPLETE")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")


def find_librispeech_root(base_dir: Path) -> Path:
    """Auto-detect LibriSpeech root from extracted Kaggle zip."""
    # Check common structures
    candidates = [
        base_dir,
        base_dir / "LibriSpeech",
        base_dir / "librispeech-clean",
    ]
    
    for candidate in candidates:
        if candidate.exists():
            # Check if it contains train-clean-* directories
            for subdir in candidate.iterdir():
                if subdir.is_dir() and "train-clean" in subdir.name:
                    return candidate
            # Check nested LibriSpeech dir
            nested = candidate / "LibriSpeech"
            if nested.exists():
                for subdir in nested.iterdir():
                    if subdir.is_dir() and "train-clean" in subdir.name:
                        return nested

    # Recursively search for train-clean directories
    for root, dirs, files in os.walk(str(base_dir)):
        for d in dirs:
            if "train-clean" in d:
                return Path(root)

    return base_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech-dir", type=str,
                        default="data/librispeech_kaggle",
                        help="Root directory containing extracted LibriSpeech data")
    parser.add_argument("--output-dir", type=str,
                        default="data/librispeech_full_mixtures/3spk",
                        help="Output directory for generated mixtures")
    parser.add_argument("--num-speakers", type=int, default=3)
    parser.add_argument("--num-train", type=int, default=20000)
    parser.add_argument("--num-val", type=int, default=2000)
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    clip_samples = int(args.clip_seconds * args.sample_rate)
    librispeech_dir = Path(args.librispeech_dir)

    print(f"Looking for LibriSpeech data in: {librispeech_dir}")
    root = find_librispeech_root(librispeech_dir)
    print(f"Found LibriSpeech root: {root}")

    speakers = discover_speakers(root)
    if not speakers:
        print(f"❌ No speakers found! Check directory structure.")
        print(f"Expected: {root}/train-clean-100/<speaker_id>/<chapter_id>/*.flac")
        sys.exit(1)

    print(f"✅ Found {len(speakers)} speakers with {sum(len(v) for v in speakers.values())} total files")

    generate_mixtures(
        speakers=speakers,
        output_dir=Path(args.output_dir),
        num_speakers=args.num_speakers,
        num_mixtures_train=args.num_train,
        num_mixtures_val=args.num_val,
        clip_samples=clip_samples,
        sr=args.sample_rate,
    )
