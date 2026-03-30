"""Augmented dataset with dynamic remixing and on-the-fly augmentation."""
from __future__ import annotations

import random
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


def _load_mono_16k(path: Path, sample_rate: int = 16000) -> np.ndarray:
    """Load mono audio at target sample rate, return numpy array."""
    audio, sr = sf.read(str(path), always_2d=False, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = resample_poly(audio, sample_rate, sr).astype(np.float32)
    return audio


def _fix_length_np(x: np.ndarray, target_len: int) -> np.ndarray:
    if len(x) > target_len:
        start = random.randint(0, len(x) - target_len)
        return x[start : start + target_len]
    if len(x) < target_len:
        return np.pad(x, (0, target_len - len(x)))
    return x


class DynamicMixDataset(Dataset):
    """
    Loads all individual source utterances from pre-generated mixture folders,
    then dynamically remixes them each __getitem__ call with augmentation.
    
    This gives effectively infinite training data from the same source pool.
    """

    def __init__(
        self,
        split_root: str | Path,
        num_speakers: int,
        clip_samples: int = 64000,
        sample_rate: int = 16000,
        gain_range_db: float = 5.0,
        speed_perturb: bool = True,
        speed_range: tuple = (0.95, 1.05),
        epoch_size: int = 0,
    ) -> None:
        self.split_root = Path(split_root)
        self.num_speakers = num_speakers
        self.clip_samples = clip_samples
        self.sample_rate = sample_rate
        self.gain_range_db = gain_range_db
        self.speed_perturb = speed_perturb
        self.speed_range = speed_range

        if not self.split_root.exists():
            raise FileNotFoundError(f"Dataset split path not found: {self.split_root}")

        # Collect ALL individual source paths from all mixture folders
        self.source_paths: list[Path] = []
        mix_dirs = sorted([p for p in self.split_root.iterdir() if p.is_dir()])
        for d in mix_dirs:
            for spk_idx in range(1, num_speakers + 1):
                src_path = d / f"s{spk_idx}.wav"
                if src_path.exists():
                    self.source_paths.append(src_path)

        if len(self.source_paths) < num_speakers:
            raise RuntimeError(f"Found only {len(self.source_paths)} sources in {self.split_root}")

        # epoch_size: how many mixtures per epoch. Default = same as original dataset
        self.epoch_size = epoch_size if epoch_size > 0 else len(mix_dirs)

        print(f"  DynamicMixDataset: {len(self.source_paths)} source utterances, "
              f"{self.epoch_size} mixtures/epoch, gain=±{gain_range_db}dB, "
              f"speed_perturb={speed_perturb}")

    def __len__(self) -> int:
        return self.epoch_size

    def _augment_source(self, audio: np.ndarray) -> np.ndarray:
        """Apply random gain and optional speed perturbation to a source."""
        # Random gain in dB
        gain_db = random.uniform(-self.gain_range_db, self.gain_range_db)
        audio = audio * (10.0 ** (gain_db / 20.0))

        # Speed perturbation
        if self.speed_perturb:
            speed = random.uniform(*self.speed_range)
            if abs(speed - 1.0) > 0.005:
                # Use integer ratio approximation for resample_poly
                orig_len = len(audio)
                new_len = int(orig_len / speed)
                if new_len > 0:
                    audio = resample_poly(audio, new_len, orig_len).astype(np.float32)

        return audio

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Pick num_speakers random sources (no replacement)
        chosen = random.sample(self.source_paths, self.num_speakers)

        sources = []
        for src_path in chosen:
            audio = _load_mono_16k(src_path, self.sample_rate)
            audio = self._augment_source(audio)
            audio = _fix_length_np(audio, self.clip_samples)
            sources.append(audio)

        # Create mixture by summing
        sources_np = np.stack(sources, axis=0)  # (C, T)
        mixture_np = sources_np.sum(axis=0)  # (T,)

        # Normalize mixture to prevent clipping, scale sources same way
        peak = np.abs(mixture_np).max()
        if peak > 0.9:
            scale = 0.9 / peak
            mixture_np = mixture_np * scale
            sources_np = sources_np * scale

        mixture = torch.from_numpy(mixture_np.astype(np.float32))
        source_tensor = torch.from_numpy(sources_np.astype(np.float32))

        return mixture, source_tensor


class StaticMixDataset(Dataset):
    """Original fixed-mixture dataset for validation (no augmentation)."""

    def __init__(
        self,
        split_root: str | Path,
        num_speakers: int,
        clip_samples: int = 64000,
        sample_rate: int = 16000,
    ) -> None:
        self.split_root = Path(split_root)
        self.num_speakers = num_speakers
        self.clip_samples = clip_samples
        self.sample_rate = sample_rate

        if not self.split_root.exists():
            raise FileNotFoundError(f"Dataset split path not found: {self.split_root}")
        self.examples = sorted([p for p in self.split_root.iterdir() if p.is_dir()])
        if not self.examples:
            raise RuntimeError(f"No mixture folders in {self.split_root}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ex_dir = self.examples[idx]
        mixture = _load_mono_16k(ex_dir / "mixture.wav", self.sample_rate)
        mixture = _fix_length_np(mixture, self.clip_samples)

        sources = []
        for spk_idx in range(1, self.num_speakers + 1):
            src = _load_mono_16k(ex_dir / f"s{spk_idx}.wav", self.sample_rate)
            src = _fix_length_np(src, self.clip_samples)
            sources.append(src)

        sources_np = np.stack(sources, axis=0)
        return (
            torch.from_numpy(mixture.astype(np.float32)),
            torch.from_numpy(sources_np.astype(np.float32)),
        )


def build_augmented_dataloaders(
    data_root: str | Path,
    num_speakers: int,
    train_batch_size: int,
    val_batch_size: int,
    num_workers: int,
    clip_samples: int = 64000,
    sample_rate: int = 16000,
    gain_range_db: float = 5.0,
    speed_perturb: bool = True,
    epoch_size: int = 0,
):
    """Build train (dynamic) and val (static) dataloaders."""
    data_root = Path(data_root)

    train_dataset = DynamicMixDataset(
        split_root=data_root / "train",
        num_speakers=num_speakers,
        clip_samples=clip_samples,
        sample_rate=sample_rate,
        gain_range_db=gain_range_db,
        speed_perturb=speed_perturb,
        epoch_size=epoch_size,
    )

    val_dataset = StaticMixDataset(
        split_root=data_root / "val",
        num_speakers=num_speakers,
        clip_samples=clip_samples,
        sample_rate=sample_rate,
    )

    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin,
        persistent_workers=num_workers > 0,
        drop_last=True,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=max(4, num_workers // 2),
        pin_memory=use_pin,
        persistent_workers=True,
        prefetch_factor=4,
    )

    return train_loader, val_loader
