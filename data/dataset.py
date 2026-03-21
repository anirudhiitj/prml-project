"""
Dataset loader for audio separation.

Supports two layouts:
    1. Pre-mixed (LibriMix-style): separate folders for mix/, s1/, s2/
    2. On-the-fly mixing: folder of clean utterances, randomly paired and summed

Each sample returns:
    mixture:  (1, T) — mixed waveform
    sources:  (C, T) — C ground-truth source waveforms
"""

import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from utils.audio_utils import load_audio


class PreMixedDataset(Dataset):
    """
    Dataset for pre-mixed audio (e.g., LibriMix).

    Expected directory structure:
        data_dir/
        ├── mix/        # mixture files
        ├── s1/         # source 1 files
        └── s2/         # source 2 files

    File names must match across all folders.

    Args:
        data_dir (str): Root directory containing mix/, s1/, s2/.
        sample_rate (int): Target sample rate.
        max_len (float): Maximum audio length in seconds (for padding/trimming).
        num_sources (int): Number of sources (C).
    """

    def __init__(self, data_dir: str, sample_rate: int = 8000,
                 max_len: float = 4.0, num_sources: int = 2):
        super().__init__()
        self.sample_rate = sample_rate
        self.max_samples = int(max_len * sample_rate)
        self.num_sources = num_sources

        mix_dir = Path(data_dir) / "mix"
        self.source_dirs = [Path(data_dir) / f"s{i+1}" for i in range(num_sources)]

        # List all mixture files
        self.filenames = sorted(os.listdir(mix_dir))
        self.mix_dir = mix_dir

        # Verify all source directories exist
        for sd in self.source_dirs:
            assert sd.exists(), f"Source directory not found: {sd}"

    def __len__(self):
        return len(self.filenames)

    def _pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pad with zeros or trim to self.max_samples."""
        T = waveform.shape[-1]
        if T > self.max_samples:
            return waveform[:, :self.max_samples]
        elif T < self.max_samples:
            pad = self.max_samples - T
            return torch.nn.functional.pad(waveform, (0, pad))
        return waveform

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # Load mixture
        mixture = load_audio(str(self.mix_dir / fname), self.sample_rate)
        mixture = self._pad_or_trim(mixture)  # (1, T)

        # Load sources
        sources = []
        for sd in self.source_dirs:
            src = load_audio(str(sd / fname), self.sample_rate)
            src = self._pad_or_trim(src)  # (1, T)
            sources.append(src)

        sources = torch.cat(sources, dim=0)  # (C, T)

        return mixture, sources  # (1, T), (C, T)


class OnTheFlyMixDataset(Dataset):
    """
    Dataset that creates mixtures on-the-fly from clean utterances.

    Randomly pairs clean audio files and sums them to create mixtures.
    Useful when you only have a folder of single-speaker recordings.

    Args:
        clean_dir (str): Directory of clean single-speaker WAV files.
        sample_rate (int): Target sample rate.
        max_len (float): Max audio length in seconds.
        num_sources (int): Number of sources to mix (C).
        num_samples (int): Number of virtual samples in the dataset per epoch.
    """

    def __init__(self, clean_dir: str, sample_rate: int = 8000,
                 max_len: float = 4.0, num_sources: int = 2,
                 num_samples: int = 3000):
        super().__init__()
        self.sample_rate = sample_rate
        self.max_samples = int(max_len * sample_rate)
        self.num_sources = num_sources
        self.num_samples = num_samples

        # Collect all audio files
        self.files = sorted([
            str(Path(clean_dir) / f)
            for f in os.listdir(clean_dir)
            if f.endswith((".wav", ".flac", ".mp3"))
        ])

        assert len(self.files) >= num_sources, (
            f"Need at least {num_sources} files, found {len(self.files)}"
        )

    def __len__(self):
        return self.num_samples

    def _pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        T = waveform.shape[-1]
        if T > self.max_samples:
            # Random crop
            start = random.randint(0, T - self.max_samples)
            return waveform[:, start:start + self.max_samples]
        elif T < self.max_samples:
            pad = self.max_samples - T
            return torch.nn.functional.pad(waveform, (0, pad))
        return waveform

    def __getitem__(self, idx):
        # Randomly pick num_sources files
        chosen = random.sample(self.files, self.num_sources)

        sources = []
        for fpath in chosen:
            src = load_audio(fpath, self.sample_rate)
            src = self._pad_or_trim(src)  # (1, T)
            sources.append(src)

        sources = torch.cat(sources, dim=0)  # (C, T)

        # Create mixture by summing sources
        mixture = sources.sum(dim=0, keepdim=True)  # (1, T)

        return mixture, sources  # (1, T), (C, T)
