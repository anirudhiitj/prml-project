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
from torch.utils.data.distributed import DistributedSampler

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


def _load_mono_16k(path: Path, sample_rate: int = 16000) -> torch.Tensor:
    """Load mono audio at target sample rate with robust error handling."""
    try:
        # Try reading with soundfile
        audio, sr = sf.read(str(path), always_2d=False, dtype=np.float32)
    except Exception as e:
        # Fallback: try with librosa
        try:
            import librosa
            audio = librosa.load(str(path), sr=sample_rate, mono=True)[0].astype(np.float32)
            sr = sample_rate
        except Exception as fallback_err:
            raise RuntimeError(f"Failed to load audio from {path}: soundfile error: {e}, librosa error: {fallback_err}")
    
    # Ensure mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    # Resample if needed
    if sr != sample_rate:
        audio = resample_poly(audio, sample_rate, sr)
    
    # Convert to tensor
    return torch.from_numpy(np.asarray(audio, dtype=np.float32))


def _fix_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.numel() > target_len:
        start = random.randint(0, x.numel() - target_len)
        return x[start : start + target_len]
    if x.numel() < target_len:
        return torch.nn.functional.pad(x, (0, target_len - x.numel()))
    return x


class MixtureDataset(Dataset):
    """Dataset for pre-generated mixture folders with mixture.wav and s1..sC.wav."""

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

        self.examples: list[Path] = sorted([p for p in self.split_root.iterdir() if p.is_dir()])
        if not self.examples:
            raise RuntimeError(f"No mixture folders found in {self.split_root}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ex_dir = self.examples[idx]
        try:
            mixture = _load_mono_16k(ex_dir / "mixture.wav", sample_rate=self.sample_rate)
            mixture = _fix_length(mixture, self.clip_samples)

            sources = []
            for spk_idx in range(1, self.num_speakers + 1):
                source = _load_mono_16k(ex_dir / f"s{spk_idx}.wav", sample_rate=self.sample_rate)
                source = _fix_length(source, self.clip_samples)
                sources.append(source)

            source_tensor = torch.stack(sources, dim=0)
            return mixture.float(), source_tensor.float()
        except Exception as e:
            print(f"\n❌ ERROR loading example {idx} from {ex_dir}: {e}")
            # Return dummy data instead of crashing
            mixture = torch.randn(self.clip_samples, dtype=torch.float32)
            sources = torch.randn(self.num_speakers, self.clip_samples, dtype=torch.float32)
            return mixture, sources


def build_dataloader(
    split_root: str | Path,
    num_speakers: int,
    batch_size: int,
    num_workers: int,
    distributed: bool,
    shuffle: bool,
    clip_samples: int = 64000,
    sample_rate: int = 16000,
) -> DataLoader:
    dataset = MixtureDataset(
        split_root=split_root,
        num_speakers=num_speakers,
        clip_samples=clip_samples,
        sample_rate=sample_rate,
    )

    sampler: Optional[DistributedSampler] = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)

    use_pin_memory = torch.cuda.is_available()

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": (shuffle and sampler is None),
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": use_pin_memory,
        "persistent_workers": num_workers > 0,
        "drop_last": shuffle,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    return DataLoader(**loader_kwargs)
