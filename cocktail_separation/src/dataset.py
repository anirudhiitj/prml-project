from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def _load_mono_16k(path: Path, sample_rate: int = 16000) -> torch.Tensor:
    audio, sr = torchaudio.load(str(path))
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    return audio.squeeze(0)


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

        self.examples: List[Path] = sorted([p for p in self.split_root.iterdir() if p.is_dir()])
        if not self.examples:
            raise RuntimeError(f"No mixture folders found in {self.split_root}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ex_dir = self.examples[idx]
        mixture = _load_mono_16k(ex_dir / "mixture.wav", sample_rate=self.sample_rate)
        mixture = _fix_length(mixture, self.clip_samples)

        sources = []
        for spk_idx in range(1, self.num_speakers + 1):
            source = _load_mono_16k(ex_dir / f"s{spk_idx}.wav", sample_rate=self.sample_rate)
            source = _fix_length(source, self.clip_samples)
            sources.append(source)

        source_tensor = torch.stack(sources, dim=0)
        return mixture.float(), source_tensor.float()


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

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": (shuffle and sampler is None),
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
        "drop_last": shuffle,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    return DataLoader(**loader_kwargs)
