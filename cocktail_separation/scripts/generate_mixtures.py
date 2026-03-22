from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed-duration speech mixtures")
    parser.add_argument("--source_root", type=str, required=True, help="Path containing speaker subfolders")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--num_mixtures", type=int, required=True)
    parser.add_argument("--num_speakers", type=int, default=5)
    parser.add_argument("--clip_samples", type=int, default=64000)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--snr_min", type=float, default=-5.0)
    parser.add_argument("--snr_max", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_mono_16k(path: Path, target_sr: int) -> torch.Tensor:
    x, sr = sf.read(str(path), always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != target_sr:
        x = resample_poly(x, target_sr, sr)
    return torch.from_numpy(np.asarray(x, dtype=np.float32))


def fix_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.numel() > target_len:
        start = random.randint(0, x.numel() - target_len)
        x = x[start : start + target_len]
    elif x.numel() < target_len:
        x = torch.nn.functional.pad(x, (0, target_len - x.numel()))
    return x


def build_speaker_index(source_root: Path) -> dict[str, list[Path]]:
    speakers = {}
    for spk_dir in source_root.iterdir():
        if not spk_dir.is_dir():
            continue
        wavs = list(spk_dir.rglob("*.wav")) + list(spk_dir.rglob("*.flac"))
        if wavs:
            speakers[spk_dir.name] = wavs
    return speakers


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    speaker_map = build_speaker_index(source_root)
    speakers = list(speaker_map.keys())
    if len(speakers) < args.num_speakers:
        raise RuntimeError("Not enough distinct speakers to build mixtures")

    for idx in tqdm(range(args.num_mixtures), desc="mixtures"):
        chosen = random.sample(speakers, args.num_speakers)
        sources = []

        for spk_pos, spk in enumerate(chosen):
            utt = random.choice(speaker_map[spk])
            src = load_mono_16k(utt, args.sample_rate)
            src = fix_length(src, args.clip_samples)

            if spk_pos == 0:
                gain = 1.0
            else:
                snr_db = random.uniform(args.snr_min, args.snr_max)
                gain = 10 ** (snr_db / 20.0)
            sources.append(gain * src)

        source_stack = torch.stack(sources, dim=0)
        mixture = source_stack.sum(dim=0)

        norm = mixture.abs().max().clamp(min=1e-8)
        mixture = mixture / norm
        source_stack = source_stack / norm

        ex_dir = output_root / f"{idx:07d}"
        ex_dir.mkdir(parents=True, exist_ok=True)

        sf.write(str(ex_dir / "mixture.wav"), mixture.cpu().numpy(), args.sample_rate)
        for sidx in range(args.num_speakers):
            sf.write(str(ex_dir / f"s{sidx + 1}.wav"), source_stack[sidx].cpu().numpy(), args.sample_rate)


if __name__ == "__main__":
    main()
