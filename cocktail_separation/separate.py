from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torchaudio
from omegaconf import OmegaConf

from src.model import DPRNNTasNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Separate speakers from a single mixture file")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    cfg = OmegaConf.load(config_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    defaults = cfg_dict.get("defaults", []) if isinstance(cfg_dict, dict) else []

    if defaults:
        merged = OmegaConf.create()
        base_dir = Path(config_path).parent
        for entry in defaults:
            if isinstance(entry, str):
                base_cfg = OmegaConf.load(base_dir / f"{entry}.yaml")
                merged = OmegaConf.merge(merged, base_cfg)
        merged = OmegaConf.merge(merged, cfg)
        return OmegaConf.to_container(merged, resolve=True)

    return cfg_dict


def load_audio_mono_16k(path: Path, sample_rate: int = 16000) -> torch.Tensor:
    audio, sr = torchaudio.load(str(path))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    return audio.squeeze(0)


def overlap_add_segments(segments: List[torch.Tensor], hop: int, total_len: int) -> torch.Tensor:
    """segments list entries are (C, seg_len)."""
    c, seg_len = segments[0].shape
    out = torch.zeros((c, total_len), dtype=segments[0].dtype)
    weight = torch.zeros((1, total_len), dtype=segments[0].dtype)

    for idx, seg in enumerate(segments):
        start = idx * hop
        end = min(start + seg_len, total_len)
        valid = end - start
        out[:, start:end] += seg[:, :valid]
        weight[:, start:end] += 1.0

    return out / torch.clamp(weight, min=1.0)


def main() -> None:
    args = parse_args()
    cfg: Dict = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DPRNNTasNet(**cfg["model"]).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sr = int(cfg["data"]["sample_rate"])
    clip_len = int(cfg["data"]["clip_samples"])
    overlap = int(0.5 * sr)
    hop = clip_len - overlap

    waveform = load_audio_mono_16k(Path(args.input), sample_rate=sr)
    orig_len = waveform.numel()

    if orig_len < clip_len:
        waveform = torch.nn.functional.pad(waveform, (0, clip_len - orig_len))

    segments = []
    starts = list(range(0, max(1, waveform.numel() - overlap), hop))

    with torch.no_grad():
        for start in starts:
            end = start + clip_len
            chunk = waveform[start:end]
            if chunk.numel() < clip_len:
                chunk = torch.nn.functional.pad(chunk, (0, clip_len - chunk.numel()))
            est = model(chunk.unsqueeze(0).to(device))[0].cpu()
            segments.append(est)

    total_len = max(orig_len, starts[-1] + clip_len)
    separated = overlap_add_segments(segments, hop=hop, total_len=total_len)
    separated = separated[:, :orig_len]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(separated.shape[0]):
        out_path = out_dir / f"speaker_{idx + 1}.wav"
        torchaudio.save(str(out_path), separated[idx].unsqueeze(0), sr)


if __name__ == "__main__":
    main()
