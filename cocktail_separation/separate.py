from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf
from scipy.signal import resample_poly

from src.model import DPRNNTasNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Separate speakers from a single mixture file")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    def _load_recursive(path: Path):
        cfg_obj = OmegaConf.load(path)
        cfg_obj_dict = OmegaConf.to_container(cfg_obj, resolve=False)
        defaults = cfg_obj_dict.get("defaults", []) if isinstance(cfg_obj_dict, dict) else []

        merged = OmegaConf.create()
        for entry in defaults:
            if isinstance(entry, str):
                dep_path = (path.parent / f"{entry}.yaml").resolve()
                merged = OmegaConf.merge(merged, _load_recursive(dep_path))

        return OmegaConf.merge(merged, cfg_obj)

    final_cfg = _load_recursive(Path(config_path).resolve())
    return OmegaConf.to_container(final_cfg, resolve=True)


def load_audio_mono_16k(path: Path, sample_rate: int = 16000) -> torch.Tensor:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = resample_poly(audio, sample_rate, sr)
    return torch.from_numpy(np.asarray(audio, dtype=np.float32))


def overlap_add_segments(segments: list[torch.Tensor], hop: int, total_len: int) -> torch.Tensor:
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
        sf.write(str(out_path), separated[idx].cpu().numpy(), sr)


if __name__ == "__main__":
    main()
