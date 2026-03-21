from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf
from pesq import pesq
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

try:
    from mir_eval.separation import bss_eval_sources
except Exception:  # pragma: no cover
    bss_eval_sources = None

from src.dataset import build_dataloader
from src.losses import si_snr
from src.model import DPRNNTasNet

warnings.filterwarnings("ignore", category=FutureWarning, module="mir_eval")
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*mir_eval\.separation\.bss_eval_sources.*Deprecated.*",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DPRNN-TasNet")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
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


def pairwise_sisnr(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    b, c, _ = est.shape
    out = torch.zeros((b, c, c), device=est.device)
    for i in range(c):
        for j in range(c):
            out[:, i, j] = si_snr(est[:, i, :], ref[:, j, :])
    return out


def reorder_by_hungarian(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    c = est.shape[0]
    score = torch.zeros((c, c))
    for i in range(c):
        for j in range(c):
            score[i, j] = si_snr(est[i].unsqueeze(0), ref[j].unsqueeze(0))
    rows, cols = linear_sum_assignment((-score).numpy())
    return est[rows], ref[cols]


def safe_pesq(ref: np.ndarray, deg: np.ndarray, fs: int) -> float:
    try:
        return float(pesq(fs, ref, deg, "wb"))
    except Exception:
        return float("nan")


def main() -> None:
    args = parse_args()
    cfg: Dict = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DPRNNTasNet(**cfg["model"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    data_root = Path(cfg["data"]["root"])
    test_loader = build_dataloader(
        split_root=data_root / "test",
        num_speakers=int(cfg["model"]["num_speakers"]),
        batch_size=1,
        num_workers=max(1, int(cfg["data"]["num_workers"]) // 2),
        distributed=False,
        shuffle=False,
        clip_samples=int(cfg["data"]["clip_samples"]),
        sample_rate=int(cfg["data"]["sample_rate"]),
    )

    si_snr_improvements: List[float] = []
    sdri_values: List[float] = []
    pesq_values: List[float] = []

    for mixture, sources in tqdm(test_loader, desc="evaluate"):
        mixture = mixture.to(device)
        sources = sources.to(device)

        with torch.no_grad():
            est = model(mixture)

        min_len = min(est.shape[-1], sources.shape[-1], mixture.shape[-1])
        est = est[..., :min_len]
        sources = sources[..., :min_len]
        mixture = mixture[..., :min_len]

        est_b = est[0].detach().cpu()
        ref_b = sources[0].detach().cpu()
        est_b, ref_b = reorder_by_hungarian(est_b, ref_b)

        for spk in range(est_b.shape[0]):
            sisnr_est = si_snr(est_b[spk].unsqueeze(0), ref_b[spk].unsqueeze(0)).item()
            sisnr_mix = si_snr(mixture[0].cpu().unsqueeze(0), ref_b[spk].unsqueeze(0)).item()
            si_snr_improvements.append(sisnr_est - sisnr_mix)

            if bss_eval_sources is not None:
                sdr_est, _, _, _ = bss_eval_sources(ref_b.numpy(), est_b.numpy())
                sdr_mix, _, _, _ = bss_eval_sources(
                    ref_b.numpy(),
                    np.repeat(mixture[0].cpu().numpy()[None, :], ref_b.shape[0], axis=0),
                )
                sdri_values.append(float(np.mean(sdr_est - sdr_mix)))

            pesq_values.append(
                safe_pesq(ref_b[spk].numpy(), est_b[spk].numpy(), int(cfg["data"]["sample_rate"]))
            )

    def mean_std(values: List[float]) -> str:
        arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
        if arr.size == 0:
            return "nan ± nan"
        return f"{arr.mean():.4f} ± {arr.std():.4f}"

    print(f"SI-SNRi: {mean_std(si_snr_improvements)} dB")
    print(f"SDRi:    {mean_std(sdri_values)} dB")
    print(f"PESQ:    {mean_std(pesq_values)}")


if __name__ == "__main__":
    main()
