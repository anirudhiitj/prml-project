"""
Evaluation metrics for speech separation.
"""

import torch

from losses.pit_loss import si_snr


def si_snri(estimate: torch.Tensor, target: torch.Tensor,
            mixture: torch.Tensor) -> torch.Tensor:
    """
    SI-SNR Improvement (SI-SNRi).

    Measures how much the separation improved the SI-SNR compared to
    the original mixture.

        SI-SNRi = SI-SNR(estimate, target) - SI-SNR(mixture, target)

    Args:
        estimate: (B, T) or (T,) — estimated source
        target:   (B, T) or (T,) — ground-truth source
        mixture:  (B, T) or (T,) — original mixture

    Returns:
        (B,) or scalar — improvement in dB (higher is better)
    """
    return si_snr(estimate, target) - si_snr(mixture, target)


def sdr(estimate: torch.Tensor, target: torch.Tensor,
        eps: float = 1e-8) -> torch.Tensor:
    """
    Signal-to-Distortion Ratio (SDR).

        SDR = 10 · log₁₀( ||target||² / ||target - estimate||² )

    Args:
        estimate: (B, T) or (T,)
        target:   (B, T) or (T,)

    Returns:
        (B,) or scalar — SDR in dB
    """
    noise = target - estimate
    sdr_val = 10 * torch.log10(
        torch.sum(target ** 2, dim=-1).clamp(min=eps) /
        torch.sum(noise ** 2, dim=-1).clamp(min=eps)
    )
    return sdr_val


def sdri(estimate: torch.Tensor, target: torch.Tensor,
         mixture: torch.Tensor) -> torch.Tensor:
    """
    SDR Improvement (SDRi).

        SDRi = SDR(estimate, target) - SDR(mixture, target)

    Args:
        estimate: (B, T) or (T,)
        target:   (B, T) or (T,)
        mixture:  (B, T) or (T,)

    Returns:
        (B,) or scalar — SDR improvement in dB
    """
    return sdr(estimate, target) - sdr(mixture, target)
