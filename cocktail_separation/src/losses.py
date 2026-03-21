from __future__ import annotations

from typing import Tuple

import torch
from scipy.optimize import linear_sum_assignment


def si_snr(estimated: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute SI-SNR per sample for waveforms of shape (B, T)."""
    if estimated.shape != target.shape:
        raise ValueError(f"Shape mismatch: estimated {estimated.shape}, target {target.shape}")

    estimated = estimated - estimated.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    dot = (estimated * target).sum(dim=-1, keepdim=True)
    target_norm_sq = (target * target).sum(dim=-1, keepdim=True) + eps
    s_target = (dot / target_norm_sq) * target

    e_noise = estimated - s_target

    ratio = (s_target * s_target).sum(dim=-1) / ((e_noise * e_noise).sum(dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)


def snr(estimated: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute standard SNR per sample for waveforms of shape (B, T)."""
    if estimated.shape != target.shape:
        raise ValueError(f"Shape mismatch: estimated {estimated.shape}, target {target.shape}")

    noise = estimated - target
    ratio = (target * target).sum(dim=-1) / ((noise * noise).sum(dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)


def _pairwise_metric(
    estimated_sources: torch.Tensor,
    true_sources: torch.Tensor,
    metric_fn,
) -> torch.Tensor:
    """Build pairwise metric matrix with shape (B, C, C)."""
    b, c, _ = estimated_sources.shape
    metric_matrix = torch.zeros((b, c, c), device=estimated_sources.device, dtype=estimated_sources.dtype)

    for i in range(c):
        for j in range(c):
            metric_matrix[:, i, j] = metric_fn(estimated_sources[:, i, :], true_sources[:, j, :])

    return metric_matrix


def _hungarian_best_assignment(pairwise_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        pairwise_scores: (C, C) where higher is better.

    Returns:
        row_idx, col_idx assignment tensors on CPU.
    """
    score_np = pairwise_scores.detach().cpu().numpy()
    row_idx, col_idx = linear_sum_assignment(-score_np)
    return torch.as_tensor(row_idx), torch.as_tensor(col_idx)


def pit_loss(
    estimated_sources: torch.Tensor,
    true_sources: torch.Tensor,
    eps: float = 1e-8,
    snr_weight: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Utterance-level PIT loss with Hungarian assignment.

    Args:
        estimated_sources: (B, C, T)
        true_sources: (B, C, T)
        snr_weight: Optional weight for SNR term.

    Returns:
        total_loss: scalar tensor for optimization
        mean_sisnr: scalar tensor for logging
    """
    if estimated_sources.shape != true_sources.shape:
        raise ValueError(
            f"Shape mismatch: estimated {estimated_sources.shape}, true {true_sources.shape}"
        )

    b, c, _ = estimated_sources.shape

    pairwise_si = _pairwise_metric(estimated_sources, true_sources, si_snr)
    pairwise_snr = _pairwise_metric(estimated_sources, true_sources, snr) if snr_weight > 0 else None

    assigned_si = []
    assigned_snr = []

    for batch_idx in range(b):
        rows, cols = _hungarian_best_assignment(pairwise_si[batch_idx])
        rows = rows.to(device=pairwise_si.device)
        cols = cols.to(device=pairwise_si.device)

        assigned_si.append(pairwise_si[batch_idx, rows, cols].mean())
        if pairwise_snr is not None:
            assigned_snr.append(pairwise_snr[batch_idx, rows, cols].mean())

    mean_sisnr = torch.stack(assigned_si).mean()
    loss = -mean_sisnr

    if pairwise_snr is not None:
        mean_snr = torch.stack(assigned_snr).mean()
        loss = loss - snr_weight * mean_snr

    return loss, mean_sisnr
