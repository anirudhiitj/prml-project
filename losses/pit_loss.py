"""
Loss functions for speech separation.

Implements:
    - SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
    - PIT (Permutation Invariant Training) with SI-SNR
"""

import torch
import torch.nn as nn
from itertools import permutations


def si_snr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Scale-Invariant Signal-to-Noise Ratio (SI-SNR).

    SI-SNR = 10 · log₁₀( ||s_target||² / ||e_noise||² )

    where:
        s_target = (<estimate, target> / ||target||²) · target
        e_noise  = estimate - s_target

    Args:
        estimate: (B, T) or (T,) — estimated signal
        target:   (B, T) or (T,) — ground-truth signal
        eps: small value for numerical stability

    Returns:
        (B,) or scalar — SI-SNR in dB (higher is better)
    """
    # Zero-mean normalization
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # s_target = (<est, tgt> / ||tgt||²) * tgt
    dot = torch.sum(estimate * target, dim=-1, keepdim=True)
    s_target_energy = torch.sum(target ** 2, dim=-1, keepdim=True).clamp(min=eps)
    s_target = (dot / s_target_energy) * target

    # e_noise = estimate - s_target
    e_noise = estimate - s_target

    # SI-SNR in dB
    si_snr_val = 10 * torch.log10(
        torch.sum(s_target ** 2, dim=-1).clamp(min=eps) /
        torch.sum(e_noise ** 2, dim=-1).clamp(min=eps)
    )

    return si_snr_val  # (B,) — one value per sample


class PITLoss(nn.Module):
    """
    Permutation Invariant Training (PIT) loss using SI-SNR.

    Since we don't know which output corresponds to which ground-truth
    speaker, we try all C! permutations and pick the assignment that
    gives the best (highest) total SI-SNR.

    The loss returned is the negative SI-SNR (so minimizing it = maximizing SI-SNR).

    Args:
        num_sources (int): Number of sources C. Defaults to 2.
    """

    def __init__(self, num_sources: int = 2):
        super().__init__()
        self.num_sources = num_sources
        # Pre-compute all permutations of source indices
        self.perms = list(permutations(range(num_sources)))

    def forward(self, estimates: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            estimates: (B, C, T) — model outputs (C separated waveforms)
            targets:   (B, C, T) — ground-truth source waveforms

        Returns:
            scalar — mean negative SI-SNR across the batch (for the best permutation)
        """
        B, C, T = estimates.shape
        assert C == self.num_sources, f"Expected {self.num_sources} sources, got {C}"

        # Compute SI-SNR for all (source, permuted_target) pairs: (B, C, C)
        # si_snr_matrix[b, i, j] = SI-SNR between estimate_i and target_j
        si_snr_matrix = torch.zeros(B, C, C, device=estimates.device)
        for i in range(C):
            for j in range(C):
                si_snr_matrix[:, i, j] = si_snr(estimates[:, i], targets[:, j])

        # Try all permutations, find the one with maximum total SI-SNR
        max_si_snr = torch.full((B,), float("-inf"), device=estimates.device)

        for perm in self.perms:
            # Total SI-SNR for this permutation: sum of diagonal-like elements
            perm_si_snr = sum(si_snr_matrix[:, i, perm[i]] for i in range(C))
            max_si_snr = torch.max(max_si_snr, perm_si_snr)

        # Average across batch, negate (because we want to maximize SI-SNR)
        loss = -max_si_snr.mean() / C

        return loss
