from __future__ import annotations

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """Learned transposed-convolution decoder for masked encoder features."""

    def __init__(self, in_channels: int = 64, out_channels: int = 1, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features with shape (B, N, T') into waveform shape (B, T)."""
        if features.dim() != 3:
            raise ValueError(f"Expected features shape (B, N, T'), got {tuple(features.shape)}")
        return self.deconv(features).squeeze(1)
