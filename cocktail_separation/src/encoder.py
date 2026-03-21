from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Learned 1D convolutional encoder for waveform inputs."""

    def __init__(self, in_channels: int = 1, out_channels: int = 64, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        self.activation = nn.ReLU()

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mixture: (B, T)

        Returns:
            Encoded features: (B, N, T')
        """
        if mixture.dim() != 2:
            raise ValueError(f"Expected mixture shape (B, T), got {tuple(mixture.shape)}")
        x = mixture.unsqueeze(1)
        return self.activation(self.conv(x))
