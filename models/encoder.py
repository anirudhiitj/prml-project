"""
Encoder module for DPRNN-TasNet.

Pipeline stage: Raw Audio → Encoder → Features
    Input:  (B, 1, T)  — raw mixture waveform
    Output: (B, N, L)  — learned feature representation

Replaces the traditional STFT with a learnable 1-D convolution.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    1-D Convolutional Encoder.
    
    Maps raw waveform to a higher-dimensional feature space using
    a single Conv1d layer followed by ReLU activation.
    
    Args:
        N (int): Number of output channels (encoder dimension).
        L (int): Kernel size (analysis window length in samples).
        stride (int): Stride (hop size). Defaults to L // 2 for 50% overlap.
    """

    def __init__(self, N: int = 64, L: int = 2, stride: int = None):
        super().__init__()
        self.N = N
        self.L = L
        self.stride = stride if stride is not None else L // 2

        # Learnable analysis transform (replaces STFT)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=N,
            kernel_size=L,
            stride=self.stride,
            bias=False,
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, T) — raw waveform

        Returns:
            (B, N, L_out) — encoded features
            where L_out = floor((T - L) / stride) + 1
        """
        return self.activation(self.conv(x))
