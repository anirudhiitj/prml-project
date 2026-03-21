"""
Decoder module for DPRNN-TasNet.

Pipeline stage: Decoder → Output Audio
    Input:  (B, N, L_out) — masked feature representation (one per source)
    Output: (B, 1, T)     — reconstructed waveform

Mirrors the encoder using a transposed convolution to go back to time-domain.
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    1-D Transposed Convolutional Decoder.

    Reconstructs a time-domain waveform from the separated feature
    representation. Acts as the inverse of the Encoder.

    Args:
        N (int): Number of input channels (must match encoder output dim).
        L (int): Kernel size (synthesis window, must match encoder).
        stride (int): Stride (must match encoder). Defaults to L // 2.
    """

    def __init__(self, N: int = 64, L: int = 2, stride: int = None):
        super().__init__()
        self.N = N
        self.L = L
        self.stride = stride if stride is not None else L // 2

        # Learnable synthesis transform (replaces inverse STFT)
        self.deconv = nn.ConvTranspose1d(
            in_channels=N,
            out_channels=1,
            kernel_size=L,
            stride=self.stride,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, L_out) — masked encoder features for one source

        Returns:
            (B, 1, T) — reconstructed waveform
        """
        return self.deconv(x)
