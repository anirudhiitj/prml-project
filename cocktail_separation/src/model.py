from __future__ import annotations

import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder
from .separator import DPRNNSeparator


class DPRNNTasNet(nn.Module):
    """End-to-end DPRNN-TasNet with learned encoder/decoder."""

    def __init__(
        self,
        num_speakers: int = 5,
        encoder_dim: int = 64,
        encoder_kernel: int = 2,
        encoder_stride: int = 2,
        bottleneck_dim: int = 64,
        chunk_size: int = 100,
        num_dprnn_blocks: int = 6,
    ) -> None:
        super().__init__()
        self.num_speakers = num_speakers

        self.encoder = Encoder(
            in_channels=1,
            out_channels=encoder_dim,
            kernel_size=encoder_kernel,
            stride=encoder_stride,
        )
        self.separator = DPRNNSeparator(
            encoder_dim=encoder_dim,
            bottleneck_dim=bottleneck_dim,
            chunk_size=chunk_size,
            num_blocks=num_dprnn_blocks,
            num_speakers=num_speakers,
        )
        self.decoder = Decoder(
            in_channels=encoder_dim,
            out_channels=1,
            kernel_size=encoder_kernel,
            stride=encoder_stride,
        )

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mixture: (B, T)

        Returns:
            Estimated sources: (B, C, T_hat)
        """
        encoded = self.encoder(mixture)
        masks = self.separator(encoded)

        decoded = []
        for speaker_idx in range(self.num_speakers):
            masked = encoded * masks[:, speaker_idx, :, :]
            waveform = self.decoder(masked)
            decoded.append(waveform)

        return torch.stack(decoded, dim=1)
