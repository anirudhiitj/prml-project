"""
DPRNN-TasNet — Top-level model.

Wires together:  Encoder → DPRNN Separator → Decoder

    Input:  (B, 1, T)   — mixed audio waveform
    Output: (B, C, T)   — C separated waveforms
"""

import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .dprnn import DPRNNSeparator


class DPRNNTasNet(nn.Module):
    """
    Dual-Path RNN Time-domain Audio Separation Network.

    End-to-end model that takes a mixed waveform and outputs C separated
    source waveforms.

    Args:
        N (int): Encoder output dimension.
        L (int): Encoder/decoder kernel size.
        H (int): RNN hidden size.
        K (int): Chunk size.
        P (int): Chunk hop size. Defaults to K // 2.
        B (int): Number of DPRNN blocks.
        C (int): Number of sources.
        rnn_type (str): "lstm" or "gru".
        num_layers (int): RNN layers per sub-block.
        bidirectional (bool): Bidirectional RNNs.
        dropout (float): Dropout rate.
        encoder_stride (int): Encoder stride. Defaults to L // 2.
        mask_activation (str): "relu" or "sigmoid".
    """

    def __init__(self, N: int = 64, L: int = 2, H: int = 128,
                 K: int = 250, P: int = None, B: int = 6,
                 C: int = 2, rnn_type: str = "lstm",
                 num_layers: int = 1, bidirectional: bool = True,
                 dropout: float = 0.0, encoder_stride: int = None,
                 mask_activation: str = "relu"):
        super().__init__()
        self.C = C

        stride = encoder_stride if encoder_stride is not None else L // 2

        self.encoder = Encoder(N, L, stride)
        self.separator = DPRNNSeparator(
            N=N, H=H, K=K, P=P, B=B, C=C,
            rnn_type=rnn_type, num_layers=num_layers,
            bidirectional=bidirectional, dropout=dropout,
            mask_activation=mask_activation,
        )
        self.decoder = Decoder(N, L, stride)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mixture: (B, 1, T) — raw mixed waveform

        Returns:
            (B, C, T) — C separated source waveforms
        """
        T_original = mixture.shape[-1]

        # 1. Encode
        enc_out = self.encoder(mixture)           # (B, N, L_out)

        # 2. Separate
        masked_features = self.separator(enc_out)  # list of C × (B, N, L_out)

        # 3. Decode each source
        separated = []
        for feat in masked_features:
            waveform = self.decoder(feat)          # (B, 1, T')
            separated.append(waveform)

        # Stack into (B, C, T')
        output = torch.cat(separated, dim=1)       # (B, C, T')

        # Trim or pad to match original input length
        T_out = output.shape[-1]
        if T_out > T_original:
            output = output[:, :, :T_original]
        elif T_out < T_original:
            output = nn.functional.pad(output, (0, T_original - T_out))

        return output  # (B, C, T)

    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
