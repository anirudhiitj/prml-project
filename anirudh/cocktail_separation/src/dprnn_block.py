from __future__ import annotations

import torch
import torch.nn as nn


class DPRNNBlock(nn.Module):
    """Dual-path recurrent block: intra-chunk then inter-chunk BiLSTM."""

    def __init__(self, channels: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = channels  # Full hidden size; BiLSTM output = 2*hidden, projected back

        self.intra_rnn = nn.LSTM(
            input_size=channels,
            hidden_size=hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.intra_linear = nn.Linear(hidden * 2, channels)
        self.intra_norm = nn.GroupNorm(1, channels, eps=1e-8)
        self.intra_dropout = nn.Dropout(dropout)

        self.inter_rnn = nn.LSTM(
            input_size=channels,
            hidden_size=hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.inter_linear = nn.Linear(hidden * 2, channels)
        self.inter_norm = nn.GroupNorm(1, channels, eps=1e-8)
        self.inter_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, K, S)

        Returns:
            Tensor with same shape (B, H, K, S)
        """
        if x.dim() != 4:
            raise ValueError(f"Expected (B, H, K, S), got {tuple(x.shape)}")

        b, h, k, s = x.shape

        # Intra-chunk: process K frames inside each chunk independently.
        intra_in = x.permute(0, 3, 2, 1).contiguous().view(b * s, k, h)
        intra_out, _ = self.intra_rnn(intra_in)
        intra_out = self.intra_dropout(self.intra_linear(intra_out))
        intra_out = intra_out.view(b, s, k, h).permute(0, 3, 2, 1).contiguous()
        x = self.intra_norm(intra_out) + x

        # Inter-chunk: process S chunks across the utterance for each position.
        inter_in = x.permute(0, 2, 3, 1).contiguous().view(b * k, s, h)
        inter_out, _ = self.inter_rnn(inter_in)
        inter_out = self.inter_dropout(self.inter_linear(inter_out))
        inter_out = inter_out.view(b, k, s, h).permute(0, 3, 1, 2).contiguous()
        x = self.inter_norm(inter_out) + x

        return x
