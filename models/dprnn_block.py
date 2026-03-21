"""
DPRNN Block — the core building block of the Dual-Path RNN.

Each block contains two sub-modules:
    1. Intra-chunk RNN  — processes each chunk independently along the time axis
                          (captures LOCAL temporal patterns)
    2. Inter-chunk RNN  — processes each time-step across all chunks
                          (captures GLOBAL / long-range dependencies)

Both use: LayerNorm → RNN → Linear projection → Residual connection.
"""

import torch
import torch.nn as nn


class IntraChunkRNN(nn.Module):
    """
    Processes each chunk independently along the time (K) dimension.

    Input:  (B, N, K, S) — B=batch, N=features, K=chunk_size, S=num_chunks
    Output: (B, N, K, S) — same shape, with local patterns captured
    """

    def __init__(self, N: int, H: int, rnn_type: str = "lstm",
                 num_layers: int = 1, bidirectional: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.norm = nn.GroupNorm(1, N)  # Equivalent to LayerNorm over N channels

        rnn_cls = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=N,
            hidden_size=H,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Project RNN output back to N dimensions
        rnn_out_dim = H * 2 if bidirectional else H
        self.linear = nn.Linear(rnn_out_dim, N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, K, S)
        Returns:
            (B, N, K, S)
        """
        B, N, K, S = x.shape
        residual = x

        # Normalize
        x = self.norm(x)  # (B, N, K, S)

        # Reshape: treat each chunk independently
        # (B, N, K, S) → (B*S, K, N) — each of the B*S chunks is a sequence of length K
        x = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)

        # RNN forward
        x, _ = self.rnn(x)  # (B*S, K, rnn_out_dim)

        # Project back to N dims
        x = self.linear(x)  # (B*S, K, N)

        # Reshape back: (B*S, K, N) → (B, N, K, S)
        x = x.view(B, S, K, N).permute(0, 3, 2, 1).contiguous()

        # Residual connection
        return x + residual


class InterChunkRNN(nn.Module):
    """
    Processes each time-step across all chunks along the S dimension.

    Input:  (B, N, K, S) — B=batch, N=features, K=chunk_size, S=num_chunks
    Output: (B, N, K, S) — same shape, with global patterns captured
    """

    def __init__(self, N: int, H: int, rnn_type: str = "lstm",
                 num_layers: int = 1, bidirectional: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.norm = nn.GroupNorm(1, N)

        rnn_cls = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=N,
            hidden_size=H,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        rnn_out_dim = H * 2 if bidirectional else H
        self.linear = nn.Linear(rnn_out_dim, N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, K, S)
        Returns:
            (B, N, K, S)
        """
        B, N, K, S = x.shape
        residual = x

        # Normalize
        x = self.norm(x)  # (B, N, K, S)

        # Reshape: treat each time-step independently
        # (B, N, K, S) → (B*K, S, N) — each of B*K time-steps is a sequence of length S
        x = x.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)

        # RNN forward
        x, _ = self.rnn(x)  # (B*K, S, rnn_out_dim)

        # Project back to N dims
        x = self.linear(x)  # (B*K, S, N)

        # Reshape back: (B*K, S, N) → (B, N, K, S)
        x = x.view(B, K, S, N).permute(0, 3, 1, 2).contiguous()

        # Residual connection
        return x + residual


class DPRNNBlock(nn.Module):
    """
    Single Dual-Path RNN block.

    Sequentially applies:
        1. Intra-chunk RNN (local modeling within each chunk)
        2. Inter-chunk RNN (global modeling across chunks)

    Args:
        N (int): Feature dimension (encoder output channels).
        H (int): RNN hidden size.
        rnn_type (str): "lstm" or "gru".
        num_layers (int): Number of RNN layers in each sub-block.
        bidirectional (bool): Whether to use bidirectional RNNs.
        dropout (float): Dropout probability.
    """

    def __init__(self, N: int, H: int, rnn_type: str = "lstm",
                 num_layers: int = 1, bidirectional: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.intra_rnn = IntraChunkRNN(N, H, rnn_type, num_layers, bidirectional, dropout)
        self.inter_rnn = InterChunkRNN(N, H, rnn_type, num_layers, bidirectional, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, K, S)
        Returns:
            (B, N, K, S)
        """
        x = self.intra_rnn(x)
        x = self.inter_rnn(x)
        return x
