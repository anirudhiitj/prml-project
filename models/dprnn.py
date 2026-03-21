"""
DPRNN Separator — full separation network between encoder and decoder.

Orchestrates:
    1. Input normalization + bottleneck conv
    2. Segmentation (chunking with overlap)
    3. Stacked DPRNN blocks
    4. Mask estimation
    5. Apply masks & overlap-add reconstruction
"""

import torch
import torch.nn as nn

from .dprnn_block import DPRNNBlock


class DPRNNSeparator(nn.Module):
    """
    Dual-Path RNN Separator.

    Takes encoder output and produces C separated feature streams
    (one per source) using chunking, stacked DPRNN blocks, and mask estimation.

    Args:
        N (int): Feature dimension (encoder output channels).
        H (int): RNN hidden size.
        K (int): Chunk size (number of frames per chunk).
        P (int): Hop size between chunks. Defaults to K // 2 for 50% overlap.
        B (int): Number of stacked DPRNN blocks.
        C (int): Number of sources to separate.
        rnn_type (str): "lstm" or "gru".
        num_layers (int): RNN layers inside each sub-block.
        bidirectional (bool): Bidirectional RNNs.
        dropout (float): Dropout rate.
        mask_activation (str): "relu" or "sigmoid".
    """

    def __init__(self, N: int = 64, H: int = 128, K: int = 250,
                 P: int = None, B: int = 6, C: int = 2,
                 rnn_type: str = "lstm", num_layers: int = 1,
                 bidirectional: bool = True, dropout: float = 0.0,
                 mask_activation: str = "relu"):
        super().__init__()
        self.N = N
        self.K = K
        self.P = P if P is not None else K // 2
        self.C = C

        # --- Input normalization ---
        self.norm = nn.GroupNorm(1, N)  # Global LayerNorm

        # --- Bottleneck 1x1 conv ---
        self.bottleneck = nn.Conv1d(N, N, kernel_size=1)

        # --- Stacked DPRNN blocks ---
        self.dprnn_blocks = nn.ModuleList([
            DPRNNBlock(N, H, rnn_type, num_layers, bidirectional, dropout)
            for _ in range(B)
        ])

        # --- Mask estimation ---
        self.prelu = nn.PReLU()
        self.mask_conv = nn.Conv2d(N, N * C, kernel_size=1)

        if mask_activation == "relu":
            self.mask_activation = nn.ReLU()
        elif mask_activation == "sigmoid":
            self.mask_activation = nn.Sigmoid()
        else:
            self.mask_activation = nn.ReLU()

    def _segment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Segment encoder output into overlapping chunks.

        Args:
            x: (B, N, L) — encoder output

        Returns:
            (B, N, K, S) — S overlapping chunks of size K
        """
        B, N, L = x.shape
        K, P = self.K, self.P

        # Pad the sequence so it divides evenly into chunks
        # Number of chunks: S = ceil((L - K) / P) + 1
        if (L - K) % P != 0:
            pad_len = P - ((L - K) % P)
            x = nn.functional.pad(x, (0, pad_len))
            L = x.shape[-1]

        S = (L - K) // P + 1

        # Use unfold to extract overlapping chunks
        # unfold(dim, size, step) → (B, N, S, K) → permute to (B, N, K, S)
        chunks = x.unfold(2, K, P)  # (B, N, S, K)
        chunks = chunks.permute(0, 1, 3, 2).contiguous()  # (B, N, K, S)

        return chunks

    def _overlap_add(self, chunks: torch.Tensor, original_length: int) -> torch.Tensor:
        """
        Reconstruct the full-length feature sequence from overlapping chunks
        using overlap-add.

        Args:
            chunks: (B, N, K, S)
            original_length: original length L of the encoder output

        Returns:
            (B, N, L) — reconstructed feature sequence
        """
        B, N, K, S = chunks.shape
        P = self.P

        # Output length after overlap-add
        out_len = (S - 1) * P + K

        output = torch.zeros(B, N, out_len, device=chunks.device, dtype=chunks.dtype)
        count = torch.zeros(1, 1, out_len, device=chunks.device, dtype=chunks.dtype)

        for i in range(S):
            start = i * P
            output[:, :, start:start + K] += chunks[:, :, :, i]
            count[:, :, start:start + K] += 1.0

        # Normalize by overlap count to average overlapping regions
        output = output / count.clamp(min=1.0)

        # Trim to original length
        return output[:, :, :original_length]

    def forward(self, enc_out: torch.Tensor) -> list:
        """
        Args:
            enc_out: (B, N, L) — encoder output

        Returns:
            list of C tensors, each (B, N, L) — one masked feature stream per source
        """
        B, N, L = enc_out.shape

        # 1. Normalize + bottleneck
        x = self.norm(enc_out)          # (B, N, L)
        x = self.bottleneck(x)          # (B, N, L)

        # 2. Segment into overlapping chunks
        x = self._segment(x)            # (B, N, K, S)
        _, _, K, S = x.shape

        # 3. Pass through stacked DPRNN blocks
        for block in self.dprnn_blocks:
            x = block(x)                # (B, N, K, S)

        # 4. Mask estimation
        x = self.prelu(x)               # (B, N, K, S)
        x = self.mask_conv(x)           # (B, N*C, K, S)
        x = x.view(B, self.C, N, K, S)  # (B, C, N, K, S)
        masks = self.mask_activation(x)  # (B, C, N, K, S)

        # 5. Apply masks — multiply each mask with the segmented encoder output
        # Segment the original encoder output the same way
        enc_segments = self._segment(enc_out)  # (B, N, K, S)
        enc_segments = enc_segments.unsqueeze(1)  # (B, 1, N, K, S)

        masked = masks * enc_segments    # (B, C, N, K, S)

        # 6. Overlap-add to reconstruct full-length features for each source
        outputs = []
        for c in range(self.C):
            out_c = self._overlap_add(masked[:, c], L)  # (B, N, L)
            outputs.append(out_c)

        return outputs  # list of C × (B, N, L)
