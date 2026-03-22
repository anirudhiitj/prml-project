from __future__ import annotations

import torch
import torch.nn as nn

from .dprnn_block import DPRNNBlock


class DPRNNSeparator(nn.Module):
    """DPRNN separator with chunking, dual-path blocks, and softmax mask head."""

    def __init__(
        self,
        encoder_dim: int = 64,
        bottleneck_dim: int = 64,
        chunk_size: int = 100,
        num_blocks: int = 6,
        num_speakers: int = 5,
    ) -> None:
        super().__init__()
        if chunk_size % 2 != 0:
            raise ValueError("chunk_size must be even for 50% overlap chunking")

        self.encoder_dim = encoder_dim
        self.bottleneck_dim = bottleneck_dim
        self.chunk_size = chunk_size
        self.hop_size = chunk_size // 2
        self.num_blocks = num_blocks
        self.num_speakers = num_speakers

        self.norm = nn.GroupNorm(1, encoder_dim, eps=1e-8)
        self.bottleneck = nn.Conv1d(encoder_dim, bottleneck_dim, kernel_size=1)

        self.blocks = nn.ModuleList([DPRNNBlock(channels=bottleneck_dim) for _ in range(num_blocks)])

        self.mask_head = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(bottleneck_dim, encoder_dim * num_speakers, kernel_size=1),
        )

    def _pad_for_chunking(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        b, h, t = x.shape
        k = self.chunk_size
        step = self.hop_size

        # Edge pad allows windows to fully cover the first and last positions.
        x = torch.nn.functional.pad(x, (step, step))
        rest = (k - ((x.shape[-1] - k) % step)) % step
        if rest > 0:
            x = torch.nn.functional.pad(x, (0, rest))

        return x, rest, step

    def _segment(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x, rest, step = self._pad_for_chunking(x)
        b, h, _ = x.shape

        segments = x.unfold(dimension=2, size=self.chunk_size, step=step)
        # unfold gives (B, H, S, K), convert to (B, H, K, S)
        segments = segments.permute(0, 1, 3, 2).contiguous()
        return segments, rest, step

    def _overlap_add(self, segments: torch.Tensor, rest: int, step: int, target_len: int) -> torch.Tensor:
        # segments: (B, H, K, S)
        b, h, k, s = segments.shape
        length = (s - 1) * step + k

        out = torch.zeros((b, h, length), device=segments.device, dtype=segments.dtype)
        overlap_count = torch.zeros((1, 1, length), device=segments.device, dtype=segments.dtype)

        seg = segments.permute(0, 1, 3, 2).contiguous()  # (B, H, S, K)
        for idx in range(s):
            start = idx * step
            end = start + k
            out[:, :, start:end] += seg[:, :, idx, :]
            overlap_count[:, :, start:end] += 1.0

        out = out / torch.clamp(overlap_count, min=1.0)

        # Remove front/back step padding and tail rest padding.
        out = out[:, :, step:]
        end_trim = step + rest
        if end_trim > 0:
            out = out[:, :, :-end_trim]

        # Final safety trim to exact encoder timeline.
        if out.shape[-1] > target_len:
            out = out[:, :, :target_len]
        elif out.shape[-1] < target_len:
            out = torch.nn.functional.pad(out, (0, target_len - out.shape[-1]))

        return out

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoded: (B, N, T')

        Returns:
            masks: (B, C, N, T')
        """
        if encoded.dim() != 3:
            raise ValueError(f"Expected encoded shape (B, N, T'), got {tuple(encoded.shape)}")

        b, n, t = encoded.shape
        x = self.bottleneck(self.norm(encoded))

        chunks, rest, step = self._segment(x)
        for block in self.blocks:
            chunks = block(chunks)

        x = self._overlap_add(chunks, rest=rest, step=step, target_len=t)
        logits = self.mask_head(x).view(b, self.num_speakers, n, t)
        masks = torch.softmax(logits, dim=1)
        return masks
