"""
Python re-implementation of Gokul's C++ Conv-TasNet.

Parameter names and forward-pass logic are kept identical to the LibTorch
C++ implementation so that weights from ``best_tasnet.pt`` (saved with
``torch::save(net, path)`` / ``torch.jit.load``) can be loaded directly.

Architecture  (matches gokul/src/conv_tasnet.{h,cpp}):
  N=512  L=16  B=256  H=416  P=3  X=8  R=3  C=2   (~8 M params)
  Sample-rate: 8 kHz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────── Layer Norms ─────────────────────────────────

class GlobalLayerNorm(nn.Module):
    """Normalise over (channels, time) — matches C++ GlobalLayerNormImpl."""

    def __init__(self, ch: int, eps: float = 1e-8):
        super().__init__()
        # named "gamma" / "beta" to match C++ register_parameter("gamma", ...)
        self.gamma = nn.Parameter(torch.ones(1, ch, 1))
        self.beta  = nn.Parameter(torch.zeros(1, ch, 1))
        self.eps   = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=[1, 2], keepdim=True)
        var  = x.var(dim=[1, 2], unbiased=False, keepdim=True)
        return self.gamma * (x - mean) / (var + self.eps).sqrt() + self.beta


class ChannelLayerNorm(nn.Module):
    """Normalise over the channel dimension only — matches C++ ChannelLayerNormImpl."""

    def __init__(self, ch: int, eps: float = 1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, ch, 1))
        self.beta  = nn.Parameter(torch.zeros(1, ch, 1))
        self.eps   = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var  = x.var(dim=1, unbiased=False, keepdim=True)
        return self.gamma * (x - mean) / (var + self.eps).sqrt() + self.beta


# ───────────────────────── Depth-wise Separable Block ────────────────────────

class DepthSepBlock(nn.Module):
    """Matches C++ DepthSepBlockImpl."""

    def __init__(self, B: int, H: int, P: int, dilation: int):
        super().__init__()
        pad = dilation * (P - 1) // 2

        self.conv_in   = nn.Conv1d(B, H, 1)
        self.prelu1    = nn.PReLU(num_parameters=H)
        self.norm1     = ChannelLayerNorm(H)

        self.dconv     = nn.Conv1d(H, H, P, dilation=dilation, padding=pad, groups=H)
        self.prelu2    = nn.PReLU(num_parameters=H)
        self.norm2     = ChannelLayerNorm(H)

        self.conv_skip = nn.Conv1d(H, B, 1)
        self.conv_res  = nn.Conv1d(H, B, 1)

    # Note: forward() is NOT called by TCN; TCN accesses sub-modules directly
    # to accumulate skip_sum separately — mirrors the C++ TCN forward loop.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(self.prelu1(self.conv_in(x)))
        h = self.norm2(self.prelu2(self.dconv(h)))
        return x + self.conv_res(h)


# ──────────────────────────────── TCN Separator ───────────────────────────────

class TCN(nn.Module):
    """Matches C++ TCNImpl."""

    def __init__(self, N: int, B: int, H: int, P: int, X: int, R: int, C: int):
        super().__init__()
        self._N = N
        self._C = C

        self.ln         = GlobalLayerNorm(N)
        self.bottleneck = nn.Conv1d(N, B, 1)
        self.blocks     = nn.ModuleList(
            DepthSepBlock(B, H, P, 2 ** x)
            for _r in range(R)
            for x in range(X)
        )
        self.mask_conv  = nn.Conv1d(B, C * N, 1)

    def forward(self, enc_out: torch.Tensor) -> torch.Tensor:
        # enc_out: [batch, N, L]
        batch = enc_out.size(0)
        L     = enc_out.size(2)

        x        = self.bottleneck(self.ln(enc_out))   # [batch, B, L]
        skip_sum = torch.zeros_like(x)

        # Replicate the C++ loop: access sub-module layers directly so
        # skip_sum and residual x are accumulated separately.
        for blk in self.blocks:
            h        = blk.norm1(blk.prelu1(blk.conv_in(x)))
            h        = blk.norm2(blk.prelu2(blk.dconv(h)))
            skip_sum = skip_sum + blk.conv_skip(h)
            x        = x + blk.conv_res(h)

        masks = self.mask_conv(skip_sum).view(batch, self._C, self._N, L)
        masks = torch.relu(masks)
        return enc_out.unsqueeze(1) * masks   # [batch, C, N, L]


# ──────────────────────────────── Conv-TasNet ────────────────────────────────

class ConvTasNet(nn.Module):
    """
    Full Conv-TasNet — matches C++ ConvTasNetImpl.

    Default hyperparameters reflect the trained checkpoint
    (``gokul/checkpoints/best_tasnet.pt``).
    """

    def __init__(
        self,
        N: int = 512,
        L: int = 16,
        B: int = 256,
        H: int = 416,
        P: int = 3,
        X: int = 8,
        R: int = 3,
        C: int = 2,
    ):
        super().__init__()
        stride = L // 2
        self._N      = N
        self._stride = stride
        self._C      = C

        self.encoder   = nn.Conv1d(1, N, L, stride=stride, bias=False)
        self.separator = TCN(N, B, H, P, X, R, C)
        self.decoder   = nn.ConvTranspose1d(N, 1, L, stride=stride, bias=False)

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mix: waveform of shape [batch, T] or [batch, 1, T]
        Returns:
            Tensor [batch, C, T] — one waveform per source
        """
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)   # [B, 1, T]
        orig_len = mix.size(2)

        # Pad to a multiple of stride (mirrors C++ padding)
        rem = orig_len % self._stride
        if rem > 0:
            mix = F.pad(mix, (0, self._stride - rem))

        enc    = torch.relu(self.encoder(mix))    # [B, N, L_enc]
        masked = self.separator(enc)              # [B, C, N, L_enc]

        batch  = masked.size(0)
        L_enc  = masked.size(3)

        flat = masked.view(batch * self._C, self._N, L_enc)
        dec  = self.decoder(flat).view(batch, self._C, -1)   # [B, C, T]

        return dec[:, :, :orig_len]


# ────────────────────────────── Helper: load ─────────────────────────────────

def load_model(
    checkpoint_path: str,
    device: str = "cpu",
    N: int = 512,
    L: int = 16,
    B: int = 256,
    H: int = 416,
    P: int = 3,
    X: int = 8,
    R: int = 3,
    C: int = 2,
) -> ConvTasNet:
    """
    Load a ``ConvTasNet`` from a checkpoint saved by the C++ training code
    (``torch::save(net, path)`` / TorchScript archive).

    The function extracts the state dict via ``torch.jit.load`` and loads it
    into a freshly constructed Python ``ConvTasNet``.
    """
    # C++ torch::save(net, path) produces a TorchScript archive; extract
    # the state dict from it and load into our Python model.
    jit_mod = torch.jit.load(checkpoint_path, map_location=device)
    state_dict = jit_mod.state_dict()

    model = ConvTasNet(N=N, L=L, B=B, H=H, P=P, X=X, R=R, C=C)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model
