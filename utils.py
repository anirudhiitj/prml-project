"""
Utility functions for the RCNN Cocktail Party Audio Separation project.

Includes:
- STFT / iSTFT helpers
- SI-SNR metric
- PIT (Permutation Invariant Training) loss
- Audio I/O utilities
"""

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from itertools import permutations


# ──────────────────────────────────────────────
#  STFT / iSTFT Helpers
# ──────────────────────────────────────────────

class STFTHelper:
    """Wrapper around torch.stft / torch.istft with fixed parameters."""

    def __init__(self, n_fft=512, hop_length=128, win_length=512, window='hann'):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window_type = window

    def _get_window(self, device):
        if self.window_type == 'hann':
            return torch.hann_window(self.win_length, device=device)
        elif self.window_type == 'hamming':
            return torch.hamming_window(self.win_length, device=device)
        else:
            return torch.ones(self.win_length, device=device)

    def stft(self, waveform):
        """
        Compute STFT.

        Args:
            waveform: (batch, samples) or (samples,)

        Returns:
            magnitude: (batch, freq_bins, time_frames)
            phase:     (batch, freq_bins, time_frames)
        """
        single = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            single = True

        window = self._get_window(waveform.device)

        # torch.stft returns complex tensor
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )  # (batch, freq_bins, time_frames)

        magnitude = spec.abs()
        phase = spec.angle()

        if single:
            magnitude = magnitude.squeeze(0)
            phase = phase.squeeze(0)

        return magnitude, phase

    def istft(self, magnitude, phase, length=None):
        """
        Inverse STFT from magnitude and phase.

        Args:
            magnitude: (batch, freq_bins, time_frames)
            phase:     (batch, freq_bins, time_frames)
            length:    desired output length in samples

        Returns:
            waveform: (batch, samples)
        """
        single = False
        if magnitude.dim() == 2:
            magnitude = magnitude.unsqueeze(0)
            phase = phase.unsqueeze(0)
            single = True

        # Reconstruct complex spectrogram
        complex_spec = magnitude * torch.exp(1j * phase)

        window = self._get_window(magnitude.device)

        waveform = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            length=length,
        )

        if single:
            waveform = waveform.squeeze(0)

        return waveform


# ──────────────────────────────────────────────
#  SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
# ──────────────────────────────────────────────

def si_snr(estimated, target, eps=1e-8):
    """
    Compute Scale-Invariant Signal-to-Noise Ratio.

    Args:
        estimated: (batch, samples)  — estimated signal
        target:    (batch, samples)  — clean reference signal

    Returns:
        si_snr: (batch,) — SI-SNR in dB for each example
    """
    # Zero-mean
    estimated = estimated - estimated.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # <s, s_hat> / ||s||^2  *  s
    dot = torch.sum(estimated * target, dim=-1, keepdim=True)
    s_target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    proj = dot * target / s_target_energy

    noise = estimated - proj

    si_snr_val = 10 * torch.log10(
        torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps) + eps
    )

    return si_snr_val


def negative_si_snr(estimated, target, eps=1e-8):
    """Negative SI-SNR for use as a loss (lower is better)."""
    return -si_snr(estimated, target, eps).mean()


# ──────────────────────────────────────────────
#  PIT (Permutation Invariant Training) Loss
# ──────────────────────────────────────────────

def pit_loss(estimated_sources, target_sources, loss_fn=negative_si_snr):
    """
    Permutation Invariant Training loss.

    Tries all assignments of estimated sources to target sources
    and picks the one with the lowest total loss.

    Args:
        estimated_sources: (batch, n_sources, samples)
        target_sources:    (batch, n_sources, samples)
        loss_fn:           pairwise loss function(estimated, target) → scalar

    Returns:
        loss: scalar — the minimum-permutation loss averaged over the batch
    """
    batch_size, n_sources, _ = estimated_sources.shape
    perms = list(permutations(range(n_sources)))

    losses = []
    for perm in perms:
        perm_loss = 0
        for est_idx, tgt_idx in enumerate(perm):
            perm_loss = perm_loss + loss_fn(
                estimated_sources[:, est_idx, :],
                target_sources[:, tgt_idx, :],
            )
        losses.append(perm_loss)

    # Stack and take minimum across permutations
    losses = torch.stack(losses, dim=0)  # (n_perms,)
    min_loss = losses.min(dim=0).values

    return min_loss


# ──────────────────────────────────────────────
#  PIT Loss on Spectrograms (for mask-based training)
# ──────────────────────────────────────────────

def pit_mse_loss(estimated_masks, target_mags, mixture_mag):
    """
    PIT loss in the spectrogram domain using MSE.

    Args:
        estimated_masks: (batch, n_sources, freq, time) — predicted masks
        target_mags:     (batch, n_sources, freq, time) — target magnitudes
        mixture_mag:     (batch, 1, freq, time)         — mixture magnitude

    Returns:
        loss: scalar
    """
    batch_size, n_sources, F, T = estimated_masks.shape
    perms = list(permutations(range(n_sources)))

    # Apply masks to get estimated magnitudes
    estimated_mags = estimated_masks * mixture_mag  # (batch, n_sources, F, T)

    best_loss = None
    for perm in perms:
        perm_loss = 0
        for est_idx, tgt_idx in enumerate(perm):
            perm_loss = perm_loss + F_mse(
                estimated_mags[:, est_idx],
                target_mags[:, tgt_idx],
            )
        if best_loss is None or perm_loss < best_loss:
            best_loss = perm_loss

    return best_loss / n_sources


def F_mse(x, y):
    """Mean squared error."""
    return ((x - y) ** 2).mean()


# ──────────────────────────────────────────────
#  Audio I/O
# ──────────────────────────────────────────────

def load_audio(filepath, target_sr=8000):
    """
    Load an audio file and resample to target sample rate.

    Returns:
        waveform: (samples,) 1D tensor
        sr: sample rate
    """
    import scipy.io.wavfile as wavfile

    sr, data = wavfile.read(filepath)

    # Convert to float32 in [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float64:
        data = data.astype(np.float32)

    waveform = torch.from_numpy(data)

    # Convert to mono if stereo
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=-1)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    return waveform, target_sr


def save_audio(filepath, waveform, sr=8000):
    """Save a waveform tensor to a wav file."""
    import scipy.io.wavfile as wavfile

    if waveform.dim() > 1:
        waveform = waveform.squeeze(0)
    data = waveform.cpu().numpy()

    # Clip to [-1, 1] and convert to int16
    data = np.clip(data, -1.0, 1.0)
    data_int16 = (data * 32767).astype(np.int16)
    wavfile.write(filepath, sr, data_int16)


def normalize_waveform(waveform, eps=1e-8):
    """Normalize waveform to [-1, 1] range."""
    return waveform / (waveform.abs().max() + eps)
