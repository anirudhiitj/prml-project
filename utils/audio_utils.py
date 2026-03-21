"""
Audio utilities — reading, writing, normalizing audio files.
"""

import torch
import torchaudio


def load_audio(path: str, target_sr: int = 8000) -> torch.Tensor:
    """
    Load a WAV file and optionally resample.

    Args:
        path: Path to the audio file.
        target_sr: Desired sample rate.

    Returns:
        (1, T) tensor — mono waveform
    """
    waveform, sr = torchaudio.load(path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform  # (1, T)


def save_audio(waveform: torch.Tensor, path: str, sample_rate: int = 8000):
    """
    Save a waveform tensor as a WAV file.

    Args:
        waveform: (1, T) or (T,) tensor
        path: Output file path.
        sample_rate: Sample rate of the audio.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(path, waveform.cpu(), sample_rate)


def normalize(waveform: torch.Tensor) -> torch.Tensor:
    """
    Zero-mean, unit-variance normalization.

    Args:
        waveform: (..., T) tensor

    Returns:
        Normalized waveform of the same shape
    """
    mean = waveform.mean(dim=-1, keepdim=True)
    std = waveform.std(dim=-1, keepdim=True).clamp(min=1e-8)
    return (waveform - mean) / std
