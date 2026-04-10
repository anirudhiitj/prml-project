"""
Dataset module for the RCNN Cocktail Party Audio Separation project.

Provides two dataset classes:
1. LibriMixDataset     — generates 2-speaker mixtures on-the-fly from LibriSpeech
2. PreGeneratedDataset — loads pre-generated mixture .pt files from disk
"""

import os
import random
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from collections import defaultdict

from utils import STFTHelper


class LibriMixDataset(Dataset):
    """
    On-the-fly 2-speaker mixture dataset built from LibriSpeech.

    Scans the LibriSpeech directory directly (no torchaudio dataset class),
    loads FLAC files with soundfile to avoid torchcodec dependency.

    Each sample returns:
        mixture_mag:  (freq_bins, time_frames) — magnitude STFT of the mix
        source1_mag:  (freq_bins, time_frames) — magnitude STFT of speaker 1
        source2_mag:  (freq_bins, time_frames) — magnitude STFT of speaker 2
        mixture_phase:(freq_bins, time_frames) — phase of the mix (for reconstruction)
        mixture_wav:  (samples,)               — raw mixture waveform
        source1_wav:  (samples,)               — raw source 1 waveform
        source2_wav:  (samples,)               — raw source 2 waveform
    """

    def __init__(
        self,
        root_dir,
        subset="train-clean-100",
        target_sr=8000,
        segment_length=32000,  # 4 seconds at 8kHz
        n_fft=512,
        hop_length=128,
        num_samples=5000,
        seed=42,
    ):
        super().__init__()
        self.target_sr = target_sr
        self.segment_length = segment_length
        self.num_samples = num_samples
        self.stft_helper = STFTHelper(n_fft=n_fft, hop_length=hop_length)

        # Scan LibriSpeech directory directly — avoids torchaudio torchcodec dependency
        subset_dir = os.path.join(root_dir, "LibriSpeech", subset)
        print(f"Scanning LibriSpeech subset at: {subset_dir} ...")

        self.speaker_to_files = defaultdict(list)
        for speaker_id in sorted(os.listdir(subset_dir)):
            speaker_path = os.path.join(subset_dir, speaker_id)
            if not os.path.isdir(speaker_path):
                continue
            for chapter_id in os.listdir(speaker_path):
                chapter_path = os.path.join(speaker_path, chapter_id)
                if not os.path.isdir(chapter_path):
                    continue
                for fname in os.listdir(chapter_path):
                    if fname.endswith('.flac'):
                        self.speaker_to_files[speaker_id].append(
                            os.path.join(chapter_path, fname)
                        )

        self.speaker_ids = list(self.speaker_to_files.keys())
        total_files = sum(len(v) for v in self.speaker_to_files.values())
        assert len(self.speaker_ids) >= 2, "Need at least 2 speakers!"
        print(f"Found {len(self.speaker_ids)} speakers, {total_files} utterances.")

        # Pre-generate random pairs for determinism
        rng = random.Random(seed)
        self.pairs = []
        for _ in range(num_samples):
            spk1, spk2 = rng.sample(self.speaker_ids, 2)
            file1 = rng.choice(self.speaker_to_files[spk1])
            file2 = rng.choice(self.speaker_to_files[spk2])
            snr_offset = rng.uniform(-5, 5)  # dB offset for mixing
            self.pairs.append((file1, file2, snr_offset))

    def __len__(self):
        return self.num_samples

    def _load_and_preprocess(self, filepath):
        """Load FLAC with soundfile, resample, and trim/pad."""
        data, sr = sf.read(filepath, dtype='float32')

        # Mono
        if data.ndim > 1:
            data = data.mean(axis=1)

        waveform = torch.from_numpy(data)  # (samples,)

        # Resample if needed
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

        # Trim or pad to fixed length
        if waveform.shape[0] >= self.segment_length:
            start = random.randint(0, waveform.shape[0] - self.segment_length)
            waveform = waveform[start:start + self.segment_length]
        else:
            pad_len = self.segment_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        return waveform

    def __getitem__(self, index):
        file1, file2, snr_offset = self.pairs[index]

        # Load individual sources
        source1 = self._load_and_preprocess(file1)
        source2 = self._load_and_preprocess(file2)

        # Normalize sources
        source1 = source1 / (source1.abs().max() + 1e-8)
        source2 = source2 / (source2.abs().max() + 1e-8)

        # Apply SNR offset to source2
        snr_linear = 10 ** (snr_offset / 20.0)
        source2 = source2 * snr_linear

        # Create mixture
        mixture = source1 + source2

        # Compute STFTs
        mix_mag, mix_phase = self.stft_helper.stft(mixture)
        s1_mag, _ = self.stft_helper.stft(source1)
        s2_mag, _ = self.stft_helper.stft(source2)

        return {
            'mixture_mag': mix_mag,
            'source1_mag': s1_mag,
            'source2_mag': s2_mag,
            'mixture_phase': mix_phase,
            'mixture_wav': mixture,
            'source1_wav': source1,
            'source2_wav': source2,
        }


class PreGeneratedDataset(Dataset):
    """
    Dataset that loads pre-generated mixture files from disk.

    Each file is a dict saved by prepare_data.py containing:
        mixture_mag, source1_mag, source2_mag, mixture_phase,
        mixture_wav, source1_wav, source2_wav
    """

    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.files = sorted([
            f for f in os.listdir(data_dir) if f.endswith('.pt')
        ])
        print(f"PreGeneratedDataset: found {len(self.files)} samples in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filepath = os.path.join(self.data_dir, self.files[index])
        data = torch.load(filepath, weights_only=False)
        return data


class WavFolderDataset(Dataset):
    """
    Dataset that loads wav-based mixture folders from disk.

    Expected folder structure:
        data_dir/
            000000/
                mixture.wav
                s1.wav
                s2.wav
            000001/
                ...

    Returns the same dict format as other datasets for compatibility.
    """

    def __init__(self, data_dir, target_sr=8000, segment_length=32000,
                 n_fft=512, hop_length=128, max_samples=None):
        super().__init__()
        self.data_dir = data_dir
        self.target_sr = target_sr
        self.segment_length = segment_length
        self.stft_helper = STFTHelper(n_fft=n_fft, hop_length=hop_length)

        # Find all subdirectories containing mixture.wav
        self.sample_dirs = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
            and os.path.exists(os.path.join(data_dir, d, 'mixture.wav'))
        ])

        if max_samples is not None:
            self.sample_dirs = self.sample_dirs[:max_samples]

        print(f"WavFolderDataset: found {len(self.sample_dirs)} samples in {data_dir}")

    def __len__(self):
        return len(self.sample_dirs)

    def _load_wav(self, filepath):
        """Load a wav file, resample, and trim/pad to segment_length."""
        import scipy.io.wavfile as wavfile
        import numpy as np

        sr, data = wavfile.read(filepath)

        # Convert to float32 in [-1, 1]
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.float64:
            data = data.astype(np.float32)

        waveform = torch.from_numpy(data)

        # Mono
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=-1)

        # Resample
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)

        # Trim or pad
        if waveform.shape[0] >= self.segment_length:
            waveform = waveform[:self.segment_length]
        else:
            pad_len = self.segment_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        return waveform

    def __getitem__(self, index):
        sample_dir = os.path.join(self.data_dir, self.sample_dirs[index])

        mixture = self._load_wav(os.path.join(sample_dir, 'mixture.wav'))
        source1 = self._load_wav(os.path.join(sample_dir, 's1.wav'))
        source2 = self._load_wav(os.path.join(sample_dir, 's2.wav'))

        # Compute STFTs
        mix_mag, mix_phase = self.stft_helper.stft(mixture)
        s1_mag, _ = self.stft_helper.stft(source1)
        s2_mag, _ = self.stft_helper.stft(source2)

        return {
            'mixture_mag': mix_mag,
            'source1_mag': s1_mag,
            'source2_mag': s2_mag,
            'mixture_phase': mix_phase,
            'mixture_wav': mixture,
            'source1_wav': source1,
            'source2_wav': source2,
        }
