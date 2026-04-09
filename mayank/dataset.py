import os
import glob
import random
import torch
import torchaudio
from torch.utils.data import Dataset

class LibriSpeechMixDataset(Dataset):
    def __init__(self, data_dirs, num_speakers=2, sample_rate=8000, segment_length=16000, 
                 epoch_length=10000, noise_dir=None, noise_snr_low=5, noise_snr_high=20):
        """
        Dynamically loads and mixes speakers from pure bare LibriSpeech folders on-the-fly.
        Optionally injects real-world background noise from MUSAN for noise-resilient training.
        Args:
            data_dirs: List of paths to LibriSpeech subsets (e.g. ['path/to/train-clean-100'])
            num_speakers: Number of clean voices to mix
            sample_rate: The uniform sample rate to resample to (LibriSpeech natively is 16kHz)
            segment_length: Target audio chunk size in samples to feed network
            epoch_length: Virtual number of batches/steps per epoch since mixtures are infinite
            noise_dir: Path to MUSAN noise directory (e.g. 'musan_dataset/musan/noise'). If None, no noise is added.
            noise_snr_low: Minimum SNR (dB) for noise injection (lower = more noise)
            noise_snr_high: Maximum SNR (dB) for noise injection (higher = less noise)
        """
        self.num_speakers = num_speakers
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.epoch_length = epoch_length
        self.all_files = []
        
        for d in data_dirs:
            pattern = os.path.join(d, "**", "*.flac")
            self.all_files.extend(glob.glob(pattern, recursive=True))
            
        if not self.all_files:
            raise ValueError(f"Could not find .flac files in {data_dirs}. Check paths!")
        
        # Load noise file paths if noise augmentation is enabled
        self.noise_files = []
        if noise_dir is not None and os.path.isdir(noise_dir):
            for ext in ("*.wav", "*.flac"):
                pattern = os.path.join(noise_dir, "**", ext)
                self.noise_files.extend(glob.glob(pattern, recursive=True))
            if self.noise_files:
                print(f"[Dataset] Noise augmentation ENABLED: Found {len(self.noise_files)} noise files in {noise_dir}")
            else:
                print(f"[Dataset] WARNING: noise_dir={noise_dir} provided but no .wav/.flac files found. Noise augmentation DISABLED.")
        
        self.noise_snr_low = noise_snr_low
        self.noise_snr_high = noise_snr_high
            
    def __len__(self):
        return self.epoch_length
        
    def _get_random_segment(self):
        # Keep sampling until we get an audio clip cleanly longer than our target segment length
        while True:
            file_path = random.choice(self.all_files)
            try:
                wav, sr = torchaudio.load(file_path)
                
                if sr != self.sample_rate:
                    wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
                    
                # To purely 1D mono
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                wav = wav.squeeze(0)
                
                # Check bounds
                if wav.shape[0] >= self.segment_length:
                    start_idx = random.randint(0, wav.shape[0] - self.segment_length)
                    return wav[start_idx : start_idx + self.segment_length]
            except RuntimeError:
                continue

    def _get_random_noise_segment(self):
        """Load a random noise clip from MUSAN and return a segment of self.segment_length."""
        while True:
            file_path = random.choice(self.noise_files)
            try:
                wav, sr = torchaudio.load(file_path)
                
                if sr != self.sample_rate:
                    wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
                    
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                wav = wav.squeeze(0)
                
                if wav.shape[0] >= self.segment_length:
                    start_idx = random.randint(0, wav.shape[0] - self.segment_length)
                    return wav[start_idx : start_idx + self.segment_length]
                else:
                    # If noise clip is too short, loop/tile it to fill the segment
                    repeats = (self.segment_length // wav.shape[0]) + 1
                    wav = wav.repeat(repeats)[:self.segment_length]
                    return wav
            except RuntimeError:
                continue

    def __getitem__(self, idx):
        # Fetch N random speakers
        sources = [self._get_random_segment() for _ in range(self.num_speakers)]
        
        # Targets shape: (num_speakers, time) — these stay CLEAN (no noise)
        targets = torch.stack(sources, dim=0)
        
        # Mathematical Mix of clean speakers
        mix = targets.sum(dim=0)
        
        # Inject background noise into the mixture (but NOT into the targets)
        if self.noise_files:
            noise = self._get_random_noise_segment()
            # Scale the noise to a random SNR relative to the clean mixture
            snr_db = random.uniform(self.noise_snr_low, self.noise_snr_high)
            signal_power = torch.mean(mix ** 2)
            noise_power = torch.mean(noise ** 2)
            if noise_power > 0:
                scale = torch.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10))))
                mix = mix + scale * noise
        
        # Soft normalization to prevent extreme clipping
        max_val = torch.max(torch.abs(mix))
        if max_val > 1.0:
            mix = mix / max_val
            targets = targets / max_val
            
        return mix, targets
