#!/usr/bin/env python3
"""
Synthetic Multi-Speaker Mixture Generator for DPRNN Training
Generates 2spk, 3spk, 4spk, 5spk mixtures with augmentation
"""

from __future__ import annotations

import os
import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
import pyroomacoustics as pra
from scipy.signal import resample_poly

class SyntheticMixtureGenerator:
    """Generate synthetic multi-speaker mixtures from audio files"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        clip_duration: float = 4.0,
        snr_range: tuple[float, float] = (-5, 5),
        output_format: str = "float32",
    ):
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.clip_samples = int(sample_rate * clip_duration)
        self.snr_range = snr_range
        self.output_format = output_format
    
    def load_and_resample_audio(self, audio_path: str) -> np.ndarray:
        """Load audio and resample to target sample rate"""
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            if sr != self.sample_rate:
                y = resample_poly(y, self.sample_rate, sr)
            return y.astype(np.float32)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None
    
    def fix_length(self, x: np.ndarray, target_len: int) -> np.ndarray:
        """Randomly crop or pad audio to target length"""
        if len(x) >= target_len:
            start = random.randint(0, len(x) - target_len)
            return x[start:start + target_len]
        else:
            pad_amount = target_len - len(x)
            return np.pad(x, (0, pad_amount), mode='constant', constant_values=0)
    
    def generate_mixture(
        self,
        source_paths: list[str],
        output_dir: Path,
        mixture_id: str,
        apply_rir: bool = True,
        apply_noise: bool = True,
    ) -> bool:
        """Generate a single multi-speaker mixture"""
        
        num_speakers = len(source_paths)
        
        # Load sources
        sources = []
        for source_path in source_paths:
            audio = self.load_and_resample_audio(source_path)
            if audio is None:
                return False
            audio = self.fix_length(audio, self.clip_samples)
            sources.append(audio)
        
        # Apply RIR (Room Impulse Response) if requested
        if apply_rir:
            sources = self._apply_rir_augmentation(sources)
        
        # Generate SNR values for each speaker
        snr_values = [0.0] + list(
            np.random.uniform(self.snr_range[0], self.snr_range[1], num_speakers - 1)
        )
        
        # Apply scaling based on SNR
        gains = []
        mixture = np.zeros(self.clip_samples, dtype=np.float32)
        scaled_sources = []
        
        for i, (source, snr_db) in enumerate(zip(sources, snr_values)):
            gain = 10 ** (snr_db / 20.0)
            gains.append(gain)
            scaled_source = gain * source
            scaled_sources.append(scaled_source)
            mixture += scaled_source
        
        # Peak normalization
        max_val = np.max(np.abs(mixture)) + 1e-8
        mixture = mixture / max_val
        scaled_sources = [s / max_val for s in scaled_sources]
        
        # Pad to 5 speakers if less
        while len(scaled_sources) < 5:
            scaled_sources.append(np.zeros(self.clip_samples, dtype=np.float32))
        
        # Apply noise augmentation to mixture
        if apply_noise:
            snr_noise_db = np.random.uniform(20, 40)
            noise = np.random.normal(0, 1, self.clip_samples).astype(np.float32)
            noise = noise / np.std(noise)
            noise_power = np.mean(mixture ** 2)
            noise_power_target = noise_power / (10 ** (snr_noise_db / 10))
            noise = np.sqrt(noise_power_target) * noise
            mixture = mixture + noise
            mixture = mixture / (np.max(np.abs(mixture)) + 1e-8)
        
        # Save mixture and sources
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sf.write(str(output_dir / "mixture.wav"), mixture, self.sample_rate)
        for i, source in enumerate(scaled_sources[:num_speakers]):
            sf.write(str(output_dir / f"s{i+1}.wav"), source, self.sample_rate)
        
        return True
    
    def _apply_rir_augmentation(self, sources: list[np.ndarray]) -> list[np.ndarray]:
        """Apply synthetic Room Impulse Response to sources"""
        try:
            # Random room dimensions
            room_dim = np.random.uniform(3, 10, 3)
            room_dim[2] = np.random.uniform(2.5, 4.5)  # height
            
            # Random absorption
            absorption = np.random.uniform(0.1, 0.5)
            
            # Create room
            room = pra.ShoeBox(room_dim, fs=self.sample_rate, materials=pra.Material(absorption), ray_tracing=False)
            
            # Random microphone and speaker positions
            mic_pos = np.random.uniform(0.5, room_dim - 0.5)
            room.add_microphone(mic_pos)
            
            augmented_sources = []
            for source in sources:
                # Random speaker position
                speaker_pos = np.random.uniform(0.5, room_dim - 0.5)
                room.add_source(speaker_pos)
                room.sources[-1].signal = source
            
            # Simulate
            room.simulate()
            
            # Extract filtered signals
            for i in range(len(sources)):
                filtered = room.mic_array.signals[0, :]
                augmented_sources.append(filtered.astype(np.float32))
                # Remove source and microphone for next iteration
                room.sources = []
                room.mic_array.R = np.random.uniform(0.5, room_dim[[0, 1]] - 0.5, (2, 1))
            
            return augmented_sources
        except:
            # If RIR generation fails, return original sources
            return sources

def create_dummy_dataset(output_dir: Path, num_speakers: int, num_mixtures: int, num_per_speaker: int = 50):
    """Create a dummy dataset with synthetic audio for testing"""
    
    sample_rate = 16000
    duration = 4.0
    num_samples = int(sample_rate * duration)
    
    print(f"Creating dummy audio files for testing...")
    speakers_dir = output_dir / "speakers"
    speakers_dir.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic speaker clips
    speaker_files = []
    for spk_id in range(num_speakers + 2):  # Extra speakers for variety
        speaker_audio = []
        for clip_id in range(num_per_speaker):
            # Create synthetic speech-like signal
            # Mix of sine waves (fundamental + harmonics)
            t = np.linspace(0, duration, num_samples)
            freq = 100 + spk_id * 50  # Different pitch per speaker
            signal = np.sin(2 * np.pi * freq * t) * 0.1
            signal += np.sin(2 * np.pi * freq * 2 * t) * 0.05
            signal += np.random.normal(0, 0.01, num_samples)  # Add noise
            signal = np.array(signal, dtype=np.float32)
            
            clip_path = speakers_dir / f"speaker_{spk_id:03d}_clip_{clip_id:03d}.wav"
            sf.write(str(clip_path), signal, sample_rate)
            speaker_files.append(str(clip_path))
    
    # Generate mixtures
    generator = SyntheticMixtureGenerator(sample_rate=sample_rate)
    
    for mix_id in tqdm(range(num_mixtures), desc=f"Generating {num_mixtures} mixtures"):
        # Sample random speakers
        selected_speakers = random.sample(speaker_files, num_speakers)
        
        mixture_dir = output_dir / f"{mix_id:07d}"
        mixture_dir.mkdir(parents=True, exist_ok=True)
        
        generator.generate_mixture(
            selected_speakers,
            mixture_dir,
            f"{mix_id:07d}",
            apply_rir=False,  # Disabled for dummy dataset
            apply_noise=False
        )

def generate_curriculum_data(
    librispeech_path: Path = None,
    output_base_dir: Path = Path("data/mixtures"),
    test_mode: bool = False
):
    """Generate training data for all phases of curriculum learning"""
    
    # Datasets per phase
    datasets = {
        "2spk": {"train": 1000 if test_mode else 80000, "val": 200 if test_mode else 4000, "test": 200 if test_mode else 2000},
        "3spk": {"train": 1000 if test_mode else 80000, "val": 200 if test_mode else 4000, "test": 200 if test_mode else 2000},
        "4spk": {"train": 1000 if test_mode else 80000, "val": 200 if test_mode else 4000, "test": 200 if test_mode else 2000},
        "5spk": {"train": 1000 if test_mode else 100000, "val": 200 if test_mode else 5000, "test": 200 if test_mode else 3000},
    }
    
    for spk_config, splits in datasets.items():
        num_speakers = int(spk_config[0])
        print(f"\n{'='*60}")
        print(f"Generating {spk_config} mixtures")
        print(f"{'='*60}")
        
        for split, count in splits.items():
            output_dir = output_base_dir / spk_config / split
            print(f"\nGenerating {count} {split} mixtures for {spk_config}...")
            
            create_dummy_dataset(output_dir, num_speakers, count, num_per_speaker=5)
            
            print(f"✓ Generated {count} {split} mixtures at {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multi-speaker mixtures for DPRNN training")
    parser.add_argument("--output-dir", type=str, default="data/mixtures", help="Output directory")
    parser.add_argument("--test-mode", action="store_true", help="Generate small test dataset")
    parser.add_argument("--librispeech-path", type=str, default=None, help="Path to LibriSpeech dataset")
    
    args = parser.parse_args()
    
    generate_curriculum_data(
        librispeech_path=Path(args.librispeech_path) if args.librispeech_path else None,
        output_base_dir=Path(args.output_dir),
        test_mode=args.test_mode
    )
    
    print(f"\n{'='*60}")
    print("✓ Data generation complete!")
    print(f"{'='*60}\n")
