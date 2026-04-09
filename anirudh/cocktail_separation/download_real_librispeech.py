#!/usr/bin/env python3
"""
Download REAL LibriSpeech corpus and generate 3-speaker mixtures
"""

import os
import urllib.request
import tarfile
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
import random

print("\n" + "="*80)
print("DOWNLOADING REAL LIBRISPEECH CORPUS")
print("="*80 + "\n")

# Base URL for LibriSpeech
LIBRISPEECH_BASE = "https://www.openslr.org/resources/12"

# We'll use train-clean-100 (100 hours of clean speech, ~400 speakers)
# Smaller than train-360 but sufficient for testing
DATASET_NAME = "train-clean-100"
DOWNLOAD_URL = f"{LIBRISPEECH_BASE}/{DATASET_NAME}.tar.gz"

data_dir = Path("data/librispeech_raw")
data_dir.mkdir(parents=True, exist_ok=True)

tar_path = data_dir / f"{DATASET_NAME}.tar.gz"
extract_path = data_dir / "extracted"

# Step 1: Download if not already present
if not tar_path.exists():
    print(f"⏳ Downloading LibriSpeech {DATASET_NAME} (~6.3 GB)...")
    print(f"   URL: {DOWNLOAD_URL}\n")
    try:
        urllib.request.urlretrieve(DOWNLOAD_URL, tar_path, lambda a, b, c: None)
        print(f"✅ Downloaded to {tar_path}\n")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("   Try manually downloading from: https://www.openslr.org/12/\n")
        exit(1)

# Step 2: Extract
if not extract_path.exists():
    print(f"⏳ Extracting tar.gz (~10 min)...")
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(extract_path)
        print(f"✅ Extracted\n")
    except Exception as e:
        print(f"❌ Extraction failed: {e}\n")
        exit(1)

# Step 3: Find all .flac files
print("🔍 Scanning for FLAC audio files...\n")
flac_files = list(extract_path.glob("**/*.flac"))
print(f"✅ Found {len(flac_files)} FLAC files\n")

if len(flac_files) < 3:
    print("❌ Not enough audio files found\n")
    exit(1)

# Step 4: Create 3-speaker mixtures
print("🎵 Creating 3-speaker mixtures from real LibriSpeech data...\n")

sr = 16000
output_dir = Path("data/real_librispeech_mixtures/3spk")
output_dir.mkdir(parents=True, exist_ok=True)

stats = {'train': 0, 'val': 0}

# Shuffle and split files
random.shuffle(flac_files)
split_idx = int(len(flac_files) * 0.9)
train_files = flac_files[:split_idx]
val_files = flac_files[split_idx:]

for split, files, max_mix in [('train', train_files, 5000), ('val', val_files, 500)]:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {min(max_mix, len(files) // 3)} {split} mixtures from {len(files)} files...\n")
    
    mix_idx = 0
    for i in tqdm(range(0, len(files) - 2, 3)):
        if mix_idx >= max_mix:
            break
        
        try:
            # Load 3 different speaker files
            speakers = []
            for j in range(3):
                # Load FLAC file
                audio, file_sr = librosa.load(str(files[i + j]), sr=sr, mono=True)
                
                # Trim to 3-4 seconds
                target_len = int(3.5 * sr)
                if len(audio) > target_len:
                    start = random.randint(0, len(audio) - target_len)
                    audio = audio[start:start + target_len]
                elif len(audio) < target_len:
                    # Pad if needed
                    audio = np.pad(audio, (0, target_len - len(audio)))
                
                speakers.append(audio)
            
            # Random gains (to simulate different speaker volumes)
            gains = [10 ** (random.uniform(-6, 6) / 20) for _ in range(3)]
            
            # Mix speakers
            mixture = sum(s * g for s, g in zip(speakers, gains))
            mixture = np.clip(mixture, -1, 1)
            
            # Save
            mix_dir = split_dir / f"{mix_idx:06d}"
            mix_dir.mkdir(parents=True, exist_ok=True)
            
            sf.write(str(mix_dir / "mixture.wav"), mixture, sr)
            
            for spk_idx, (src, gain) in enumerate(zip(speakers, gains)):
                src_scaled = src * gain
                src_scaled = np.clip(src_scaled, -1, 1)
                sf.write(str(mix_dir / f"s{spk_idx+1}.wav"), src_scaled, sr)
            
            stats[split] += 1
            mix_idx += 1
            
        except Exception as e:
            continue
    
    print(f"✅ Generated {stats[split]} {split} mixtures\n")

print("="*80)
print("✅ REAL LIBRISPEECH DATA READY")
print(f"   Train: {stats['train']} mixtures")
print(f"   Val: {stats['val']} mixtures")
print(f"   Location: {output_dir}")
print(f"   Type: REAL HUMAN SPEECH (LibriSpeech)")
print("="*80)
print("\nNext: Start training with:")
print("  python3 train_librispeech.py\n")
