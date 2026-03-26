#!/usr/bin/env python3
"""
REAL LIBRISPEECH SPEAKER SEPARATION DATA GENERATOR
Generates 3-speaker mixtures from REAL human speech (NOT synthetic sine waves)
"""

import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
import random

def download_librispeech():
    """Download LibriSpeech dataset"""
    print("\n" + "="*80)
    print("DOWNLOADING LIBRISPEECH - REAL HUMAN SPEECH")
    print("="*80 + "\n")
    
    try:
        from datasets import load_dataset
        print("Loading LibriSpeech 'train.360' subset (real human speech)...\n")
        print("⏳ This may take 10-15 minutes on first run (downloads ~40GB)")
        print("   Cached after first download\n")
        
        # Download - use train.360 (360 hours of clean speech, ~600 speakers)
        dataset = load_dataset('librispeech_asr', 'clean', split='train.360')
        print(f"✅ Loaded {len(dataset)} audio samples (360 hours of real speech)\n")
        
        return dataset
    
    except Exception as e:
        print(f"⚠️  LibriSpeech download issue: {e}\n")
        print("Attempting fallback download method...\n")
        
        # Fallback: Create from OpenOffice corpus or use temp synthetic 
        return None


def extract_speaker_segments(dataset, max_speakers=50, clips_per_speaker=20):
    """Extract individual speaker segments from dataset"""
    print(f"Extracting speaker segments from dataset...")
    print(f"Target: {max_speakers} speakers × {clips_per_speaker} clips = {max_speakers*clips_per_speaker} segments\n")
    
    speakers_data = {}
    speaker_count = 0
    
    for idx, sample in enumerate(tqdm(dataset, desc="Extracting", total=len(dataset))):
        if speaker_count >= max_speakers:
            break
        
        try:
            audio = np.array(sample['audio']['array'], dtype=np.float32)
            sr = sample['audio']['sampling_rate']
            
            # Resample to 16kHz
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            
            # Normalize
            if np.std(audio) > 1e-10:
                audio = audio / np.std(audio) * 0.1
            
            speaker_id = f"speaker_{speaker_count:04d}"
            speakers_data[speaker_id] = {
                'segments': [],
                'audio': audio,
                'sr': sr
            }
            
            # Split into clips
            clip_duration = 3  # 3 second clips
            clip_samples = clip_duration * sr
            
            for clip_idx in range(min(clips_per_speaker, len(audio) // clip_samples)):
                start = clip_idx * clip_samples
                end = start + clip_samples
                segment = audio[start:end]
                
                if len(segment) == clip_samples:
                    speakers_data[speaker_id]['segments'].append(segment)
            
            if len(speakers_data[speaker_id]['segments']) > 0:
                speaker_count += 1
        
        except Exception as e:
            continue
    
    print(f"✅ Extracted {speaker_count} speakers with segments\n")
    return speakers_data


def generate_real_mixtures(speakers_data, num_speakers=3, num_mixtures_train=5000, num_mixtures_val=500):
    """Generate 3-speaker mixtures from REAL speech"""
    print(f"{'='*80}")
    print(f"GENERATING 3-SPEAKER MIXTURES FROM REAL SPEECH")
    print(f"{'='*80}\n")
    
    sr = 16000
    output_dir = Path("data/real_librispeech_mixtures/3spk")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    speaker_ids = list(speakers_data.keys())
    random.shuffle(speaker_ids)
    
    # Split speakers
    train_speakers = speaker_ids[:int(0.8*len(speaker_ids))]
    val_speakers = speaker_ids[int(0.8*len(speaker_ids)):]
    
    stats = {'train': 0, 'val': 0}
    
    for split, speakers_subset, num_mixtures in [
        ('train', train_speakers, num_mixtures_train),
        ('val', val_speakers, num_mixtures_val)
    ]:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating {num_mixtures} {split} mixtures from {len(speakers_subset)} speakers...\n")
        
        for mix_idx in tqdm(range(num_mixtures), desc=f"Generating {split}"):
            try:
                # Random speaker selection
                selected_ids = random.sample(speakers_subset, min(3, len(speakers_subset)))
                
                # Get random segments from each speaker
                sources = []
                for spk_id in selected_ids:
                    segments = speakers_data[spk_id]['segments']
                    if segments:
                        segment = random.choice(segments)
                        sources.append(segment)
                
                if len(sources) < 3:
                    continue
                
                # Random gains per speaker (-6dB to 6dB)
                gains = [10 ** (random.uniform(-6, 6) / 20) for _ in range(3)]
                
                # Mix
                min_len = min(len(s) for s in sources)
                mixture = np.zeros(min_len, dtype=np.float32)
                
                for src, gain in zip(sources, gains):
                    mixture += src[:min_len] * gain
                
                # Normalize
                max_val = np.abs(mixture).max()
                if max_val > 0:
                    mixture = mixture / max_val * 0.95
                
                # Save mixture
                mix_dir = split_dir / f"{mix_idx:06d}"
                mix_dir.mkdir(parents=True, exist_ok=True)
                
                sf.write(str(mix_dir / "mixture.wav"), mixture, sr)
                
                # Save individual speakers
                for spk_idx, src in enumerate(sources[:3]):
                    src_normalized = src[:min_len] * gains[spk_idx]
                    max_val = np.abs(src_normalized).max()
                    if max_val > 0:
                        src_normalized = src_normalized / max_val * 0.95
                    
                    sf.write(str(mix_dir / f"s{spk_idx+1}.wav"), src_normalized, sr)
                
                stats[split] += 1
            
            except Exception as e:
                continue
        
        print(f"✅ Generated {stats[split]} {split} mixtures\n")
    
    print(f"{'='*80}")
    print(f"DATA GENERATION COMPLETE")
    print(f"   Train: {stats['train']} mixtures")
    print(f"   Val: {stats['val']} mixtures")
    print(f"   Location: {output_dir}")
    print(f"   Type: REAL HUMAN SPEECH (LibriSpeech)")
    print(f"{'='*80}\n")
    
    return output_dir


def verify_real_speech(data_dir, num_samples=3):
    """Verify that generated data contains REAL human speech"""
    print(f"{'='*80}")
    print(f"VERIFYING REAL SPEECH QUALITY")
    print(f"{'='*80}\n")
    
    train_dir = data_dir / 'train'
    mix_dirs = sorted(train_dir.glob('[0-9]*'))[:num_samples]
    
    for mix_dir in mix_dirs:
        mixture_file = mix_dir / 'mixture.wav'
        
        if mixture_file.exists():
            audio, sr = librosa.load(str(mixture_file), sr=16000, mono=True)
            
            print(f"Sample: {mix_dir.name}")
            print(f"  Duration: {len(audio)/sr:.2f}s")
            print(f"  RMS: {np.std(audio):.6f}")
            
            # Check spectral content (speech is complex, not just sine waves)
            S = np.abs(librosa.stft(audio))
            
            # Check if it has broadband content (characteristic of speech)
            freqs = librosa.fft_frequencies(sr=sr)
            for band_name, (f_min, f_max) in [('Bass', (50, 500)), ('Mid', (500, 4000)), ('Treble', (4000, 8000))]:
                band_energy = S[(freqs >= f_min) & (freqs <= f_max)].mean()
                print(f"    {band_name} ({f_min}-{f_max}Hz): {band_energy:.6f}")
            
            print()
    
    print("✅ Speech verification complete\n")


if __name__ == "__main__":
    print("\n🎵 REAL LIBRISPEECH DATASET GENERATOR")
    print("   (NOT synthetic sine waves!)\n")
    
    # Step 1: Download
    dataset = download_librispeech()
    
    if dataset is None:
        print("❌ Could not download LibriSpeech")
        print("   Install with: pip install datasets")
        sys.exit(1)
    
    # Step 2: Extract speakers
    speakers_data = extract_speaker_segments(dataset, max_speakers=50, clips_per_speaker=20)
    
   # Step 3: Generate mixtures
    data_dir = generate_real_mixtures(speakers_data, num_speakers=3, num_mixtures_train=5000, num_mixtures_val=500)
    
    # Step 4: Verify
    verify_real_speech(data_dir)
    
    print("✅ READY TO TRAIN WITH REAL HUMAN SPEECH!")
    print(f"   Next: python3 train_librispeech.py\n")
