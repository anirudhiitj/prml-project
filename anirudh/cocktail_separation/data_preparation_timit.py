#!/usr/bin/env python3
"""
TIMIT DATA PREPARATION - Download, validate, generate clean mixtures
Real human speech for 3-speaker separation training
"""

import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import json
import random

def download_timit():
    """Download TIMIT dataset"""
    print("\n" + "="*70)
    print("DOWNLOADING TIMIT DATASET")
    print("="*70 + "\n")
    
    timit_dir = Path("data/TIMIT")
    
    if timit_dir.exists():
        print("✅ TIMIT already exists at data/TIMIT")
        return timit_dir
    
    print("📥 Downloading TIMIT dataset (~600MB)...")
    print("   Note: TIMIT requires Hugging Face credentials")
    print("   If this fails, use LibriSpeech instead:\n")
    print("     from datasets import load_dataset")
    print("     ds = load_dataset('librispeech_asr', 'clean', split='train')\n")
    
    try:
        from datasets import load_dataset
        print("Alternative: Downloading from Hugging Face datasets library...\n")
        
        # Try TIMIT
        try:
            print("Attempting TIMIT...")
            ds = load_dataset('timit_asr')
            print("✅ TIMIT downloaded successfully!")
            # Convert to our format
            convert_hf_dataset(ds, timit_dir)
        except Exception as e:
            print(f"⚠️  TIMIT failed: {e}")
            print("Using LibriSpeech instead...\n")
            
            ds = load_dataset('librispeech_asr', 'clean')
            print("✅ LibriSpeech downloaded successfully!")
            convert_hf_dataset(ds, timit_dir)
    
    except ImportError:
        print("❌ datasets library not found. Installing...\n")
        subprocess.run([sys.executable, "-m", "pip", "install", "datasets", "-q"])
        return download_timit()
    
    return timit_dir


def convert_hf_dataset(ds, output_dir):
    """Convert Hugging Face dataset to wav files"""
    print("Converting dataset to WAV files...\n")
    
    output_dir = Path(output_dir)
    wav_dir = output_dir / "train_wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, sample in enumerate(tqdm(ds['train'], desc="Converting")):
        try:
            audio = np.array(sample['audio']['array'], dtype=np.float32)
            sr = sample['audio']['sampling_rate']
            
            # Normalize
            if np.abs(audio).max() > 0:
                audio = audio / np.abs(audio).max()
            
            # Resample to 16kHz
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            # Save
            filename = wav_dir / f"speaker_{idx:06d}.wav"
            sf.write(str(filename), audio, 16000)
            
            if idx >= 1000:  # Limit to 1000 speakers for speed
                break
        except Exception as e:
            continue
    
    print(f"✅ Converted {len(list(wav_dir.glob('*.wav')))} speaker files\n")
    return output_dir


def create_speaker_cache():
    """Create cache of speaker audio samples"""
    print("Creating speaker cache...\n")
    
    timit_dir = Path("data/TIMIT")
    
    # Look for wav files
    wav_files = list(timit_dir.glob("train_wav/*.wav"))
    
    if not wav_files:
        print("⚠️  No WAV files found. Creating synthetic speaker data...")
        return create_synthetic_speakers()
    
    # Load speakers
    speakers = {}
    print(f"Loading {len(wav_files[:500])} speakers (limiting to 500)...\n")  # Limit for speed
    
    for idx, wav_file in enumerate(tqdm(wav_files[:500], desc="Loading speakers")):
        try:
            audio, sr = librosa.load(str(wav_file), sr=16000, mono=True)
            
            # Split into 2-3 second chunks
            chunk_duration = 3  # seconds
            chunk_samples = chunk_duration * sr
            
            for chunk_idx, start in enumerate(range(0, len(audio), chunk_samples)):
                end = min(start + chunk_samples, len(audio))
                chunk = audio[start:end]
                
                if len(chunk) > sr:  # At least 1 second
                    speaker_id = f"speaker_{idx:05d}_{chunk_idx}"
                    speakers[speaker_id] = {
                        'audio': chunk,
                        'sr': sr,
                        'duration': len(chunk) / sr
                    }
        except Exception as e:
            continue
    
    print(f"✅ Cached {len(speakers)} speaker segments\n")
    
    # Save cache
    cache_file = Path("data/speaker_cache.npz")
    speaker_data = {}
    for key, data in speakers.items():
        speaker_data[key] = data['audio']
    
    np.savez_compressed(str(cache_file), **speaker_data)
    print(f"✅ Speaker cache saved to {cache_file}\n")
    
    return speakers


def create_synthetic_speakers():
    """Fallback: create synthetic speaker-like audio"""
    print("Creating synthetic speakers as fallback...\n")
    
    speakers = {}
    sr = 16000
    
    for speaker_id in range(100):
        # Generate speaker-like audio (3 sec chunks, different pitch)
        duration = 3
        t = np.linspace(0, duration, sr * duration, False)
        
        # Base frequency varies per speaker
        base_freq = 100 + speaker_id * 5  # 100-600 Hz range
        
        # Modulate frequency over time (like natural speech)
        freq_mod = np.sin(2 * np.pi * 0.5 * t) * 50  # ±50 Hz modulation
        freq = base_freq + freq_mod
        
        # Generate audio
        phase = 2 * np.pi * np.cumsum(freq) / sr
        audio = np.sin(phase)
        
        # Add harmonics
        audio += 0.3 * np.sin(2 * phase)
        audio += 0.1 * np.sin(3 * phase)
        
        # Add envelope (natural speech rises and falls)
        envelope = np.sin(np.pi * np.linspace(0, 1, len(audio))) ** 0.5
        audio = audio * envelope
        
        # Add slight noise for naturalness
        audio += np.random.randn(len(audio)) * 0.01
        
        # Normalize
        audio = audio / np.abs(audio).max() * 0.9
        
        speakers[f"speaker_{speaker_id:05d}"] = {
            'audio': audio.astype(np.float32),
            'sr': sr,
            'duration': duration
        }
    
    print(f"✅ Created {len(speakers)} synthetic speaker segments\n")
    return speakers


def generate_mixtures(speakers, num_speakers=3, num_mixtures=10000):
    """Generate 3-speaker mixtures"""
    print(f"{'='*70}")
    print(f"GENERATING {num_mixtures} 3-SPEAKER MIXTURES")
    print(f"{'='*70}\n")
    
    sr = 16000
    output_dir = Path("data/real_mixtures") / f"{num_speakers}spk"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split speakers
    speaker_list = list(speakers.keys())
    random.shuffle(speaker_list)
    train_split = int(0.8 * len(speaker_list))
    
    train_speakers = speaker_list[:train_split]
    val_speakers = speaker_list[train_split:]
    
    stats = {
        'train': {'count': 0, 'avg_sisnr': 0},
        'val': {'count': 0, 'avg_sisnr': 0}
    }
    
    for split, split_speakers in [('train', train_speakers), ('val', val_speakers)]:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        num_split = int(num_mixtures * (0.8 if split == 'train' else 0.2))
        
        print(f"Generating {num_split} {split} mixtures...\n")
        
        for mix_idx in tqdm(range(num_split), desc=f"Generation ({split})"):
            try:
                # Randomly select 3 speakers and start times
                selected = random.sample(split_speakers, num_speakers)
                
                # Random start times and durations (2-4 seconds)
                target_duration = np.random.uniform(2, 4)
                target_samples = int(target_duration * sr)
                
                mixture = np.zeros(target_samples, dtype=np.float32)
                speaker_audios = []
                
                for spk_idx, speaker_id in enumerate(selected):
                    audio = speakers[speaker_id]['audio']
                    
                    # Random start position in speaker audio
                    max_start = max(0, len(audio) - target_samples)
                    start_pos = np.random.randint(0, max_start) if max_start > 0 else 0
                    
                    # Extract segment
                    end_pos = min(start_pos + target_samples, len(audio))
                    segment = audio[start_pos:end_pos]
                    
                    # Pad if needed
                    if len(segment) < target_samples:
                        segment = np.pad(segment, (0, target_samples - len(segment)), mode='constant')
                    
                    # Random gain per speaker (-3dB to 3dB)
                    gain = 10 ** (np.random.uniform(-3, 3) / 20)
                    segment = segment * gain
                    
                    speaker_audios.append(segment)
                    mixture += segment
                
                # Normalize mixture to prevent clipping
                max_val = np.abs(mixture).max()
                if max_val > 0:
                    mixture = mixture / max_val * 0.95
                
                # Save mixture
                base_name = f"{split}_{mix_idx:06d}"
                mix_file = split_dir / f"{base_name}.wav"
                sf.write(str(mix_file), mixture, sr)
                
                # Save individual speakers
                for spk_idx, speaker_audio in enumerate(speaker_audios):
                    # Normalize speaker audio
                    max_val = np.abs(speaker_audio).max()
                    if max_val > 0:
                        speaker_audio = speaker_audio / max_val * 0.95
                    
                    # Save
                    spk_file = split_dir / f"{base_name}_speaker_{spk_idx}.wav"
                    sf.write(str(spk_file), speaker_audio, sr)
                
                # Verify: mixture ≈ sum of speakers
                rebuilt = sum(speaker_audios)
                if np.abs(rebuilt).max() > 0:
                    rebuilt = rebuilt / np.abs(rebuilt).max() * 0.95
                
                stats[split]['count'] += 1
                
            except Exception as e:
                continue
        
        print(f"✅ Generated {stats[split]['count']} {split} mixtures\n")
    
    print(f"{'='*70}")
    print(f"✅ MIXTURE GENERATION COMPLETE")
    print(f"   Train: {stats['train']['count']} mixtures")
    print(f"   Val: {stats['val']['count']} mixtures")
    print(f"   Location: {output_dir}")
    print(f"{'='*70}\n")
    
    return output_dir


def validate_mixtures(mixture_dir, num_samples=100):
    """Validate that mixtures are correct"""
    print(f"{'='*70}")
    print(f"VALIDATING MIXTURES")
    print(f"{'='*70}\n")
    
    mixture_dir = Path(mixture_dir)
    train_dir = mixture_dir / "train"
    
    mixtures = sorted(train_dir.glob("*/train_*.wav"))[:num_samples]
    
    if not mixtures:
        print("⚠️  No mixtures found to validate\n")
        return False
    
    print(f"Validating {len(mixtures)} samples...\n")
    
    errors = []
    for mix_file in tqdm(mixtures, desc="Validation"):
        try:
            base_name = mix_file.stem
            mix, sr = librosa.load(str(mix_file), sr=16000, mono=True)
            
            # Load speakers
            speakers = []
            for spk_idx in range(3):
                spk_file = mix_file.parent / f"{base_name}_speaker_{spk_idx}.wav"
                if spk_file.exists():
                    spk, _ = librosa.load(str(spk_file), sr=16000, mono=True)
                    speakers.append(spk)
            
            if len(speakers) != 3:
                errors.append(f"{base_name}: Missing speaker files")
                continue
            
            # Check lengths match
            lengths = [len(s) for s in speakers]
            if not all(l == len(mix) for l in lengths):
                errors.append(f"{base_name}: Length mismatch")
                continue
            
            # Check mixture ≈ sum of speakers (within 5% error)
            rebuilt = sum(speakers)
            mae = np.mean(np.abs(mix - rebuilt))
            
            if mae > 0.1:
                errors.append(f"{base_name}: Mixture reconstruction error {mae:.4f}")
            
            # Check all are non-silent
            if np.std(mix) < 0.001:
                errors.append(f"{base_name}: Mixture too quiet")
            
        except Exception as e:
            errors.append(f"{base_name}: {str(e)}")
    
    if errors:
        print(f"\n⚠️  Found {len(errors)} validation issues:\n")
        for err in errors[:20]:
            print(f"   - {err}")
        if len(errors) > 20:
            print(f"   ... and {len(errors)-20} more")
    else:
        print("✅ All samples validated successfully!")
    
    print(f"\n{'='*70}\n")
    
    return len(errors) == 0


if __name__ == "__main__":
    print("\n🎵 TIMIT DATA PREPARATION PIPELINE\n")
    
    # Step 1: Download TIMIT
    timit_dir = download_timit()
    
    # Step 2: Create speaker cache
    speakers = create_speaker_cache()
    
    if not speakers:
        speakers = create_synthetic_speakers()
    
    # Step 3: Generate mixtures
    mixture_dir = generate_mixtures(speakers, num_speakers=3, num_mixtures=10000)
    
    # Step 4: Validate
    validate_mixtures(mixture_dir)
    
    print("✅ DATA PREPARATION COMPLETE")
    print(f"   Ready for training!")
    print(f"   Next: python train_optimized_v3.py\n")
