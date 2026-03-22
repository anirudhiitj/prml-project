#!/usr/bin/env python3
"""
DPRNN-TasNet Inference Script - DEBUG VERSION
Shows what's happening during separation
"""

import argparse
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import time

from src.model import DPRNNTasNet


def load_audio(audio_path: str, sample_rate: int = 16000, max_duration: float = 10.0) -> torch.Tensor:
    """
    Load audio file and convert to mono 16kHz
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate (default 16kHz)
        max_duration: Maximum duration in seconds (default 10 sec)
    
    Returns:
        Tensor of shape [1, num_samples]
    """
    import librosa
    
    print(f"📂 Loading audio: {audio_path}")
    
    # Try torchaudio first, fallback to librosa
    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"   ⚠️  torchaudio failed: {e}")
        print(f"   📚 Using librosa instead...")
        waveform_np, sr = librosa.load(audio_path, sr=None, mono=True)
        waveform = torch.from_numpy(waveform_np).unsqueeze(0).float()
        
    # Load audio
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.from_numpy(waveform).float()
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Limit duration to max_duration seconds
    max_samples = int(sample_rate * max_duration)
    if waveform.shape[1] > max_samples:
        print(f"⚠️  Audio longer than {max_duration}s, truncating to {max_duration}s")
        waveform = waveform[:, :max_samples]
    
    duration = waveform.shape[1] / sample_rate
    print(f"✅ Loaded: {waveform.shape[1]:,} samples ({duration:.2f}s) @ {sample_rate}Hz")
    print(f"   Input range: [{waveform.min():.4f}, {waveform.max():.4f}]")
    
    return waveform


def load_model(checkpoint_path: str, num_speakers: int, device: str = "cuda") -> torch.nn.Module:
    """Load trained model from checkpoint"""
    print(f"🔄 Loading model from: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    
    # Create model with correct config keys
    model = DPRNNTasNet(
        num_speakers=int(cfg["model"]["num_speakers"]),
        encoder_dim=int(cfg["model"]["encoder_dim"]),
        encoder_kernel=int(cfg["model"]["encoder_kernel"]),
        encoder_stride=int(cfg["model"]["encoder_stride"]),
        bottleneck_dim=int(cfg["model"]["bottleneck_dim"]),
        chunk_size=int(cfg["model"]["chunk_size"]),
        num_dprnn_blocks=int(cfg["model"]["num_dprnn_blocks"]),
    ).to(device)
    
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    print(f"✅ Model loaded (epoch {ckpt['epoch']}, val_sisnr={float(ckpt['val_sisnr']):.2f} dB, {int(cfg['model']['num_speakers'])} speakers)")
    return model, cfg


@torch.no_grad()
def separate_speakers(
    model: torch.nn.Module,
    mixture: torch.Tensor,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Separate mixture into individual speakers
    
    Args:
        model: Model instance
        mixture: Tensor of shape [1, num_samples]
        device: Device to run on
    
    Returns:
        Tensor of shape [num_speakers, num_samples]
    """
    mixture = mixture.to(device)
    
    print(f"🔊 Separating {mixture.shape[1]:,} samples...")
    start_time = time.time()
    
    # Run model
    estimates = model(mixture)  # [1, num_speakers, num_samples]
    estimates = estimates.squeeze(0)  # [num_speakers, num_samples]
    
    elapsed = time.time() - start_time
    print(f"⏱️  Inference time: {elapsed:.2f}s")
    
    # DEBUG: Analyze outputs
    print(f"\n📊 OUTPUT ANALYSIS:")
    print(f"   Model output shape: {estimates.shape}")
    
    for i, est in enumerate(estimates):
        min_val = est.min().item()
        max_val = est.max().item()
        mean_val = est.mean().item()
        std_val = est.std().item()
        energy = (est ** 2).mean().sqrt().item()
        
        print(f"\n   Speaker {i+1}:")
        print(f"      Range: [{min_val:.6f}, {max_val:.6f}]")
        print(f"      Mean: {mean_val:.6f}, Std: {std_val:.6f}")
        print(f"      Energy (RMS): {energy:.6f}")
        
        # Check if similar to input
        correlation = torch.nn.functional.cosine_similarity(
            mixture.squeeze(0).to(device), 
            est.to(device), 
            dim=0
        ).item()
        print(f"      Correlation with input: {correlation:.4f}")
    
    return estimates.cpu()


def save_separated_audio(
    estimates: torch.Tensor,
    output_dir: str,
    sample_rate: int = 16000
) -> None:
    """Save separated speakers to individual files"""
    import soundfile as sf
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving separated audio to: {output_dir}")
    
    for i, estimate in enumerate(estimates):
        output_file = output_path / f"speaker_{i+1}.wav"
        
        # Convert to numpy - DO NOT normalize globally
        audio_np = estimate.numpy()
        
        # Store pre-normalization stats
        pre_min = audio_np.min()
        pre_max = audio_np.max()
        pre_rms = np.sqrt((audio_np ** 2).mean())
        
        # Check for clipping FIRST, normalize if needed
        max_abs = np.abs(audio_np).max()
        
        print(f"   Speaker {i+1}:")
        print(f"      Pre-norm range: [{pre_min:.6f}, {pre_max:.6f}]")
        print(f"      RMS energy: {pre_rms:.6f}")
        
        if max_abs > 1.0:
            print(f"      ⚠️  Clipping detected (max={max_abs:.6f}), normalizing...")
            audio_np = audio_np / max_abs
        
        sf.write(str(output_file), audio_np, sample_rate, subtype='PCM_16')
        print(f"      ✅ Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Separate multi-speaker audio using DPRNN-TasNet")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num-speakers", type=int, required=True, help="Number of speakers (2-5)")
    parser.add_argument("--output-dir", type=str, default="separated_output", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Validate
    if not Path(args.audio).exists():
        print(f"❌ Audio file not found: {args.audio}")
        return
    
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    if args.num_speakers < 2 or args.num_speakers > 5:
        print(f"❌ Invalid num_speakers: {args.num_speakers}. Must be 2-5.")
        return
    
    print(f"\n{'='*70}")
    print(f"DPRNN-TasNet Speaker Separation (DEBUG MODE)")
    print(f"{'='*70}")
    print(f"Audio:         {args.audio}")
    print(f"Checkpoint:    {args.checkpoint}")
    print(f"Num Speakers:  {args.num_speakers}")
    print(f"Device:        {args.device}")
    print(f"Output Dir:    {args.output_dir}")
    print(f"{'='*70}\n")
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load audio and model
    mixture = load_audio(args.audio, max_duration=30.0)  # Allow 30 sec
    model, cfg = load_model(args.checkpoint, args.num_speakers, device=str(device))
    
    # Separate speakers
    estimates = separate_speakers(model, mixture, device=str(device))
    
    # Save output
    save_separated_audio(estimates, args.output_dir)
    
    print(f"\n{'='*70}")
    print(f"✅ Separation complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
