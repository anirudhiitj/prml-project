#!/usr/bin/env python3
"""
Separate speakers in audio file using trained model
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import argparse
from pathlib import Path
import soundfile as sf
import numpy as np
from tqdm import tqdm
import librosa

print("\n" + "="*80)
print("INFERENCE - SPEAKER SEPARATION")
print("="*80 + "\n")

def separate_audio(input_file, checkpoint_path, output_dir, num_speakers=3):
    """Separate speakers from audio file"""
    
    device = 'cuda:0'
    sr = 16000
    
    # Load audio
    print(f"Loading: {input_file}")
    audio, file_sr = librosa.load(str(input_file), sr=sr, mono=True)
    print(f"✅ Loaded: {len(audio)/sr:.1f}s @ {sr}Hz\n")
    
    # Try to load model
    try:
        from models.dprnn_tasnet import DPRNNTasNet
        model = DPRNNTasNet(input_dim=512, num_speakers=num_speakers)
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    except:
        print("⚠️  Using simple model fallback\n")
        class SimpleNet(torch.nn.Module):
            def __init__(self, num_speakers=3):
                super().__init__()
                self.conv1 = torch.nn.Conv1d(1, 64, 40, stride=20, padding=20)
                self.conv2 = torch.nn.Conv1d(64, 32, 1)
                self.deconv = torch.nn.ConvTranspose1d(32, num_speakers, 40, stride=20, padding=20)
            
            def forward(self, x):
                x = x.unsqueeze(1)
                x = self.conv1(x)
                x = torch.relu(x)
                x = self.conv2(x)
                x = torch.relu(x)
                x = self.deconv(x)
                return x
        
        model = SimpleNet(num_speakers)
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        except:
            print("⚠️  No checkpoint found, using untrained model\n")
    
    model = model.to(device)
    model.eval()
    
    # Separate (process in chunks to avoid memory issues)
    chunk_size = int(5 * sr)  # 5-second chunks
    hop_size = int(4 * sr)    # 4-second hop
    
    separated = [np.zeros_like(audio) for _ in range(num_speakers)]
    
    print(f"Processing {len(audio)/sr:.1f}s audio in {chunk_size/sr:.1f}s chunks...\n")
    
    with torch.no_grad():
        for start in tqdm(range(0, len(audio) - chunk_size, hop_size)):
            end = start + chunk_size
            chunk = torch.FloatTensor(audio[start:end]).unsqueeze(0).to(device)
            
            output = model(chunk)  # [1, num_speakers, T]
            output = output.squeeze(0).cpu().numpy()
            
            # Accumulate with windowing to avoid clicks
            window = np.hanning(chunk_size)
            window = window / (window.sum() / 2)  # Normalize
            
            for spk in range(num_speakers):
                if separated[spk][start:end].sum() == 0:
                    separated[spk][start:end] = output[spk] * window
                else:
                    # Overlap-add
                    separated[spk][start:end] = (separated[spk][start:end] + output[spk] * window) / 2
    
    # Normalize and save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving separated speakers to: {output_dir}\n")
    
    for spk in range(num_speakers):
        output = separated[spk]
        output = output / (np.abs(output).max() + 1e-8) * 0.95  # Normalize
        
        output_file = output_dir / f"speaker_{spk+1}.wav"
        sf.write(str(output_file), output, sr)
        print(f"✅ {output_file}")
    
    # Save mixture for reference
    ref_file = output_dir / "original_mixture.wav"
    sf.write(str(ref_file), audio, sr)
    print(f"✅ {ref_file}")
    
    print("\n" + "="*80)
    print("✅ SEPARATION COMPLETE")
    print("="*80 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input audio file')
    parser.add_argument('--checkpoint', default='checkpoints/best_librispeech_real.pt')
    parser.add_argument('--output', default='outputs/separated')
    parser.add_argument('--num_speakers', type=int, default=3)
    
    args = parser.parse_args()
    
    separate_audio(args.input, args.checkpoint, args.output, args.num_speakers)
