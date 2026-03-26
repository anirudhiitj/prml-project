#!/usr/bin/env python3
"""
Fine-tune DPRNN-TasNet on real LibriSpeech data using GPU 3 only
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

# Import model (assuming it exists in codebase)
try:
    from models.dprnn_tasnet import DPRNNTasNet
except:
    print("⚠️  Model import failed, will use placeholder")
    DPRNNTasNet = None

print("\n" + "="*80)
print("FINE-TUNING ON REAL LIBRISPEECH DATA - GPU 3")
print("="*80 + "\n")

# Configuration
CONFIG = {
    'input_dim': 512,
    'num_speakers': 3,
    'batch_size': 8,
    'epochs': 20,
    'lr': 1e-4,
    'device': 'cuda:0',  # GPU 3 is visible as cuda:0 due to CUDA_VISIBLE_DEVICES
    'data_dir': Path('data/real_librispeech_mixtures/3spk'),
}

# Dataset
class LibriSpeechMixtureDataset(Dataset):
    def __init__(self, split='train', data_dir=None):
        self.split_dir = data_dir / split
        self.samples = sorted(list(self.split_dir.glob('*/mixture.wav')))
        self.sr = 16000
        print(f"   Loaded {len(self.samples)} {split} samples")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        mix_dir = self.samples[idx].parent
        
        # Load mixture and sources
        mixture, _ = sf.read(str(mix_dir / 'mixture.wav'))
        s1, _ = sf.read(str(mix_dir / 's1.wav'))
        s2, _ = sf.read(str(mix_dir / 's2.wav'))
        s3, _ = sf.read(str(mix_dir / 's3.wav'))
        
        # Convert to tensors
        mixture = torch.FloatTensor(mixture)
        sources = torch.stack([
            torch.FloatTensor(s1),
            torch.FloatTensor(s2),
            torch.FloatTensor(s3)
        ])
        
        return mixture, sources

# SI-SNR Loss
def sisnr(pred, target):
    """Scale-Invariant Signal-to-Noise Ratio"""
    eps = 1e-8
    
    # Zero mean
    pred = pred - torch.mean(pred, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)
    
    # Scaling factor
    s = torch.sum(pred * target, dim=-1) / (torch.sum(target ** 2, dim=-1) + eps)
    
    # SI-SNR
    e_target = s.unsqueeze(-1) * target
    e_noise = pred - e_target
    
    sisnr_value = 10 * torch.log10(torch.sum(e_target ** 2, dim=-1) / (torch.sum(e_noise ** 2, dim=-1) + eps) + eps)
    
    return -torch.mean(sisnr_value)  # Negative for minimization

# Training
def train():
    device = CONFIG['device']
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = LibriSpeechMixtureDataset('train', CONFIG['data_dir'])
    val_dataset = LibriSpeechMixtureDataset('val', CONFIG['data_dir'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                            shuffle=False, num_workers=2)
    
    print(f"✅ Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples\n")
    
    # Model (using simple Conv1D if DPRNNTasNet not available)
    if DPRNNTasNet is not None:
        model = DPRNNTasNet(input_dim=CONFIG['input_dim'], 
                           num_speakers=CONFIG['num_speakers'])
    else:
        print("⚠️  Using simple model (DPRNNTasNet not found)\n")
        model = SimpleNet()
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    history = {'train_loss': [], 'train_sisnr': [], 'val_loss': [], 'val_sisnr': []}
    
    # Training loop
    for epoch in range(CONFIG['epochs']):
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        # Train
        model.train()
        train_loss_all = []
        train_sisnr_all = []
        
        with tqdm(train_loader, desc="Train") as pbar:
            for mixture, sources in pbar:
                mixture = mixture.to(device)
                sources = sources.to(device)
                
                # Forward (simple separation)
                output = model(mixture)  # [B, 3, T]
                
                # Loss
                loss = sisnr(output, sources)
                sisnr_val = -loss.item()
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss_all.append(loss.item())
                train_sisnr_all.append(sisnr_val)
                pbar.set_postfix({'loss': np.mean(train_loss_all[-10:])})
        
        # Validate
        model.eval()
        val_loss_all = []
        val_sisnr_all = []
        
        with torch.no_grad():
            for mixture, sources in tqdm(val_loader, desc="Val"):
                mixture = mixture.to(device)
                sources = sources.to(device)
                
                output = model(mixture)
                loss = sisnr(output, sources)
                
                val_loss_all.append(loss.item())
                val_sisnr_all.append(-loss.item())
        
        # Log
        train_loss_mean = np.mean(train_loss_all)
        train_sisnr_mean = np.mean(train_sisnr_all)
        val_loss_mean = np.mean(val_loss_all)
        val_sisnr_mean = np.mean(val_sisnr_all)
        
        history['train_loss'].append(train_loss_mean)
        history['train_sisnr'].append(train_sisnr_mean)
        history['val_loss'].append(val_loss_mean)
        history['val_sisnr'].append(val_sisnr_mean)
        
        print(f"  Train Loss: {train_loss_mean:.4f} | SI-SNR: {train_sisnr_mean:.2f}")
        print(f"  Val Loss: {val_loss_mean:.4f} | SI-SNR: {val_sisnr_mean:.2f}\n")
        
        # Save best
        if val_sisnr_mean == max(history['val_sisnr']):
            torch.save(model.state_dict(), checkpoint_dir / 'best_librispeech_real.pt')
    
    # Save history
    with open(checkpoint_dir / 'training_history_real.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("="*80)
    print("✅ TRAINING COMPLETE")
    print(f"   Best checkpoint: checkpoints/best_librispeech_real.pt")
    print(f"   Best SI-SNR: {max(history['val_sisnr']):.2f}")
    print("="*80)

class SimpleNet(nn.Module):
    """Simple fallback model if DPRNNTasNet unavailable"""
    def __init__(self, num_speakers=3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 40, stride=20, padding=20)
        self.conv2 = nn.Conv1d(64, 32, 1)
        self.deconv = nn.ConvTranspose1d(32, num_speakers, 40, stride=20, padding=20)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.deconv(x)
        return x

if __name__ == '__main__':
    train()
