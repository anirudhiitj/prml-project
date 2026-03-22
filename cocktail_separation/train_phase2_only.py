#!/usr/bin/env python3
"""
Retrain ONLY Phase 2 (3-speaker) - The best performing model
This removes the curriculum learning that degraded performance with 4-5 speakers
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import json
from datetime import datetime
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import build_dataloader
from src.model import DPRNNTasNet
from src.losses import pit_loss

def train_phase_2():
    """Train ONLY Phase 2 (3-speaker separation)"""
    
    print("\n" + "="*70)
    print("PHASE 2 (3-Speaker Separation) - ISOLATED TRAINING")
    print("="*70 + "\n")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_speakers = 3
    num_epochs = 100
    batch_size = 8
    learning_rate = 1e-3
    
    print(f"Configuration:")
    print(f"  Speakers: {num_speakers}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}\n")
    
    # Create dataset loaders
    print("Loading training data...")
    train_root = Path("data/mixtures") / f"{num_speakers}spk" / "train"
    eval_root = Path("data/mixtures") / f"{num_speakers}spk" / "val"
    
    if not train_root.exists() or not eval_root.exists():
        print(f"❌ Dataset path not found. Looking for:")
        print(f"   Train: {train_root}")
        print(f"   Eval: {eval_root}")
        return
    
    train_loader = build_dataloader(
        split_root=train_root,
        num_speakers=num_speakers,
        batch_size=batch_size,
        num_workers=0,
        distributed=False,
        shuffle=True
    )
    
    eval_loader = build_dataloader(
        split_root=eval_root,
        num_speakers=num_speakers,
        batch_size=batch_size,
        num_workers=0,
        distributed=False,
        shuffle=False
    )
    
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Eval samples: {len(eval_loader.dataset)}\n")
    
    # Create model
    print("Creating model...")
    model = DPRNNTasNet(
        num_speakers=num_speakers,
        encoder_dim=64,
        encoder_kernel=2,
        encoder_stride=2,
        bottleneck_dim=64,
        chunk_size=100,
        num_dprnn_blocks=6
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Training loop
    best_sisnr = -float('inf')
    best_epoch = 0
    history = {
        'epoch': [],
        'train_loss': [],
        'train_sisnr': [],
        'val_loss': [],
        'val_sisnr': []
    }
    
    print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0
        train_sisnr = 0
        train_count = 0
        
        for batch_idx, (mixture, speakers) in enumerate(train_loader):
            mixture = mixture.to(device)
            speakers = speakers.to(device)
            
            optimizer.zero_grad()
            estimates = model(mixture)
            
            # Handle length mismatches
            min_len = min(estimates.shape[-1], speakers.shape[-1])
            estimates = estimates[..., :min_len]
            speakers = speakers[..., :min_len]
            
            loss, sisnr = pit_loss(estimates, speakers, snr_weight=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss += loss.item() * mixture.shape[0]
            train_sisnr += sisnr.item() * mixture.shape[0]
            train_count += mixture.shape[0]
            
            # Progress
            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | SI-SNR: {sisnr.item():.2f} dB", end='\r')
        
        train_loss /= train_count
        train_sisnr /= train_count
        
        # Validation
        model.eval()
        val_loss = 0
        val_sisnr = 0
        val_count = 0
        
        with torch.no_grad():
            for mixture, speakers in eval_loader:
                mixture = mixture.to(device)
                speakers = speakers.to(device)
                
                estimates = model(mixture)
                
                # Handle length mismatches
                min_len = min(estimates.shape[-1], speakers.shape[-1])
                estimates = estimates[..., :min_len]
                speakers = speakers[..., :min_len]
                
                loss, sisnr = pit_loss(estimates, speakers, snr_weight=0.1)
                
                val_loss += loss.item() * mixture.shape[0]
                val_sisnr += sisnr.item() * mixture.shape[0]
                val_count += mixture.shape[0]
        
        val_loss /= val_count
        val_sisnr /= val_count
        
        # Update scheduler
        scheduler.step(val_sisnr)
        
        # Save history
        history['epoch'].append(epoch)
        history['train_loss'].append(float(train_loss))
        history['train_sisnr'].append(float(train_sisnr))
        history['val_loss'].append(float(val_loss))
        history['val_sisnr'].append(float(val_sisnr))
        
        # Best model checkpoint
        if val_sisnr > best_sisnr:
            best_sisnr = val_sisnr
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': {
                    'model': {
                        'num_speakers': num_speakers,
                        'encoder_dim': 64,
                        'encoder_kernel': 2,
                        'encoder_stride': 2,
                        'bottleneck_dim': 64,
                        'chunk_size': 100,
                        'num_dprnn_blocks': 6
                    }
                },
                'val_sisnr': float(val_sisnr),
                'train_sisnr': float(train_sisnr)
            }
            torch.save(checkpoint, 'checkpoints/best_phase2.pt')
        
        elapsed = time.time() - epoch_start
        print(f"✅ Epoch {epoch+1:3d} | Train Loss: {train_loss:7.4f} | Train SI-SNR: {train_sisnr:6.2f} dB | "
              f"Val SI-SNR: {val_sisnr:6.2f} dB | Best: {best_sisnr:6.2f} dB (epoch {best_epoch+1}) | Time: {elapsed:.1f}s")
    
    # Save final checkpoint
    torch.save(checkpoint, 'checkpoints/phase2_final.pt')
    
    # Save status
    status = {
        'phase': 2,
        'num_speakers': 3,
        'start_time': datetime.now().isoformat(),
        'status': 'COMPLETED',
        'best_val_sisnr': float(best_sisnr),
        'best_sisnr_epoch': best_epoch,
        'training_history': history
    }
    
    with open('training_results/phase_2_3spk/phase2_isolated_status.json', 'w') as f:
        json.dump(status, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE")
    print(f"   Best SI-SNR: {best_sisnr:.2f} dB (epoch {best_epoch+1})")
    print(f"   Checkpoints saved:")
    print(f"      - checkpoints/best_phase2.pt")
    print(f"      - checkpoints/phase2_final.pt")
    print(f"   Status: training_results/phase_2_3spk/phase2_isolated_status.json")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    train_phase_2()
