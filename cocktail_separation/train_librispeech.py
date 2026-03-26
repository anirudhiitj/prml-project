#!/usr/bin/env python3
"""
TRAINING WITH REAL LIBRISPEECH HUMAN SPEECH
Train DPRNN-TasNet on actual human-spoken audio for 3-speaker separation
GPU 3 optimized with 7 parallel workers
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['OMP_NUM_THREADS'] = '8'

import sys
sys.path.insert(0, '/mnt/raid/rl_gaming/RL4VLM2/cocktail_separation')

import torch
import time
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

print(f"\n{'='*80}")
print(f"LIBRISPEECH REAL SPEECH TRAINING - GPU 3")
print(f"{'='*80}\n")

# GPU check
device = torch.device("cuda:0")
print(f"🖥️  Device: {device}")
print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB\n")

# Import model
try:
    from src.model import DPRNNTasNet
    from src.losses import pit_loss
    from src.dataset import build_dataloader
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    print("✅ All imports successful\n")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def train_librispeech():
    """Train on REAL LibriSpeech human speech"""
    
    # Configuration
    num_speakers = 3
    num_epochs = 60
    batch_size = 12  # Slightly smaller for compatibility
    num_workers = 7  # 7 PARALLEL WORKERS
    learning_rate = 1e-3
    
    print("Configuration:")
    print(f"  Speakers: {num_speakers}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Workers: {num_workers} (parallel)")
    print(f"  Learning Rate: {learning_rate}\n")
    
    # Data paths
    train_root = Path("data/real_librispeech_mixtures/3spk/train")
    eval_root = Path("data/real_librispeech_mixtures/3spk/val")
    
    if not train_root.exists():
        print(f"⚠️  LibriSpeech data not found at {train_root}")
        print(f"    Waiting for data preparation to complete...")
        print(f"    (Check: python3 generate_librispeech_mixtures.py)\n")
        
        # Wait for data
        import time
        for i in range(120):  # Wait up to 2 minutes
            if train_root.exists():
                break
            time.sleep(5)
        
        if not train_root.exists():
            print(f"❌ Data directory still not found after 2 minutes")
            print(f"   Ensure: python3 generate_librispeech_mixtures.py is running")
            sys.exit(1)
    
    print(f"✅ LibriSpeech data found\n")
    
    # Create model
    print("Creating DPRNN-TasNet model...")
    model = DPRNNTasNet(num_speakers=num_speakers).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}\n")
    
    # Setup optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Create dataloaders with 7 workers (PARALLEL)
    print(f"Creating dataloaders with {num_workers} workers...\n")
    
    train_loader = build_dataloader(
        split_root=train_root,
        num_speakers=num_speakers,
        batch_size=batch_size,
        num_workers=num_workers,  # 7 PARALLEL WORKERS
        distributed=False,
        shuffle=True
    )
    
    eval_loader = build_dataloader(
        split_root=eval_root,
        num_speakers=num_speakers,
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=False,
        shuffle=False
    )
    
    # Training
    best_sisnr = -float('inf')
    best_epoch = 0
    history = []
    
    print(f"Starting training at {datetime.now().strftime('%H:%M:%S')}\n")
    import time as time_module
    start_tm = time_module.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_sisnr = 0
        train_count = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]", leave=False)
        for batch_idx, batch in enumerate(train_bar):
            try:
                if len(batch) != 2:
                    continue
                
                mixture, speakers = batch
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
                
                train_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'SI-SNR': f'{sisnr.item():.2f} dB'})
            
            except Exception as e:
                continue
        
        train_loss = train_loss / max(train_count, 1)
        train_sisnr = train_sisnr / max(train_count, 1)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_sisnr = 0
        val_count = 0
        
        val_bar = tqdm(eval_loader, desc=f"Epoch {epoch+1}/{num_epochs} [VAL]", leave=False)
        with torch.no_grad():
            for batch in val_bar:
                try:
                    if len(batch) != 2:
                        continue
                    
                    mixture, speakers = batch
                    mixture = mixture.to(device)
                    speakers = speakers.to(device)
                    
                    estimates = model(mixture)
                    
                    min_len = min(estimates.shape[-1], speakers.shape[-1])
                    estimates = estimates[..., :min_len]
                    speakers = speakers[..., :min_len]
                    
                    loss, sisnr = pit_loss(estimates, speakers, snr_weight=0.1)
                    
                    val_loss += loss.item() * mixture.shape[0]
                    val_sisnr += sisnr.item() * mixture.shape[0]
                    val_count += mixture.shape[0]
                    
                    val_bar.set_postfix({'SI-SNR': f'{sisnr.item():.2f} dB'})
                
                except Exception as e:
                    continue
        
        val_loss = val_loss / max(val_count, 1)
        val_sisnr = val_sisnr / max(val_count, 1)
        
        # Learning rate scheduling
        scheduler.step(val_sisnr)
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_sisnr': float(train_sisnr),
            'val_loss': float(val_loss),
            'val_sisnr': float(val_sisnr)
        })
        
        # Checkpoint if best
        if val_sisnr > best_sisnr:
            best_sisnr = val_sisnr
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': {'num_speakers': num_speakers},
                'val_sisnr': float(val_sisnr)
            }
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save(checkpoint, 'checkpoints/best_librispeech.pt')
        
        # Print epoch result
        elapsed = time_module.time() - start_tm
        eta_min = (elapsed / (epoch + 1)) * (num_epochs - epoch - 1) / 60
        
        print(f"✅ Epoch {epoch+1:3d} | Train SI-SNR: {train_sisnr:6.2f} dB | Val SI-SNR: {val_sisnr:6.2f} dB | "
              f"Best: {best_sisnr:6.2f} (#{best_epoch+1}) | ETA: {eta_min:.1f}m")
    
    # Complete
    total_time = (time_module.time() - start_tm) / 60
    print(f"\n{'='*80}")
    print(f"✅ LIBRISPEECH TRAINING COMPLETE")
    print(f"   Total time: {total_time:.1f} minutes")
    print(f"   Best SI-SNR: {best_sisnr:.2f} dB (Epoch {best_epoch+1})")
    print(f"   Checkpoint: checkpoints/best_librispeech.pt")
    print(f"   Data: REAL HUMAN SPEECH (LibriSpeech)")
    print(f"{'='*80}\n")
    
    with open('training_history_librispeech.json', 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    train_librispeech()
