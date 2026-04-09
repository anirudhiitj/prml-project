#!/usr/bin/env python3
"""
Monitor GPU 3 training progress in real-time
"""

import subprocess
import json
from pathlib import Path
import time

def check_training():
    """Check GPU 3 training status"""
    
    print("\n" + "="*80)
    print("GPU 3 TRAINING STATUS")
    print("="*80 + "\n")
    
    # Check GPU status
    print("GPU Status:")
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', 
         '--format=csv,noheader'],
        capture_output=True, text=True
    )
    
    for line in result.stdout.strip().split('\n'):
        if 'NVIDIA' in line or '3,' in line:
            print(f"  {line}")
    
    print()
    
    # Check training history
    history_file = Path("training_history_gpu3.json")
    if history_file.exists():
        try:
            with open(history_file) as f:
                history = json.load(f)
            
            print(f"Training Progress ({len(history)} epochs completed):")
            for entry in history[-5:]:  # Last 5 epochs
                print(f"  Epoch {entry['epoch']+1:3d} | Loss: {entry['train_loss']:7.4f} | Val SI-SNR: {entry['val_sisnr']:6.2f} dB")
            
            if history:
                best = max(history, key=lambda x: x['val_sisnr'])
                print(f"\n  Best: Epoch {best['epoch']+1} with SI-SNR {best['val_sisnr']:.2f} dB\n")
        except:
            pass
    
    # Check processes
    print("Running Processes:")
    result = subprocess.run(
        ['pgrep', '-fa', 'train_gpu3_fast'],
        capture_output=True, text=True
    )
    
    if result.stdout.strip():
        print(f"  {result.stdout.strip()[:100]}")
        print("  ✅ Training is running...\n")
    else:
        print("  ⚠️  No training process found\n")
    
    # Check checkpoint
    checkpoint = Path("checkpoints/best_gpu3.pt")
    if checkpoint.exists():
        size_mb = checkpoint.stat().st_size / 1e6
        print(f"Checkpoint: {checkpoint.name} ({size_mb:.1f}MB)\n")

if __name__ == "__main__":
    check_training()
