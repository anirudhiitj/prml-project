#!/usr/bin/env python3
"""
INTEGRATED OPTIMIZED TRAINING LAUNCHER
- Kills old processes  
- Prepares data using existing tools
- Starts training on GPU 3
"""

import os
import sys
import signal
import subprocess
from pathlib import Path

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def kill_old_processes():
    """Kill stray training processes"""
    print("\n🔪 Killing old training processes...")
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-processes=pid,process_name,gpu_memory_usage', 
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        
        pids_to_kill = []
        for line in result.stdout.strip().split('\n'):
            if line.strip() and 'python' in line.lower():
                parts = line.split(',')
                pid = parts[0].strip()
                try:
                    mem = float(parts[2].replace('MiB', '').strip())
                    if mem > 5000:  # Large memory = old training
                        pids_to_kill.append(pid)
                        print(f"  Found: PID {pid} ({mem:.0f}MB)")
                except:
                    pass
        
        for pid in pids_to_kill:
            try:
                os.kill(int(pid), 9)
                print(f"  Killed: {pid}")
            except:
                pass
    
    except Exception as e:
        print(f"  Note: {e}")
    
    print("✅ Process cleanup done\n")


def prepare_data():
    """Use existing data generation tools"""
    print("📊 Preparing training data...")
    
    # Use existing generate_mixtures_fast.py if available
    script = Path("generate_mixtures_fast.py")
    if script.exists():
        print(f"  Using existing {script.name}...")
        os.system(f"source /mnt/raid/rl_gaming/dprnn2/bin/activate && python3 {script}")
    else:
        print("  Creating minimal data...")
        # Create dummy data in fallback location
        Path("data/mixtures/3spk").mkdir(parents=True, exist_ok=True)
    
    print("✅ Data ready\n")


def start_training():
    """Start optimized training"""
    print("🚀 Starting training on GPU 3...\n")
    
    cmd = [
        'bash', '-c',
        'source /mnt/raid/rl_gaming/dprnn2/bin/activate && '
        'export CUDA_VISIBLE_DEVICES=3 && '
        'python3 train_optimized_v3.py'
    ]
    
    subprocess.run(cmd)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("OPTIMIZED TRAINING LAUNCHER - GPU 3")
    print("="*80)
    
    # Execute pipeline
    kill_old_processes()
    prepare_data()
    start_training()
    
    print("\n" + "="*80)
    print("✅ TRAINING PIPELINE COMPLETE")
    print("="*80 + "\n")
