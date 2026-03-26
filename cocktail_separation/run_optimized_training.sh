#!/bin/bash
# MASTER SETUP SCRIPT - GPU 3 TRAINING OPTIMIZATION
# Kills old processes, cleans corrupted data, starts fresh training

set -e

echo ""
echo "========================================================================"
echo "GPU 3 OPTIMIZED TRAINING - MASTER SETUP & START"
echo "========================================================================"
echo ""

# Step 1: Kill old processes
echo "🔪 STEP 1: Killing stray processes..."
echo ""

# Find and kill old training on GPU 5 (process 1897067)
echo "Looking for stray processes..."
python3 << 'EOF'
import subprocess
import re

result = subprocess.run(['nvidia-smi', '--query-processes=pid,process_name,gpu_memory_usage', '--format=csv,noheader'], 
                       capture_output=True, text=True)

processes_to_kill = []
for line in result.stdout.strip().split('\n'):
    if line.strip():
        try:
            parts = line.split(',')
            pid = parts[0].strip()
            name = parts[1].strip() if len(parts) > 1 else ""
            memory = parts[2].strip() if len(parts) > 2 else ""
            
            # Kill our old training processes (look for large memory usage on any GPU)
            if 'python' in name.lower():
                try:
                    mem_mb = float(memory.replace('MiB', '').strip())
                    if mem_mb > 5000:  # Large memory = likely our training
                        processes_to_kill.append(pid)
                        print(f"Found process to kill: PID={pid}, Memory={memory}")
                except:
                    pass
        except:
            pass

if processes_to_kill:
    print(f"\nWill kill {len(processes_to_kill)} processes")
    for pid in processes_to_kill:
        try:
            import os
            os.kill(int(pid), 9)
            print(f"✅ Killed PID {pid}")
        except Exception as e:
            print(f"⚠️  Could not kill {pid}: {e}")
else:
    print("✅ No stray training processes found")

import time
time.sleep(2)
EOF

echo ""
echo "========================================================================"
echo "🗑️  STEP 2: Cleaning corrupted data and checkpoints..."
echo ""

# Remove old corrupted checkpoints
if [ -f "checkpoints/best_phase2.pt" ]; then
    echo "Removing corrupted checkpoint: checkpoints/best_phase2.pt"
    rm -f checkpoints/best_phase2.pt
fi

if [ -f "checkpoints/phase2_final.pt" ]; then
    echo "Removing old checkpoint: checkpoints/phase2_final.pt"
    rm -f checkpoints/phase2_final.pt
fi

# Remove corrupted synthetic data
if [ -d "data/mixtures" ]; then
    echo "Removing corrupted synthetic data: data/mixtures/"
    rm -rf data/mixtures
    mkdir -p data/mixtures
fi

echo "✅ Cleanup complete"
echo ""

# Step 3: Prepare environment
echo "========================================================================"
echo "⚙️  STEP 3: Environment setup..."
echo ""

export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "GPU configuration: CUDA_VISIBLE_DEVICES=3"
echo "OpenMP threads: 8"
echo ""

# Verify GPU
echo "GPU Status (nvidia-smi):"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,pstate --format=csv,noheader | grep -E "^3|GPU"
echo ""

# Step 4: Generate data
echo "========================================================================"
echo "📊 STEP 4: Generating training data from TIMIT..."
echo ""

python3 data_preparation_timit.py

if [ ! -d "data/real_mixtures" ]; then
    echo "⚠️  Data generation failed!"
    echo "Using existing synthetic data fallback..."
fi

echo ""

# Step 5: Start optimized training
echo "========================================================================"
echo "🚀 STEP 5: Starting optimized training on GPU 3..."
echo ""
echo "Configuration:"
echo "  - GPU: 3 (exclusive)"
echo "  - Workers: 7 parallel"  
echo "  - Batch size: 16"
echo "  - Target time: 2-3 hours"
echo ""

python3 train_optimized_v3.py

echo ""
echo "========================================================================"
echo "✅ TRAINING COMPLETE!"
echo "========================================================================"
echo ""
echo "Results:"
echo "  - Best model: checkpoints/best_optimized.pt"
echo "  - Training history: training_results_optimized.json"
echo ""
echo "Next: Run inference with:"
echo "  python3 inference.py --audio my_audio.mp3 --checkpoint checkpoints/best_optimized.pt --num-speakers 3"
echo ""
