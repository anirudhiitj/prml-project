#!/usr/bin/env python3
"""
Re-initialize Phase 2 checkpoint as the primary checkpoint
This was the best-performing model during training
"""

import torch
import json
from pathlib import Path

# Read Phase 2 status to get metrics
status_file = Path("training_results/phase_2_3spk/training_status_phase_2_3spk.json")

with open(status_file, 'r') as f:
    status = json.load(f)

print(f"Phase 2 (3-speaker) metrics:")
print(f"  Best SI-SNR: {status['best_val_sisnr']:.2f} dB")
print(f"  Final epoch: {status['current_epoch']}")

# The issue: we don't have the Phase 2 checkpoint saved!
# We only have the final Phase 4 checkpoint which is worse

print(f"\n❌ Problem: Phase 2 checkpoint was not saved separately")
print(f"   We need to retrain Phase 2 in isolation to preserve the best model")
