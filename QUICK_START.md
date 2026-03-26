# DPRNN-TasNet Training Quick Start Guide
## GPU 5 & 6 - Speaker Separation from Your Audio

### Environment Setup ✓
```bash
source ~/dprnn2/bin/activate
cd /mnt/raid/rl_gaming/RL4VLM2/cocktail_separation
```

### Step 1: Generate Training Data (5-10 minutes for test mode)
```bash
# Generate synthetic test data first (fast - 10-30 min)
python generate_mixtures_fast.py --output-dir data/mixtures --test-mode

# Or generate full dataset (requires time, can run in background)
python generate_mixtures_fast.py --output-dir data/mixtures
```

### Step 2: Start Training (Full Curriculum: 2→3→4→5 speakers)
```bash
# Run full training pipeline with progress tracking
python train_enhanced.py --full-curriculum

# Or specific phase:
# python train_enhanced.py --phase 1 --config configs/2spk.yaml
# python train_enhanced.py --phase 2 --config configs/3spk.yaml
# python train_enhanced.py --phase 3 --config configs/4spk.yaml
# python train_enhanced.py --phase 4 --config configs/5spk.yaml
```

### Step 3: Monitor Training Progress (in another terminal)
```bash
# Watch training in real-time
watch -n 1 'cat training_results/phase_*/training_status_*.json | python -m json.tool'

# Or use the provided monitoring script:
python monitor_training.py --watch
```

### Step 4: Test Your Model on Your Audio
```bash
python separate.py \
    --input /path/to/your/audio.wav \
    --output separated_speakers/ \
    --checkpoint training_results/phase_4_5spk/best.pt
```

---

## Expected Training Times (with GPUs 5 & 6)

### Phase 1 (2-speaker)
- **Duration**: ~2-4 hours
- **Expected SI-SNR**: >15 dB
- **Batch Memory**: ~2-3 GB per GPU

### Phase 2 (3-speaker)
- **Duration**: ~2-3 hours
- **Expected SI-SNR**: >12 dB
- **Batch Memory**: ~3-4 GB per GPU

### Phase 3 (4-speaker)
- **Duration**: ~1.5-2 hours
- **Expected SI-SNR**: >10 dB
- **Batch Memory**: ~4-5 GB per GPU

### Phase 4 (5-speaker) - FINAL TARGET
- **Duration**: ~3-5 hours
- **Expected SI-SNR**: >10 dB
- **Batch Memory**: ~5-6 GB per GPU

**Total Curriculum**: ~8-14 hours

---

## File Structure
```
cocktail_separation/
├── data/
│   └── mixtures/
│       ├── 2spk/  (train/val/test)
│       ├── 3spk/  (train/val/test)
│       ├── 4spk/  (train/val/test)
│       └── 5spk/  (train/val/test)
├── training_results/
│   ├── phase_1_2spk/
│   │   ├── best.pt (best model)
│   │   ├── latest.pt (checkpoint)
│   │   ├── epoch_*.pt (periodic saves)
│   │   └── training_status_phase_1_2spk.json (live metrics)
│   ├── phase_2_3spk/
│   ├── phase_3_4spk/
│   └── phase_4_5spk/
└── separated_speakers/
    ├── speaker_1.wav
    ├── speaker_2.wav
    ...
    └── speaker_5.wav
```

---

## Real-Time Monitoring

The training script automatically saves progress:
```json
{
  "current_epoch": 12,
  "best_val_sisnr": 14.2,
  "eta_hours": 1.5,
  "training_history": {
    "epoch": [0, 1, 2, ...],
    "train_loss": [1.23, 1.15, ...],
    "val_sisnr": [8.5, 10.2, ...]
  }
}
```

Access this file:
```bash
tail -f training_results/phase_1_2spk/training_status_phase_1_2spk.json | python -m json.tool
```

---

## Troubleshooting

### OOM (Out of Memory)
- Reduce `train_batch_size` in `configs/base.yaml` from 8 to 4
- Or reduce `chunk_size` from 100 to 50

### Slow Training
- Check GPU utilization: `nvidia-smi -l 1`
- Verify GPUs 5 & 6 are being used
- Increase `num_workers` in config if CPU bottleneck

### Model Not Improving
- Ensure data generation is correct (check mix amplitude)
- Verify curriculum order (don't skip phases)
- Check SI-SNR loss implementation

---

## Next Steps After Training

1. **Evaluate on Test Set**:
```bash
python evaluate.py \
    --checkpoint training_results/phase_4_5spk/best.pt \
    --test-dir data/mixtures/5spk/test
```

2. **Separate Your Audio**:
```bash
python separate.py \
    --input your_audio.wav \
    --output separated/ \
    --checkpoint training_results/phase_4_5spk/best.pt
```

3. **Fine-tune on Your Data** (optional):
- Collect samples of you and friends speaking
- Create custom dataset
- Fine-tune final model with low learning rate

---

## Key Configuration Files

- **configs/base.yaml**: Shared settings (model, data, training)
- **configs/2spk.yaml**: Phase 1 overrides (2-speaker specific)
- **configs/3spk.yaml**: Phase 2 overrides (3-speaker specific)
- **configs/4spk.yaml**: Phase 3 overrides (4-speaker specific)
- **configs/5spk.yaml**: Phase 4 overrides (5-speaker specific, final)

---

## GPU Configuration

Current setup:
- **GPU 5**: 143 GB available (used for training)
- **GPU 6**: 140 GB available (used for training)
- Both GPUs used in distributed training

To use only one GPU:
```bash
CUDA_VISIBLE_DEVICES=5 python train.py --config configs/5spk.yaml
```

---

## References

- **DPRNN Paper**: Dual-Path RNN for efficient audio source separation
- **SI-SNR Loss**: Scale-Invariant Signal-to-Noise Ratio
- **PIT**: Permutation Invariant Training (Hungarian algorithm for optimal speaker assignment)
- **Curriculum Learning**: Progressive training 2→3→4→5 speakers for better convergence

---

**Status**: Ready to train! 🚀

Next: `python generate_mixtures_fast.py --output-dir data/mixtures --test-mode`
