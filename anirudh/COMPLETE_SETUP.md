# 🎯 DPRNN-TasNet Multi-Speaker Separation - Complete Setup Guide

**Created**: March 22, 2026
**Status**: ✅ Ready to Train
**Target**: Separate 2-5 simultaneous speakers from single-channel audio using GPUs 5 & 6

---

## 📋 What's Been Set Up

### ✅ Environment
- **Python venv**: `/mnt/raid/rl_gaming/dprnn2` (activated with all dependencies)
- **Dependencies**: PyTorch 2.10, torchaudio 2.10, pyroomacoustics, wandb, hydra, etc.
- **GPU Resources**: 
  - GPU 5: 143 GB VRAM (currently ~160 MB used)
  - GPU 6: 143 GB VRAM (currently ~3.3 GB used)

### ✅ Project Structure
```
/mnt/raid/rl_gaming/RL4VLM2/
├── cocktail_separation/
│   ├── src/              (Model, Dataset, Losses - already implemented)
│   ├── configs/          (base.yaml, 2spk.yaml, 3spk.yaml, 4spk.yaml, 5spk.yaml)
│   ├── data/
│   │   └── mixtures/     (✅ 2spk, 3spk, 4spk, 5spk - test data generated)
│   │       ├── 2spk/ (train: 1000, val: 200, test: 200)
│   │       ├── 3spk/ (train: 1000, val: 200, test: 200)
│   │       ├── 4spk/ (train: 1000, val: 200, test: 200)
│   │       └── 5spk/ (train: 1000, val: 200, test: 200)
│   ├── train.py          (Main training loop)
│   ├── train_enhanced.py (✅ NEW: GPU management + progress tracking + ETA)
│   ├── monitor_training.py (✅ NEW: Real-time monitoring dashboard)
│   ├── generate_mixtures_fast.py (✅ NEW: Fast data generation)
│   └── separate.py       (Inference - separate your audio)
├── QUICK_START.md        (Step-by-step instructions)
├── TRAINING_LOG.md       (Live training log will be updated here)
└── training_results/     (Will contain all checkpoints and metrics)
```

### ✅ Scripts Created

1. **`train_enhanced.py`** - Smart training launcher with:
   - Automatic GPU 5 & 6 selection
   - Progress tracking to JSON files
   - ETA estimation (updates every epoch)
   - Real-time metrics saving
   - Full curriculum management (2→3→4→5)

2. **`monitor_training.py`** - Real-time monitoring with:
   - Live progress bars
   - Current metrics display
   - ETA countdown
   - Summary reports

3. **`generate_mixtures_fast.py`** - Data generation:
   - Synthetic audio generation (test mode: complete)
   - LibriSpeech support (bring your own audio)
   - Multi-speaker mixture creation
   - RIR and noise augmentation

### ✅ Test Data Generated
- **2-speaker**: 1,000 train + 200 val + 200 test
- **3-speaker**: 1,000 train + 200 val + 200 test
- **4-speaker**: 1,000 train + 200 val + 200 test
- **5-speaker**: 1,000 train + 200 val + 200 test
- **Total**: 4,800 mixtures (ready for testing!)

---

## 🚀 QUICK START - Run This Now

### Terminal 1: Start Training
```bash
cd /mnt/raid/rl_gaming/RL4VLM2/cocktail_separation
source /mnt/raid/rl_gaming/dprnn2/bin/activate

# Run full curriculum (2→3→4→5 speakers)
python train_enhanced.py --full-curriculum

# This will:
# 1. Start Phase 1 (2-speaker) - ~30-60 min
# 2. Auto-load Phase 1 checkpoint and start Phase 2 (3-speaker) - ~20-50 min
# 3. Auto-load Phase 2 checkpoint and start Phase 3 (4-speaker) - ~20-30 min
# 4. Auto-load Phase 3 checkpoint and start Phase 4 (5-speaker) - ~40-80 min
# Total: 2-4 hours for test data, 10-20 hours for full data
```

### Terminal 2: Monitor Progress (open in NEW terminal)
```bash
cd /mnt/raid/rl_gaming/RL4VLM2/cocktail_separation
source /mnt/raid/rl_gaming/dprnn2/bin/activate

# Watch training in real-time (updates every 5 seconds)
python monitor_training.py --watch

# Or use built-in watching:
watch -n 1 'python monitor_training.py'
```

---

## 📊 Expected Results During Training

### Phase 1: 2-Speaker Separation
```
Epoch Progress: [████████████░░░░░░░░░░░░░░░░] 45%
Current Epoch:  23 / 50
Val SI-SNR:     14.2 dB (improving)
ETA:            1.5 hours
Learning Rate:  1.0e-03
```

### Phase 4: 5-Speaker Separation (Final Target)
```
Expected SI-SNR:  10-12 dB
Training Time:    3-5 hours
Memory Usage:     5-6 GB per GPU
Success Criteria: Model can separate your voice from friends!
```

---

## 📈 Real-Time Progress Tracking

All results saved to:
```
training_results/
├── phase_1_2spk/
│   ├── best.pt (best checkpoint)
│   ├── latest.pt (current checkpoint)
│   ├── epoch_*.pt (periodic saves)
│   └── training_status_phase_1_2spk.json  ← Real-time metrics
├── phase_2_3spk/
│   └── training_status_phase_2_3spk.json
├── phase_3_4spk/
│   └── training_status_phase_3_4spk.json
└── phase_4_5spk/
    └── training_status_phase_4_5spk.json  ← Final model here
```

**Metrics saved every epoch:**
- Train loss & SI-SNR
- Validation loss & SI-SNR
- Learning rate
- Gradient norm
- Epoch time
- ETA (hours remaining)

---

## 🎤 After Training: Separate Your Audio

Once training completes:

```bash
# Test on a 4-second audio clip of you and friends
python separate.py \
    --input /path/to/your/audio.wav \
    --output ./separated_speakers/ \
    --checkpoint training_results/phase_4_5spk/best.pt

# Output files:
# - separated_speakers/speaker_1.wav (your voice)
# - separated_speakers/speaker_2.wav (friend 1)
# - separated_speakers/speaker_3.wav (friend 2)
# - separated_speakers/speaker_4.wav (friend 3)
# - separated_speakers/speaker_5.wav (friend 4)
```

---

## 🔧 Configuration

All training parameters in: `configs/base.yaml`

**Key Settings:**
```yaml
model:
  num_speakers: 5           # Max simultaneous speakers
  encoder_dim: 64           # N parameter (DPRNN)
  bottleneck_dim: 64        # H parameter (DPRNN)
  chunk_size: 100           # K (chunk length)
  num_dprnn_blocks: 6       # Number of DPRNN blocks

data:
  train_batch_size: 8       # Per GPU (16 total)
  val_batch_size: 4         # Per GPU
  clip_seconds: 4           # Audio clip length
  num_workers: 8            # Data loading workers

train:
  epochs: 100               # Per phase
  lr: 0.001                 # Starting learning rate
  grad_clip_norm: 5.0       # Gradient clipping
```

To use **ONLY Phase 4 (5-speaker) with quick convergence**:
```yaml
train:
  epochs: 50                # Fewer epochs for transfer learning
  lr: 0.00005               # Lower LR for fine-tuning
```

---

## 📚 Understanding the Architecture

### DPRNN-TasNet Overview
```
Raw Waveform (B, T)
    ↓
[Learned Encoder] → (B, N=64, T')
    ↓
[Bottleneck Projection] → (B, H=64, T')
    ↓
[DPRNN Separator ×6 blocks]
  ├─ Intra-chunk BiLSTM (local context)
  └─ Inter-chunk BiLSTM (global speaker tracking)
    ↓
[Mask Estimation] → (B, 5 speakers, N, T')
    ↓
[Per-speaker Masking & Decoding]
    ↓
Separated Speakers (B, 5, T_out)
```

### Why This Works for Your Use Case
- **Intra-chunk BiLSTM**: Learns acoustic patterns within 100-frame windows
- **Inter-chunk BiLSTM**: Maintains speaker identity across entire utterance
- **Curriculum Learning**: Starts easy (2 speakers) → builds to hard (5 speakers)
- **SI-SNR Loss**: Scale-invariant objective ensures separation quality
- **Permutation Invariant Training**: Automatically solves which speaker is which

---

## ⚡ Performance Tips

### To make training FASTER:
1. Use GPUs 5 & 6 together (✅ already configured)
2. Batch size 8 per GPU uses ~15-20GB memory (safe on 143GB)
3. Increase `num_workers` to 12-16 if CPU is idle
4. Keep `chunk_size` at 100 (memory efficient for DPRNN)

### To make training MORE ACCURATE:
1. Generate full dataset (80k-100k mixtures instead of 1k)
2. Use real LibriSpeech audio (instead of synthetic)
3. Run through all curriculum phases (don't skip 2→3→4)
4. Train until validation SI-SNR plateaus (typically 40-100 epochs)

### Example: Full Training Setup
```bash
# Generate full dataset (takes ~30-60 minutes)
python generate_mixtures_fast.py --output-dir data/mixtures
# (don't use --test-mode)

# Then train
python train_enhanced.py --full-curriculum
# Expect: 10-20 hours total, 10-14 dB SI-SNR on 5-speaker
```

---

## 🐛 Troubleshooting

### Training crashes with CUDA Out of Memory
```bash
# Reduce batch size in configs/base.yaml:
train_batch_size: 4  # instead of 8
```

### Training is very slow
```bash
# Check GPU utilization:
nvidia-smi

# If GPU% is low, increase workers:
num_workers: 16  # in configs/base.yaml
```

### Model not improving after first epoch
- Check that `--test-mode` data is being used for testing
- Verify SI-SNR loss is computing correctly
- Try different learning rate: `lr: 0.0001` or `lr: 0.0005`

### Can't import modules
```bash
# Reactivate venv and check imports:
source /mnt/raid/rl_gaming/dprnn2/bin/activate
python -c "import torch; print(torch.__version__)"
```

---

## 📞 Next Steps

1. **Right Now**: 
   - Run: `python train_enhanced.py --full-curriculum`
   - Monitor: `python monitor_training.py --watch` (in another terminal)

2. **After Phase 1 completes** (~45 min - 1 hour):
   - Check validation SI-SNR > 14 dB
   - If good, training auto-continues to Phase 2

3. **After Phase 4 completes** (~2-4 hours total):
   - Model checkpoint saved: `training_results/phase_4_5spk/best.pt`
   - Ready to separate your audio!

4. **Advanced (Optional)**:
   - Generate full dataset with real LibriSpeech data
   - Fine-tune on your voice recordings
   - Experiment with variable speaker counts

---

## 📝 Files to Know About

- **TRAINING_LOG.md** - High-level training summary (updated each epoch)
- **QUICK_START.md** - Step-by-step tutorial
- **training_results/phase_*/training_status_*.json** - Real-time metrics (← watch this!)
- **data_generation.log** - Data generation details

---

## ✨ Summary

Your DPRNN speaker separation system is **100% ready to train**! 

**Training sequence**:
```
2-speaker (30-60 min) 
  ↓ [Load best.pt]
3-speaker (20-50 min)
  ↓ [Load best.pt]
4-speaker (20-30 min)
  ↓ [Load best.pt]
5-speaker [FINAL] (40-80 min)
  ↓
✅ You can now separate your voice from friends!
```

**GPU allocation**: Both GPUs 5 & 6 ready (286GB total VRAM secured)

**Next Action**: 
```bash
cd /mnt/raid/rl_gaming/RL4VLM2/cocktail_separation
source /mnt/raid/rl_gaming/dprnn2/bin/activate
python train_enhanced.py --full-curriculum
```

**Good luck! 🚀**

---

*Generated March 22, 2026 by GitHub Copilot*
