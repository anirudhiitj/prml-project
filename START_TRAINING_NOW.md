# 🎯 IMMEDIATE ACTION PLAN - Start Training NOW

## Current Status: ✅ READY TO TRAIN

```
✅ Venv created: /mnt/raid/rl_gaming/dprnn2
✅ Dependencies installed: torch, torchaudio, pyroomacoustics, wandb, hydra, etc.
✅ GPU 5: 143 GB available (0% used)
✅ GPU 6: 143 GB available (~2% used)
✅ Test data generated: 4,800 mixtures (2spk, 3spk, 4spk, 5spk)
✅ Training scripts ready: train_enhanced.py, monitor_training.py
✅ Configuration files ready: base.yaml, 2spk.yaml, 3spk.yaml, 4spk.yaml, 5spk.yaml
```

---

## 🚀 STEP-BY-STEP: START TRAINING IN 30 SECONDS

### Step 1: Open Terminal 1 (TRAINING)
```bash
cd /mnt/raid/rl_gaming/RL4VLM2/cocktail_separation
source /mnt/raid/rl_gaming/dprnn2/bin/activate
python train_enhanced.py --full-curriculum
```

**What will happen:**
- Initializes training on GPUs 5 & 6
- Shows: `GPU Availability Check` → `GPU 5: 143.8 GB available ✓`
- Starts Phase 1 (2-speaker) - live progress bar appears
- Automatically continues to Phases 2, 3, 4 after each completes

---

### Step 2: Open Terminal 2 (MONITORING) - *In NEW terminal*
```bash
cd /mnt/raid/rl_gaming/RL4VLM2/cocktail_separation
source /mnt/raid/rl_gaming/dprnn2/bin/activate
python monitor_training.py --watch
```

**What will appear:**
```
╔════════════════════════════════════════════════════════╗
║  DPRNN-TasNet Training Monitor - 2026-03-22 14:30:45 ║
║  GPU 5 & 6 - Multi-Speaker Separation                ║
╚════════════════════════════════════════════════════════╝

════════════════════════════════════════════════════════
PHASE 1: 2-Speaker Separation
════════════════════════════════════════════════════════

  Status:        RUNNING
  GPUs:          [5, 6]
  Progress:      [████████████░░░░░░░░░] 45.0%
  Current Epoch: 23 / 50
  
  Latest Metrics:
    Train Loss:    1.1234
    Train SI-SNR:  14.5 dB
    Val Loss:      1.3456
    Val SI-SNR:    14.2 dB
    Learn Rate:    1.00e-03
    
  Best Performance:
    Best Val SI-SNR: 14.5 dB (Epoch 15)
  
  Estimated Time Remaining: 1.2 hours
```

---

## 📊 EXPECTED TIMELINE

```
Time 00:00  → Phase 1 (2-speaker) STARTS
            → GPU 5,6 ramp up to full usage
            → Training loss decreases rapidly
            
Time 00:45  → Phase 1 COMPLETES (✅ Val SI-SNR > 15 dB)
            → Checkpoint saved: phase_1_2spk/best.pt
            → Phase 2 auto-starts with Phase 1 weights

Time 01:35  → Phase 2 (3-speaker) COMPLETES (✅ Val SI-SNR > 12 dB)
            → Phase 3 auto-starts

Time 02:15  → Phase 3 (4-speaker) COMPLETES (✅ Val SI-SNR > 10 dB)
            → Phase 4 auto-starts

Time 03:45  → Phase 4 (5-speaker) COMPLETES ✨
            → FINAL MODEL READY: training_results/phase_4_5spk/best.pt
            → You can now separate your audio!
```

**Total Time: 3-4 hours for test data**

---

## 📁 LIVE FILES TO TRACK

While training is running, these files update in real-time:

```bash
# Terminal 1: Watch the JSON metrics (updates every epoch)
watch -n 5 'cat training_results/phase_*/training_status_*.json | python -m json.tool | head -50'

# Terminal 3: Quick check during training
tail -f training_results/phase_1_2spk/training_status_phase_1_2spk.json
```

**What to look for:**
```json
{
  "current_epoch": 23,
  "best_val_sisnr": 14.5,
  "eta_hours": 1.2,
  "training_history": {
    "epoch": [0, 1, 2, ..., 23],
    "val_sisnr": [8.5, 9.2, 10.1, ..., 14.5],
    "epoch_time": [45.2, 44.8, 43.9, ..., 41.5]
  }
}
```

**Good Signs:**
- ✅ `val_sisnr` increasing each epoch (learning!)
- ✅ `epoch_time` stable around 40-50 seconds (no memory issues)
- ✅ GPU% near 100% in `nvidia-smi` (full utilization)

---

## 🎯 SUCCESS CRITERIA

### Phase 1 (2-speaker) - COMPLETE when:
- Val SI-SNR reaches **> 15 dB**
- Training time: **30-60 minutes**
- Automatically stops and moves to Phase 2

### Phase 2 (3-speaker) - COMPLETE when:
- Val SI-SNR reaches **> 12 dB**
- Training time: **20-50 minutes**
- Automatically stops and moves to Phase 3

### Phase 3 (4-speaker) - COMPLETE when:
- Val SI-SNR reaches **> 10 dB**
- Training time: **20-30 minutes**
- Automatically stops and moves to Phase 4

### Phase 4 (5-speaker) - COMPLETE when:
- Val SI-SNR reaches **> 10 dB**
- Training time: **40-80 minutes**
- 🎉 **YOU'RE DONE! Model ready for inference!**

---

## 🎤 AFTER TRAINING: SEPARATE YOUR AUDIO (2-minute setup)

Once Phase 4 completes and you see `best.pt` in `training_results/phase_4_5spk/`:

```bash
# Prepare your audio file (4sec clip with 2-5 people speaking)
# Save as: my_audio.wav

# Separate it:
python separate.py \
    --input my_audio.wav \
    --output separated_speakers/ \
    --checkpoint training_results/phase_4_5spk/best.pt

# Results:
ls -lh separated_speakers/
# separated_speakers/speaker_1.wav
# separated_speakers/speaker_2.wav
# separated_speakers/speaker_3.wav
# separated_speakers/speaker_4.wav
# separated_speakers/speaker_5.wav

# Play them:
ffplay separated_speakers/speaker_1.wav  # Your voice
ffplay separated_speakers/speaker_2.wav  # Friend 1
# ... etc
```

---

## ⚡ QUICK REFERENCE COMMANDS

```bash
# In terminal 1 (training):
python train_enhanced.py --full-curriculum

# In terminal 2 (monitoring):
python monitor_training.py --watch

# Check GPU usage (terminal 3):
nvidia-smi -l 1  # Updates every 1 second

# Kill training gracefully:
# Press Ctrl+C in terminal 1
# Status will save with: "status": "INTERRUPTED"

# Resume training from checkpoint:
python train_enhanced.py --phase 2 --resume training_results/phase_1_2spk/best.pt

# Generate more data for production:
python generate_mixtures_fast.py  # Removes --test-mode for full 80-100k mixtures
```

---

## 🛑 IF SOMETHING GOES WRONG

### Training crashes immediately:
1. Check GPU: `nvidia-smi` (both GPUs should show available)
2. Check venv: `which python` (should be `/mnt/raid/rl_gaming/dprnn2/bin/python`)
3. Check data: `ls data/mixtures/2spk/train/ | head -5` (should show 0000001/, etc.)

### Out of Memory (OOM):
```bash
# Reduce batch size in configs/base.yaml:
train_batch_size: 4  # was 8
# Restart training
```

### Very slow training (< 30 samples/sec):
```bash
# Increase workers in configs/base.yaml:
num_workers: 16  # was 8
```

### Validation metrics not improving:
- Wait longer (sometimes takes 5-10 epochs to start improving)
- Check data: verify mixture amplitudes are in [-1, 1] range
- Lower learning rate: `lr: 0.0001` in Phase 1

---

## 📞 SUPPORT

All metrics are automatically saved. Check:
- **Live training**: `training_results/phase_*/training_status_*.json`
- **Checkpoints**: `training_results/phase_*/best.pt` (best model per phase)
- **Logs**: `data_generation.log` (data gen details)
- **Monitor output**: Run `python monitor_training.py` anytime to check status

---

## ✅ FINAL CHECKLIST BEFORE YOU START

- [ ] Terminals are open and ready
- [ ] Venv activated: `which python` shows `/mnt/raid/rl_gaming/dprnn2/bin/python`
- [ ] Data exists: `ls cocktail_separation/data/mixtures/2spk/train/ | wc -l` shows ~1000
- [ ] GPU ready: `nvidia-smi` shows GPUs 5 & 6 with >140GB free
- [ ] Read this file completely before starting (you are here ✅)

---

## 🚀 NOW GO START TRAINING!

**Terminal 1:**
```bash
cd /mnt/raid/rl_gaming/RL4VLM2/cocktail_separation
source /mnt/raid/rl_gaming/dprnn2/bin/activate
python train_enhanced.py --full-curriculum
```

**Terminal 2 (after Terminal 1 starts):**
```bash
cd /mnt/raid/rl_gaming/RL4VLM2/cocktail_separation
source /mnt/raid/rl_gaming/dprnn2/bin/activate
python monitor_training.py --watch
```

**Good luck! You'll have a working speaker separator in 3-4 hours. 🎉**

---

*Last Updated: March 22, 2026*
*Next Run: 📍 SEE "NOW GO START TRAINING!" ABOVE*
