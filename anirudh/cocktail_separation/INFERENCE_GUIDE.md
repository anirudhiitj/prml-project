# 🎯 Using the Model for Inference (Audio Separation)

## Timeline
- **Phase 1 (2-speaker)**: ~50 min from now (3:15-3:20 AM) ✅ Already running
- **Phase 2 (3-speaker)**: ~1 hour after Phase 1 completes (can separate 3 humans speaking) 🎯
- **Auto-progression**: All phases happen automatically, you just wait

---

## 📋 Steps to Separate Your Audio

### Step 1️⃣: Prepare Your Audio File
```bash
# Your audio file can be:
✅ MP3, WAV, FLAC, AAC, OGG, etc.
✅ Mono or Stereo (auto-converted to mono)
✅ Any sample rate (auto-resampled to 16kHz)
✅ **Max 10 seconds** (longer files auto-truncated)

Example:
/path/to/my_audio.mp3  (3 humans speaking)
```

### Step 2️⃣: Find the Best Checkpoint
```bash
# After Phase 3 completes (~3:50-4:00 AM), use the 3-speaker checkpoint:
cd /mnt/raid/rl_gaming/RL4VLM2/cocktail_separation
ls -lh checkpoints/*/best.pt

# Use the checkpoint with highest SI-SNR for your num_speakers:
# - Phase 1: checkpoints/2spk/best.pt  (2 speakers)
# - Phase 2: checkpoints/3spk/best.pt  (3 speakers) ← For your use case
# - Phase 3: checkpoints/4spk/best.pt  (4 speakers)
# - Phase 4: checkpoints/5spk/best.pt  (5 speakers)
```

### Step 3️⃣: Run Inference
```bash
cd /mnt/raid/rl_gaming/RL4VLM2/cocktail_separation
source /mnt/raid/rl_gaming/dprnn2/bin/activate

# For 3 humans speaking:
python inference.py \
  --audio /path/to/my_audio.mp3 \
  --checkpoint checkpoints/3spk/best.pt \
  --num-speakers 3 \
  --output-dir ./separated_output
```

### Step 4️⃣: Check Results
```bash
ls -la separated_output/
# Output:
# speaker_1.wav  (first person)
# speaker_2.wav  (second person)
# speaker_3.wav  (third person)

# Play them:
ffplay separated_output/speaker_1.wav
ffplay separated_output/speaker_2.wav
ffplay separated_output/speaker_3.wav
```

---

## 🎙️ Full Example (Copy & Paste)

### Scenario: Separate 3 friends talking

```bash
# 1. Activate environment
cd /mnt/raid/rl_gaming/RL4VLM2/cocktail_separation
source /mnt/raid/rl_gaming/dprnn2/bin/activate

# 2. Run separation (use your actual file path)
python inference.py \
  --audio /path/to/3_friends_recording.mp3 \
  --checkpoint checkpoints/3spk/best.pt \
  --num-speakers 3 \
  --output-dir ./my_separated_speakers

# 3. Listen to results
ffplay my_separated_speakers/speaker_1.wav
ffplay my_separated_speakers/speaker_2.wav
ffplay my_separated_speakers/speaker_3.wav
```

---

## ⚙️ Parameters Explained

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--audio` | Path to your file | MP3, WAV, FLAC, etc. (any format) |
| `--checkpoint` | Path to .pt file | Use checkpoint for your `num-speakers` |
| `--num-speakers` | 2-5 | Must match checkpoint (2spk/3spk/4spk/5spk) |
| `--output-dir` | Directory | Where speaker_1.wav, speaker_2.wav, etc. are saved |
| `--device` | cuda / cpu | Default: cuda (GPU is ~50x faster) |

---

## 📊 Expected Quality

| Speakers | Phase | Est. Time | SI-SNR | Quality |
|----------|-------|-----------|--------|---------|
| 2 | Phase 1 | 50 min | >12 dB | Excellent |
| 3 | Phase 2 | 1h 50m | >11 dB | Excellent |
| 4 | Phase 3 | 2h 50m | >10 dB | Very Good |
| 5 | Phase 4 | 3h 50m | >10 dB | Very Good |

**SI-SNR = Signal-to-Interference Noise Ratio**
- >10 dB: Very good separation
- 8-10 dB: Good separation
- 5-8 dB: Acceptable
- <5 dB: Poor

---

## 🚨 Troubleshooting

### "Checkpoint not found"
```bash
# Make sure you're using the correct path
# Should look like: checkpoints/3spk/best.pt
# NOT: checkpoints/3spk/latest.pt (use best.pt, not latest)
```

### "Audio file longer than 10s"
```bash
# The script auto-truncates to 10 seconds
# To use longer audio, trim it manually:
ffmpeg -i long_audio.mp3 -ss 0 -t 10 trimmed_audio.mp3
```

### "CUDA out of memory"
```bash
# Use CPU instead (slower but works):
python inference.py \
  --audio my_audio.mp3 \
  --checkpoint checkpoints/3spk/best.pt \
  --num-speakers 3 \
  --device cpu
```

### "Unknown file format"
```bash
# Convert to WAV first:
ffmpeg -i your_file.m4a -acodec pcm_s16le -ar 16000 output.wav
python inference.py --audio output.wav ...
```

---

## 💡 Pro Tips

1. **Test with short clips first** (2-3 sec) to verify everything works
2. **Best results**: Clean audio, each speaker talking roughly equally
3. **Multiple speakers talking at once**: Better separation = less overlap
4. **Remove silence**: Trim leading/trailing silence for cleaner output
5. **Check metrics**: Higher SI-SNR values (12+) indicate better separation quality

---

## 📈 What Happens Behind the Scenes

```
Your Audio (mixture)
    ↓
[DNN Encoder] → Converts to feature space
    ↓
[DPRNN Separator] → Learns speaker patterns (6 recurrent blocks)
    ↓
[Hungarian Matcher] → Matches speakers optimally
    ↓
[DNN Decoder] → Converts back to time domain
    ↓
Speaker 1, Speaker 2, Speaker 3 (separated!)
```

---

## ⏱️ Current Timeline

```
Now:          Phase 1 (2spk) training: 41/100 epochs (41%)
3:15-3:20 AM: Phase 1 completes → Phase 2 starts
4:15-4:20 AM: Phase 2 completes → Phase 3 starts ← Ready for 3 speakers!
5:15-5:20 AM: Phase 3 completes → Phase 4 starts
6:00-6:15 AM: Full curriculum done (2/3/4/5 speaker models ready)
```

**You can start using the model for 3-speaker separation around 4:20 AM!** 🎉

