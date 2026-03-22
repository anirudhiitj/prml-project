#!/bin/bash
# 🎯 QUICK INFERENCE COMMANDS - COPY & PASTE

# ============================================================
# FOR 3 HUMANS SPEAKING (after ~4:20 AM)
# ============================================================

cd /mnt/raid/rl_gaming/RL4VLM2/cocktail_separation
source /mnt/raid/rl_gaming/dprnn2/bin/activate

# Replace YOUR_FILE.wav with your actual audio file
python inference.py \
  --audio YOUR_FILE.wav \
  --checkpoint checkpoints/3spk/best.pt \
  --num-speakers 3 \
  --output-dir separated_speakers_3

# Listen to results:
ffplay separated_speakers_3/speaker_1.wav
ffplay separated_speakers_3/speaker_2.wav
ffplay separated_speakers_3/speaker_3.wav


# ============================================================
# FOR 2 HUMANS SPEAKING (after ~3:20 AM)
# ============================================================

python inference.py \
  --audio YOUR_FILE.wav \
  --checkpoint checkpoints/2spk/best.pt \
  --num-speakers 2 \
  --output-dir separated_speakers_2


# ============================================================
# FOR 4-5 HUMANS SPEAKING (after ~5:20 AM)
# ============================================================

# 4 speakers:
python inference.py \
  --audio YOUR_FILE.wav \
  --checkpoint checkpoints/4spk/best.pt \
  --num-speakers 4 \
  --output-dir separated_speakers_4

# 5 speakers:
python inference.py \
  --audio YOUR_FILE.wav \
  --checkpoint checkpoints/5spk/best.pt \
  --num-speakers 5 \
  --output-dir separated_speakers_5


# ============================================================
# IF AUDIO FILE FORMAT NOT SUPPORTED, CONVERT FIRST
# ============================================================

# Convert any audio to WAV:
ffmpeg -i your_file.mp3 -acodec pcm_s16le -ar 16000 output.wav

# Then use: --audio output.wav


# ============================================================
# REAL EXAMPLE (Copy this entire block)
# ============================================================

cd /mnt/raid/rl_gaming/RL4VLM2/cocktail_separation
source /mnt/raid/rl_gaming/dprnn2/bin/activate

# Separate 3 friends in a recording
python inference.py \
  --audio /home/user/3_friends_talk.mp3 \
  --checkpoint checkpoints/3spk/best.pt \
  --num-speakers 3 \
  --output-dir ./my_separated_audio
