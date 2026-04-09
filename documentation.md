# RCNN-Based Cocktail Party Audio Source Separation
source /mnt/raid/rl_gaming/RL4VLM2/cocktail_separation/dprnn2/bin/activate

## Documentation & Concept Guide

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Background & Motivation](#2-background--motivation)
3. [RCNN Concept — Why Recurrent + Convolutional?](#3-rcnn-concept--why-recurrent--convolutional)
4. [Model Architecture](#4-model-architecture)
5. [Dataset Pipeline](#5-dataset-pipeline)
6. [Training Pipeline](#6-training-pipeline)
7. [Inference Pipeline](#7-inference-pipeline)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Complete Pipeline Flow](#9-complete-pipeline-flow)
10. [How to Run](#10-how-to-run)
11. [Project File Structure](#11-project-file-structure)
12. [References](#12-references)

---

## 1. Problem Statement

### The Cocktail Party Problem

The **cocktail party problem** refers to the challenge of isolating a single speaker's voice from a mixture of overlapping voices and background noise — the same task our brain effortlessly performs at a busy party.

**Formally**: Given a mixed audio signal `y(t) = s₁(t) + s₂(t) + ... + sₙ(t)`, where `sᵢ(t)` are individual speaker signals, the goal is to recover each `sᵢ(t)` from only the observed mixture `y(t)`.

This is a **blind source separation (BSS)** problem because:
- We have only **one microphone** (single-channel).
- We have **no prior knowledge** of the speakers' voices.
- We don't know the mixing conditions (room, distance, etc.).

### Why Is This Hard?

| Challenge | Description |
|-----------|-------------|
| **Spectral overlap** | Human voices have overlapping frequency ranges (100–8000 Hz). |
| **Speaker variability** | Pitch, speaking rate, accent vary enormously. |
| **Permutation ambiguity** | The model doesn't know which output corresponds to which speaker. |
| **Phase reconstruction** | Even if we estimate magnitudes perfectly, reconstructing the phase is non-trivial. |

---

## 2. Background & Motivation

Traditional approaches to source separation include:
- **Independent Component Analysis (ICA)** — assumes statistical independence.
- **Non-negative Matrix Factorization (NMF)** — decomposes spectrograms into basis components.
- **Beamforming** — uses multiple microphones (not applicable for single-channel).

**Deep learning** revolutionized this field with approaches like:
- **Deep Clustering** (Hershey et al., 2016)
- **Permutation Invariant Training (PIT)** (Yu et al., 2017)
- **Conv-TasNet** (Luo & Mesgarani, 2019)
- **DPRNN** (Luo et al., 2020)

Our approach uses an **RCNN (Recurrent Convolutional Neural Network)**, which combines:
- **CNNs** for learning local spectral patterns (formants, harmonics).
- **RNNs (LSTM)** for modeling temporal structure across time frames.

This hybrid architecture is well-suited because speech has both **local spectral structure** (captured by convolutions) and **long-range temporal dependencies** (captured by recurrent layers).

---

## 3. RCNN Concept — Why Recurrent + Convolutional?

### Convolutional Layers (CNN)

2D Convolutions applied to spectrograms can:
- Detect **local spectral patterns** (e.g., harmonic structures, formant transitions).
- Learn **shift-invariant features** across both frequency and time.
- Compress the frequency dimension through strided convolutions.

**Analogy**: Like how CNNs detect edges and textures in images, in spectrograms they detect fundamental frequencies, harmonics, and speech formants.

### Recurrent Layers (RNN / LSTM)

Bidirectional LSTMs applied across time frames can:
- Capture **temporal context** — speech is inherently sequential.
- Model **long-range dependencies** — a word spoken 2 seconds ago provides context.
- Handle **variable-length sequences** naturally.
- The **bidirectional** aspect means the model sees both past and future context.

### Why Combine Them?

| Alone | Limitation |
|-------|-----------|
| CNN only | Cannot model long-range temporal dependencies; receptive field is limited. |
| RNN only | Processes raw features without learning good spectral representations first. |
| **RCNN** | **CNN extracts meaningful spectral features → RNN models temporal relationships between them.** |

The RCNN approach gives us:
1. **Feature hierarchy**: Low-level spectral features → high-level temporal patterns.
2. **Computational efficiency**: CNNs downsample the frequency dimension, reducing the LSTM input size.
3. **Skip connections**: Encoder features are added to decoder inputs, preserving fine-grained information.

---

## 4. Model Architecture

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT                                     │
│  Mixture Magnitude Spectrogram: (batch, 1, 257, T)              │
│  (257 freq bins from 512-point FFT)                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ENCODER                                      │
│                                                                  │
│  Conv2D(1→32,  3×3, stride=(2,1)) → BN → ReLU                  │
│  Conv2D(32→64, 3×3, stride=(2,1)) → BN → ReLU                  │
│  Conv2D(64→128,3×3, stride=(2,1)) → BN → ReLU                  │
│                                                                  │
│  Output: (batch, 128, freq', T)  where freq' ≈ 33              │
│  Downsamples frequency by 8× while preserving time dimension    │
└────────────────────────┬────────────────────────────────────────┘
                         │
            ┌────────────┤ (skip connection)
            │            ▼
            │  ┌──────────────────────────────────────────────────┐
            │  │               RESHAPE                            │
            │  │  (batch, 128, freq', T) → (batch, T, 128×freq') │
            │  └────────────────────┬─────────────────────────────┘
            │                       ▼
            │  ┌──────────────────────────────────────────────────┐
            │  │          BIDIRECTIONAL LSTM                       │
            │  │                                                  │
            │  │  BiLSTM(input=128×freq', hidden=256, layers=2)   │
            │  │  Output: (batch, T, 512)                         │
            │  │                                                  │
            │  │  Linear Projection: 512 → 128×freq'              │
            │  │  Output: (batch, T, 128×freq')                   │
            │  └────────────────────┬─────────────────────────────┘
            │                       ▼
            │  ┌──────────────────────────────────────────────────┐
            │  │               RESHAPE                            │
            │  │  (batch, T, 128×freq') → (batch, 128, freq', T) │
            │  └────────────────────┬─────────────────────────────┘
            │                       ▼
            └──────────────► (+) ADD (skip connection)
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DECODER                                      │
│                                                                  │
│  ConvTranspose2D(128→64, 3×3, stride=(2,1)) → BN → ReLU        │
│  ConvTranspose2D(64→32,  3×3, stride=(2,1)) → BN → ReLU        │
│  ConvTranspose2D(32→1,   3×3, stride=(2,1)) → BN → ReLU        │
│                                                                  │
│  Output: (batch, 1, 257, T)  (frequency upsampled back)         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MASK GENERATION                                 │
│                                                                  │
│  Conv2D(1 → n_sources, 1×1) → Sigmoid                          │
│  Output: (batch, 2, 257, T)  — one mask per source              │
│  Mask values in [0, 1]                                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MASK APPLICATION                                │
│                                                                  │
│  estimated_source_i = mask_i × mixture_magnitude                │
│  Then iSTFT with original phase → separated waveform            │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Domain** | STFT magnitude spectrogram | Time-frequency representation reveals structure invisible in raw waveform |
| **Encoder stride** | (2,1) — downsample freq, keep time | Reduces LSTM input while preserving full temporal resolution |
| **BiLSTM** | 2-layer, 256 hidden per direction | Sufficient capacity for 2-speaker separation |
| **Skip connection** | Add encoder output to decoder input | Prevents information loss through the bottleneck |
| **Mask type** | Sigmoid (soft mask) | Allows partial energy attribution between sources |
| **Phase** | Reuse mixture phase (no estimation) | Simplification — magnitude mask + original phase works well for 2 speakers |

---

## 5. Dataset Pipeline

### Data Source: LibriSpeech

We use **LibriSpeech**, a corpus of ~1000 hours of read English speech. Specifically:
- **Training**: `train-clean-100` (100 hours, 251 speakers)
- **Validation**: `dev-clean` (5 hours, 40 speakers)

### Mixture Generation Process

```
Speaker 1 utterance ──→ Resample to 8 kHz ──→ Trim/Pad to 4 sec ──→ Normalize
                                                                        │
Speaker 2 utterance ──→ Resample to 8 kHz ──→ Trim/Pad to 4 sec ──→ Normalize + SNR offset
                                                                        │
                                                            ────────────┘
                                                           │
                                                           ▼
                                                   mixture = s1 + s2
                                                           │
                                                           ▼
                                                       STFT (512-pt)
                                                           │
                                                    ┌──────┴──────┐
                                                    │             │
                                                Magnitude      Phase
```

**Parameters**:
| Parameter | Value | Notes |
|-----------|-------|-------|
| Sample rate | 8 kHz | Downsampled from 16 kHz for faster processing |
| Segment length | 4 seconds (32,000 samples) | Standard for speech separation benchmarks |
| FFT size | 512 | → 257 frequency bins |
| Hop length | 128 | → ~250 time frames per 4-second segment |
| SNR offset | Uniform[-5, +5] dB | Random relative loudness between speakers |
| Training mixtures | 5,000 | Pre-generated for fast, deterministic training |
| Validation mixtures | 500 | From a different speaker set (dev-clean) |

### Two Modes of Operation

1. **Pre-generated** (`prepare_data.py`): Download LibriSpeech → generate fixed `.pt` files → fast DataLoader.
2. **On-the-fly** (`dataset.py`): Generate mixtures dynamically during training (slower but infinite variety).

---

## 6. Training Pipeline

### Loss Function: SI-SNR with PIT

#### Scale-Invariant Signal-to-Noise Ratio (SI-SNR)

SI-SNR measures the quality of a separated signal relative to the clean reference:

```
SI-SNR(ŝ, s) = 10 · log₁₀( ||s_target||² / ||e_noise||² )

where:
    s_target = (<ŝ, s> / ||s||²) · s       ← projection of estimate onto target
    e_noise  = ŝ - s_target                 ← residual noise
```

- **Higher is better** (measured in dB).
- Scale-invariant: doesn't penalize overall volume differences.
- Standard metric in speech separation literature.

#### Permutation Invariant Training (PIT)

**Problem**: The model outputs 2 signals, but we don't know which output corresponds to which speaker.

**Solution**: Try all permutations of (output → target) assignment, pick the one with the lowest loss:

```
loss = min over all permutations P of:
           Σᵢ -SI-SNR(estimated_i, target_P(i))
```

For 2 sources, there are only 2! = 2 permutations, so this is computationally trivial.

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| LR scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Batch size | 8 |
| Gradient clipping | Max norm 5.0 |
| Mixed precision | AMP (FP16) |
| Epochs | 50 (configurable) |

### Training Flow

```
For each epoch:
    For each batch:
        1. Load (mixture_mag, source_mags, mixture_phase, waveforms)
        2. Forward pass: mixture_mag → RCNN → masks
        3. Apply masks: estimated_mag = mask × mixture_mag
        4. Reconstruct waveforms: iSTFT(estimated_mag, mixture_phase)
        5. Compute SI-SNR loss with PIT
        6. Backpropagate (with AMP scaling)
        7. Clip gradients
        8. Update weights
    Validate on held-out set
    Update learning rate (ReduceLROnPlateau)
    Save checkpoint if best validation loss
```

---

## 7. Inference Pipeline

```
Input Audio File (.wav / .mp3 / .flac)
         │
         ▼
  Load & Resample to 8 kHz
         │
         ▼
  ┌──────────────────────────────────────┐
  │  Segment into 4-second chunks       │
  │  (50% overlap for smooth stitching) │
  └──────────────┬───────────────────────┘
                 │
                 ▼
  For each segment:
         │
    ┌────┴─────────────────┐
    │  STFT → magnitude    │
    │         + phase       │
    └────┬─────────────────┘
         │
    ┌────┴─────────────────┐
    │  RCNN Model          │
    │  → 2 masks           │
    └────┬─────────────────┘
         │
    ┌────┴─────────────────────────────┐
    │  mask × mixture_mag → est_mag   │
    │  iSTFT(est_mag, phase) → wav    │
    └────┬─────────────────────────────┘
         │
         ▼
  Overlap-add all segments
         │
         ▼
  Normalize & Save as .wav
  + Generate visualization (waveform + spectrogram plots)
```

---

## 8. Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **SI-SNR** | `10·log₁₀(||s_target||²/||e_noise||²)` | Higher = better. Standard metric. |
| **SI-SNRi** | `SI-SNR(separated) - SI-SNR(mixture)` | Improvement over raw mixture. |
| **SDR** | Signal-to-Distortion Ratio (BSSEval) | Accounts for interference + artifacts + noise. |

Typical results for 2-speaker separation on WSJ0-2mix:
| Method | SI-SNRi (dB) |
|--------|-------------|
| Deep Clustering | 10.8 |
| PIT | 10.0 |
| Conv-TasNet | 15.3 |
| **RCNN (ours, expected)** | **8–12** |

---

## 9. Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE PIPELINE                                │
│                                                                         │
│  ┌───────────────┐    ┌───────────────┐    ┌──────────────────────┐    │
│  │   PREPARE     │───▶│    TRAIN      │───▶│     INFERENCE        │    │
│  │               │    │               │    │                      │    │
│  │ prepare_data  │    │  train.py     │    │  inference.py        │    │
│  │    .py        │    │               │    │                      │    │
│  │               │    │ SI-SNR + PIT  │    │ Load checkpoint      │    │
│  │ Download      │    │ AMP + Grad    │    │ Segment + STFT       │    │
│  │ LibriSpeech   │    │ Clip          │    │ RCNN → masks         │    │
│  │               │    │ LR Schedule   │    │ iSTFT → wavs         │    │
│  │ Generate      │    │ TensorBoard   │    │ Save separated .wav  │    │
│  │ 2-speaker     │    │ Checkpoints   │    │ + visualization      │    │
│  │ mixtures      │    │               │    │                      │    │
│  │               │    │               │    │                      │    │
│  │ 5000 train    │    │ best_model.pt │    │ separated_source_1   │    │
│  │ 500 val       │    │               │    │ separated_source_2   │    │
│  └───────────────┘    └───────────────┘    └──────────────────────┘    │
│                                                                         │
│  Hardware: NVIDIA H200 GPU (GPU 3), CUDA 12.9                          │
│  Framework: PyTorch                                                     │
│  Sample Rate: 8 kHz | FFT: 512 | Segment: 4 seconds                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. How to Run

### Step 1: Install Dependencies

```bash
cd /mnt/raid/rl_gaming/LLM4DyG-forked2/cocktail_party_rcnn
pip install -r requirements.txt
```

### Step 2: Prepare Dataset

```bash
# Download LibriSpeech and generate mixtures (takes ~15-30 min first time)
CUDA_VISIBLE_DEVICES=3 python prepare_data.py \
    --data_root ./data \
    --output_root ./data/generated \
    --num_train 5000 \
    --num_val 500
```

### Step 3: Train the Model

```bash
# Full training run (~50 epochs)
CUDA_VISIBLE_DEVICES=3 python train.py \
    --data_dir ./data/generated \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-3 \
    --checkpoint_dir ./checkpoints \
    --log_dir ./runs

# Quick smoke test (1 epoch, small dataset)
CUDA_VISIBLE_DEVICES=3 python train.py \
    --librispeech_root ./data \
    --epochs 1 \
    --num_train 100 \
    --num_val 20 \
    --batch_size 4
```

### Step 4: Monitor Training

```bash
tensorboard --logdir ./runs --port 6006
```

### Step 5: Run Inference

```bash
CUDA_VISIBLE_DEVICES=3 python inference.py \
    --input path/to/mixed_audio.wav \
    --checkpoint ./checkpoints/best_model.pt \
    --output_dir ./separated_output
```

### Step 6: Check Results

The `separated_output/` directory will contain:
- `separated_source_1.wav` — first separated speaker
- `separated_source_2.wav` — second separated speaker
- `original_mixture.wav` — the input mixture (for comparison)
- `separation_visualization.png` — waveform + spectrogram plots

---

## 11. Project File Structure

```
cocktail_party_rcnn/
├── model.py            # RCNN model architecture (encoder-BiLSTM-decoder)
├── dataset.py          # Dataset classes (on-the-fly + pre-generated)
├── train.py            # Training script with SI-SNR + PIT loss
├── inference.py        # Inference & separation script
├── prepare_data.py     # Data preparation (download + mixture generation)
├── utils.py            # Utilities (STFT, SI-SNR, PIT, audio I/O)
├── requirements.txt    # Python dependencies
├── documentation.md    # This document
├── data/               # (created at runtime)
│   ├── LibriSpeech/    # Downloaded LibriSpeech data
│   └── generated/      # Pre-generated mixture files
│       ├── train/      # Training .pt files
│       └── val/        # Validation .pt files
├── checkpoints/        # (created at runtime) Saved model checkpoints
├── runs/               # (created at runtime) TensorBoard logs
└── separated_output/   # (created at runtime) Inference outputs
```

---

## 12. References

1. **Cocktail Party Problem**: Cherry, E.C. (1953). "Some Experiments on the Recognition of Speech, with One and with Two Ears."
2. **Deep Clustering**: Hershey, J.R. et al. (2016). "Deep Clustering: Discriminative Embeddings for Segmentation and Separation." ICASSP.
3. **PIT**: Yu, D. et al. (2017). "Permutation Invariant Training of Deep Models for Speaker-Independent Multi-talker Speech Separation." ICASSP.
4. **Conv-TasNet**: Luo, Y. & Mesgarani, N. (2019). "Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation." IEEE/ACM TASLP.
5. **SI-SNR**: Le Roux, J. et al. (2019). "SDR – Half-baked or Well Done?" ICASSP.
6. **LibriSpeech**: Panayotov, V. et al. (2015). "LibriSpeech: An ASR Corpus Based on Public Domain Audio Books."
7. **RCNN for Audio**: Combination of CNN feature extraction (LeCun, 1998) with LSTM sequence modeling (Hochreiter & Schmidhuber, 1997) applied to audio spectrograms.
