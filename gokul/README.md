# Cocktail-Party Speech Separation

**End-to-end monaural speech separation using Conv-TasNet — implemented entirely in C++ with LibTorch.**

> PRML Final Project · Separates a single-channel recording of two overlapping speakers into individual clean speech signals.

---

## Table of Contents

- [Overview](#overview)
- [Why This Problem](#why-this-problem)
- [Architecture](#architecture)
- [Mathematical Foundations](#mathematical-foundations)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation Metrics](#evaluation-metrics)
- [Build & Run](#build--run)
- [GUI Application](#gui-application)
- [Project Structure](#project-structure)
- [Results & Expectations](#results--expectations)
- [References](#references)

---

## Overview

Given a single mixed audio signal **x(t) = s₁(t) + s₂(t)** containing two overlapping speakers, this system recovers the individual source signals **ŝ₁(t)** and **ŝ₂(t)**.

```
Input:  mixture.wav  (2 speakers talking simultaneously)
          ↓
     Conv-TasNet  (8.2M parameters, ~15 dB SI-SNRi)
          ↓
Output: source_1.wav  (Speaker A isolated)
        source_2.wav  (Speaker B isolated)
```

The entire pipeline — model, training loop, inference, audio I/O, STFT, metrics, and a desktop GUI — is written in **C++17** using LibTorch and libsndfile. No Python is needed at runtime.

---

## Why This Problem

The **cocktail party problem** is one of the most fundamental challenges in signal processing. A single microphone captures a linear superposition of sources — one equation, two unknowns. It is mathematically ill-posed without additional priors.

Deep learning makes it tractable by learning the statistical structure of speech signals, enabling the network to infer which time-frequency components belong to which speaker.

**Applications:** hearing aids, voice assistants, teleconferencing, ASR preprocessing in multi-speaker environments.

---

## Architecture

### Primary Model: Conv-TasNet

Conv-TasNet (Luo & Mesgarani, 2019) operates entirely in the **time domain**, bypassing the traditional Short-Time Fourier Transform (STFT) by learning its own encoder and decoder. This removes phase estimation errors inherent in spectrogram-based approaches.

```
Mixture waveform x[t]    (1D signal, 8 kHz)
        ↓
   ┌───────────┐
   │  Encoder   │   1D Conv (1 → 512, kernel=16, stride=8)
   │            │   "Learned STFT" — each of 512 filters ≈ a frequency band
   └─────┬─────┘
         ↓
   ┌───────────┐
   │    TCN     │   3 repeats × 8 dilated depth-separable conv blocks
   │ Separator  │   Exponentially growing receptive field (1,2,4,...,128)
   │            │   Global + Channel LayerNorm, PReLU activations
   └─────┬─────┘
         ↓
   2 masks (ReLU) → element-wise multiply with encoder output
         ↓
   ┌───────────┐
   │  Decoder   │   Transposed 1D Conv (512 → 1, kernel=16, stride=8)
   │            │   "Learned inverse STFT"
   └───────────┘
         ↓
   ŝ₁[t], ŝ₂[t]   separated source waveforms
```

**Hyperparameters:**

| Symbol | Name | Value | Purpose |
|--------|------|-------|---------|
| N | Encoder filters | 512 | Frequency resolution of learned filter bank |
| L | Encoder kernel | 16 | Temporal resolution (2 ms @ 8 kHz) |
| B | Bottleneck channels | 256 | Dimensionality reduction before TCN |
| H | Hidden channels | 416 | Capacity of each separable conv block |
| P | Depth-wise kernel | 3 | Local temporal context per block |
| X | Blocks per repeat | 8 | Dilation pattern: 1, 2, 4, …, 128 |
| R | Repeats | 3 | Total receptive field ≈ 1.5 s |
| C | Number of speakers | 2 | Output masks / separated sources |

**Total parameters:** ~8.2 million

**Key components:**
- **Depth-wise separable convolution** — factorizes the standard 1D convolution into a per-channel spatial convolution + a 1×1 cross-channel convolution, reducing parameters by a factor of approximately H/P
- **Global Layer Normalization (gLN)** — normalizes across both channel and time dimensions at the bottleneck
- **Channel Layer Normalization (cLN)** — normalizes across channels only inside each conv block

### Baseline Model: Spectrogram U-Net

A classical 4-level encoder-decoder U-Net that operates on STFT magnitude spectrograms. It predicts sigmoid masks in [0, 1] for each speaker, multiplies them with the mixture magnitude, and reconstructs via inverse STFT using the mixture phase.

This serves as a comparison point: it demonstrates that **learned representations (Conv-TasNet) outperform hand-designed ones (STFT + magnitude masking)**.

---

## Mathematical Foundations

### Short-Time Fourier Transform (STFT)

Speech is non-stationary — phonemes change every 10–30 ms. The STFT windows the signal into overlapping frames:

```
STFT{x}(t, f) = Σₙ x[n] · w[n − tH] · e^{−j2πfn/N}
```

| Setting | Value | Rationale |
|---------|-------|-----------|
| Window | Hann, 256 samples (32 ms) | Smooth edges reduce spectral leakage |
| Hop size | 64 samples (8 ms) | 75% overlap → perfect reconstruction (COLA) |
| FFT size | 256 | Matches window length at 8 kHz |
| Freq bins | 129 | One-sided real FFT |

The U-Net baseline uses the STFT directly. Conv-TasNet replaces it with a learned encoder — and this is one of the core insights of the project.

### SI-SNR (Scale-Invariant Signal-to-Noise Ratio)

The primary training objective and evaluation metric:

```
ŝ = s − mean(s),   ê = e − mean(e)       (zero-mean)
s_target = (⟨ŝ, ê⟩ / ‖ê‖²) · ê           (optimal rescaling via projection)
SI-SNR   = 10 · log₁₀( ‖s_target‖² / ‖ŝ − s_target‖² )
```

SI-SNR is **scale-invariant** — it measures separation quality regardless of amplitude. This aligns with human perception and provides smoother gradients than SDR.

- **0 dB** → error power equals signal power (bad)
- **10 dB** → signal 10× stronger than error (good)
- **20 dB** → signal 100× stronger than error (excellent)

### Permutation Invariant Training (PIT)

The network outputs [ŝ₁, ŝ₂] but does not know which output corresponds to which ground-truth speaker. PIT resolves this:

```
L_PIT = − max( SI-SNR(ŝ₁, s₁) + SI-SNR(ŝ₂, s₂),
               SI-SNR(ŝ₁, s₂) + SI-SNR(ŝ₂, s₁) )
```

Both possible speaker-to-output assignments are evaluated. Training uses the one that yields the highest total SI-SNR.

---

## Dataset

**Libri2Mix** (Cosentino et al., 2020), derived from LibriSpeech:

| Split | Mixtures | Purpose |
|-------|----------|---------|
| train-360 | 50,800 | Training |
| dev | 3,000 | Validation (LR scheduling, early stopping) |
| test | 3,000 | Final evaluation |

- **Sample rate:** 8 kHz (standard for speech separation benchmarks)
- **Speakers per mixture:** 2
- **Mode:** `min` — mixtures trimmed to the length of the shortest source
- **Generation:** `bash scripts/generate_librimix.sh ./data` (requires ~100 GB disk, downloads LibriSpeech and generates all mixtures)

### Audio Preprocessing

```
Raw WAV → Load (libsndfile) → Mono → Enforce 8 kHz → Peak normalize (0.9) → VAD silence trim
```

Preprocessing normalizes for consistency but never modifies content. No band-pass filtering, denoising, or compression — these would distort the additive mixture model x = s₁ + s₂ that the loss function assumes.

### Training Augmentations

| Augmentation | Details | Applied to |
|---|---|---|
| Random gain | ±6 dB uniform | Mix + both sources (consistent) |
| Gaussian noise | 20–40 dB SNR | Mixture only |
| Circular shift | ±10% of segment length | Mix + both sources |
| Polarity flip | 50% probability | Mix + both sources |

---

## Training

### Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| Base LR | 1 × 10⁻³ |
| LR warmup | Linear over epochs 1–5 |
| LR schedule | Halve on plateau (patience 5 epochs on val SI-SNR) |
| Min LR | 1 × 10⁻⁶ |
| Gradient clipping | Max-norm 5.0 |
| Batch size | 4 (micro-batch on GPU) |
| Gradient accumulation | 1–12× (configurable for VRAM-limited GPUs) |
| Segment length | 32,000 samples (4 seconds @ 8 kHz) |
| Epochs | 100 |

### Learning Rate Schedule

1. **Warmup (epochs 1–5):** LR ramps linearly from `base_lr / 5` to `base_lr`
2. **Plateau (epochs 6+):** LR held until val SI-SNR stalls for 5 consecutive epochs, then halved
3. **Floor:** LR never drops below 1 × 10⁻⁶

### Checkpointing

| File | When saved | Purpose |
|---|---|---|
| `best_tasnet.pt` | New best val SI-SNR | **Use this for inference** |
| `latest_tasnet.pt` | Every epoch | Crash recovery |
| `tasnet_epN.pt` | Every 5 epochs | Training history |
| `final_tasnet.pt` | End of training | Last-epoch model |
| `*.optim` | With each `.pt` | Optimizer state for resume |
| `*.meta` | With each `.pt` | Epoch, best SI-SNR, LR |

### Resuming Training

```bash
./build/train \
    --model tasnet \
    --data_dir ./data/Libri2Mix/Libri2Mix/wav8k/min/train-360 \
    --val_dir  ./data/Libri2Mix/Libri2Mix/wav8k/min/dev \
    --resume checkpoints/latest_tasnet.pt
```

All state (model weights, optimizer, LR, epoch counter, best metric) is restored automatically.

---

## Evaluation Metrics

| Metric | Measures | Range | Target |
|--------|----------|-------|--------|
| **SI-SNRi** | Scale-invariant SNR improvement over mixture | −∞ to +∞ dB | > 12 dB |
| **SDRi** | Signal-to-distortion ratio improvement | −∞ to +∞ dB | > 10 dB |
| **STOI** | Short-time objective intelligibility | 0 to 1 | > 0.85 |

**SI-SNRi** is the primary metric — it measures how much the separation improves the SNR relative to the raw mixture input.

### Expected Performance

| Model | SI-SNRi | SDRi | STOI | Params |
|---|---|---|---|---|
| Mixture (input) | 0.0 dB | 0.0 dB | ~0.55 | — |
| Spectrogram U-Net | ~10.5 dB | ~10.0 dB | ~0.82 | 7.8M |
| **Conv-TasNet** | **~14.5 dB** | **~14.0 dB** | **~0.92** | **8.2M** |
| Conv-TasNet (paper) | 15.3 dB | 15.6 dB | — | 5.1M |

---

## Build & Run

### Prerequisites

| Tool | Version | Install |
|---|---|---|
| g++ | ≥ 11 | `sudo apt install g++` |
| CMake | ≥ 3.18 | `sudo apt install cmake` |
| libsndfile | ≥ 1.0.31 | `sudo apt install libsndfile1-dev` |
| GTK+ 3 | ≥ 3.20 | `sudo apt install libgtk-3-dev` (for GUI) |
| LibTorch | 2.6.0+cu124 | Already in `libtorch/` |
| CUDA | ≥ 12.4 | Driver + toolkit |
| ffmpeg | any | `sudo apt install ffmpeg` (for the GUI's media conversion) |

### Build

```bash
cd prml-project
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH=../libtorch ..
make -j$(nproc)
```

This produces three binaries:
- `./build/train` — model training
- `./build/inference` — separation + optional evaluation
- `./build/gui` — GTK3 desktop GUI

### Smoke Test

```bash
./build/train --smoke_test
./build/inference --smoke_test
```

Validates architectures, STFT roundtrip, loss functions, and metrics — no dataset required.

### Generate Dataset

```bash
bash scripts/generate_librimix.sh ./data
```

Downloads LibriSpeech (~30 GB) and generates Libri2Mix mixtures (~36 GB total). Requires Python 3 and sox.

### Train

```bash
# Background training (recommended)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup ./build/train \
    --model tasnet \
    --data_dir ./data/Libri2Mix/Libri2Mix/wav8k/min/train-360 \
    --val_dir  ./data/Libri2Mix/Libri2Mix/wav8k/min/dev \
    --epochs 100 --batch_size 4 --accumulate 1 --lr 1e-3 \
    > training.log 2>&1 &
echo $! > training_pid.txt

# Monitor
tail -f training.log
```

### Training Flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `tasnet` | `tasnet` or `unet` |
| `--data_dir` | (required) | Path to training split |
| `--val_dir` | (optional) | Path to validation split |
| `--epochs` | `100` | Number of epochs |
| `--batch_size` | `2` | Micro-batch size |
| `--accumulate` | `3` | Gradient accumulation steps |
| `--lr` | `1e-3` | Base learning rate |
| `--warmup_epochs` | `5` | Linear warmup epochs |
| `--min_lr` | `1e-6` | LR floor |
| `--grad_clip` | `5.0` | Max gradient norm |
| `--seg_len` | `32000` | Segment length in samples (4 s) |
| `--sr` | `8000` | Sample rate |
| `--workers` | `4` | Data loader threads |
| `--no_augment` | (flag) | Disable augmentation |
| `--resume` | (path) | Resume from checkpoint |
| `--ckpt_dir` | `./checkpoints` | Checkpoint directory |
| `--log_interval` | `25` | Print every N batches |
| `--save_interval` | `5` | Periodic save every N epochs |

### Inference

```bash
# Basic separation
./build/inference \
    --model tasnet \
    --checkpoint checkpoints/best_tasnet.pt \
    --input mixture.wav

# With quality evaluation (requires ground-truth sources)
./build/inference \
    --model tasnet \
    --checkpoint checkpoints/best_tasnet.pt \
    --input mixture.wav \
    --ref_s1 clean_speaker1.wav \
    --ref_s2 clean_speaker2.wav
```

Output: `output/source_1.wav` and `output/source_2.wav`

### Using MP4/Video/Non-WAV Input

The inference binary works with 8 kHz mono WAV. For other formats, convert first:

```bash
ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 8000 -ac 1 mixture.wav
./build/inference --model tasnet --checkpoint checkpoints/best_tasnet.pt --input mixture.wav
```

Or simply use the [GUI application](#gui-application), which handles conversion automatically.

---

## GUI Application

A native **GTK3 desktop application** (written in C++) for separating audio from any media file. No Python required.

```bash
./build/gui
```

**Workflow:**
1. Click **Select File** — supports MP4, MKV, AVI, MP3, WAV, FLAC, OGG, and more
2. Click **Separate Speakers** — the GUI runs ffmpeg conversion + Conv-TasNet inference
3. **Play** or **Save** each separated speaker track

The GUI automatically locates the inference binary, the best available checkpoint, and handles the ffmpeg conversion pipeline. It provides status feedback and error reporting through the GTK interface.

---

## Project Structure

```
prml-project/
├── CMakeLists.txt                  Build configuration
├── README.md                       This file
├── .gitignore
│
├── src/
│   ├── conv_tasnet.h / .cpp        Conv-TasNet model (encoder, TCN separator, decoder)
│   ├── unet.h / .cpp               Spectrogram U-Net baseline
│   ├── train.cpp                   Training entry point
│   ├── inference.cpp               Inference + evaluation entry point
│   ├── gui.cpp                     GTK3 desktop GUI
│   ├── dataset.h / .cpp            LibriMix CSV dataset loader
│   ├── losses.h / .cpp             SI-SNR and PIT loss
│   ├── metrics.h / .cpp            SI-SNRi, SDRi, STOI
│   ├── audio_utils.h / .cpp        WAV I/O via libsndfile
│   ├── stft.h / .cpp               STFT / iSTFT
│   ├── preprocessing.h / .cpp      Normalize, VAD trim
│   └── augmentation.h / .cpp       Training augmentations
│
├── scripts/
│   └── generate_librimix.sh        Dataset download & generation
│
├── libtorch/                       LibTorch runtime (gitignored)
├── third_party/                    Local libsndfile (fallback)
├── checkpoints/                    Saved model weights
├── data/                           Libri2Mix dataset
└── output/                         Separated audio output
```

---

## Results & Expectations

### Will it reach 20 dB SI-SNR?

Based on the current training trajectory:

| Milestone | Estimated time | Justification |
|---|---|---|
| **10 dB** (val SI-SNR) | ~2–3 epochs (~3 h) | Already at ~11.7 dB val after epoch 1 |
| **13 dB** | ~10–15 epochs (~15 h) | Rapid improvement phase |
| **14–15 dB** | ~30–50 epochs (~50 h) | Refinement, LR reductions |
| **20 dB** | **Unlikely** | Exceeds published SOTA for this architecture |

The Conv-TasNet paper reports **15.3 dB SI-SNRi** on WSJ0-2mix. Libri2Mix is a harder dataset (more speakers, more diverse content). For this architecture, **13–15 dB on Libri2Mix is a strong result**. Reaching 20 dB would require a fundamentally different (and much larger) architecture like SepFormer (~26M params with self-attention).

### Training Timeline (RTX 4070, batch=4)

| Phase | Epochs | Time | What happens |
|---|---|---|---|
| Warmup | 1–5 | ~8 h | LR ramps up, model learns basic patterns |
| Rapid improvement | 6–20 | ~24 h | Biggest SI-SNR gains |
| Refinement | 21–50 | ~48 h | Slower gains, LR reductions begin |
| Polish | 51–100 | ~80 h | Diminishing returns |

---

## References

1. Y. Luo & N. Mesgarani. *Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation.* IEEE/ACM TASLP, 2019.
2. J. R. Hershey et al. *Deep Clustering: Discriminative Embeddings for Segmentation and Separation.* ICASSP, 2016.
3. D. Yu et al. *Permutation Invariant Training of Deep Models for Speaker-Independent Multi-Talker Speech Separation.* ICASSP, 2017.
4. J. Le Roux et al. *SDR — Half-Baked or Well Done?* ICASSP, 2019.
5. J. Cosentino et al. *LibriMix: An Open-Source Dataset for Generalizable Speech Separation.* arXiv, 2020.
6. O. Ronneberger et al. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI, 2015.
7. C. Subakan et al. *Attention Is All You Need in Speech Separation.* ICASSP, 2021 (SepFormer).

---

**License:** MIT
