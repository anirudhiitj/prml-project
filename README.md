# Cocktail-Party Speech Separation

**Single-microphone speech separation using Conv-TasNet and Spectrogram U-Net in C++ / LibTorch**

> PRML Final Project — monaural (single-channel) separation of overlapping speakers into individual clean speech signals.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Audio Preprocessing](#2-audio-preprocessing)
3. [Fourier / Time-Frequency Representation](#3-fourier--time-frequency-representation)
4. [Deep Learning Separation Models](#4-deep-learning-separation-models)
5. [Training Strategy](#5-training-strategy)
6. [Evaluation and Results](#6-evaluation-and-results)
7. [Implementation Guide](#7-implementation-guide)
8. [Project Narrative for Reports](#8-project-narrative-for-reports)
9. [Quick Start](#9-quick-start)
10. [References](#10-references)

---

## 1. Problem Statement

Given a single mixed audio signal `x(t) = s₁(t) + s₂(t)` with two overlapping speakers, recover the individual signals `ŝ₁(t)` and `ŝ₂(t)`.

This is the **cocktail party problem** — one of the oldest problems in signal processing. It is ill-posed: one equation, two unknowns. Deep learning makes it tractable by learning statistical priors over speech structure.

### Why it matters

- Hearing aids, voice assistants, teleconferencing
- Preprocessing for ASR in multi-speaker environments
- Foundational problem connecting signal processing + machine learning

---

## 2. Audio Preprocessing

### Pipeline

```
Raw WAV → Load → Mono conversion → Peak normalization → Silence trimming (VAD)
```

| Step | What | Why | Implementation |
|------|------|-----|----------------|
| **Load** | Read WAV via libsndfile | Standard lossless audio format | `audio::load_wav()` |
| **Mono** | Average channels to 1 | Monaural separation problem definition | `audio::load_wav()` |
| **Resample** | Enforce target sample rate (8 kHz) | Consistent temporal resolution; 8 kHz standard for speech separation benchmarks | Validated at load time |
| **Normalize** | Scale peak to 0.9 | Prevents clipping; gives network consistent amplitude range | `preprocess::normalize()` |
| **VAD Trim** | Remove leading/trailing silence | Training segments should contain speech, not blank audio | `preprocess::vad_trim()` |

### What we do NOT do (and why)

| Avoided step | Reason |
|--|--|
| **Band-pass filtering** | Destroys speech harmonics above the cutoff. The neural network must see full-bandwidth speech to learn separation. |
| **Aggressive denoising** | Pre-denoising (e.g. spectral subtraction) introduces musical noise artifacts and removes fine spectral detail the model needs. |
| **Compression / AGC** | Non-linear processing distorts the additive mixture model `x = s₁ + s₂` that the loss function assumes. |

**Principle**: Preprocess for consistency (amplitude, silence), but never for content modification. Let the neural network handle separation.

---

## 3. Fourier / Time-Frequency Representation

### Why STFT, not plain FFT?

Speech is **non-stationary**: phonemes, pitch, and energy change every 10-30 ms. A single FFT gives one global spectrum across the entire signal — it cannot distinguish speakers who occupy different time-frequency regions.

The **Short-Time Fourier Transform (STFT)** windows the signal into overlapping short frames and computes the DFT of each:

```
STFT{x}(t, f) = Σₙ x[n] · w[n - tH] · e^{-j2πfn/N}
```

This gives a 2D time-frequency representation where each speaker occupies distinct T-F bins, enabling **masking-based separation**.

### STFT Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Window function | Hann | Best sidelobe suppression for speech; smooth edges reduce spectral leakage |
| Window size | 256 samples (32 ms @ 8 kHz) | Captures ~3 pitch periods of typical speech (100-300 Hz fundamental) |
| Hop size | 64 samples (8 ms) | 75% overlap → smooth overlap-add reconstruction |
| FFT size | 256 | Equal to window → no zero-padding needed at 8 kHz |
| Frequency bins | 129 (= 256/2 + 1) | One-sided real FFT output |

### Magnitude vs. Complex Features

| Approach | Pros | Cons | Used by |
|----------|------|------|---------|
| **Magnitude-only** | Simple, interpretable; most energy info | Loses phase; needs external phase source for reconstruction | U-Net baseline |
| **Complex (real + imag)** | Complete information; no phase loss | 2× input channels; harder to interpret | (Advanced models) |
| **Magnitude + phase** | Separable processing | Phase estimation is hard to learn | Hybrid approaches |

**Our design:**
- **U-Net baseline**: Learns magnitude masks, reuses mixture phase for reconstruction (classical approach)
- **Conv-TasNet**: Bypasses STFT entirely with learned encoder/decoder (end-to-end)

### Reconstruction: Inverse STFT

```
x̂[n] = Σₜ IDFT{S(t,f)} · w[n - tH]  /  Σₜ w²[n - tH]
```

The overlap-add (OLA) method sums windowed IDFT frames. The Hann window + 75% overlap guarantees perfect reconstruction (COLA condition).

**Implementation**: `stft_utils::stft()` and `stft_utils::istft()` in `src/stft.h`.

---

## 4. Deep Learning Separation Models

### Architecture Comparison

| | **Conv-TasNet** (main) | **Spectrogram U-Net** (baseline) | SepFormer |
|---|---|---|---|
| **Domain** | Time (learned encoder) | STFT spectrogram | Time + attention |
| **Parameters** | ~5.1M | ~7.8M | ~26M |
| **SI-SNRi** (WSJ0-2mix) | 15.3 dB | 10-12 dB | 20.4 dB |
| **LibTorch feasibility** | ✅ Straightforward | ✅ Simple | ⚠️ Complex |
| **Explainability** | Good (encoder ≈ learned filter bank) | Excellent (STFT masks visible) | Hard |
| **Implementation effort** | Medium | Low | High |

### Main Model: Conv-TasNet

**Why Conv-TasNet**: Best balance of performance (~15 dB SI-SNRi), parameter efficiency (~5M), and implementation feasibility in LibTorch. The architecture has a clean mathematical story: learned filter bank → temporal convolutional masking → synthesis.

```
Mixture waveform x[t]
        ↓
    ┌──────────┐
    │  Encoder  │   1D Conv (1 → N, kernel L, stride L/2)
    │  (ReLU)   │   "Learned STFT" — each filter ≈ a frequency band
    └────┬─────┘
         ↓
    ┌──────────┐
    │   TCN     │   R × X stacked dilated depth-separable conv blocks
    │ Separator │   Exponentially growing receptive field
    │           │   Global + Channel LayerNorm
    └────┬─────┘
         ↓
    C masks (ReLU)  → element-wise multiply with encoder output
         ↓
    ┌──────────┐
    │  Decoder  │   Transposed 1D Conv (N → 1, kernel L, stride L/2)
    │           │   "Learned inverse STFT"
    └──────────┘
         ↓
    ŝ₁[t], ŝ₂[t]   separated sources
```

**Hyperparameters:**

| Symbol | Name | Value | Role |
|--------|------|-------|------|
| N | Encoder filters | 256 | Frequency resolution of learned filter bank |
| L | Encoder kernel | 16 | Temporal resolution (2 ms @ 8 kHz) |
| B | Bottleneck channels | 128 | Dimension reduction before TCN |
| H | Hidden channels | 256 | Capacity of each conv block |
| P | Depth-wise kernel | 3 | Local temporal context per block |
| X | Blocks per repeat | 8 | Dilation pattern: 1,2,4,...,128 |
| R | Repeats | 3 | Total receptive field: R × Σ 2ⁱ × P ≈ 1.5s |
| C | Speakers | 2 | Number of output masks |

**Key components:**
- **Global Layer Normalization (gLN)**: Normalizes across both channel and time dimensions. Critical for the bottleneck.
- **Channel Layer Normalization (cLN)**: Normalizes across channels only. Used inside conv blocks for causal-compatible normalization.
- **Depth-wise separable convolution**: Factorizes standard convolution into depth-wise (per-channel spatial) + point-wise (1×1 cross-channel). Reduces parameters by ~H/P×.

### Baseline: Spectrogram U-Net

**Why as baseline**: Makes the STFT pipeline explicit and interpretable. Demonstrates the classical masking approach, providing a strong comparison point.

```
Mixture magnitude spectrogram |X(t,f)|
        ↓
    ┌──────────┐
    │  Encoder  │   4× (Conv2d 3×3 + BN + ReLU + MaxPool2×2)
    │  32→64→   │   Progressive downsampling
    │  128→256  │
    └────┬─────┘
         ↓
    ┌──────────┐
    │Bottleneck │   Conv2d block at 512 channels
    └────┬─────┘
         ↓
    ┌──────────┐
    │  Decoder  │   4× (ConvTranspose2d 2×2 + skip cat + Conv2d + BN + ReLU)
    │  256→128→ │
    │  64→32    │
    └────┬─────┘
         ↓
    Conv2d 1×1 → sigmoid
         ↓
    C masks Mₖ(t,f) ∈ [0,1]
         ↓
    ŝₖ = iSTFT( Mₖ · |X| · e^{j∠X} )
```

**Reconstruction**: Apply each mask to the mixture magnitude, combine with the mixture phase (phase is not learned — a known limitation), then inverse STFT.

---

## 5. Training Strategy

### Dataset

| | Details |
|---|---|
| **Dataset** | Libri2Mix (derived from LibriSpeech) |
| **Split** | train-360 (~13,900 mixtures), dev (~3,000), test (~3,000) |
| **Sample rate** | 8 kHz (standard for separation benchmarks) |
| **Speakers** | 2 overlapping speakers per mixture |
| **Mode** | `min` — mixtures trimmed to shortest source |
| **Generation** | `scripts/generate_librimix.sh` |

### Mixture Creation

Libri2Mix handles this automatically:
1. Sample two random utterances from different LibriSpeech speakers
2. Apply random relative SNR between speakers
3. Mix: `x[t] = s₁[t] + s₂[t]` (additive in time domain)
4. Store CSVs with paths to mixture, source 1, source 2

### Augmentations

Applied during training to improve generalization:

| Augmentation | Details | Applied to |
|---|---|---|
| **Random gain** | ±6 dB uniform | Mix + sources (consistently) |
| **Gaussian noise** | 20-40 dB SNR | Mixture only |
| **Circular shift** | ±10% of segment | Mix + sources |
| **Polarity flip** | 50% probability | Mix + sources |

### Loss Function

**SI-SNR (Scale-Invariant Signal-to-Noise Ratio):**

```
ŝ = s - mean(s),  ê = e - mean(e)

s_target = (<ŝ, ê> / ||ê||²) · ê

SI-SNR  = 10 · log₁₀( ||s_target||² / ||ŝ - s_target||² )
```

**Why SI-SNR over SDR**: SI-SNR is scale-invariant — it measures signal quality regardless of amplitude. This aligns with human perception and provides smoother gradients during training.

**PIT (Permutation Invariant Training):**

The network outputs [ŝ₁, ŝ₂] but doesn't know which source is which. PIT solves this:

```
L_PIT = -max( SI-SNR(ŝ₁,s₁) + SI-SNR(ŝ₂,s₂),
              SI-SNR(ŝ₁,s₂) + SI-SNR(ŝ₂,s₁) )
```

We evaluate both orderings and train with the one giving higher total SI-SNR.

**For U-Net baseline**: PIT loss on L1 spectrogram distance (spectral domain PIT).

### Optimizer & Schedule

| Parameter | Value |
|---|---|
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| Initial LR | 1 × 10⁻³ |
| LR schedule | Halve on plateau (patience=3 epochs on val SI-SNR) |
| Gradient clipping | Max norm 5.0 |
| Batch size | 4 (8 GB VRAM) or 8+ on servers |
| Segment length | 4 seconds (32,000 samples @ 8 kHz) |
| Epochs | 100 |

### Variable-Length Handling

During training, all segments are cropped/padded to fixed `segment_len` (32000 samples). During inference, the full waveform is processed (padded to encoder stride multiple, then trimmed back).

---

## 6. Evaluation and Results

### Objective Metrics

| Metric | What it measures | Range | Target |
|--------|-----------------|-------|--------|
| **SI-SNRi** | Scale-invariant SNR improvement (dB) | -∞ to +∞ | >12 dB |
| **SDRi** | Signal-to-distortion ratio improvement | -∞ to +∞ | >10 dB |
| **STOI** | Short-time objective intelligibility | 0 to 1 | >0.85 |
| **PESQ** | Perceptual quality (requires external tool) | 1 to 5 | >2.5 |

**SI-SNRi** is the primary metric: it measures how much the separation network improved the SNR relative to the input mixture.

### Figures for Report

| Figure | What to show | Why it impresses |
|--------|-------------|------------------|
| **Waveform comparison** | Mixture → separated sources overlaid with ground truth | Visual proof of separation quality |
| **Spectrogram before/after** | STFT magnitude of mix, separated, and clean | Shows T-F masking in action |
| **Training loss curve** | Negative SI-SNR vs. epoch for both models | Shows convergence and training stability |
| **Metric comparison table** | SI-SNRi, SDRi, STOI for Conv-TasNet vs. U-Net | Quantitative advantage of the main model |
| **Encoder filter visualization** | Conv-TasNet encoder weights plotted as frequency responses | Proves the learned encoder ≈ filter bank |
| **Mask visualization** | TCN output masks across time | Shows how the network attends to each speaker |
| **Listening examples** | Before/after audio clips | The most convincing evidence of quality |

### Example Results Table (expected)

| Model | SI-SNRi (dB) | SDRi (dB) | STOI | Params |
|-------|-------------|-----------|------|--------|
| Mixture (input) | 0.0 | 0.0 | ~0.55 | — |
| Spectrogram U-Net | ~10.5 | ~10.0 | ~0.82 | 7.8M |
| **Conv-TasNet** | **~14.5** | **~14.0** | **~0.92** | 5.1M |

### Experiments Shortlist

1. **Main result**: Train Conv-TasNet 100 epochs, report test SI-SNRi
2. **Baseline comparison**: Train U-Net same epochs, compare
3. **Ablation — encoder kernel**: L=8 vs 16 vs 32 (temporal resolution)
4. **Ablation — repeats**: R=1 vs 2 vs 3 (receptive field)
5. **Augmentation effect**: With vs. without augmentation
6. **Generalization**: Evaluate on unseen noise conditions

---

## 7. Implementation Guide

### Project Structure

```
prml-project/
├── CMakeLists.txt               # Build system
├── README.md                    # This file
├── .gitignore
├── scripts/
│   └── generate_librimix.sh     # Dataset generation
├── src/
│   ├── audio_utils.h/.cpp       # WAV I/O (libsndfile)
│   ├── preprocessing.h/.cpp     # Normalize, VAD trim
│   ├── stft.h/.cpp              # STFT/iSTFT, magnitude/phase
│   ├── augmentation.h/.cpp      # Training augmentations
│   ├── dataset.h/.cpp           # LibriMix CSV dataset loader
│   ├── conv_tasnet.h/.cpp       # Main: Conv-TasNet
│   ├── unet.h/.cpp              # Baseline: Spectrogram U-Net
│   ├── losses.h/.cpp            # SI-SNR, PIT, spectral loss
│   ├── metrics.h/.cpp           # SI-SNRi, SDRi, STOI
│   ├── train.cpp                # Training entry point
│   └── inference.cpp            # Inference + evaluation
├── libtorch/                    # LibTorch (gitignored)
├── checkpoints/                 # Saved models
└── output/                      # Separated audio files
```

### Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| **LibTorch** | 2.6.0+cu124 | Neural network framework |
| **libsndfile** | ≥ 1.0.31 | WAV file I/O |
| **CMake** | ≥ 3.18 | Build system |
| **g++** | ≥ 11 | C++17 compiler |
| **CUDA** | ≥ 12.4 | GPU acceleration |

### Build

```bash
# 1. Download LibTorch (if not present)
wget -q https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-LATEST.zip
unzip -q libtorch-cxx11-abi-shared-LATEST.zip -d .

# 2. Install libsndfile
sudo apt install libsndfile1-dev

# 3. Build
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=../libtorch ..
make -j$(nproc)
```

### Smoke Test

```bash
# Validates model architecture, STFT roundtrip, loss functions, metrics
./build/train --smoke_test
./build/inference --smoke_test
```

### Training

```bash
# Conv-TasNet (main model)
./build/train \
    --model tasnet \
    --data_dir ./data/Libri2Mix/wav8k/min/train-360 \
    --val_dir  ./data/Libri2Mix/wav8k/min/dev \
    --epochs 100 --batch_size 4 --lr 1e-3

# Spectrogram U-Net (baseline)
./build/train \
    --model unet \
    --data_dir ./data/Libri2Mix/wav8k/min/train-360 \
    --val_dir  ./data/Libri2Mix/wav8k/min/dev \
    --epochs 100 --batch_size 4 --lr 1e-3
```

### Inference

```bash
# Separate a mixture
./build/inference \
    --model tasnet \
    --checkpoint checkpoints/best_tasnet.pt \
    --input test_mixture.wav

# With evaluation against ground truth
./build/inference \
    --model tasnet \
    --checkpoint checkpoints/best_tasnet.pt \
    --input test_mixture.wav \
    --ref_s1 clean_speaker1.wav \
    --ref_s2 clean_speaker2.wav
```

### Dataset Generation

```bash
bash scripts/generate_librimix.sh ./data
# Requires ~100 GB disk, Python 3, and sox
```

### Checkpoint Management

- `best_tasnet.pt` / `best_unet.pt` — best validation SI-SNR
- `tasnet_epN.pt` / `unet_epN.pt` — every 10 epochs
- `final_tasnet.pt` / `final_unet.pt` — end of training

---

## 8. Project Narrative for Reports

### Main Scientific Idea

We combine **signal processing** (STFT, time-frequency analysis) with **deep learning** (temporal convolutional networks) to solve the cocktail party problem. The project demonstrates that:

1. Classical Fourier analysis provides interpretable time-frequency representations
2. Learned representations (Conv-TasNet encoder) outperform hand-designed ones (STFT)
3. Permutation-invariant training solves the label ambiguity problem elegantly

### How to Frame in a PRML Report

**Section 1 — Introduction**: The cocktail party problem. Why single-channel separation is fundamentally ill-posed. How can we solve an underdetermined system?

**Section 2 — Signal Processing Foundation**: STFT theory, windowing, overlap-add reconstruction. Why non-stationarity of speech requires time-frequency analysis. This establishes the mathematical foundation.

**Section 3 — Masking-Based Separation**: How T-F masks can separate speakers. The ideal binary mask (IBM), ideal ratio mask (IRM), and neural mask estimation. This connects STFT to the supervised learning problem.

**Section 4 — Conv-TasNet Architecture**: The key insight: replace STFT with a **learned encoder** (1D convolution). Show that the encoder learns filter bank-like representations. TCN for temporal modeling with exponentially growing receptive field.

**Section 5 — Training**: PIT loss derivation (combinatorial optimization over permutations). SI-SNR as a scale-invariant objective. Connection to maximum likelihood estimation.

**Section 6 — Experiments**: Quantitative comparison (Conv-TasNet vs. U-Net), ablations, visualizations.

**Mathematical highlights to include:**
- STFT as a windowed inner product: `X(t,f) = ⟨x · w_t, e_f⟩`
- SI-SNR derivation from projection operators
- PIT as minimization over the symmetric group S_C
- Depth-separable convolution as a low-rank approximation of standard convolution

### Baseline Justification

The U-Net provides a fair comparison because:
1. Same dataset, same loss, same training
2. Uses classical STFT → demonstrates that learned representations outperform hand-crafted ones
3. Interpretable masks that can be visualized

### What Makes This Project Impressive

1. **End-to-end C++ implementation** — real engineering, not just notebook code
2. **Dual architecture** — shows depth of understanding (learned vs. hand-crafted features)
3. **Full evaluation pipeline** — SI-SNR, SDR, STOI, spectrograms, audio demos
4. **Signal processing + ML** — bridges the gap between PRML theory and practice
5. **Production-quality code** — modular, documented, testable

### Presentation Outline

1. **Problem demo** (30s): Play mixed audio, then separated. Hook the audience.
2. **Background** (2 min): Speech as a signal. STFT. Why separation is hard.
3. **Our approach** (3 min): Conv-TasNet architecture. Learned encoder vs. STFT.
4. **Training** (2 min): PIT loss, SI-SNR, dataset.
5. **Results** (3 min): Comparison table, spectrograms, loss curves.
6. **Demo** (1 min): Live separation on unseen audio.
7. **Conclusion** (30s): Learned > hand-crafted. Future work.

---

## 9. Quick Start

```bash
# Clone and build
git clone <repo-url> prml-project && cd prml-project
# Ensure libtorch/ exists (download if needed)
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=../libtorch .. && make -j$(nproc)

# Smoke test
./train --smoke_test
./inference --smoke_test

# Generate dataset (requires ~100 GB)
cd .. && bash scripts/generate_librimix.sh ./data

# Train Conv-TasNet
./build/train --model tasnet \
    --data_dir ./data/Libri2Mix/wav8k/min/train-360 \
    --val_dir  ./data/Libri2Mix/wav8k/min/dev

# Separate audio
./build/inference --model tasnet \
    --checkpoint checkpoints/best_tasnet.pt \
    --input my_mixture.wav
```

---

## 10. References

1. Luo, Y. & Mesgarani, N. (2019). *Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation*. IEEE/ACM TASLP.
2. Hershey, J.R. et al. (2016). *Deep Clustering: Discriminative Embeddings for Segmentation and Separation*. ICASSP.
3. Yu, D. et al. (2017). *Permutation Invariant Training of Deep Models for Speaker-Independent Multi-Talker Speech Separation*. ICASSP.
4. Le Roux, J. et al. (2019). *SDR – Half-Baked or Well Done?*. ICASSP.
5. Cosentino, J. et al. (2020). *LibriMix: An Open-Source Dataset for Generalizable Speech Separation*. arXiv.
6. Ronneberger, O. et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI.
7. Subakan, C. et al. (2021). *Attention is All You Need in Speech Separation*. ICASSP (SepFormer).

---

**License**: MIT
