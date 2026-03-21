# DPRNN-TasNet — Audio Source Separation

A PyTorch implementation of **Dual-Path RNN (DPRNN)** for the **Cocktail Party Problem** — separating mixed audio into individual speaker sources.

## Architecture

```
Raw Audio → Encoder (Conv1D) → Chunking → DPRNN Blocks (Intra + Inter RNN) → Mask Estimation → Decoder (ConvTranspose1D) → Separated Audio
```

## Project Structure

```
├── configs/default.yaml       # Hyperparameters
├── data/dataset.py            # Dataset loaders
├── models/
│   ├── encoder.py             # 1-D Conv encoder
│   ├── decoder.py             # 1-D Transposed Conv decoder
│   ├── dprnn_block.py         # Intra/Inter-chunk RNN block
│   ├── dprnn.py               # DPRNN separator
│   └── dprnn_tasnet.py        # Top-level model
├── losses/pit_loss.py         # SI-SNR + PIT loss
├── utils/
│   ├── audio_utils.py         # Audio I/O
│   └── metrics.py             # SI-SNRi, SDRi
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
└── inference.py               # Single-file separation
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Prepare your data in the following structure (LibriMix-style):

```
data/train/
├── mix/    # Mixture audio files
├── s1/     # Source 1 audio files
└── s2/     # Source 2 audio files
```

## Training

```bash
python train.py                              # Train with defaults
python train.py --config configs/custom.yaml # Custom config
python train.py --overfit_one_batch --epochs 50  # Sanity check
```

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

## Inference

```bash
python inference.py --input mix.wav --checkpoint checkpoints/best_model.pth --output_dir results/
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `encoder_dim` (N) | 64 | Encoder output channels |
| `encoder_kernel` (L) | 2 | Conv1d kernel size |
| `chunk_size` (K) | 250 | Frames per chunk |
| `hidden_size` (H) | 128 | RNN hidden dim |
| `num_dprnn_blocks` (B) | 6 | Stacked DPRNN blocks |
| `num_sources` (C) | 2 | Number of speakers |