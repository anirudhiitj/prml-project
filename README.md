# Cocktail Party Problem — Speech Separation

> **PRML Final Project** | Single-channel (monaural) speech separation — isolating individual voices from a mixed audio signal using deep learning.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Project Structure](#project-structure)
3. [Models](#models)
4. [Architecture Overview](#architecture-overview)
5. [Real-World Applications](#real-world-applications)
6. [Setup & Installation](#setup--installation)
7. [Running the Application](#running-the-application)
8. [API Reference](#api-reference)
9. [Team](#team)

---

## Problem Statement

Given a single mixed audio signal:

$$x(t) = s_1(t) + s_2(t) + \cdots + s_n(t)$$

the goal is to recover each individual speaker signal $\hat{s}_i(t)$ using only the observed mixture — with no prior knowledge of the speakers or room conditions. This is the classic **cocktail party problem**, a fundamental challenge in blind source separation (BSS).

---

## Project Structure

```
prml-project/
├── app.py                  # Unified FastAPI backend — routes to all models
├── requirements.txt        # Project dependencies
├── frontend/
│   └── index.html          # Web UI for audio upload & playback
├── anirudh/
│   └── cocktail_separation/ # DPRNN model (Python / PyTorch)
├── mayank/                 # Conv-TasNet via Asteroid (Python / PyTorch)
├── mohit/                  # RCNN model — Conv2D + BiLSTM (Python / PyTorch)
└── gokul/                  # Conv-TasNet in C++ / LibTorch
```

---

## Models

Four independent model implementations, all accessible through a single unified API endpoint:

| Model Key        | Contributor | Architecture           | Domain                | Framework        |
| ---------------- | ----------- | ---------------------- | --------------------- | ---------------- |
| `dprnn`          | Anirudh     | Dual-Path RNN          | Waveform (end-to-end) | Python / PyTorch |
| `convtasnet`     | Mayank      | Conv-TasNet (Asteroid) | Waveform (end-to-end) | Python / PyTorch |
| `rcnn`           | Mohit       | RCNN — Conv2D + BiLSTM | Time-Frequency (STFT) | Python / PyTorch |
| `convtasnet_cpp` | Gokul       | Conv-TasNet            | Waveform (end-to-end) | C++ / LibTorch   |

### Model Descriptions

---

#### DPRNN — Dual-Path RNN `(Anirudh)`

**Concept**
DPRNN splits a long audio sequence into overlapping chunks and applies two RNN passes alternately — one **intra-chunk** (local, short-range patterns) and one **inter-chunk** (global, long-range context). This avoids the quadratic cost of processing the entire sequence at once, making it practical for long audio. The model operates end-to-end on raw waveforms using a learnable encoder/decoder.

**Key idea:** _segment → local RNN → global RNN → repeat → reconstruct_. Separation is learned via Permutation Invariant Training (PIT) so the model does not need pre-assigned speaker labels.

**Achievement level**

- Full training pipeline on LibriSpeech 2-speaker mixtures
- End-to-end inference via REST API
- Evaluation with SI-SNRi and SDR metrics

---

#### Conv-TasNet — Convolutional Time-domain Audio Separation `(Mayank)`

**Concept**
Conv-TasNet replaces the traditional STFT with a **learnable 1D convolutional filterbank** (encoder) that maps raw waveform into a latent representation optimised for separation. A stack of dilated depthwise-separable convolutions (TCN) then estimates per-source masks in that latent space. The decoder (transposed conv) converts masked representations back to waveforms. No hand-crafted features — fully data-driven.

**Key idea:** _learnable filterbank → TCN mask estimator → inverse filterbank_. Implemented via [Asteroid](https://github.com/asteroid-team/asteroid), a production-grade speech separation library.

**Achievement level**

- Pre-trained ConvTasNet weights loaded for inference
- 2-speaker separation on 8 kHz speech
- Integrated into the unified API endpoint

---

#### RCNN — Conv2D + BiLSTM `(Mohit)`

**Concept**
This approach works in the **time-frequency (T-F) domain**. The mixture waveform is converted to a spectrogram via STFT. A 2D convolutional encoder (like a miniature image feature extractor) captures local spectral patterns. A Bidirectional LSTM then models the sequential (temporal) structure across frames in both directions. A transposed-conv decoder produces one soft **mask per source**, which is multiplied with the mixture spectrogram; the Griffith-Lim algorithm (or phase copying) converts the masked magnitude back to audio.

**Key idea:** _treat spectrogram as image → CNN features → BiLSTM temporal context → mask per speaker_.

**Achievement level**

- Full model architecture implemented and trained
- Mask-based separation working on 2-speaker mixtures
- Inference and REST API integration complete

---

#### Conv-TasNet C++ / LibTorch `(Gokul)`

**Concept**
A faithful C++ reimplementation of Conv-TasNet using **LibTorch** (the C++ frontend of PyTorch). The same learnable filterbank → TCN separator → inverse filterbank pipeline runs natively without the Python interpreter. This demonstrates that trained PyTorch models can be exported (TorchScript) and deployed in resource-constrained or latency-critical environments — embedded devices, real-time applications, server inference without Python overhead.

**Key idea:** _same Conv-TasNet architecture, zero Python runtime — pure C++ inference_.

**Achievement level**

- Full C++ implementation of encoder, TCN separator, and decoder
- CMake build system for cross-platform compilation
- LibTorch-based inference pipeline with REST API wrapper

---

## Real-World Applications

Speech separation is a foundational capability that enables dozens of real-world technologies:

| Domain                                 | Application                     | How separation helps                                                                 |
| -------------------------------------- | ------------------------------- | ------------------------------------------------------------------------------------ |
| **Voice Assistants**                   | Alexa, Google Assistant, Siri   | Isolate the target speaker's command from TV/background speech                       |
| **Teleconferencing**                   | Zoom, Teams, Meet               | Clean up multi-speaker calls; remove crosstalk and background voices                 |
| **Hearing Aids**                       | Cochlear implants, smart aids   | Focus on the voice the user is facing; suppress room noise                           |
| **Automatic Speech Recognition (ASR)** | Transcription, subtitles        | Separate speakers before feeding audio to an ASR engine — improves WER significantly |
| **Speaker Diarisation**                | "Who spoke when?"               | Separation pre-processing makes it easier to assign speech segments to identities    |
| **Forensic Audio**                     | Law enforcement, legal evidence | Recover intelligible speech from degraded or overlapping recordings                  |
| **Music / Broadcast**                  | Podcast editing, live mixing    | Isolate vocals, instruments, or commentators from a mixed recording                  |
| **Surveillance & Safety**              | Emergency response              | Detect and isolate a distress call from ambient crowd noise                          |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Web Frontend (HTML)                  │
│          Upload audio  ·  Select model  ·  Playback     │
└──────────────────────────┬──────────────────────────────┘
                           │ POST /api/separate
                           ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI Backend  (app.py)                  │
│                                                         │
│   /api/separate?model=dprnn|convtasnet|rcnn|...         │
│       │                                                 │
│       ├──▶ anirudh/api.py   →  DPRNN                   │
│       ├──▶ mayank/api.py    →  Conv-TasNet (Asteroid)  │
│       ├──▶ mohit/api.py     →  RCNN                    │
│       └──▶ gokul/api.py     →  Conv-TasNet C++         │
└─────────────────────────────────────────────────────────┘
```

---

## API Reference

### `POST /api/separate`

Separates a mixed audio file using the selected model.

**Form fields**

| Field          | Type    | Default | Description                                                   |
| -------------- | ------- | ------- | ------------------------------------------------------------- |
| `audio`        | file    | —       | Mixed audio file (`.wav`)                                     |
| `model`        | string  | `dprnn` | Model to use: `dprnn`, `convtasnet`, `rcnn`, `convtasnet_cpp` |
| `num_speakers` | integer | `2`     | Number of speakers to separate                                |

**Response**

```json
{
  "files": ["path/to/speaker1.wav", "path/to/speaker2.wav"]
}
```

---
