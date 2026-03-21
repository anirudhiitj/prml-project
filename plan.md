# FULL BUILD PROMPT — DPRNN Cocktail Party Audio Separation System
### For GitHub Copilot / AI-assisted development
### Target: 5-speaker separation · 8×H200 clusters · PyTorch

---

## MISSION STATEMENT

Build a complete, production-grade, end-to-end deep learning system for
**blind audio source separation of up to 5 simultaneous speakers** from a
single-channel mixture recording (the "cocktail party problem").

The system uses **DPRNN (Dual-Path Recurrent Neural Network)** as the
separator module, wrapped inside the **TasNet encoder/decoder shell**
(time-domain, learned basis functions — no STFT, no spectrograms).

Everything must be implemented from scratch in PyTorch. No use of
asteroid, speechbrain, or other high-level separation libraries.
The codebase must be clean, modular, fully commented, and ready to
train across 8 NVIDIA H200 GPU clusters using PyTorch DDP.

---

## THEORETICAL BACKGROUND — read this before writing a single line

### Why time-domain and not spectrograms

Traditional speech separation worked on magnitude spectrograms: take the
STFT of the mixture, estimate a mask, apply it, reconstruct with the
original (noisy) phase via ISTFT. This has two fundamental problems.
First, the STFT is a fixed human-designed transform — it is not optimised
for separation. Second, discarding phase information and then trying to
reconstruct it causes audible artefacts regardless of how good the mask is.

TasNet solves both problems by replacing the STFT/ISTFT pair with a
**learned encoder/decoder pair** trained end-to-end with the separator.
The encoder is a 1D convolution that learns its own basis functions —
whatever linear projection of the raw waveform makes the separation task
easiest. The decoder is a transpose convolution that inverts exactly what
the encoder learned. Because they are trained jointly, the decoder always
knows how to perfectly reconstruct from the encoded representation.
Phase is never discarded because we never leave the time domain.

### Why DPRNN and not Conv-TasNet (TCN)

Conv-TasNet uses a Temporal Convolution Network (TCN) as its separator.
The TCN is a stack of dilated depthwise-separable 1D convolutions with
exponentially increasing dilation: 1, 2, 4, 8, 16, 32, 64, 128, repeated
across 3 stacks. The receptive field grows exponentially, reaching ~8
seconds of audio context. The TCN processes all time steps in parallel —
it is a purely spatial operation with no hidden state.

The TCN has a hard structural ceiling for 5-speaker separation. It has
no memory — every time step is processed with the same fixed window of
context, symmetrically in both directions. When 5 speakers are active
simultaneously and two of them have similar short-term spectral patterns,
the only way to resolve the assignment correctly is to remember their
longer acoustic trajectories. The TCN cannot do this. Its mask estimates
become diffuse and interfering in 5-speaker fully-overlapping segments.

DPRNN solves this with a hierarchical recurrent structure. It splits
the encoded feature sequence into overlapping chunks, then applies two
separate BiLSTM modules: one that runs within each chunk (local, no
memory across chunks) and one that runs across chunks (global, maintains
hidden state across the entire utterance). The inter-chunk BiLSTM's
hidden state is the crucial mechanism — it accumulates a compressed
representation of each speaker's voice characteristics as the utterance
progresses, making mask estimation at later chunks informed by everything
heard so far. This is the speaker identity tracking that enables clean
5-speaker separation.

Empirically: Conv-TasNet reaches ~8–10 dB SI-SNR improvement on
5-speaker LibriSpeech mixtures. DPRNN reaches ~11–14 dB. That 3–5 dB gap
is perceptually the difference between barely intelligible and genuinely
useful separation.

### The masking paradigm

Both TasNet and DPRNN use soft masking in the encoded feature space.
The separator estimates C masks (one per speaker), each with values in
(0, 1), with the same shape as the encoder output (B, N, T').
Each mask is element-wise multiplied with the encoder output, producing
C separate feature streams. Each stream is then passed through the
decoder to produce a separated waveform.

The key insight: the masks are estimated in the learned encoded space,
not in raw waveform space or spectrogram space. The encoder is designed
(by training pressure) to produce a representation where masking
corresponds to meaningful source isolation in waveform space.

### SI-SNR loss

Scale-Invariant Signal-to-Noise Ratio is the training objective.
Given estimated source ŝ and true source s:

    s_target = (<ŝ, s> / <s, s>) * s          (project ŝ onto s)
    e_noise  = ŝ - s_target                    (residual error)
    SI-SNR   = 10 * log10(||s_target||² / ||e_noise||²)

Scale-invariant means the loss does not penalise amplitude differences —
only shape/content differences. This is correct for mask-based separation
because masks naturally lose absolute amplitude information.

Higher SI-SNR = better separation. We maximise it, so the training loss
is the negative mean SI-SNR.

### Permutation Invariant Training (PIT)

The model outputs C estimated sources in some arbitrary order.
The ground-truth sources are also in some arbitrary order.
We must match estimated sources to ground-truth sources optimally.

For 5 speakers there are 5! = 120 possible permutations.
PIT tries all permutations and picks the one that maximises the total
SI-SNR across all speaker pairs. Gradients are computed only through
the optimal permutation.

CRITICAL: use utterance-level PIT, not chunk-level PIT. Solve the
permutation once per utterance using the Hungarian algorithm over
full-utterance SI-SNR values, then apply that fixed permutation for
the entire utterance's gradient computation. Chunk-level PIT allows
the model to swap speaker assignments mid-utterance (since a different
permutation might be locally optimal), destroying temporal coherence.
With 5 speakers this is catastrophic. The Hungarian algorithm over
5 speakers is O(5³) = trivially fast.

---

## COMPLETE ARCHITECTURE SPECIFICATION

### Overview

```
Mixture waveform  (B, T)
        ↓
[ ENCODER ]          — shared with Conv-TasNet, identical
        ↓
(B, N, T')
        ↓
[ LayerNorm + 1×1 Conv ]    — project N→H
        ↓
(B, H, T')
        ↓
[ SEGMENTATION ]     — chunk into overlapping windows
        ↓
(B, H, K, S)         — K=chunk_size, S=num_chunks
        ↓
┌──────────────────────────────────┐
│  DPRNN SEPARATOR  (×6 blocks)   │
│                                  │
│  ┌─────────────────────────┐    │
│  │ Intra-chunk BiLSTM      │    │  ← local, within-chunk
│  │ LayerNorm + residual    │    │
│  └────────────┬────────────┘    │
│               ↓                  │
│  ┌─────────────────────────┐    │
│  │ Inter-chunk BiLSTM      │    │  ← global, across-chunk
│  │ LayerNorm + residual    │    │
│  └─────────────────────────┘    │
│                                  │
│  (repeat block ×6)               │
└──────────────────────────────────┘
        ↓
(B, H, K, S)
        ↓
[ OVERLAP-ADD ]      — reassemble (B, H, T')
        ↓
[ PReLU + 1×1 Conv × C ]   — C=5 mask heads
        ↓
[ Softmax across speakers ]
        ↓
mask₁ … mask₅   each (B, N, T')
        ↓
encoded_features × maskᵢ   (element-wise, per speaker)
        ↓
[ DECODER ]          — shared with Conv-TasNet, identical
        ↓
ŝ₁(t) … ŝ₅(t)   (B, C, T_out)
```

### Encoder — exact specification

```
Component:    nn.Conv1d
in_channels:  1
out_channels: N = 64            ← NOTE: DPRNN uses N=64, not 512
kernel_size:  L = 2
stride:       L = 2             ← non-overlapping in original DPRNN paper
padding:      0
bias:         False
activation:   nn.ReLU()

Input:   (B, T)         → unsqueeze(1) → (B, 1, T)
Output:  (B, 64, T')    where T' = T / L
```

Note: some implementations use 50% overlap (stride = L//2). This gives
smoother gradients but doubles T', increasing DPRNN memory cost. Use
non-overlapping (stride = L) for memory efficiency on 5-speaker runs.

### Bottleneck projection

```
nn.GroupNorm(1, N, eps=1e-8)    ← global layer norm over channels
nn.Conv1d(N, H, kernel_size=1)  ← H = 64 (DPRNN hidden dim)

Output: (B, H, T')
```

### Segmentation (chunking)

This is the operation unique to DPRNN. It converts the 3D feature tensor
into a 4D tensor by sliding a window of length K with 50% overlap.

```
Input:  (B, H, T')
Output: (B, H, K, S)

K = chunk_size (e.g. 100 frames)
S = number of chunks = ceil(T' / (K/2))

Procedure:
1. Pad T' dimension so it is divisible by K//2
2. Use unfold(dimension=2, size=K, step=K//2)
   → produces (B, H, S, K)
3. Permute to (B, H, K, S)
```

The 50% overlap ensures no information is lost at chunk boundaries.
The overlap-add step later recombines the processed chunks correctly.

### DPRNN block — exact specification

Each DPRNN block contains two sub-blocks in sequence.

**Intra-chunk sub-block:**
```
Input:  (B, H, K, S)

For each of the S chunks independently:
  - reshape chunk to (B*S, K, H)  ← treat each chunk as a batch item
  - nn.LSTM(H, H//2, bidirectional=True, batch_first=True)
    → output: (B*S, K, H)
  - reshape back to (B, H, K, S)
  - nn.GroupNorm(1, H)
  - residual add: output + input
```

**Inter-chunk sub-block:**
```
Input:  (B, H, K, S)

For each of the K positions within a chunk independently:
  - reshape to (B*K, S, H)  ← treat each position as a batch item
  - nn.LSTM(H, H//2, bidirectional=True, batch_first=True)
    → output: (B*K, S, H)
  - reshape back to (B, H, K, S)
  - nn.GroupNorm(1, H)
  - residual add: output + input
```

Stack 6 of these DPRNN blocks. Each block refines the separation.
Early blocks: coarse speaker assignment. Late blocks: fine-grained mask.

### Overlap-add reassembly

```
Input:  (B, H, K, S)   ← processed chunks
Output: (B, H, T')

Procedure:
1. Permute to (B, H, S, K)
2. For each chunk s, the output at position t contributes to
   the overlap-add at positions [s*step : s*step + K]
3. Divide by the overlap count at each position to normalize
   (positions covered by 2 chunks get divided by 2)
4. Trim padding added during segmentation
```

Implement this with torch.nn.functional.fold or manually with
torch operations — do NOT use a for loop over S for performance.

### Mask estimation head

```
nn.PReLU()
nn.Conv1d(H, N*C, kernel_size=1)   ← C=5 speakers, N=64 encoder filters

reshape: (B, N*C, T') → (B, C, N, T')

activation: F.softmax(dim=1)       ← softmax across the C=5 speakers
                                      enforces masks sum to 1 per position
```

The softmax constraint is important for 5 speakers. Sigmoid (per-speaker)
allows masks to sum to >1, causing energy leakage in heavily overlapping
regions. Softmax enforces a physical prior: the mixture energy at each
position came from exactly the C sources being separated, no more.

### Decoder — exact specification

```
nn.ConvTranspose1d(
    in_channels:  N = 64
    out_channels: 1
    kernel_size:  L = 2
    stride:       L = 2
    bias:         False
)

For each speaker i:
  masked = encoder_output × mask[:, i, :, :]   ← (B, N, T')
  waveform = decoder(masked)                    ← (B, 1, T_out)
  sources[i] = waveform.squeeze(1)              ← (B, T_out)

Return: torch.stack(sources, dim=1)             ← (B, C, T_out)
```

### Full hyperparameter table

```
N (encoder filters):         64
L (encoder kernel/stride):   2
H (DPRNN hidden channels):   64
K (chunk size):              100
DPRNN blocks:                6
C (number of speakers):      5
BiLSTM hidden per direction: H//2 = 32  (concat → H=64)
Dropout (LSTM):              0.0  (dropout hurts separation tasks)
```

---

## DATA PIPELINE — complete specification

### Source datasets

Download in this priority order:
1. LibriSpeech train-clean-360  (360h, 921 speakers)  ← primary
2. LibriSpeech train-clean-100  (100h, 251 speakers)  ← supplement
3. VoxCeleb2 dev set            (~2000h, 5994 speakers) ← diversity
4. VCTK Corpus                  (44h, 110 speakers, multiple accents)

All audio must be resampled to 16000 Hz mono before mixture generation.

### Mixture generation — exact procedure

```
SAMPLE_RATE    = 16000
CLIP_DURATION  = 4       seconds
CLIP_SAMPLES   = 64000   samples
N_SPEAKERS     = 5
SNR_RANGE      = (-5, 5) dB   relative between speakers
SPLITS:
  train: 100,000 mixtures
  val:   5,000   mixtures
  test:  3,000   mixtures
```

For each mixture:
1. Sample N_SPEAKERS distinct speakers (never reuse a speaker in one mix)
2. For each speaker, randomly select one utterance
3. Randomly crop or pad to exactly CLIP_SAMPLES samples
4. Apply random gain scaling per speaker:
   - anchor speaker 0 at gain = 1.0
   - for speakers 1–4: draw SNR_dB ~ Uniform(-5, 5)
     gain_i = 10^(SNR_dB / 20)
5. Sum all scaled sources: mixture = Σ gain_i * source_i
6. Peak-normalize mixture: mixture /= max(abs(mixture)) + 1e-8
   Apply same normalization factor to all individual sources
7. Save: mixture.wav, s1.wav, s2.wav, s3.wav, s4.wav, s5.wav

CRITICAL: speaker identity must never repeat within a single mixture.
Track which speakers are used across the training split — each pair of
speakers should appear together roughly equally. Do not generate mixtures
from only the most common speakers.

### Augmentation pipeline (apply DURING training, not pre-generation)

Apply these transforms on-the-fly in the DataLoader workers:

**Room Impulse Response (RIR) convolution — MANDATORY for real-world perf:**
Use pyroomacoustics to generate synthetic RIRs on the fly.
Room dimensions: Uniform(3m, 10m) per axis, height Uniform(2.5m, 4.5m)
Absorption coefficient: Uniform(0.1, 0.5) (0.1=reverberant, 0.5=dry)
Apply BEFORE mixing. Each source gets its own independent RIR.
Apply with probability 0.7 during training.

**Additive noise:**
Noise type: white Gaussian noise
SNR: Uniform(20, 40) dB (low noise — don't overwhelm speech signal)
Apply to the mixture AFTER mixing.
Apply with probability 0.5 during training.

**Speed perturbation:**
Rate: Uniform(0.9, 1.1)
Use torchaudio.functional.resample to implement.
Resample source back to 16000 Hz after perturbation.
Apply with probability 0.3.

### Dataset class specification

```python
class MixtureDataset(torch.utils.data.Dataset):

    __getitem__ returns:
        mixture:  torch.FloatTensor of shape (T,)      ← 64000 samples
        sources:  torch.FloatTensor of shape (C, T)    ← 5 × 64000

    __len__ returns number of mixtures in split

    Paths:
        root/train/0000001/mixture.wav
        root/train/0000001/s1.wav  ...  s5.wav
        root/val/...
        root/test/...
```

Use torchaudio.load for all audio loading. Do NOT use librosa in the
training loop — it is too slow. librosa is acceptable for preprocessing.

### DataLoader configuration

```
train DataLoader:
    batch_size:    8 per GPU (64 total across 8 GPUs)
    num_workers:   8 per GPU
    pin_memory:    True
    persistent_workers: True
    prefetch_factor: 4
    shuffle:       True

val/test DataLoader:
    batch_size:    4 per GPU
    shuffle:       False
```

---

## LOSS FUNCTION — complete implementation

### SI-SNR

```python
def si_snr(estimated, target, eps=1e-8):
    """
    estimated: (B, T) — one speaker's estimated waveform
    target:    (B, T) — one speaker's ground truth waveform
    returns:   (B,)   — SI-SNR value per sample (higher = better)
    """
    # Zero-mean both signals
    estimated = estimated - estimated.mean(dim=-1, keepdim=True)
    target    = target    - target.mean(dim=-1, keepdim=True)

    # Project estimated onto target
    dot       = (estimated * target).sum(dim=-1, keepdim=True)
    norm_sq   = (target * target).sum(dim=-1, keepdim=True) + eps
    s_target  = (dot / norm_sq) * target

    # Noise is the residual
    e_noise   = estimated - s_target

    # SI-SNR in dB
    ratio = (s_target * s_target).sum(dim=-1) / \
            ((e_noise * e_noise).sum(dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)
```

### Utterance-level PIT with Hungarian algorithm

```python
from scipy.optimize import linear_sum_assignment

def pit_loss(estimated_sources, true_sources, eps=1e-8):
    """
    estimated_sources: (B, C, T) — model output
    true_sources:      (B, C, T) — ground truth
    C = 5 speakers

    Returns: scalar loss (negative mean SI-SNR, minimised during training)
    """
    B, C, T = estimated_sources.shape
    total_loss = 0.0

    for b in range(B):
        # Build C×C cost matrix of SI-SNR values
        cost_matrix = torch.zeros(C, C)
        for i in range(C):
            for j in range(C):
                cost_matrix[i, j] = si_snr(
                    estimated_sources[b, i].unsqueeze(0),
                    true_sources[b, j].unsqueeze(0)
                )

        # Hungarian algorithm: maximise SI-SNR = minimise negative SI-SNR
        cost_np = -cost_matrix.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)

        # Accumulate loss using the optimal permutation
        sample_loss = 0.0
        for r, c in zip(row_ind, col_ind):
            sample_loss += si_snr(
                estimated_sources[b, r].unsqueeze(0),
                true_sources[b, c].unsqueeze(0)
            )
        total_loss += sample_loss / C

    return -(total_loss / B)   ← negative because we minimise
```

### Optional secondary loss

Add a small weight (0.1) of standard SNR loss alongside SI-SNR.
This prevents the model from normalising all outputs to equal amplitudes
when the true sources have different loudness levels.

```
total_loss = si_snr_pit_loss + 0.1 * snr_pit_loss
```

---

## TRAINING INFRASTRUCTURE — complete specification

### Distributed training setup (8×H200)

Use PyTorch DistributedDataParallel (DDP) with NCCL backend.
Do NOT use DataParallel — it is deprecated and slower.

```
Launch command (from master node):
torchrun \
    --nproc_per_node=8 \
    --nnodes=<number_of_machines> \
    --node_rank=<this_machine_rank> \
    --master_addr=<master_ip> \
    --master_port=29500 \
    train.py --config configs/5spk.yaml
```

Each process handles its own GPU. The model is wrapped in DDP after
moving to device. Gradients are automatically averaged across GPUs.

Use DistributedSampler for the training DataLoader to ensure
each GPU sees a non-overlapping subset of the data each epoch.

### Optimiser and scheduler

```
Optimiser: Adam
    lr:           1e-3
    betas:        (0.9, 0.999)
    weight_decay: 1e-5

Scheduler: ReduceLROnPlateau
    mode:         max           ← we maximise validation SI-SNR
    factor:       0.5
    patience:     5             ← epochs with no improvement
    min_lr:       1e-6

Gradient clipping:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    Apply before every optimiser step.
```

### Training curriculum — MANDATORY for 5-speaker success

Do NOT start training directly on 5-speaker mixtures from random
initialisation. The 5-speaker loss landscape has deep local minima
corresponding to partial solutions. Random init will land in them and
gradient descent cannot escape.

```
Phase 1: 2-speaker separation
    Dataset: regenerate 80k train / 4k val / 2k test with C=2
    Model:   C=2 in mask head (everything else identical)
    Train until: val SI-SNR improvement > 15 dB  (~50–80 epochs)
    Checkpoint: save as phase1_best.pt

Phase 2: 3-speaker separation
    Dataset: regenerate with C=3
    Init:    load phase1_best.pt, reinitialise mask head for C=3
    LR:      1e-3 × 0.2 = 2e-4  (reduced from phase 1 LR)
    Train until: val SI-SNR improvement > 12 dB  (~30–50 epochs)
    Checkpoint: save as phase2_best.pt

Phase 3: 4-speaker separation
    Dataset: regenerate with C=4
    Init:    load phase2_best.pt, reinitialise mask head for C=4
    LR:      1e-4
    Train until: val SI-SNR improvement > 10 dB  (~30 epochs)
    Checkpoint: save as phase3_best.pt

Phase 4: 5-speaker separation  ← final target
    Dataset: full 100k/5k/3k with C=5
    Init:    load phase3_best.pt, reinitialise mask head for C=5
    LR:      5e-5
    Train until: convergence or 100 epochs
```

WHY this works: the encoder learns genuine speech representations during
phase 1/2. The intra-chunk BiLSTM learns local acoustic modeling. The
inter-chunk BiLSTM learns to track speaker trajectories. When you extend
to more speakers, only the combinatorial complexity increases — the core
acoustic modeling is already learned and provides a strong initialisation.

### Checkpointing

Save checkpoints at:
- Every epoch end (overwrite latest.pt)
- Every time validation SI-SNR improves (best.pt)
- Every 10 epochs (epoch_N.pt, keep last 3)

Checkpoint contents:
```python
{
    'epoch':           int,
    'model_state':     model.module.state_dict(),  ← unwrap DDP
    'optim_state':     optimizer.state_dict(),
    'sched_state':     scheduler.state_dict(),
    'val_sisnr':       float,
    'config':          dict,
}
```

### Logging

Use Weights & Biases (wandb). Log every step:
- train/loss
- train/si_snr
- val/loss (every epoch)
- val/si_snr (every epoch)
- learning_rate
- grad_norm

Log audio samples every 10 epochs:
- mixture.wav
- estimated_s1.wav … estimated_s5.wav
- true_s1.wav … true_s5.wav
(Pick 3 random samples from the validation set)

---

## EVALUATION — complete specification

Implement ALL three metrics. Report all three.

### 1. SI-SNR improvement (SI-SNRi)

```
SI-SNRi = SI-SNR(estimated, true) - SI-SNR(mixture, true)
```

This measures improvement over the unprocessed mixture.
Target on 5-speaker LibriSpeech: >10 dB

### 2. Signal-to-Distortion Ratio improvement (SDRi)

Use mir_eval.separation.bss_eval_sources.
SDR is the standard metric in the BSS Eval toolkit, widely reported
in the literature. Required for comparison with published results.

### 3. PESQ (Perceptual Evaluation of Speech Quality)

Use pesq library (pip install pesq).
PESQ ranges from -0.5 to 4.5 (higher = better speech quality).
Use wideband mode (wb) with 16kHz audio.
Target on clean LibriSpeech: >2.5

### Evaluation procedure

```
For each test mixture:
1. Run model.eval() with torch.no_grad()
2. Get estimated sources (B, C, T)
3. Solve optimal permutation with Hungarian PIT
4. Compute SI-SNRi, SDRi, PESQ for each speaker
5. Average across all speakers and all test mixtures
6. Report: mean ± std for each metric
```

---

## FILE STRUCTURE — implement exactly this layout

```
cocktail_separation/
│
├── configs/
│   ├── base.yaml           ← shared hyperparameters
│   ├── 2spk.yaml           ← phase 1 overrides
│   ├── 3spk.yaml           ← phase 2 overrides
│   ├── 4spk.yaml           ← phase 3 overrides
│   └── 5spk.yaml           ← phase 4, final config
│
├── data/
│   ├── raw/                ← downloaded source corpora
│   └── mixtures/
│       ├── 2spk/
│       ├── 3spk/
│       ├── 4spk/
│       └── 5spk/
│           ├── train/
│           ├── val/
│           └── test/
│
├── src/
│   ├── __init__.py
│   ├── encoder.py          ← Encoder class
│   ├── decoder.py          ← Decoder class
│   ├── dprnn_block.py      ← single DPRNN block (intra + inter)
│   ├── separator.py        ← full DPRNN separator (6 blocks + chunking)
│   ├── model.py            ← DPRNNTasNet (encoder + separator + decoder)
│   ├── losses.py           ← si_snr(), pit_loss()
│   └── dataset.py          ← MixtureDataset, build_dataloader()
│
├── scripts/
│   ├── download_data.sh
│   ├── generate_mixtures.py
│   └── prepare_rir.py
│
├── train.py                ← main training script (DDP-aware)
├── evaluate.py             ← full evaluation on test set
├── separate.py             ← inference script (single audio file)
│
├── requirements.txt
└── README.md
```

---

## REQUIREMENTS.TXT

```
torch>=2.3.0
torchaudio>=2.3.0
numpy>=1.24.0
soundfile>=0.12.1
librosa>=0.10.0
pyroomacoustics>=0.7.3
scipy>=1.11.0
mir_eval>=0.7
pesq>=0.0.4
wandb>=0.16.0
hydra-core>=1.3.0
einops>=0.7.0
tqdm>=4.66.0
```

---

## INFERENCE — separate.py specification

The inference script must:
1. Accept a single mixed audio file path as input
2. Load and resample to 16000 Hz mono if needed
3. Chunk long files into 4-second segments with 0.5-second overlap
4. Run DPRNN on each segment
5. Stitch segments back together with overlap-add
6. Write speaker_1.wav … speaker_5.wav to an output directory

Handle the edge case where the audio is shorter than 4 seconds:
zero-pad to 4 seconds, run model, trim output to original length.

Handle the edge case where N speakers < 5 are actually present:
output silence for unused speaker slots (the model will naturally
assign near-zero masks to speakers that are not present if trained
on variable-speaker mixtures — see optional extension below).

---

## OPTIONAL EXTENSIONS (implement after core system is working)

### Variable number of speakers

Real cocktail party audio does not always have exactly 5 speakers.
Extend the system to handle 1–5 speakers by:
1. Generating mixtures with variable C during training (random C each sample)
2. Adding a speaker count predictor head on top of the DPRNN features
3. Using the predicted C to zero out the extra mask outputs
This requires careful batching since C varies per sample.

### Speaker conditioning

If reference audio for a target speaker is available (e.g. "isolate
this person"), add a speaker encoder (d-vector or x-vector) that
encodes the reference and conditions the DPRNN via FiLM (Feature-wise
Linear Modulation). This converts the blind separation problem into a
target speaker extraction problem, which is both easier and more useful.

### Online / streaming inference

The current system requires the full utterance before processing (the
inter-chunk BiLSTM is bidirectional). For streaming use, replace the
inter-chunk BiLSTM with a unidirectional LSTM. This loses ~1–2 dB
SI-SNR but enables real-time processing with a latency of K frames.

---

## KNOWN PITFALLS — avoid these

1. **Chunk boundary artefacts**: if the overlap-add is implemented
   incorrectly, you will hear clicks at chunk boundaries in the output.
   Verify by running inference on a pure sine wave — output should also
   be a pure sine wave.

2. **Speaker permutation inconsistency**: if you accidentally use
   chunk-level PIT instead of utterance-level PIT, the model will
   produce speaker-swapped outputs. Verify by checking that speaker
   identity is consistent across a 4-second output file.

3. **Gradient explosion in BiLSTM**: LSTM gradients can spike during
   early training. The gradient clipping (max_norm=5.0) is mandatory,
   not optional. Monitor grad_norm in wandb — if it frequently hits
   the clip ceiling, reduce learning rate.

4. **Memory OOM with large batches**: the DPRNN is memory-hungry because
   the chunked (B, H, K, S) tensor can be large. On H200 (141GB) with
   batch_size=8, K=100, T'=32000 (4s clip with stride-2 encoder):
   S = ceil(32000 / 50) = 640 chunks. Tensor size = 8×64×100×640×4 bytes
   = ~1.3 GB per batch per DPRNN block. With 6 blocks and activations
   stored for backprop, peak memory is ~15–20 GB — well within H200.
   If you increase N or H significantly, recheck this calculation.

5. **Wrong normalisation in overlap-add**: divide by 2 at positions
   covered by two overlapping chunks, divide by 1 at the edges.
   Incorrect normalisation causes amplitude inconsistency across the
   output waveform.

6. **SI-SNR implementation bug**: the most common bug is forgetting to
   zero-mean both signals before computing the projection. Always subtract
   the mean of each signal over the time dimension before any SI-SNR
   computation.

7. **Training directly on 5 speakers**: do not skip the curriculum.
   Without the 2→3→4→5 speaker curriculum, the model will consistently
   converge to a local minimum where it separates 1–2 dominant speakers
   and produces noise for the rest.

---

## EXPECTED RESULTS (benchmarks to validate against)

After full training through the curriculum on LibriSpeech mixtures:

```
2-speaker test set:
  SI-SNRi: ~15.3 dB
  SDRi:    ~15.6 dB

3-speaker test set:
  SI-SNRi: ~12.8 dB
  SDRi:    ~13.1 dB

5-speaker test set:
  SI-SNRi: ~10.5–12.0 dB
  SDRi:    ~10.8–12.3 dB
```

If your results are significantly below these numbers, check in this order:
1. SI-SNR implementation (zero-mean? correct projection formula?)
2. PIT implementation (utterance-level? Hungarian algorithm?)
3. Data pipeline (correct SNR normalisation? speaker diversity?)
4. Training curriculum (did you skip straight to 5 speakers?)
5. Architecture (chunk size K? number of DPRNN blocks?)

---

*End of prompt. Build this system completely and correctly.*
*Every component described above must be implemented.*
*No shortcuts, no placeholder functions, no TODOs left in final code.*