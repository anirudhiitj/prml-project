# RCNN Cocktail Party — Project Status

## ✅ What Is Implemented

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| **RCNN Model** | `model.py` | ✅ Done | Conv2D encoder → BiLSTM → TransposedConv decoder → sigmoid masks. 13.1M params. Forward pass verified. |
| **Utilities** | `utils.py` | ✅ Done | STFT/iSTFT helpers, SI-SNR metric, PIT loss, audio I/O (load/save/resample/normalize). |
| **Dataset** | `dataset.py` | ✅ Done | On-the-fly LibriSpeech 2-speaker mixture generation + pre-generated `.pt` file loader. |
| **Data Prep** | `prepare_data.py` | ✅ Done | Downloads LibriSpeech (`train-clean-100`, `dev-clean`), generates 5000 train + 500 val mixtures. |
| **Training** | `train.py` | ✅ Done | SI-SNR + PIT loss, AMP (FP16), gradient clipping, LR scheduling, TensorBoard, checkpointing. |
| **Inference** | `inference.py` | ✅ Done | Overlap-add segmented processing for long audio, waveform + spectrogram visualization. |
| **Dependencies** | `requirements.txt` | ✅ Done | torch, torchaudio, numpy, scipy, soundfile, tensorboard, tqdm, matplotlib. |
| **Documentation** | `documentation.md` | ✅ Done | Full concept explanation, architecture diagram, pipeline flow, metrics, references. |

## ❌ What Remains To Be Done

| Step | Command | Time Estimate |
|------|---------|---------------|
| **1. Download & generate dataset** | `CUDA_VISIBLE_DEVICES=3 python prepare_data.py --data_root ./data --num_train 5000 --num_val 500` | ~15–30 min (first-time download of LibriSpeech ~6 GB) |
| **2. Train the model** | `CUDA_VISIBLE_DEVICES=3 python train.py --data_dir ./data/generated --epochs 50 --batch_size 8` | ~1–3 hours (GPU 3 — H200) |
| **3. Run inference** | `CUDA_VISIBLE_DEVICES=3 python inference.py --input <your_audio.wav> --checkpoint ./checkpoints/best_model.pt` | ~5 sec per file |

## Pipeline Summary

```
LibriSpeech Download → Mixture Generation → STFT → RCNN Training (SI-SNR+PIT) → Checkpoint → Inference → Separated .wav files
```

## Environment

- **Conda env**: `rl4vlm_clean` (activate with `conda activate rl4vlm_clean`)
- **GPU**: Use GPU 3 (`CUDA_VISIBLE_DEVICES=3`) — NVIDIA H200, ~68 GB free
- **Python**: 3.13 | **PyTorch**: available in env | **CUDA**: 12.9
