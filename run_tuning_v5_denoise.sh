#!/bin/bash
# Script to launch V5 Noise-Resilient Training (V3 Architecture + MUSAN Noise Injection)

export OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=0

LIBRISPEECH_ROOT="/mnt/raid/rl_gaming/TGL-LLM2/data/LibriSpeech"
MUSAN_NOISE_DIR="/mnt/raid/rl_gaming/TGL-LLM2/musan_dataset/musan/noise"

echo "Starting V5 Noise-Resilient training on GPU 4..."

# Binding to GPU 4
export CUDA_VISIBLE_DEVICES=4

torchrun \
    --nproc_per_node=1 \
    --master_port=29504 \
    train.py \
    --exp_name "v5_denoise" \
    --train_dirs "$LIBRISPEECH_ROOT/train-clean-100,$LIBRISPEECH_ROOT/train-clean-360" \
    --val_dirs "$LIBRISPEECH_ROOT/dev-clean" \
    --save_dir ./checkpoints \
    --mixed_precision bf16 \
    --n_src 2 \
    --sample_rate 8000 \
    --batch_size 16 \
    --epochs 200 \
    --learning_rate 1e-3 \
    --train_steps 4000 \
    --val_steps 400 \
    --n_blocks 8 \
    --n_repeats 3 \
    --bn_chan 128 \
    --hid_chan 512 \
    --noise_dir "$MUSAN_NOISE_DIR" \
    --noise_snr_low 5 \
    --noise_snr_high 20
