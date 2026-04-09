#!/bin/bash
# Script to launch V3 Improved Tuning (Baseline Architecture with Hyper-scaled Training Params)

export OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=0 # Enable if using NVLink

LIBRISPEECH_ROOT="/mnt/raid/rl_gaming/TGL-LLM2/data/LibriSpeech"

echo "Starting distributed tuning on GPU 6 for V3 Improved Model..."

# Binding to the free GPU
export CUDA_VISIBLE_DEVICES=6

# Using torchrun for PyTorch DistributedDataParallel
torchrun \
    --nproc_per_node=1 \
    --master_port=29502 \
    train.py \
    --exp_name "v3_improved" \
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
    --hid_chan 512
