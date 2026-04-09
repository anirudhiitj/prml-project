#!/bin/bash
# Script to launch training across 8 GPUs for the H200 cluster

export OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=0 # Enable if using NVLink

LIBRISPEECH_ROOT="/mnt/raid/rl_gaming/TGL-LLM2/data/LibriSpeech"

echo "Starting distributed training on GPU 6..."

# Select ONLY GPU 6
export CUDA_VISIBLE_DEVICES=6

# Using torchrun for PyTorch DistributedDataParallel
# nproc_per_node must match the number of GPUs you isolated
torchrun \
    --nproc_per_node=1 \
    --master_port=29500 \
    train.py \
    --train_dirs "$LIBRISPEECH_ROOT/train-clean-100,$LIBRISPEECH_ROOT/train-clean-360" \
    --val_dirs "$LIBRISPEECH_ROOT/dev-clean" \
    --save_dir ./checkpoints \
    --mixed_precision bf16 \
    --n_src 2 \
    --sample_rate 8000 \
    --batch_size 4 \
    --epochs 100 \
    --learning_rate 1e-3
