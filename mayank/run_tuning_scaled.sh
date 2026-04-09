#!/bin/bash
# Script to launch Scaled Hyperparameter Tuning Experiment

export OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=0 # Enable if using NVLink

LIBRISPEECH_ROOT="/mnt/raid/rl_gaming/TGL-LLM2/data/LibriSpeech"

echo "Starting distributed tuning on GPU 6 for Scaled Architecture..."

# Binding to the free GPU
export CUDA_VISIBLE_DEVICES=6

# Using torchrun for PyTorch DistributedDataParallel
torchrun \
    --nproc_per_node=1 \
    --master_port=29501 \
    train.py \
    --exp_name "v2_scaled_arch" \
    --train_dirs "$LIBRISPEECH_ROOT/train-clean-100,$LIBRISPEECH_ROOT/train-clean-360" \
    --val_dirs "$LIBRISPEECH_ROOT/dev-clean" \
    --save_dir ./checkpoints \
    --mixed_precision bf16 \
    --n_src 2 \
    --sample_rate 8000 \
    --batch_size 4 \
    --epochs 100 \
    --learning_rate 1e-3 \
    --n_blocks 10 \
    --n_repeats 4 \
    --bn_chan 256 \
    --hid_chan 1024
