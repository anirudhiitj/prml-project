#!/bin/bash
# Wait for download completion, then run full pipeline
# Run this in a tmux session for unattended execution
set -e

cd /mnt/raid/rl_gaming/RL4VLM2/cocktail_separation

VENV="dprnn2/bin/python"
GPUS="4,5"
DATA_DIR="data/librispeech_kaggle"
ZIP_FILE="${DATA_DIR}/librispeech-clean.zip"
MIX_DIR="data/librispeech_full_mixtures/3spk"
CONFIG="configs/3spk_full.yaml"

echo "=============================================="
echo "COCKTAIL PARTY SEPARATION — FULL PIPELINE"
echo "Started: $(date)"
echo "=============================================="
echo ""

# Step 0: Wait for download to finish
echo "Step 0: Waiting for download to complete..."
while pgrep -f "curl.*librispeech" > /dev/null 2>&1; do
    SIZE=$(ls -lh ${ZIP_FILE} 2>/dev/null | awk '{print $5}')
    echo "  $(date '+%H:%M:%S') — Download size: ${SIZE}"
    sleep 60
done
SIZE=$(ls -lh ${ZIP_FILE} 2>/dev/null | awk '{print $5}')
echo "✅ Download complete! Final size: ${SIZE}"
echo ""

# Step 1: Extract
echo "Step 1: Extracting LibriSpeech zip (this may take 10-20 min)..."
echo "  $(date '+%H:%M:%S') — Starting extraction..."
cd ${DATA_DIR}
unzip -o -q librispeech-clean.zip
cd /mnt/raid/rl_gaming/RL4VLM2/cocktail_separation
echo "  $(date '+%H:%M:%S') — Extraction complete!"
echo ""

# Show what was extracted
echo "Extracted contents:"
find ${DATA_DIR} -maxdepth 3 -type d | head -20
echo ""

# Find FLAC files
FLAC_COUNT=$(find ${DATA_DIR} -name "*.flac" 2>/dev/null | wc -l)
echo "Found ${FLAC_COUNT} FLAC files"
echo ""

# Step 2: Generate mixtures (20k train + 2k val)
echo "Step 2: Generating 3-speaker mixtures..."
echo "  $(date '+%H:%M:%S') — Starting mixture generation..."
${VENV} generate_mixtures_from_librispeech.py \
    --librispeech-dir ${DATA_DIR} \
    --output-dir ${MIX_DIR} \
    --num-speakers 3 \
    --num-train 20000 \
    --num-val 2000 \
    --clip-seconds 4.0 \
    --seed 42
echo "  $(date '+%H:%M:%S') — Mixture generation complete!"
echo ""

# Step 3: Smoke test
echo "Step 3: Smoke test on GPUs ${GPUS}..."
CUDA_VISIBLE_DEVICES=${GPUS} ${VENV} -c "
import torch
from src.model import DPRNNTasNet
from src.dataset import MixtureDataset

model = DPRNNTasNet(num_speakers=3, encoder_dim=512, encoder_kernel=20, encoder_stride=10, bottleneck_dim=256, chunk_size=200, num_dprnn_blocks=6)
model = torch.nn.DataParallel(model).cuda()
x = torch.randn(4, 64000).cuda()
y = model(x)
params = sum(p.numel() for p in model.parameters())
print(f'Model: {y.shape}, Params: {params/1e6:.1f}M')

ds = MixtureDataset('${MIX_DIR}/train', num_speakers=3, clip_samples=64000)
print(f'Dataset: {len(ds)} samples')
print('✅ Smoke test PASSED')
"
echo ""

# Step 4: Train
echo "Step 4: Starting training on GPUs ${GPUS}..."
echo "  $(date '+%H:%M:%S') — Training started"
CUDA_VISIBLE_DEVICES=${GPUS} ${VENV} train_full.py \
    --config ${CONFIG} \
    2>&1 | tee training_full.log
echo ""

# Step 5: Inference
echo "Step 5: Running inference on my_audio.mp3..."
CUDA_VISIBLE_DEVICES=4 ${VENV} separate.py \
    --config ${CONFIG} \
    --checkpoint checkpoints/3spk_full/best.pt \
    --input my_audio.mp3 \
    --output_dir my_results_full/
echo ""

echo "=============================================="
echo "PIPELINE COMPLETE — $(date)"
echo "Check my_results_full/ for separated speakers"
echo "=============================================="
