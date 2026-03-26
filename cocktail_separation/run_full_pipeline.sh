#!/bin/bash
# Full pipeline: extract data, generate mixtures, train model, run inference
# Usage: bash run_full_pipeline.sh
set -e

VENV="dprnn2/bin/python"
GPUS="4,5"
DATA_DIR="data/librispeech_kaggle"
ZIP_FILE="${DATA_DIR}/librispeech-clean.zip"
MIX_DIR="data/librispeech_full_mixtures/3spk"
CONFIG="configs/3spk_full.yaml"

echo "=============================================="
echo "COCKTAIL PARTY SEPARATION — FULL PIPELINE"
echo "=============================================="
echo ""

# Step 1: Extract downloaded zip
if [ ! -d "${DATA_DIR}/LibriSpeech" ] && [ ! -d "${DATA_DIR}/train-clean-100" ]; then
    echo "Step 1: Extracting LibriSpeech zip..."
    cd ${DATA_DIR}
    unzip -o librispeech-clean.zip
    cd -
    echo "✅ Extraction complete"
else
    echo "Step 1: LibriSpeech already extracted, skipping"
fi
echo ""

# Step 2: Generate mixtures
if [ ! -d "${MIX_DIR}/train" ] || [ "$(ls ${MIX_DIR}/train/ 2>/dev/null | wc -l)" -lt 1000 ]; then
    echo "Step 2: Generating 3-speaker mixtures..."
    ${VENV} generate_mixtures_from_librispeech.py \
        --librispeech-dir ${DATA_DIR} \
        --output-dir ${MIX_DIR} \
        --num-speakers 3 \
        --num-train 20000 \
        --num-val 2000 \
        --clip-seconds 4.0 \
        --seed 42
    echo "✅ Mixtures generated"
else
    echo "Step 2: Mixtures already exist ($(ls ${MIX_DIR}/train/ | wc -l) train), skipping"
fi
echo ""

# Step 3: Smoke test
echo "Step 3: Smoke test..."
CUDA_VISIBLE_DEVICES=${GPUS} ${VENV} -c "
import torch
from src.model import DPRNNTasNet
from src.dataset import MixtureDataset

# Test model
model = DPRNNTasNet(num_speakers=3, encoder_dim=512, encoder_kernel=20, encoder_stride=10, bottleneck_dim=256, chunk_size=200, num_dprnn_blocks=6)
model = torch.nn.DataParallel(model).cuda()
x = torch.randn(4, 64000).cuda()
y = model(x)
params = sum(p.numel() for p in model.parameters())
print(f'Model output: {y.shape}, Params: {params/1e6:.1f}M')

# Test dataset
ds = MixtureDataset('${MIX_DIR}/train', num_speakers=3, clip_samples=64000)
mix, src = ds[0]
print(f'Dataset: {len(ds)} samples, Mixture: {mix.shape}, Sources: {src.shape}')
print('✅ Smoke test PASSED')
"
echo ""

# Step 4: Train
echo "Step 4: Starting training on GPUs ${GPUS}..."
echo "Log file: training_full.log"
CUDA_VISIBLE_DEVICES=${GPUS} ${VENV} train_full.py \
    --config ${CONFIG} \
    2>&1 | tee training_full.log

echo ""
echo "Step 5: Running inference on my_audio.mp3..."
CUDA_VISIBLE_DEVICES=4 ${VENV} separate.py \
    --config ${CONFIG} \
    --checkpoint checkpoints/3spk_full/best.pt \
    --input my_audio.mp3 \
    --output_dir my_results_full/

echo ""
echo "=============================================="
echo "PIPELINE COMPLETE"
echo "Check my_results_full/ for separated speakers"
echo "=============================================="
