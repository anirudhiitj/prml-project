#!/bin/bash
# run_pipeline.sh
# Master script to run fine-tuning and inference for 3-speakers

echo "=================================================="
echo " DPRNN-TasNet Cocktail Separation (3 Speakers)"
echo "=================================================="

# 1. Provide instructions to activate environments
echo "Make sure your virtual environment is active."
echo "If not, run: source /path/to/dprnn2/bin/activate (or pipenv/conda)"
echo ""

cd "$(dirname "$0")" || exit 1

# 2. Run the DDP Fine-tuning
echo "[1] STARTING 3-GPU FINE-TUNING..."
echo "Running on GPUs, maxing out your 143GB VRAM pool!"
echo "command: torchrun --nproc_per_node=3 finetune_real.py"

# IMPORTANT: Ensure your data/mixtures/3spk exists before training!
if [ ! -d "data/mixtures/3spk/train" ]; then
    echo "⚠️ Warning: data/mixtures/3spk/train not found! You might need to generate data first."
    echo "Run: python scripts/generate_mixtures.py --num_speakers 3 ..."
    echo "Skipping fine-tuning for now."
else
    torchrun --nproc_per_node=3 finetune_real.py
fi

echo ""
echo "[2] RUNNING INFERENCE ON 30-SEC AUDIO"
echo "We are now using the permutation-tracking inference to prevent speaker swapping."

INPUT_FILE="path/to/your_30sec_audio.wav"
OUTPUT_DIR="outputs/"

if [ ! -f "$INPUT_FILE" ]; then
    echo "⚠️ Please edit run_pipeline.sh to point to your actual 30-sec .wav file!"
    echo "Current path: $INPUT_FILE"
else
    python separate.py \
    --config configs/3spk.yaml \
    --checkpoint checkpoints/best_phase2.pt \
    --input "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR"
    
    echo "✅ Inference complete! Check $OUTPUT_DIR for speaker_1.wav, speaker_2.wav, and speaker_3.wav"
fi
echo "Done."
