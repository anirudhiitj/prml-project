#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# generate_librimix.sh — Download & generate Libri2Mix dataset
#
# Usage: bash scripts/generate_librimix.sh ./data
#
# This generates the full Libri2Mix train-360 + dev + test at 8kHz.
# Requires ~100 GB disk space, Python 3, and sox.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DATA_DIR="${1:-./data}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "═══════════════════════════════════════════════════"
echo " Libri2Mix Dataset Generation"
echo "═══════════════════════════════════════════════════"

# ── Dependencies ─────────────────────────────────────────────────────────────
echo "[1/4] Checking dependencies..."
command -v python3 >/dev/null || { echo "Error: python3 required"; exit 1; }
command -v sox     >/dev/null || { echo "Error: sox required (apt install sox)"; exit 1; }
pip install --quiet soundfile pysoundfile 2>/dev/null || true

# ── Clone LibriMix ────────────────────────────────────────────────────────────
if [ ! -d "LibriMix" ]; then
    echo "[2/4] Cloning LibriMix..."
    git clone https://github.com/JorisCos/LibriMix.git
else
    echo "[2/4] LibriMix already cloned."
fi

# ── Configuration ─────────────────────────────────────────────────────────────
# Edit LibriMix/metadata/LibriSpeech/generate_librimix.sh parameters:
# - n_src=2          (2-speaker)
# - freqs="8k"       (8 kHz sample rate)
# - modes="min"      (trim to shortest source)
# - types="mix_clean mix_both" (clean + noisy mixtures)

echo "[3/4] Generating Libri2Mix (train-360 + dev + test, 8kHz)..."
cd LibriMix
bash generate_librimix.sh "$DATA_DIR/LibriSpeech"

echo "[4/4] Done!"
echo ""
echo "Dataset location: $DATA_DIR/Libri2Mix/wav8k/min/"
echo ""
echo "Expected structure:"
echo "  train-360/mix_clean/  — clean mixtures"
echo "  train-360/s1/         — speaker 1 (clean)"
echo "  train-360/s2/         — speaker 2 (clean)"
echo "  dev/                  — validation split"
echo "  test/                 — test split"
echo ""
echo "CSV files: mixture_train-360_mix_clean.csv, mixture_dev_mix_clean.csv, etc."
