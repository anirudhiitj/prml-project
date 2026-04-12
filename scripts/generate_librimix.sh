#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# generate_librimix.sh — Download & generate Libri2Mix dataset
#
# Usage: bash scripts/generate_librimix.sh [DATA_DIR]
#
# Generates Libri2Mix: train-360 + dev + test at 8kHz (min mode).
# Requires ~100 GB disk space, Python 3, and sox.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DATA_DIR="${1:-./data}"
mkdir -p "$DATA_DIR"

echo "═══════════════════════════════════════════════════"
echo " Libri2Mix Dataset Generation"
echo "═══════════════════════════════════════════════════"
echo " Output:   $DATA_DIR"
echo " Disk:     $(df -h "$DATA_DIR" | tail -1 | awk '{print $4}') available"
echo ""

# ── Dependencies ─────────────────────────────────────────────────────────────
echo "[1/5] Checking dependencies..."
command -v python3 >/dev/null || { echo "Error: python3 required"; exit 1; }
command -v sox     >/dev/null || { echo "Error: sox required (apt install sox)"; exit 1; }

# Ensure Python packages
pip3 install --quiet soundfile numpy scipy 2>/dev/null || true

# ── Clone LibriMix ────────────────────────────────────────────────────────────
cd "$DATA_DIR"

if [ ! -d "LibriMix" ]; then
    echo "[2/5] Cloning LibriMix repository..."
    git clone --depth 1 https://github.com/JorisCos/LibriMix.git
else
    echo "[2/5] LibriMix already cloned."
fi

# ── Download LibriSpeech ──────────────────────────────────────────────────────
echo "[3/5] Downloading LibriSpeech data..."
LIBRISPEECH_DIR="$DATA_DIR/LibriSpeech"
mkdir -p "$LIBRISPEECH_DIR"

download_and_extract() {
    local url="$1"
    local target_dir="$2"
    local name=$(basename "$url" .tar.gz)

    if [ -d "$target_dir/$name" ] || [ -d "$target_dir/LibriSpeech/$name" ]; then
        echo "  ✓ $name already exists"
        return 0
    fi

    echo "  Downloading $name..."
    wget -q --show-progress -c "$url" -O "/tmp/${name}.tar.gz" || {
        echo "  Retrying with curl..."
        curl -L -C - -o "/tmp/${name}.tar.gz" "$url"
    }
    echo "  Extracting $name..."
    tar -xzf "/tmp/${name}.tar.gz" -C "$target_dir"
    rm -f "/tmp/${name}.tar.gz"
    echo "  ✓ $name extracted"
}

# Download the required LibriSpeech splits
download_and_extract \
    "https://www.openslr.org/resources/12/train-clean-360.tar.gz" \
    "$LIBRISPEECH_DIR"

download_and_extract \
    "https://www.openslr.org/resources/12/dev-clean.tar.gz" \
    "$LIBRISPEECH_DIR"

download_and_extract \
    "https://www.openslr.org/resources/12/test-clean.tar.gz" \
    "$LIBRISPEECH_DIR"

# Fix directory structure if tar extracted into LibriSpeech subdirectory
if [ -d "$LIBRISPEECH_DIR/LibriSpeech" ]; then
    echo "  Fixing directory structure..."
    mv "$LIBRISPEECH_DIR/LibriSpeech"/* "$LIBRISPEECH_DIR/" 2>/dev/null || true
    rmdir "$LIBRISPEECH_DIR/LibriSpeech" 2>/dev/null || true
fi

# ── Download WHAM noise ───────────────────────────────────────────────────────
echo "[4/5] Downloading WHAM noise data..."
WHAM_DIR="$DATA_DIR/wham_noise"

if [ ! -d "$WHAM_DIR" ]; then
    echo "  Downloading WHAM noise (~4GB)..."
    wget -q --show-progress -c \
        "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip" \
        -O "/tmp/wham_noise.zip" || {
        curl -L -C - -o "/tmp/wham_noise.zip" \
            "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip"
    }
    echo "  Extracting WHAM noise..."
    unzip -q "/tmp/wham_noise.zip" -d "$DATA_DIR"
    rm -f "/tmp/wham_noise.zip"
    echo "  ✓ WHAM noise extracted"
else
    echo "  ✓ WHAM noise already exists"
fi

# ── Generate Libri2Mix ────────────────────────────────────────────────────────
echo "[5/5] Generating Libri2Mix mixtures (this takes a while)..."

cd "$DATA_DIR/LibriMix"

# Create the metadata/LibriSpeech directory if needed
mkdir -p metadata

# Run the generation script with the right parameters
# The LibriMix repo's generate_librimix.sh handles everything
python3 scripts/create_librimix_from_metadata.py \
    --librispeech_dir "$LIBRISPEECH_DIR" \
    --wham_dir "$WHAM_DIR" \
    --metadata_dir metadata/Libri2Mix \
    --librimix_outdir "$DATA_DIR/Libri2Mix" \
    --n_src 2 \
    --freqs 8k \
    --modes min \
    --types mix_clean mix_both 2>&1 || {
    echo "  Direct script failed, trying LibriMix's own generation..."
    # Fallback: use the repo's own generate script
    bash generate_librimix.sh "$LIBRISPEECH_DIR"
}

echo ""
echo "═══════════════════════════════════════════════════"
echo " ✓ Dataset generation complete!"
echo "═══════════════════════════════════════════════════"
echo ""
echo "Dataset location: $DATA_DIR/Libri2Mix/wav8k/min/"
echo ""
echo "Expected structure:"
echo "  train-360/  — training split (~50k mixtures)"
echo "  dev/        — validation split"
echo "  test/       — test split"
echo ""
echo "Each split contains:"
echo "  mix_clean/  — clean 2-speaker mixtures"
echo "  s1/         — speaker 1 (clean)"
echo "  s2/         — speaker 2 (clean)"
echo "  mixture_*.csv — metadata"
echo ""

# List CSV files
echo "CSV files found:"
find "$DATA_DIR" -name "mixture_*.csv" -type f 2>/dev/null | sort
echo ""

# Count files per split
for split in train-360 dev test; do
    count=$(find "$DATA_DIR/Libri2Mix/wav8k/min/$split" -name "*.wav" 2>/dev/null | wc -l)
    echo "  $split: $count wav files"
done
