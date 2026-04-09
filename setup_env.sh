#!/bin/bash
# Exit on error
set -e

ENV_DIR="/mnt/raid/rl_gaming/TGL-LLM2/conv_tasnet_env"

# Override Conda's default package cache directory to the RAID drive
export CONDA_PKGS_DIRS="/mnt/raid/rl_gaming/TGL-LLM2/conda_pkgs"
mkdir -p $CONDA_PKGS_DIRS

echo "Creating conda environment in $ENV_DIR..."
conda create --prefix $ENV_DIR python=3.10 -y

echo "Activating environment..."
# Retrieve conda base dynamically
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $ENV_DIR

echo "Installing dependencies..."
# Create a temporary directory on the large RAID drive to avoid "No space left on device"
mkdir -p /mnt/raid/rl_gaming/tmp_pip
export TMPDIR=/mnt/raid/rl_gaming/tmp_pip

# Install with --no-cache-dir to save disk space
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir -r requirements.txt

# Clean up tmp
rm -rf /mnt/raid/rl_gaming/tmp_pip

echo "Setup complete! To activate the environment, run: conda activate $ENV_DIR"
