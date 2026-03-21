#!/usr/bin/env bash
set -euo pipefail

# This script is intentionally lightweight because dataset access methods vary by environment.
# Use official mirrors/tools for each dataset and place files under data/raw.

echo "Download datasets manually or with your internal data pipeline:"
echo "  1) LibriSpeech train-clean-100, train-clean-360"
echo "  2) VoxCeleb2 dev"
echo "  3) VCTK"
echo "Then organize under data/raw and run generate_mixtures.py"
