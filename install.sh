#!/usr/bin/env bash
set -euo pipefail

# ============================================
# Image Mixer - Installation Script
# ============================================
# This script handles the correct installation order
# for all dependencies including git-based packages.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  Image Mixer - Dependency Installation"
echo "============================================"

# Step 0: Check for CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. This project requires a CUDA GPU."
    echo "Continuing installation anyway..."
fi

# Step 1: Create or activate virtual environment
if [ ! -d "venv" ]; then
    echo "
[1/5] Creating virtual environment..."
    python3 -m venv venv
else
    echo "
[1/5] Virtual environment already exists, reusing..."
fi
source venv/bin/activate

# Step 2: Upgrade core build tools
echo "
[2/5] Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# Step 3: Install CLIP (requires --no-build-isolation due to numpy/torch build deps)
echo "
[3/5] Installing OpenAI CLIP..."
pip install --no-build-isolation git+https://github.com/openai/CLIP.git@main

# Step 4: Install main requirements
echo "
[4/5] Installing requirements..."
pip install -r requirements.txt

# Step 5: (Optional) Install taming-transformers if VQ models are needed
# Not required for Image Mixer (uses AutoencoderKL, not VQModel)
# Uncomment below if you need VQ model support:
# echo "
[5/5] Installing taming-transformers (optional)..."
# pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
echo "
[5/5] Skipping taming-transformers (not needed for Image Mixer)"

echo "
============================================"
echo "  Installation complete!"
echo "  Run: source venv/bin/activate"
echo "  Then: python scripts/gradio_image_mixer.py"
echo "============================================"
