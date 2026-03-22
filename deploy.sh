#!/usr/bin/env bash
# ============================================
# Image Mixer - Automated Deployment Script
# ============================================
# This script automates:
# 1. Dependencies installation
# 2. Ngrok installation & configuration
# 3. Background execution of the Web UI and Tunnel

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  Image Mixer - Automated Deployment"
echo "============================================"

# Step 1: Install Python dependencies
echo "
[1/4] Running installation script..."
chmod +x install.sh
./install.sh

# Step 2: Install ngrok
echo "
[2/4] Installing ngrok agent..."
if ! command -v ngrok &> /dev/null; then
    curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
      | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
    
    # Adding the ngrok repository
    echo "deb https://ngrok-agent.s3.amazonaws.com bookworm main" \
      | sudo tee /etc/apt/sources.list.d/ngrok.list
    
    sudo apt update
    sudo apt install ngrok -y
else
    echo "ngrok is already installed, skipping installation."
fi

# Step 3: Configure ngrok
echo "
[3/4] Configuring ngrok auth..."
ngrok config add-authtoken 2nUUPXE54je7RQ5ZM4cScjWSsBn_2X5yNcDntrCDk4pq3io6b

# Step 4: Launch services
echo "
[4/4] Launching Image Mixer and ngrok..."

# Cleaning up existing processes to avoid port conflicts
echo "Cleaning up any old instances..."
pkill -f "gradio_image_mixer.py" || true
pkill -f "ngrok http" || true

# Activate environment and run app in background
source venv/bin/activate

echo "Starting Web UI (logging to app.log)..."
nohup python scripts/gradio_image_mixer.py > app.log 2>&1 &

echo "Starting ngrok tunnel (logging to ngrok.log)..."
nohup ngrok http --url=rich-liger-equally.ngrok-free.app 7860 > ngrok.log 2>&1 &

echo "
============================================"
echo "  Deployment Successful!"
echo "  App URL: https://rich-liger-equally.ngrok-free.app"
echo "  Logs: tail -f app.log or tail -f ngrok.log"
echo "============================================"
