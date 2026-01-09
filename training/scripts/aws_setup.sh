#!/bin/bash
# AWS GPU Instance Setup Script for Simplex Cognitive Training
# Run this on a fresh AWS g5.xlarge or p3.2xlarge instance

set -e

echo "============================================"
echo "Simplex Cognitive Model Training Setup"
echo "============================================"

# Update system
echo "[1/8] Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install basic dependencies
echo "[2/8] Installing system dependencies..."
sudo apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    build-essential \
    python3-pip \
    python3-venv

# Install CUDA drivers if needed (for fresh instances)
echo "[3/8] Checking NVIDIA drivers..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA drivers..."
    sudo apt-get install -y nvidia-driver-535
    echo "Drivers installed. You may need to reboot."
fi

nvidia-smi

# Create project directory
echo "[4/8] Setting up project directory..."
mkdir -p ~/simplex-training
cd ~/simplex-training

# Create Python virtual environment
echo "[5/8] Creating Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
echo "[6/8] Installing PyTorch..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install training dependencies
echo "[7/8] Installing training dependencies..."
pip install \
    transformers>=4.40.0 \
    accelerate>=0.27.0 \
    datasets>=2.18.0 \
    peft>=0.10.0 \
    bitsandbytes>=0.43.0 \
    trl>=0.8.0 \
    wandb>=0.16.0 \
    tensorboard>=2.16.0 \
    pandas>=2.2.0 \
    numpy>=1.26.0 \
    jsonlines>=4.0.0 \
    tqdm>=4.66.0 \
    scikit-learn>=1.4.0 \
    scipy>=1.12.0 \
    faker>=24.0.0 \
    pyyaml

# Login to HuggingFace (for model access)
echo "[8/8] Setup complete!"
echo ""
echo "============================================"
echo "Next steps:"
echo "1. Upload training scripts: scp -r training/* ubuntu@<instance>:~/simplex-training/"
echo "2. Activate environment: source ~/simplex-training/venv/bin/activate"
echo "3. Login to HuggingFace: huggingface-cli login"
echo "4. Login to Wandb: wandb login"
echo "5. Run training: python scripts/train_context_protocol.py --generate-data"
echo "============================================"
