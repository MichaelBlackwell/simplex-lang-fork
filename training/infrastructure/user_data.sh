#!/bin/bash
# Simplex Cognitive Training - Instance Bootstrap Script
# This script runs on first boot to configure the training environment

set -e

# Log all output
exec > >(tee /var/log/simplex-setup.log) 2>&1

echo "============================================"
echo "Simplex Cognitive Training Setup"
echo "Started: $(date)"
echo "============================================"

# Update system
echo "[1/7] Updating system packages..."
apt-get update && apt-get upgrade -y

# Install additional dependencies
echo "[2/7] Installing dependencies..."
apt-get install -y \
    awscli \
    jq \
    tmux \
    htop \
    nvtop

# Verify NVIDIA drivers
echo "[3/7] Verifying GPU..."
nvidia-smi

# Create working directory
echo "[4/7] Setting up working directory..."
mkdir -p /home/ubuntu/simplex-training
cd /home/ubuntu/simplex-training

# Create Python virtual environment
echo "[5/7] Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install PyTorch and dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install \
    transformers>=4.40.0 \
    accelerate>=0.27.0 \
    datasets>=2.18.0 \
    peft>=0.10.0 \
    bitsandbytes>=0.43.0 \
    trl>=0.8.0 \
    wandb>=0.16.0 \
    tensorboard>=2.16.0 \
    pandas numpy jsonlines tqdm \
    scikit-learn scipy faker pyyaml

# Configure credentials if provided
echo "[6/7] Configuring credentials..."

%{ if wandb_api_key != "" ~}
echo "Configuring Weights & Biases..."
wandb login ${wandb_api_key}
%{ endif ~}

%{ if hf_token != "" ~}
echo "Configuring HuggingFace..."
huggingface-cli login --token ${hf_token}
%{ endif ~}

# Download training code from S3
echo "[7/7] Downloading training code..."
%{ if s3_bucket != "" ~}
aws s3 sync s3://${s3_bucket}/training /home/ubuntu/simplex-training/ || echo "No training code in S3 yet"
%{ endif ~}

# Fix permissions
chown -R ubuntu:ubuntu /home/ubuntu/simplex-training

# Create convenience script
cat > /home/ubuntu/start-training.sh << 'EOF'
#!/bin/bash
cd /home/ubuntu/simplex-training
source venv/bin/activate

echo "Simplex Training Environment Ready"
echo ""
echo "Commands:"
echo "  python scripts/run_full_training.py --all    # Full training"
echo "  python scripts/train_context_protocol.py    # Stage 1 only"
echo "  nvidia-smi                                   # GPU status"
echo "  nvtop                                        # GPU monitor"
echo ""

exec bash
EOF
chmod +x /home/ubuntu/start-training.sh

echo "============================================"
echo "Setup complete: $(date)"
echo "Run: /home/ubuntu/start-training.sh"
echo "============================================"
