#!/bin/bash
# Run full training pipeline on EC2 with S3 upload
# Usage: ./run_full_training.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="/home/ubuntu/training"

echo "============================================================"
echo "SIMPLEX COGNITIVE - FULL TRAINING PIPELINE"
echo "============================================================"
echo "Start time: $(date)"
echo ""

# Check if running on EC2
if [ ! -d "$TRAINING_DIR" ]; then
    echo "Error: Not running on EC2 or training directory not found"
    echo "Expected: $TRAINING_DIR"
    exit 1
fi

cd "$TRAINING_DIR"

# Configure HuggingFace cache to use NVMe drive (more space)
export HF_HOME=/opt/dlami/nvme/cache
export TRANSFORMERS_CACHE=/opt/dlami/nvme/cache/transformers
export HF_DATASETS_CACHE=/opt/dlami/nvme/cache/datasets
mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HF_DATASETS_CACHE

# Check GPU
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    echo "Warning: nvidia-smi not found, may be running on CPU"
fi

# Setup Python environment
echo ""
echo "Setting up Python environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip --quiet
pip install torch transformers peft datasets trl accelerate --quiet
pip install faker boto3 --quiet

echo ""
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Configure AWS
echo ""
echo "Configuring AWS..."
aws configure set region ap-southeast-2

# Verify S3 access
echo "Verifying S3 access..."
if aws s3 ls s3://simplex-model-repo/ --region ap-southeast-2 &> /dev/null; then
    echo "  S3 bucket accessible"
else
    echo "  Warning: Cannot access S3 bucket, uploads may fail"
fi

# Run full training
echo ""
echo "============================================================"
echo "STARTING FULL TRAINING"
echo "============================================================"
echo ""

cd scripts

# Create output dir on NVMe drive
OUTPUT_DIR="/opt/dlami/nvme/training/outputs/specialists"
mkdir -p $OUTPUT_DIR

# Train all specialists
python run_full_pipeline.py --all \
    --output-dir $OUTPUT_DIR \
    --num-examples 10000 \
    2>&1 | tee ../training.log

echo ""
echo "============================================================"
echo "TRAINING COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo ""

# Summary
echo "Checking outputs..."
ls -la $OUTPUT_DIR || echo "No outputs found"

# Final S3 sync
echo ""
echo "Final S3 sync..."
aws s3 sync $OUTPUT_DIR s3://simplex-model-repo/specialists/ \
    --region ap-southeast-2 \
    --only-show-errors

echo ""
echo "All models uploaded to s3://simplex-model-repo/specialists/"
echo ""
echo "To download models later:"
echo "  aws s3 sync s3://simplex-model-repo/specialists/ ./models/ --region ap-southeast-2"
