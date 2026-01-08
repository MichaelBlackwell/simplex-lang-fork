# Simplex Cognitive Model Training Pipeline

Training infrastructure for the Simplex Cognitive model family.

## Overview

This pipeline fine-tunes Qwen3-8B for Simplex Cognitive Hive AI with:

1. **Context Protocol Training** - Simplex memory format understanding
2. **Confidence Calibration** - Well-calibrated confidence outputs
3. **Belief Revision** - Updating beliefs on new evidence

## Directory Structure

```
training/
├── configs/
│   └── training_config.yaml     # Training hyperparameters
├── data/                        # Generated training data
├── scripts/
│   ├── train_context_protocol.py
│   ├── train_confidence_calibration.py
│   ├── train_belief_revision.py
│   ├── run_full_training.py     # Orchestration script
│   ├── export_to_gguf.py        # Export to Ollama
│   └── aws_setup.sh             # AWS instance setup
├── outputs/                     # Trained models
└── requirements.txt
```

## Quick Start

### Local Testing (CPU - Slow)

```bash
cd training

# Install dependencies
pip install -r requirements.txt

# Generate a small test dataset and run 1 epoch
python scripts/train_context_protocol.py --local-test --generate-data
```

### AWS GPU Training (Recommended)

1. **Launch AWS Instance**
   ```bash
   # Recommended: g5.xlarge (1x A10G 24GB) - ~$1/hr
   # Alternative: p3.2xlarge (1x V100 16GB) - ~$3/hr
   ```

2. **Setup Instance**
   ```bash
   # Copy setup script
   scp scripts/aws_setup.sh ubuntu@<instance>:~/

   # SSH and run setup
   ssh ubuntu@<instance>
   chmod +x aws_setup.sh && ./aws_setup.sh
   ```

3. **Upload Training Code**
   ```bash
   scp -r training/* ubuntu@<instance>:~/simplex-training/
   ```

4. **Run Training**
   ```bash
   source ~/simplex-training/venv/bin/activate
   cd ~/simplex-training

   # Full pipeline (all 3 stages)
   python scripts/run_full_training.py --all

   # Or individual stages
   python scripts/train_context_protocol.py --generate-data
   python scripts/train_confidence_calibration.py --generate-data
   python scripts/train_belief_revision.py --generate-data
   ```

5. **Export to GGUF**
   ```bash
   python scripts/export_to_gguf.py \
     --adapter-path outputs/context_protocol/final \
     --model-name simplex-cognitive-8b
   ```

## Training Stages

### Stage 1: Context Protocol (4-6 hrs)

Teaches the model to understand Simplex memory context format:
- `<context>...</context>` - Individual Anima memory
- `<hive name="...">...</hive>` - Shared hive knowledge
- Confidence thresholds (30% Anima, 50% Hive, 70% Divine)

**Training Data**: 100K synthetic examples

### Stage 2: Confidence Calibration (2-4 hrs)

Trains for well-calibrated confidence outputs:
- High confidence for factual questions
- Medium confidence for ambiguous questions
- Low confidence for unknowable questions

**Target**: Expected Calibration Error (ECE) < 0.05

### Stage 3: Belief Revision (2-4 hrs)

Trains the model to update beliefs appropriately:
- Revise on strong evidence
- Maintain on weak evidence
- Resist bad evidence (logical fallacies, outdated info)

## Configuration

Edit `configs/training_config.yaml` to adjust:

```yaml
# LoRA settings
lora:
  r: 16              # LoRA rank
  lora_alpha: 32     # LoRA alpha
  lora_dropout: 0.05

# Training settings
training:
  num_train_epochs: 3
  learning_rate: 2.0e-4
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
```

## Estimated Costs

| Stage | GPU Hours | Cost (g5.xlarge) |
|-------|-----------|------------------|
| Context Protocol | 4-6 hrs | $4-6 |
| Confidence Calibration | 2-4 hrs | $2-4 |
| Belief Revision | 2-4 hrs | $2-4 |
| **Total** | **8-14 hrs** | **$8-14** |

## Export to Ollama

After training, export the model:

```bash
# Merge LoRA and convert to GGUF
python scripts/export_to_gguf.py \
  --adapter-path outputs/context_protocol/final \
  --quantization q4_k_m \
  --model-name simplex-cognitive-8b

# Import to Ollama
ollama create simplex-cognitive-8b -f exports/Modelfile

# Test
ollama run simplex-cognitive-8b "Test the model..."
```

## Troubleshooting

### Out of Memory
- Reduce `per_device_train_batch_size`
- Enable `gradient_checkpointing: true`
- Use smaller LoRA rank (`r: 8`)

### Slow Training
- Enable `bf16: true` and `tf32: true`
- Increase batch size if memory allows
- Use `optim: adamw_8bit`

### Poor Calibration
- Increase training epochs
- Balance dataset distribution
- Add temperature scaling post-training
