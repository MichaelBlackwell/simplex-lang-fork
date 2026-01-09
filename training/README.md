# Simplex Cognitive Models - Training Pipeline

Training infrastructure for fine-tuning specialist AI models using LoRA (Low-Rank Adaptation) on open-source base models.

## Overview

This pipeline trains 50+ specialist models across business domains by:
1. Loading real-world datasets for each specialty
2. Establishing baseline performance of base models
3. Fine-tuning using LoRA for efficient, targeted adaptation
4. Validating improvements meet quality thresholds
5. Exporting production-ready adapter weights

## Architecture

```
training/
├── aws/                    # AWS EC2 infrastructure scripts
│   ├── setup-pilot.sh      # Create VPC, security groups, keys
│   ├── start-pilot*.sh     # Launch GPU instances
│   ├── stop-pilot.sh       # Terminate instances (save costs)
│   └── connect-pilot.sh    # SSH into running instances
├── validation/             # Pilot validation pipeline
│   ├── data_loaders/       # Real dataset integrations
│   ├── evaluation/         # Task-specific evaluators
│   ├── pilots/             # Training pilot scripts
│   └── test_sets/          # Curated test cases
├── tests/                  # Unit and integration tests
├── scripts/                # Training automation
└── outputs/                # Generated models and results
```

## Base Model Selection

| Model | Parameters | Use Cases | GPU Memory |
|-------|------------|-----------|------------|
| Qwen/Qwen2.5-3B-Instruct | 3B | Simple classification, extraction | 8GB |
| Qwen/Qwen2.5-7B-Instruct | 7B | Standard NLU tasks | 16GB |
| Qwen/Qwen2.5-Coder-7B | 7B | Code generation, SQL | 16GB |
| Qwen/Qwen2.5-32B-Instruct | 32B | Complex reasoning, writing | 40GB+ |

**Selection Criteria:**
- **3B models**: Fast inference, suitable for high-volume simple tasks
- **7B models**: Balance of capability and cost for most business tasks
- **32B models**: Complex tasks requiring nuanced understanding

## Specialist Categories

### Business Operations
| Specialist | Base Model | Dataset | Metrics |
|------------|------------|---------|---------|
| Sentiment Analysis | 3B | SST-2, IMDB | Accuracy, Macro F1 |
| Text Classification | 3B | AG News, DBpedia | Accuracy |
| Named Entity Recognition | 3B | CoNLL-2003 | F1 Score |
| Invoice Processing | 3B | CORD, SROIE | Field F1 |

### Data & Analytics
| Specialist | Base Model | Dataset | Metrics |
|------------|------------|---------|---------|
| SQL Generation | 7B Coder | WikiSQL, Spider | Exact Match, Valid SQL |
| Data Summarization | 7B | CNN/DailyMail | ROUGE-L |
| Report Generation | 7B | Custom | BLEU |

### Customer Service
| Specialist | Base Model | Dataset | Metrics |
|------------|------------|---------|---------|
| Intent Classification | 3B | CLINC150, Banking77 | Accuracy |
| FAQ Answering | 7B | SQuAD, Custom | F1, EM |
| Ticket Routing | 3B | Custom | Accuracy |

### Content & Writing
| Specialist | Base Model | Dataset | Metrics |
|------------|------------|---------|---------|
| Content Writer | 32B | Custom prompts | Human eval |
| Email Composer | 7B | Enron, Custom | BLEU |
| Translation | 7B | WMT, OPUS | BLEU, chrF |

### Code & Technical
| Specialist | Base Model | Dataset | Metrics |
|------------|------------|---------|---------|
| Code Generation | 7B Coder | HumanEval, MBPP | Pass@1 |
| Code Review | 7B Coder | Custom | Accuracy |
| Documentation | 7B | Custom | Human eval |

## Training Algorithm

### LoRA (Low-Rank Adaptation)

We use LoRA for efficient fine-tuning:

```python
LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,              # Rank (8-32 based on task complexity)
    lora_alpha=16,    # Scaling factor
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)
```

**LoRA Rank Selection:**
- `r=8`: Simple classification tasks
- `r=16`: Standard NLU tasks (SQL, extraction)
- `r=32`: Complex generation tasks (writing, code)

### Training Hyperparameters

```python
SFTConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch = 16
    learning_rate=2e-4,
    num_train_epochs=3,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    bf16=True,                      # BFloat16 for stability
    gradient_checkpointing=True,    # Memory optimization
)
```

## Datasets

### Real Dataset Sources

| Task | Primary Dataset | Secondary | Size |
|------|-----------------|-----------|------|
| Sentiment | SST-2 | IMDB | 70K train |
| SQL | WikiSQL | Spider | 80K train |
| Invoice | CORD | SROIE | 10K train |
| NER | CoNLL-2003 | OntoNotes | 15K train |
| Classification | AG News | DBpedia | 120K train |

### Data Pipeline

1. **Load** - Fetch from HuggingFace datasets
2. **Clean** - Remove duplicates, fix encoding
3. **Format** - Convert to instruction format
4. **Split** - Train/Val/Test (80/10/10)
5. **Save** - JSONL for reproducibility

### Instruction Format

```json
{
  "prompt": "Analyze the sentiment of this review:\n\n\"Great product!\"\n\nReturn: sentiment, confidence.",
  "response": "**Sentiment:** POSITIVE\n**Confidence:** 0.95",
  "metadata": {"source": "sst2", "label": "positive"}
}
```

## Evaluation Metrics

### Classification Tasks
- **Accuracy**: Overall correctness
- **Macro F1**: Balanced across classes
- **Confusion Matrix**: Error analysis

### Generation Tasks
- **Exact Match**: SQL, extraction
- **BLEU/ROUGE**: Text similarity
- **Valid SQL Rate**: Syntax correctness
- **Human Evaluation**: Quality assessment

### Quality Thresholds

| Task | Metric | Baseline Target | Fine-tuned Target |
|------|--------|-----------------|-------------------|
| Sentiment | Accuracy | 75% | 88% |
| SQL | Exact Match | 40% | 65% |
| Invoice | Field F1 | 60% | 80% |

## AWS Infrastructure

### Instance Types

| Instance | GPU | Memory | Cost/hr (spot) | Use |
|----------|-----|--------|----------------|-----|
| g4dn.xlarge | T4 16GB | 16GB | $0.19 | Inference, 3B training |
| g5.xlarge | A10G 24GB | 16GB | $0.42 | 7B training |
| g5.2xlarge | A10G 24GB | 32GB | $0.84 | 7B + larger batch |
| p4d.24xlarge | 8xA100 | 320GB | $12.50 | 32B training |

### Quick Start

```bash
# Setup infrastructure (one-time)
cd training/aws
./setup-pilot.sh

# Run pilot validation
./start-pilot-ondemand.sh sentiment --baseline-only
./connect-pilot.sh  # SSH to monitor

# When done - IMPORTANT!
./stop-pilot.sh  # Terminate to stop charges
```

### Cost Estimates

| Task | Instances | Duration | Cost |
|------|-----------|----------|------|
| Pilot (3 specialists) | 1x g4dn | 2 hours | ~$1.00 |
| All specialists (33) | 1x g5 | 40 hours | ~$35 |
| Full training run | Mixed | 100 hours | ~$100 |

## Validation Pipeline

### Pilot Workflow

1. **Load Real Data** - SST-2, IMDB for sentiment
2. **Baseline Evaluation** - Test base model performance
3. **Fine-tune** - LoRA training on train split
4. **Post-training Evaluation** - Measure improvement
5. **Report** - Generate comparison metrics

### Running Pilots

```bash
# Sentiment pilot (recommended first)
python validation/pilots/train_pilot.py --specialist sentiment

# SQL pilot
python validation/pilots/train_pilot.py --specialist sql

# Invoice pilot
python validation/pilots/train_pilot.py --specialist invoice

# Baseline only (no training)
python validation/pilots/train_pilot.py --specialist sentiment --baseline-only

# Local test mode (small dataset)
python validation/pilots/train_pilot.py --specialist sentiment --local-test
```

## Testing

```bash
# Run quick tests (no GPU required)
cd training
./tests/run_tests.sh quick

# Run all tests
./tests/run_tests.sh all

# Specific test suites
./tests/run_tests.sh loader   # Data loader tests
./tests/run_tests.sh metrics  # Evaluation metrics
./tests/run_tests.sh training # Training integration
```

## Production Deployment

### Model Export

After successful validation:

```python
# Save LoRA adapter
model.save_pretrained("outputs/sentiment-specialist")

# Merge with base (optional, increases size)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("outputs/sentiment-merged")
```

### Inference

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base + adapter
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = PeftModel.from_pretrained(base, "outputs/sentiment-specialist")

# Inference
output = model.generate(tokenizer.encode(prompt, return_tensors="pt"))
```

## Roadmap

### Phase 1: Validation (Current)
- [x] AWS infrastructure setup
- [x] Pilot validation pipeline
- [ ] Complete 3 pilot specialists
- [ ] Validate quality thresholds

### Phase 2: Training
- [ ] Train all 33 specialists
- [ ] Quality assurance pass
- [ ] Performance optimization

### Phase 3: Production
- [ ] Deploy to inference endpoints
- [ ] API integration
- [ ] Monitoring and feedback loop

## Contributing

See the main [Simplex documentation](../simplex-docs/README.md) for contribution guidelines.

## License

Apache 2.0 - See LICENSE file.
