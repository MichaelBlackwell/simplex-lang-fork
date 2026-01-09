# Pilot Validation Pipeline

**Purpose**: Validate the training approach before spending money on all 33 specialists.

## Why This Exists

Before committing $60+ to train all specialists, we need to answer:

1. **Does training on real data improve the model?**
2. **Are our evaluation metrics meaningful?**
3. **Is the base model (Qwen3) appropriate for these tasks?**
4. **Are our success thresholds achievable?**

This pipeline validates with 3 diverse pilots:
- **Sentiment Analysis** - Classification task (easy)
- **SQL Generation** - Structured generation (medium)
- **Invoice Processing** - Information extraction (hard)

## Quick Start

```bash
# Install dependencies
cd /Users/rod/code/simplex-lang/training
pip install -r requirements.txt
pip install datasets sqlparse

# Run quick local test (CPU, ~30 min total)
cd validation
chmod +x run_pilots.sh
./run_pilots.sh --local-test

# Run baseline only (measure before training)
./run_pilots.sh --baseline-only

# Run full training (needs GPU)
./run_pilots.sh --full
```

## Pipeline Steps

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Load Real Datasets                                  │
│  - SST-2 + IMDB (sentiment)                                 │
│  - WikiSQL + Spider (SQL)                                   │
│  - CORD + synthetic (invoice)                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Baseline Evaluation                                 │
│  - Run untrained model on test set                          │
│  - Measure accuracy/F1 BEFORE training                      │
│  - Establish improvement target                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Training with Validation                            │
│  - LoRA fine-tuning on real data                            │
│  - Early stopping on validation loss                        │
│  - Save best checkpoint                                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Post-Training Evaluation                            │
│  - Run trained model on same test set                       │
│  - Compare to baseline                                      │
│  - Check against success thresholds                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 5: Generate Report                                     │
│  - Baseline vs Trained comparison                           │
│  - Pass/Fail verdict                                        │
│  - Recommendations                                          │
└─────────────────────────────────────────────────────────────┘
```

## Success Criteria

| Specialist | Metric | Threshold | Rationale |
|------------|--------|-----------|-----------|
| Sentiment | Accuracy | ≥ 85% | SST-2 SOTA is ~96%, 85% is achievable |
| Sentiment | Macro F1 | ≥ 84% | Balanced across classes |
| SQL | Exact Match | ≥ 60% | SQL is hard, 60% is good for LoRA |
| SQL | Valid SQL | ≥ 90% | Basic syntax should work |
| Invoice | Field F1 | ≥ 75% | Extraction is noisy |
| Invoice | Total Accuracy | ≥ 80% | Numbers should be correct |

## Directory Structure

```
validation/
├── README.md                    # This file
├── run_pilots.sh                # Main runner script
│
├── datasets/
│   └── loaders.py               # Real dataset loaders
│
├── evaluation/
│   └── metrics.py               # Task-specific metrics
│
├── pilots/
│   └── train_pilot.py           # Training script
│
├── test_sets/
│   └── curated_tests.json       # Manually curated test cases
│
└── results/                     # Output directory
    ├── sentiment_baseline.json
    ├── sentiment_trained.json
    ├── sentiment_report.txt
    └── ...
```

## Datasets Used

### Sentiment Analysis
| Dataset | Size | License | Use |
|---------|------|---------|-----|
| SST-2 | 67K | Open | Train/Val |
| IMDB | 50K | Open | Train/Test |

### SQL Generation
| Dataset | Size | License | Use |
|---------|------|---------|-----|
| WikiSQL | 80K | CC BY-SA | Train/Val/Test |
| Spider | 10K | Academic | Train/Val |

### Invoice Processing
| Dataset | Size | License | Use |
|---------|------|---------|-----|
| CORD | 1K | Research | Train/Val/Test |
| Synthetic | 2K | Generated | Supplement |

## Expected Results

### Baseline (Untrained Qwen3-8B)
- Sentiment: ~70-75% accuracy (zero-shot is decent)
- SQL: ~30-40% exact match (needs training)
- Invoice: ~40-50% field F1 (needs training)

### After Training (Target)
- Sentiment: ≥85% accuracy (+10-15%)
- SQL: ≥60% exact match (+20-30%)
- Invoice: ≥75% field F1 (+25-35%)

## Cost Estimate

| Mode | Time | Cost |
|------|------|------|
| Local test (CPU) | ~30 min/specialist | Free |
| Baseline only | ~10 min/specialist | Free |
| Full training (GPU) | ~2 hr/specialist | ~$3-5 |
| **All 3 pilots (full)** | **~6 hours** | **~$10-15** |

## Interpreting Results

### PASSED
```
VERDICT: SUCCESS - Model meets quality thresholds
```
Training works. Proceed to train all 33 specialists.

### PARTIAL
```
VERDICT: PARTIAL - Training improved but below threshold
```
Training helps but not enough. Consider:
- More training data
- Longer training
- Different base model
- Adjusted thresholds

### FAILED
```
VERDICT: FAILED - Training did not improve model
```
Something is wrong. Investigate:
- Data quality issues
- Wrong hyperparameters
- Task too hard for model size

## Next Steps

After pilots validate:

1. **If all pass**: Run full training
   ```bash
   cd ../scripts
   python train_all_specialists.py --all
   ```

2. **If partial**: Iterate on failing specialists
   - Try larger base model
   - Add more real data
   - Adjust LoRA rank

3. **If failed**: Revisit approach
   - Different base model
   - Different training strategy
   - Lower expectations

## Troubleshooting

### "CUDA out of memory"
Reduce batch size or use CPU:
```bash
./run_pilots.sh --local-test
```

### "datasets not found"
Install HuggingFace datasets:
```bash
pip install datasets
```

### Slow training
Use GPU or cloud instance:
```bash
# On AWS g5.xlarge
./run_pilots.sh --full
```

### Baseline already passes threshold
The base model is good enough! Consider:
- Raising the threshold
- Skipping training for this specialist
- Using the base model directly
