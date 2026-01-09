# TASK-003: Convert Training Pipeline to Simplex

**Status**: Planned
**Priority**: Medium (Long-term)
**Created**: 2026-01-09
**Updated**: 2026-01-09
**Target Version**: 0.8.0+
**Depends On**: TASK-001 (Neural IR), TASK-002 (Cognitive Models)

---

## Overview

Port the entire model training pipeline from Python to pure Simplex. This achieves two critical goals:

1. **Dogfooding**: Use Simplex to train Simplex cognitive models - proving the language's capabilities for production ML workloads
2. **Self-Improvement Loop**: Simplex-generated seed models can improve Simplex itself, creating a virtuous cycle

### Current State: Python

```
training/
├── scripts/
│   ├── train_all_specialists.py      # 33 specialist generators (~1600 LOC)
│   ├── train_context_protocol.py     # Context protocol training
│   ├── train_confidence_calibration.py # Confidence calibration
│   ├── train_belief_revision.py      # Belief revision training
│   ├── train_neural_ir_gates.py      # Neural IR/Gates training
│   ├── train_specialists.py          # Original specialist trainer
│   ├── curate_datasets.py            # Dataset curation
│   ├── curate_all_datasets.py        # Full dataset curator
│   ├── evaluate_model.py             # Evaluation framework
│   ├── export_to_gguf.py             # GGUF export for Ollama
│   └── run_full_training.py          # Orchestration script
├── configs/
│   └── specialists_catalog.yaml      # Specialist definitions
└── tests/
    ├── test_specialists.py
    └── test_data_generation.py
```

### Target State: Simplex

```
training/
├── src/
│   ├── main.sx                       # CLI entry point
│   ├── train/
│   │   ├── mod.sx
│   │   ├── specialists.sx            # All specialist training
│   │   ├── context_protocol.sx       # Context protocol training
│   │   ├── confidence.sx             # Confidence calibration
│   │   ├── belief_revision.sx        # Belief revision training
│   │   └── neural_ir.sx              # Neural IR/Gates training
│   ├── data/
│   │   ├── mod.sx
│   │   ├── generators.sx             # Synthetic data generators
│   │   └── curators.sx               # Dataset curation
│   ├── eval/
│   │   ├── mod.sx
│   │   └── benchmarks.sx             # Evaluation framework
│   └── export/
│       ├── mod.sx
│       └── gguf.sx                   # GGUF export
├── configs/
│   └── specialists.sx                # Specialist definitions (typed)
└── tests/
    └── training_tests.sx
```

---

## Background

### Why This Matters

1. **Proof of Capability**: If Simplex cannot train its own models, it's not a serious language for AI/ML
2. **Unified Toolchain**: Developers use one language for both application logic and model training
3. **Cognitive Integration**: Training code can leverage Simplex cognitive primitives (beliefs, Anima, Mnemonic)
4. **Performance**: Native Neural IR compilation can potentially outperform Python/PyTorch for inference
5. **Self-Hosting**: Simplex models trained by Simplex - the ultimate dogfooding

### Current Python Dependencies

| Library | Purpose | Simplex Replacement |
|---------|---------|---------------------|
| PyTorch | Tensor operations, autograd | simplex-tensor (new) |
| Transformers | Model loading, tokenization | simplex-models (new) |
| PEFT | LoRA adapters | Native LoRA support |
| Datasets | Data loading | simplex-data (new) |
| bitsandbytes | Quantization | Native quantization |
| Faker | Synthetic data | simplex-fake (new) |
| trl | SFT training | Native SFT |

---

## Technical Challenges

### 1. Tensor Operations Library

**Problem**: Simplex has no native tensor library. PyTorch provides 2000+ operations.

**Solution**: Create `simplex-tensor` with three tiers:

```simplex
// Tier 1: Core tensor type with Neural IR integration
type Tensor<T, Shape: TensorShape> {
    data: Buffer<T>,
    shape: Shape,
    device: Device,
    requires_grad: bool,
}

// Tier 2: Essential operations (start here)
impl<T: Numeric> Tensor<T, S> {
    fn matmul<S2>(self, other: Tensor<T, S2>) -> Tensor<T, _>;
    fn add(self, other: Tensor<T, S>) -> Tensor<T, S>;
    fn mul(self, other: Tensor<T, S>) -> Tensor<T, S>;
    fn softmax(self, dim: i32) -> Tensor<T, S>;
    fn layer_norm(self, normalized_shape: &[i32]) -> Tensor<T, S>;
    fn gelu(self) -> Tensor<T, S>;
    fn attention(q: Self, k: Self, v: Self, mask: Option<Self>) -> Self;
}

// Tier 3: GPU backend via Neural IR
@gpu
fn batched_matmul(a: Tensor, b: Tensor) -> Tensor {
    // Compiles to CUDA/ROCm via Neural IR
}
```

**Implementation Strategy**:
- Phase 1: CPU reference implementation in pure Simplex
- Phase 2: BLAS/LAPACK FFI bindings for performance
- Phase 3: GPU backend via Neural IR compilation

### 2. Automatic Differentiation

**Problem**: Training requires backpropagation. PyTorch autograd is ~50K LOC.

**Solution**: Leverage Neural IR's built-in differentiability.

```simplex
// Neural gates are already differentiable in training mode
neural_gate softmax_gate(logits: Tensor) -> Tensor {
    logits.softmax(dim: -1)
}

// Training mode automatically builds computation graph
// Simplex compiler handles gradient computation

// Explicit gradient API when needed
fn train_step(model: &mut Model, batch: Batch) -> Loss {
    let loss = model.forward(batch);

    // Gradients computed via Neural IR
    let grads = loss.backward();

    // Apply gradients
    optimizer.step(model, grads);

    loss
}
```

**Note**: Neural IR (TASK-001) already solves the hard problem. This task connects it to training.

### 3. Model Architecture Definition

**Problem**: Need to define transformer architectures in Simplex.

**Solution**: Type-safe model building DSL.

```simplex
// Declarative model definition
struct SimplexCognitive7B {
    embed: Embedding<vocab_size: 151936, dim: 4096>,
    layers: [TransformerBlock; 32],
    head: LMHead<dim: 4096, vocab_size: 151936>,
}

impl SimplexCognitive7B {
    fn forward(&self, input_ids: Tensor<i64, [B, L]>) -> Tensor<f32, [B, L, V]> {
        let hidden = self.embed.forward(input_ids);

        for layer in &self.layers {
            hidden = layer.forward(hidden);
        }

        self.head.forward(hidden)
    }
}

// TransformerBlock with attention
struct TransformerBlock {
    attention: MultiHeadAttention<heads: 32, dim: 4096>,
    mlp: MLP<dim: 4096, hidden: 14336>,
    norm1: RMSNorm<dim: 4096>,
    norm2: RMSNorm<dim: 4096>,
}

impl TransformerBlock {
    fn forward(&self, x: Tensor) -> Tensor {
        // Pre-norm architecture
        let attn_out = self.attention.forward(self.norm1.forward(x));
        let x = x + attn_out;
        let mlp_out = self.mlp.forward(self.norm2.forward(x));
        x + mlp_out
    }
}
```

### 4. LoRA Adapter Support

**Problem**: Need efficient fine-tuning via LoRA adapters.

**Solution**: First-class LoRA support in Simplex.

```simplex
// LoRA configuration
struct LoraConfig {
    r: i32,           // Rank
    alpha: f32,       // Scaling factor
    dropout: f32,
    target_modules: List<String>,
}

// Apply LoRA to a model
fn apply_lora<M: Model>(model: M, config: LoraConfig) -> LoraModel<M> {
    let lora_layers = model.named_modules()
        .filter(|(name, _)| config.target_modules.contains(name))
        .map(|(name, module)| {
            (name, LoraLayer::new(module, config.r, config.alpha))
        })
        .collect();

    LoraModel { base: model, lora_layers }
}

// LoRA layer implementation
struct LoraLayer<T: Linear> {
    base: T,
    lora_a: Tensor<f32, [R, In]>,   // Low-rank matrix A
    lora_b: Tensor<f32, [Out, R]>,  // Low-rank matrix B
    scaling: f32,
}

impl<T: Linear> LoraLayer<T> {
    fn forward(&self, x: Tensor) -> Tensor {
        let base_out = self.base.forward(x);
        let lora_out = x.matmul(self.lora_a).matmul(self.lora_b) * self.scaling;
        base_out + lora_out
    }
}
```

### 5. Tokenization

**Problem**: Need tokenizer support (BPE, SentencePiece, etc.).

**Solution**: `simplex-tokenizers` library with FFI fallback.

```simplex
// Tokenizer trait
trait Tokenizer {
    fn encode(&self, text: &str) -> List<i64>;
    fn decode(&self, ids: &[i64]) -> String;
    fn vocab_size(&self) -> usize;
}

// BPE tokenizer (pure Simplex)
struct BPETokenizer {
    vocab: HashMap<String, i64>,
    merges: Vec<(String, String)>,
    special_tokens: HashMap<String, i64>,
}

impl Tokenizer for BPETokenizer {
    fn encode(&self, text: &str) -> List<i64> {
        // BPE encoding algorithm
        let tokens = self.pre_tokenize(text);
        let mut ids = Vec::new();

        for token in tokens {
            let merged = self.apply_bpe(token);
            ids.extend(merged.iter().map(|t| self.vocab[t]));
        }

        ids
    }
}

// FFI fallback to HuggingFace tokenizers for complex cases
@ffi("tokenizers")
extern fn load_hf_tokenizer(path: &str) -> *mut HFTokenizer;
```

### 6. Dataset Loading

**Problem**: Need to load various dataset formats efficiently.

**Solution**: `simplex-data` with streaming support.

```simplex
// Dataset trait
trait Dataset<T> {
    fn len(&self) -> usize;
    fn get(&self, idx: usize) -> T;
    fn iter(&self) -> impl Iterator<Item = T>;
}

// Streaming dataset for large files
struct JsonlDataset {
    path: Path,
    line_offsets: Vec<u64>,  // Precomputed for random access
}

impl Dataset<TrainingExample> for JsonlDataset {
    fn get(&self, idx: usize) -> TrainingExample {
        let offset = self.line_offsets[idx];
        let line = self.read_line_at(offset);
        TrainingExample::from_json(&line)
    }
}

// Dataloader with batching and shuffling
struct DataLoader<D: Dataset<T>, T> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    num_workers: usize,
}

impl<D: Dataset<T>, T> DataLoader<D, T> {
    async fn iter_batches(&self) -> impl AsyncIterator<Item = Vec<T>> {
        // Parallel batch loading via actors
        let workers: Vec<DataWorker> = (0..self.num_workers)
            .map(|_| spawn DataWorker::new(&self.dataset))
            .collect();

        // Yield batches as they're ready
        for batch_indices in self.batch_indices() {
            let batch = workers.fetch_batch(batch_indices).await;
            yield batch;
        }
    }
}
```

### 7. Synthetic Data Generation

**Problem**: Port 33 specialist generators from Python to Simplex.

**Solution**: Leverage Simplex's expressive syntax.

```simplex
// Generator trait
trait DataGenerator {
    type Output;
    fn generate(&self, rng: &mut Rng) -> Self::Output;
}

// Invoice processing generator
struct InvoiceGenerator {
    fake: Faker,  // simplex-fake library
}

impl DataGenerator for InvoiceGenerator {
    type Output = TrainingExample;

    fn generate(&self, rng: &mut Rng) -> TrainingExample {
        let vendor = self.fake.company();
        let inv_num = format!("INV-{}", rng.gen_range(10000..99999));
        let items = rng.gen_range(2..5);
        let subtotal = rng.gen_range(100.0..10000.0);
        let tax = subtotal * 0.1;
        let total = subtotal + tax;

        let prompt = format!(r#"
Extract line items and totals from this invoice:

INVOICE #{inv_num}
From: {vendor}
Date: {date}

Items:
{items_list}

Subtotal: ${subtotal:.2}
Tax (10%): ${tax:.2}
TOTAL: ${total:.2}
"#, /* ... */);

        let response = json!({
            "invoice_number": inv_num,
            "vendor": vendor,
            "line_items": items,
            "subtotal": subtotal,
            "tax": tax,
            "total": total,
            "currency": "USD"
        });

        TrainingExample {
            text: format!("{prompt}\n\nAssistant: {response}\n[confidence: 0.95]")
        }
    }
}

// Register all generators
const GENERATORS: Map<&str, Box<dyn DataGenerator>> = map! {
    "invoice_processing" => Box::new(InvoiceGenerator::new()),
    "document_extraction" => Box::new(DocumentExtractionGenerator::new()),
    "contract_analysis" => Box::new(ContractAnalysisGenerator::new()),
    // ... 30 more generators
};
```

---

## Implementation Phases

### Phase 1: Foundation Libraries (Q2 2026)

**Deliverables**:
1. `simplex-tensor` - Core tensor type and CPU operations
2. `simplex-fake` - Synthetic data generation (port of Faker)
3. `simplex-data` - Dataset loading and streaming
4. `simplex-tokenizers` - BPE tokenizer implementation

**Success Criteria**:
- [ ] Tensor matmul, add, mul, softmax implemented
- [ ] Can generate synthetic training data for 5 specialists
- [ ] Can load JSONL datasets with streaming
- [ ] BPE tokenizer passes compatibility tests

**LOC Estimate**: ~5,000 lines Simplex

### Phase 2: Model Architecture (Q2-Q3 2026)

**Deliverables**:
1. `simplex-models` - Model building blocks (Linear, Attention, MLP, etc.)
2. Transformer architecture definition
3. Model loading from safetensors/GGUF
4. LoRA adapter support

**Success Criteria**:
- [ ] Can define SimplexCognitive7B architecture
- [ ] Can load pretrained Qwen weights
- [ ] Forward pass produces correct logits
- [ ] LoRA application and merging works

**LOC Estimate**: ~8,000 lines Simplex

### Phase 3: Training Loop (Q3 2026)

**Deliverables**:
1. Autograd integration with Neural IR
2. Optimizer implementations (AdamW, SGD)
3. Learning rate schedulers
4. Gradient accumulation and mixed precision
5. Checkpointing and resumption

**Success Criteria**:
- [ ] Can train a simple model on toy data
- [ ] Gradients match PyTorch reference
- [ ] AdamW optimizer converges correctly
- [ ] Training can resume from checkpoint

**LOC Estimate**: ~4,000 lines Simplex

### Phase 4: Port All Generators (Q3-Q4 2026)

**Deliverables**:
1. Port all 33 specialist generators
2. Data quality validation
3. Generation throughput benchmarks
4. CLI for data generation

**Success Criteria**:
- [ ] All 33 generators produce valid training data
- [ ] Generated data quality matches Python version
- [ ] Generation speed within 2x of Python
- [ ] CLI: `sxc train --generate --specialist all`

**LOC Estimate**: ~6,000 lines Simplex

### Phase 5: Training Scripts (Q4 2026)

**Deliverables**:
1. Port `train_all_specialists.sx`
2. Port `train_context_protocol.sx`
3. Port `train_confidence_calibration.sx`
4. Port `train_belief_revision.sx`
5. Port `train_neural_ir_gates.sx`
6. Port `evaluate_model.sx`

**Success Criteria**:
- [ ] Can train specialist LoRA adapter end-to-end
- [ ] Trained models pass evaluation benchmarks
- [ ] Training time within 1.5x of Python baseline
- [ ] Full training pipeline runs unattended

**LOC Estimate**: ~5,000 lines Simplex

### Phase 6: Export and Distribution (Q4 2026 - Q1 2027)

**Deliverables**:
1. GGUF export (`export_to_gguf.sx`)
2. Ollama integration
3. Model card generation
4. S3 upload automation
5. models.senuamedia.com integration

**Success Criteria**:
- [ ] Can export to GGUF Q4_K_M, Q8_0
- [ ] Exported models work in Ollama
- [ ] Automated pipeline: train → evaluate → export → deploy
- [ ] Integration with model repository website

**LOC Estimate**: ~2,000 lines Simplex

### Phase 7: GPU Acceleration (Q1 2027)

**Deliverables**:
1. Neural IR GPU backend for training
2. CUDA/ROCm code generation
3. Multi-GPU support (data parallel)
4. Memory optimization (gradient checkpointing)

**Success Criteria**:
- [ ] Training runs on NVIDIA GPU
- [ ] 5x+ speedup vs CPU
- [ ] Can train 8B model on 24GB GPU
- [ ] Multi-GPU scales linearly

**LOC Estimate**: ~3,000 lines Simplex (mostly compiler work)

---

## Technical Dependencies

| Dependency | Type | Status | Notes |
|------------|------|--------|-------|
| TASK-001 Neural IR | Internal | Complete | Required for autograd |
| TASK-002 Cognitive Models | Internal | In Progress | Defines model targets |
| LLVM 18+ | External | Available | Backend for CPU/GPU |
| CUDA Toolkit | External | Optional | GPU acceleration |
| safetensors | External | Available | Weight loading |
| GGUF spec | External | Available | Export format |

---

## File Mapping

| Python File | Simplex File | LOC (Est.) |
|-------------|--------------|------------|
| `train_all_specialists.py` | `train/specialists.sx` | 800 |
| `train_context_protocol.py` | `train/context_protocol.sx` | 400 |
| `train_confidence_calibration.py` | `train/confidence.sx` | 400 |
| `train_belief_revision.py` | `train/belief_revision.sx` | 400 |
| `train_neural_ir_gates.py` | `train/neural_ir.sx` | 500 |
| `curate_datasets.py` | `data/curators.sx` | 600 |
| `curate_all_datasets.py` | `data/curators.sx` | (merged) |
| `evaluate_model.py` | `eval/benchmarks.sx` | 500 |
| `export_to_gguf.py` | `export/gguf.sx` | 400 |
| `run_full_training.py` | `main.sx` | 300 |
| **Total** | | **~4,300** |

Plus libraries:
| Library | LOC (Est.) |
|---------|------------|
| `simplex-tensor` | 5,000 |
| `simplex-models` | 8,000 |
| `simplex-data` | 2,000 |
| `simplex-tokenizers` | 2,000 |
| `simplex-fake` | 1,500 |
| **Total Libraries** | **~18,500** |

**Grand Total**: ~23,000 lines of Simplex

---

## Milestones

| Milestone | Target | Deliverable |
|-----------|--------|-------------|
| M1: First Tensor Op | Q2 2026 | `simplex-tensor` matmul works |
| M2: First Generator | Q2 2026 | One specialist generator in Simplex |
| M3: Model Forward | Q3 2026 | Can run inference on loaded model |
| M4: First Gradient | Q3 2026 | Backward pass computes gradients |
| M5: First Training | Q3 2026 | Train toy model successfully |
| M6: First Specialist | Q4 2026 | Train one specialist LoRA in Simplex |
| M7: Full Pipeline | Q4 2026 | All 33 specialists trainable |
| M8: GPU Training | Q1 2027 | Production-speed GPU training |
| M9: Self-Trained Model | Q1 2027 | Simplex model trained entirely by Simplex |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Neural IR autograd insufficient | Low | High | Already designed for differentiability |
| GPU backend too complex | Medium | High | Start CPU-only, add GPU incrementally |
| Performance too slow | Medium | Medium | FFI escape hatch to optimized C/CUDA |
| Tokenizer compatibility | Low | Medium | FFI to HuggingFace tokenizers |
| Model weight loading issues | Low | Medium | Use safetensors (simple format) |

---

## Open Questions

1. **FFI Strategy**: How much to implement in pure Simplex vs FFI to optimized libraries?
   - Recommendation: Pure Simplex for correctness, FFI for performance-critical paths

2. **Distributed Training**: Do we need multi-node support?
   - Recommendation: Defer to post-v1.0; single-node multi-GPU sufficient initially

3. **Mixed Precision**: BF16, FP16, or FP32 default?
   - Recommendation: BF16 for training (matches PyTorch), FP32 for CPU inference

4. **Checkpointing Format**: Custom or compatible with PyTorch?
   - Recommendation: safetensors for weights, custom for optimizer state

5. **CI/CD Integration**: How to test training code in CI?
   - Recommendation: Small model + tiny dataset for smoke tests; full training offline

---

## Success Definition

This task is complete when:

1. **All training scripts run in pure Simplex** - No Python required
2. **Trained models match quality** - Evaluation metrics within 5% of Python baseline
3. **Performance is acceptable** - Training time within 2x of Python/PyTorch
4. **Self-hosting achieved** - At least one production simplex-cognitive model trained entirely by Simplex
5. **Documentation complete** - Users can train custom specialists in Simplex

---

## References

- TASK-001: Neural IR and Neural Gates
- TASK-002: Simplex Cognitive Models
- PyTorch internals documentation
- GGUF specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- safetensors format: https://huggingface.co/docs/safetensors
- LoRA paper: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)

---

*This task represents the ultimate dogfooding milestone: Simplex training Simplex.*
