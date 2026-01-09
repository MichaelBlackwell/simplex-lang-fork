# Simplex v0.6.0 Release Notes

**Release Date:** 2026-01-08
**Codename:** Neural IR

---

## Overview

Simplex v0.6.0 introduces Neural IR (Intermediate Representation) and Neural Gates, enabling differentiable program execution. This release bridges the gap between traditional programming and machine learning by making program logic itself learnable and optimizable via gradient descent.

---

## Major Features

### Neural Gates with Gumbel-Softmax

Neural Gates transform discrete control flow into differentiable operations during training, then compile back to efficient discrete branches for inference.

```simplex
// Define a learnable gate
neural_gate should_retry(confidence: f64) -> bool {
    confidence > 0.7
}

// Training mode: compiles to sigmoid((confidence - 0.7) * temperature)
// Inference mode: compiles to standard conditional (zero overhead)
```

**Key Capabilities:**
- Gumbel-Softmax for categorical choices
- Temperature annealing during training
- Straight-Through Estimator (STE) for hard constraints
- Zero overhead in inference mode

### Dual Compilation Modes

```bash
# Training mode - differentiable execution
sxc build --mode=train model.sx

# Inference mode - optimized discrete execution
sxc build --mode=infer model.sx
```

### Contract Logic for Probabilistic Verification

Contracts ensure safety guarantees even with probabilistic gates:

```simplex
neural_gate memory_safe_path(analysis: SecurityAnalysis) -> bool
    requires analysis.confidence > 0.95  // Minimum confidence
    ensures result => no_buffer_overflow  // Guaranteed property
    fallback => conservative_path()       // Fallback when unsure
{
    analysis.is_safe
}
```

**Contract Types:**
- `requires`: Pre-conditions on gate inputs
- `ensures`: Post-conditions when gate fires
- `invariant`: Properties across gate transitions
- `fallback`: Handler when confidence is below threshold

### Hardware-Aware Compilation

Automatic graph partitioning targets the right hardware:

```simplex
@gpu
neural_gate batch_classifier(inputs: List<Embedding>) -> List<Label> {
    // Runs on GPU - batch tensor operations
}

@cpu
fn process_result(label: Label) -> Action {
    // Runs on CPU - branching logic
}

// Automatic targeting (compiler decides)
neural_gate smart_router(query: String) -> Specialist {
    // Compiler analyzes operation types and places accordingly
}
```

**Supported Targets:**
| Target | Best For |
|--------|----------|
| CPU | Control flow, I/O, syscalls |
| GPU | Matrix ops, batch parallel |
| NPU | Cognitive inference, SLM calls |

### Structural Pruning

After training, the compiler eliminates dead paths:

- **Weight Magnitude Pruning**: Remove gates where |weight| < threshold
- **Activation Pruning**: Remove paths that fire < threshold% of training
- **Gradient Pruning**: Remove gates with consistently zero gradients
- **Structured Pruning**: Remove entire subgraphs, not just edges

**Result:** Final binaries approach C/Rust footprint with inference speed matching traditional compilation.

### Superposition Memory Model

Handles "50% true / 50% false" states during training:

```simplex
// Lazy Evaluation (default) - allocate only dominant path
let result = match branch_selector(x) {
    A => compute_a(),  // Only allocated if P(A) > threshold
    B => compute_b(),
}

// Speculative Execution - allocate all, weight results
@speculative
let result = match branch_selector(x) {
    A => compute_a(),  // All allocated, weighted, low-weight GC'd
    B => compute_b(),
}
```

**New Type: WeightedRef<T>**
```simplex
type WeightedRef<T> = {
    ptr: *T,
    weight: f64,  // 0.0 to 1.0
    allocated: bool,
}
```

---

## Technical Details

### Gradient Flow

Neural Gates support full backpropagation:
- Automatic differentiation through Gumbel-Softmax
- Gradient clipping at gate boundaries
- Configurable temperature schedules
- Integration with existing Simplex autograd

### Compilation Pipeline

```
Source Code
    │
    ▼
Lexer/Parser (neural_gate keyword)
    │
    ▼
Anima Graph Construction
    │
    ├─────────────────────────────────┐
    ▼                                 ▼
Training Mode                    Inference Mode
(Differentiable)                 (Discrete)
    │                                 │
    ▼                                 ▼
Gumbel-Softmax                   Standard Branch
    │                                 │
    ▼                                 ▼
LLVM IR + Autograd               LLVM IR (optimized)
    │                                 │
    ▼                                 ▼
Training Binary                  Production Binary
```

### Performance

| Metric | Training Mode | Inference Mode |
|--------|---------------|----------------|
| Gate overhead | ~5% vs discrete | 0% (identical to if) |
| Memory | +20% for gradients | Standard |
| Binary size | +15% for autograd | Within 2x of C |

---

## Migration Guide

### From v0.5.x

1. **No breaking changes** - existing code compiles unchanged
2. **New keyword**: `neural_gate` (reserved, don't use as identifier)
3. **New annotations**: `@cpu`, `@gpu`, `@npu`, `@speculative`, `@checkpoint`
4. **New contracts**: `requires`, `ensures`, `invariant`, `fallback`

### Adopting Neural Gates

```simplex
// Before: Static threshold
fn should_process(score: f64) -> bool {
    score > 0.5
}

// After: Learnable threshold
neural_gate should_process(score: f64) -> bool {
    score > 0.5  // Threshold becomes learnable during training
}
```

---

## Known Limitations

1. GPU targeting requires CUDA 11.0+ or ROCm 5.0+
2. NPU support currently limited to Apple Neural Engine
3. Checkpoint mode has 2x memory overhead
4. Maximum gate depth of 64 in single compilation unit

---

## What's Next

v0.7.0 will introduce **Real-Time Continuous Learning** (TASK-004), enabling AI specialists to learn during inference without batch retraining.

---

## Credits

Neural IR was designed and implemented by Rod Higgins ([@senuamedia](https://github.com/senuamedia)).

Key influences:
- Gumbel-Softmax: Jang et al., "Categorical Reparameterization with Gumbel-Softmax" (2017)
- Differentiable Programming: Innes et al. (2019)
- Enzyme: Automatic differentiation for LLVM
