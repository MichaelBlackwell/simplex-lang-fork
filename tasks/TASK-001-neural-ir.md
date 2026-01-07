# TASK-001: Neural IR and Neural Gates

**Status**: Planning
**Priority**: High
**Created**: 2026-01-08
**Updated**: 2026-01-08
**Target Version**: 0.6.0

## Overview

Implement Neural IR (Intermediate Representation) and Neural Gates to enable differentiable program execution in Simplex. This allows programs to be optimized via gradient descent and enables learnable control flow.

## Background

Simplex v0.5.1 has cognitive primitives (Anima, beliefs, HiveMnemonic) that conceptually align with neural computation:

| Current Feature | Neural IR Analog |
|-----------------|------------------|
| Belief confidence (0.0-1.0) | Learnable weights |
| Belief thresholds (30/50/70%) | Activation functions |
| Semantic routing | Soft attention |
| HiveMnemonic embeddings | Shared embedding layer |
| Per-hive SLM | Shared transformer backbone |

The gap: these are not currently differentiable. We cannot backpropagate through program logic.

---

## Core Technical Challenges

### 1. Differentiability of Control Flow (The "If" Problem)

**Problem**: Standard discrete branches (`if-else`) have zero gradient. You cannot backpropagate through a hard conditional.

**Solution**: Implement **Gumbel-Softmax** and **Continuous Relaxation** in the compiler.

```simplex
// Source code
neural_gate should_retry(confidence: f64) -> bool {
    confidence > 0.7
}

// Training mode compilation (differentiable)
// Transforms to: sigmoid((confidence - 0.7) * temperature)
// Temperature anneals during training (high → low)

// Inference mode compilation (discrete)
// Snaps back to: confidence > 0.7
// Full performance, no gradient overhead
```

**Implementation**:
- Compiler flag: `--mode=train` vs `--mode=infer`
- Training mode: all Neural Gates compile to sigmoid/softmax approximations
- Inference mode: gates compile to discrete branches (zero overhead)
- Gumbel-Softmax for categorical choices (selecting from N options)
- Straight-Through Estimator (STE) as fallback for hard constraints

### 2. Probabilistic Formal Verification

**Problem**: Soft logic loses predictability. If a gate is "85% true," how do you verify correctness? How do you prevent the neural part from causing system crashes?

**Solution**: Implement **Contract Logic** with confidence bounds.

```simplex
neural_gate memory_safe_path(analysis: SecurityAnalysis) -> bool
    requires analysis.confidence > 0.95  // Must exceed 95% to take safe path
    ensures result => no_buffer_overflow  // If true, guarantee holds
{
    analysis.is_safe
}

// Compiler enforces:
// - Gate cannot return true unless confidence > 0.95
// - Violation triggers fallback path or runtime error
// - Contracts are checked at compile time where possible
```

**Contract Types**:
- `requires`: Pre-conditions on gate inputs (minimum confidence thresholds)
- `ensures`: Post-conditions guaranteed when gate fires
- `invariant`: Properties that must hold across gate transitions
- `fallback`: Explicit handler when confidence is below threshold

**Verification Modes**:
- Static: Prove bounds at compile time via abstract interpretation
- Dynamic: Runtime confidence checks with graceful degradation
- Probabilistic: Monte Carlo verification for complex gate compositions

### 3. Hardware-Aware Anima Mapping

**Problem**: CPUs excel at branching; GPUs/TPUs excel at tensor ops. A Neural IR that runs everything on one device wastes resources.

**Solution**: **Hybrid Targeting** with automatic graph partitioning.

```
Anima Graph Analysis
        │
        ▼
┌───────────────────────────────────────┐
│         Graph Partitioner             │
│  - Identify Hard Gates (deterministic)│
│  - Identify Neural Gates (tensor ops) │
│  - Identify Memory Gates (I/O bound)  │
└───────────────────────────────────────┘
        │
        ├──────────────┬──────────────┐
        ▼              ▼              ▼
   ┌────────┐    ┌──────────┐   ┌─────────┐
   │  CPU   │    │ GPU/TPU  │   │   NPU   │
   │ Target │    │  Target  │   │ Target  │
   └────────┘    └──────────┘   └─────────┘
   Hard Gates    Neural Gates   Cognitive
   Control flow  Matrix ops     Inference
   I/O, syscalls Batch parallel SLM calls
```

**Implementation**:
- Annotate gates with `@cpu`, `@gpu`, `@npu` hints (optional)
- Compiler analyzes data dependencies and operation types
- Automatic placement when no hint provided
- Data marshalling between devices handled by runtime
- Async execution with dependency tracking

```simplex
// Explicit targeting
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
    // Compiler analyzes: embedding lookup + softmax = GPU
    // Routes accordingly
}
```

### 4. Structural Pruning at IR Level

**Problem**: Neural networks are dense (everything connects); code is sparse (only call what you need). Naively "neuralizing" code creates bloat.

**Solution**: **Dead Path Elimination** in the Anima Graph.

```simplex
// After training, some gate weights may be ~0
neural_gate rarely_used_path(x: f64) -> bool {
    // Training shows this gate fires <0.1% of the time
    // Weight effectively zero
}

// Pruning pass identifies:
// - Gates with weight < threshold (e.g., 0.01)
// - Paths that are statistically dead
// - Unreachable branches after weight collapse
```

**Pruning Strategies**:
1. **Weight Magnitude Pruning**: Remove gates where |weight| < ε
2. **Activation Pruning**: Remove paths that fire < threshold% of training
3. **Gradient Pruning**: Remove gates with consistently zero gradients
4. **Structured Pruning**: Remove entire subgraphs, not just individual edges

**Compilation Phases**:
```
Source → Anima Graph → Training → Pruned Graph → Optimized Binary
                           │
                           ▼
                    Pruning Analysis
                    - Weight magnitudes
                    - Activation statistics
                    - Gradient flow
```

**Benefits**:
- Final binary approaches C/Rust footprint
- Inference speed matches traditional compilation
- Only "hot" neural paths remain

### 5. Superposition Memory Model

**Problem**: If a gate is 50% true and 50% false, does the program allocate memory for both branches? This leads to "ghost allocations" and memory leaks.

**Solution**: Define explicit **Weighted Pointer** semantics and **Lazy Branching**.

```simplex
// Superposition state during training
neural_gate branch_selector(x: f64) -> Branch {
    // Returns weighted distribution over branches
    // e.g., {A: 0.6, B: 0.3, C: 0.1}
}

// Memory model options:

// 1. Lazy Evaluation (default)
// Only allocate when branch is "observed" (confidence > threshold)
let result = match branch_selector(x) {
    A => compute_a(),  // Only allocated if P(A) > lazy_threshold
    B => compute_b(),
    C => compute_c(),
}

// 2. Speculative Execution
// Allocate all, weight results, GC unused
@speculative
let result = match branch_selector(x) {
    A => compute_a(),  // All allocated
    B => compute_b(),  // Results weighted
    C => compute_c(),  // Low-weight branches GC'd
}

// 3. Checkpoint-Restore
// Snapshot state, explore branches, restore
@checkpoint
let result = match branch_selector(x) {
    A => { checkpoint(); compute_a() },
    B => { restore(); compute_b() },
    C => { restore(); compute_c() },
}
```

**Memory Semantics**:

| Mode | Memory Behavior | Use Case |
|------|-----------------|----------|
| Lazy | Allocate only dominant path | Production inference |
| Speculative | Allocate all, weight, GC | Training with memory budget |
| Checkpoint | Snapshot/restore | Exact gradient computation |
| Pooled | Pre-allocate max, reuse | Real-time systems |

**Weighted Pointer Type**:
```simplex
// New type: WeightedRef<T>
// Represents a reference with associated probability

type WeightedRef<T> = {
    ptr: *T,
    weight: f64,  // 0.0 to 1.0
    allocated: bool,
}

// Runtime tracks weighted references
// GC reclaims when weight drops below threshold
// Or when superposition collapses to single branch
```

---

## Implementation Phases

### Phase 1: Neural Gates with Gumbel-Softmax (Jan 2026)

**Deliverables**:
1. `neural_gate` keyword in lexer/parser
2. Dual compilation: training mode (differentiable) / inference mode (discrete)
3. Gumbel-Softmax implementation for categorical gates
4. Temperature annealing during training
5. Basic gradient tracking and backprop

**Success Criteria**:
- [ ] Simple gate compiles in both modes
- [ ] Gradients flow correctly through soft gates
- [ ] Training loop updates gate parameters
- [ ] Inference mode has zero overhead vs normal `if`

### Phase 2: Contract Logic and Verification (Feb 2026)

**Deliverables**:
1. `requires`, `ensures`, `invariant`, `fallback` syntax
2. Static analysis for provable bounds
3. Runtime confidence checking
4. Graceful degradation when contracts fail

**Success Criteria**:
- [ ] Contracts compile and enforce
- [ ] Static analyzer catches obvious violations
- [ ] Runtime checks don't exceed 5% overhead
- [ ] Fallback paths execute correctly

### Phase 3: Hardware-Aware Compilation (Feb - Mar 2026)

**Deliverables**:
1. Graph partitioning algorithm
2. CPU/GPU/NPU code generation
3. `@cpu`, `@gpu`, `@npu` annotations
4. Automatic placement heuristics
5. Cross-device data marshalling

**Success Criteria**:
- [ ] Manual annotations work correctly
- [ ] Automatic placement matches manual for common patterns
- [ ] Data transfer overhead < 10% of computation time
- [ ] Mixed CPU/GPU program runs correctly

### Phase 4: Structural Pruning (Mar 2026)

**Deliverables**:
1. Weight magnitude pruning pass
2. Activation statistics collection
3. Dead path elimination
4. Binary size optimization

**Success Criteria**:
- [ ] Pruned binary within 2x of equivalent C
- [ ] Inference speed within 1.5x of static compilation
- [ ] Pruning removes >50% of trained gates (typical)

### Phase 5: Superposition Memory Model (Mar - Apr 2026)

**Deliverables**:
1. `WeightedRef<T>` type implementation
2. Lazy, Speculative, Checkpoint execution modes
3. Weighted GC algorithm
4. Memory safety proofs for each mode

**Success Criteria**:
- [ ] No memory leaks in any execution mode
- [ ] Lazy mode matches traditional memory usage
- [ ] Speculative mode bounded by configurable budget
- [ ] Checkpoint mode provides exact gradients

---

## Technical Dependencies

- LLVM for CPU code generation
- CUDA/ROCm for GPU targeting (optional)
- Custom autograd implementation (or integrate Enzyme)
- Modified GC for weighted references

## Research References

- Gumbel-Softmax: Jang et al., "Categorical Reparameterization with Gumbel-Softmax" (2017)
- Straight-Through Estimator: Bengio et al. (2013)
- Differentiable Programming: Innes et al., "Differentiable Programming" (2019)
- Neural Program Synthesis: various (for verification approaches)
- Enzyme: Automatic differentiation for LLVM

## Resources

- Blog post: https://blog.senuamedia.com/posts/neural-ir-next-step.html
- Anima spec: simplex-docs/spec/12-anima.md
- Cognitive Hive spec: simplex-docs/spec/09-cognitive-hive.md

---

## Notes

The key insight is that Simplex's cognitive architecture already thinks in terms of continuous values (belief confidence), soft decisions (semantic routing), and learned representations (embeddings). Neural IR formalizes this and makes it optimizable.

The five technical challenges—differentiability, verification, hardware mapping, pruning, and memory—are the engineering work needed to make Neural IR a practical reality rather than a theoretical concept.
