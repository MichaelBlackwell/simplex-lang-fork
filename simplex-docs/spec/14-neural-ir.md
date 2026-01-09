# Neural IR and Differentiable Execution

**Version 0.6.0**

---

## Overview

Neural IR (Intermediate Representation) introduces differentiable program execution to Simplex. This enables program logic itself to become learnable and optimizable via gradient descent, bridging traditional programming with machine learning.

---

## Core Concepts

### Neural Gates

Neural Gates transform discrete control flow into differentiable operations during training, then compile back to efficient discrete branches for inference.

```simplex
// Define a learnable gate
neural_gate should_retry(confidence: f64) -> bool {
    confidence > 0.7
}

// Training mode: compiles to sigmoid((confidence - 0.7) * temperature)
// Inference mode: compiles to standard conditional (zero overhead)
```

### Gumbel-Softmax

For categorical choices, Simplex uses the Gumbel-Softmax trick to enable gradient flow:

```simplex
neural_gate select_strategy(scores: List<f64>) -> Strategy {
    match argmax(scores) {
        0 => Strategy::Conservative,
        1 => Strategy::Balanced,
        _ => Strategy::Aggressive,
    }
}
```

During training:
- Samples from Gumbel distribution
- Applies softmax with temperature parameter
- Gradients flow through all branches (weighted by probability)

During inference:
- Standard argmax selection
- Zero additional overhead

---

## Dual Compilation Modes

### Training Mode

```bash
sxc build --mode=train model.sx
```

- Differentiable execution path
- Gumbel-Softmax for categorical choices
- Gradient tracking enabled
- Temperature annealing support

### Inference Mode

```bash
sxc build --mode=infer model.sx
```

- Standard discrete execution
- Dead path elimination
- Zero neural gate overhead
- Production-optimized binary

---

## Contract Logic

Contracts ensure safety guarantees even with probabilistic gates:

### Requires (Preconditions)

```simplex
neural_gate memory_safe_path(analysis: SecurityAnalysis) -> bool
    requires analysis.confidence > 0.95
{
    analysis.is_safe
}
```

The gate only fires when precondition is met.

### Ensures (Postconditions)

```simplex
neural_gate select_allocator(size: usize) -> Allocator
    ensures result.can_allocate(size)
{
    if size > LARGE_THRESHOLD { Allocator::Large }
    else { Allocator::Small }
}
```

### Fallback Handlers

```simplex
neural_gate risky_optimization(data: Data) -> Output
    requires data.is_valid()
    fallback => safe_default(data)
{
    optimized_path(data)
}
```

When confidence is below threshold or precondition fails, fallback executes instead.

### Invariants

```simplex
neural_gate state_transition(state: State, event: Event) -> State
    invariant state.is_consistent()
{
    compute_next_state(state, event)
}
```

---

## Hardware Targeting

### Explicit Annotations

```simplex
@gpu
neural_gate batch_classifier(inputs: List<Embedding>) -> List<Label> {
    // Runs on GPU - batch tensor operations
    inputs.map(|e| classify(e))
}

@cpu
fn process_result(label: Label) -> Action {
    // Runs on CPU - branching logic
    match label {
        Label::Safe => Action::Allow,
        Label::Suspicious => Action::Review,
        Label::Malicious => Action::Block,
    }
}

@npu
neural_gate cognitive_inference(prompt: String) -> Response {
    // Runs on NPU - SLM inference
    slm.generate(prompt)
}
```

### Automatic Targeting

When no annotation is provided, the compiler analyzes operation types:

| Operation Type | Default Target |
|----------------|----------------|
| Matrix multiplication | GPU |
| Element-wise tensor ops | GPU |
| Control flow, branching | CPU |
| I/O, syscalls | CPU |
| SLM inference | NPU |

---

## Structural Pruning

After training, the compiler eliminates dead paths:

### Weight Magnitude Pruning

```simplex
// Gates with |weight| < threshold are removed
neural_gate rarely_used(x: f64) -> bool {
    x > 0.99  // If weight → 0 during training, gate is pruned
}
```

### Activation Pruning

```simplex
// Paths that fire < threshold% during training are eliminated
neural_gate conditional_path(analysis: Analysis) -> Path {
    match analysis.risk_level {
        Low => Path::Fast,      // Fires 95% - kept
        Medium => Path::Normal, // Fires 4% - kept
        High => Path::Careful,  // Fires 0.1% - may be pruned
    }
}
```

### Gradient Pruning

Gates with consistently zero gradients during training are candidates for removal.

---

## Superposition Memory Model

During training, states may be in superposition (e.g., 50% true / 50% false):

### Lazy Evaluation (Default)

```simplex
let result = match branch_selector(x) {
    A => compute_a(),  // Only allocated if P(A) > threshold
    B => compute_b(),
}
```

### Speculative Execution

```simplex
@speculative
let result = match branch_selector(x) {
    A => compute_a(),  // All branches allocated
    B => compute_b(),  // Results weighted by probability
}
// Low-weight results garbage collected
```

### WeightedRef Type

```simplex
type WeightedRef<T> = {
    ptr: *T,
    weight: f64,  // 0.0 to 1.0
    allocated: bool,
}
```

---

## Temperature Annealing

Temperature controls the "softness" of gates during training:

```simplex
// High temperature (early training): soft decisions, explore
// Low temperature (late training): hard decisions, exploit

let schedule = TemperatureSchedule::exponential(
    initial: 1.0,
    final: 0.1,
    decay_steps: 10000,
);

neural_gate learnable_branch(x: f64) -> bool
    @temperature(schedule)
{
    x > threshold
}
```

---

## Integration with Autograd

Neural gates integrate seamlessly with Simplex's autograd:

```simplex
fn train_step(model: &mut Model, input: Tensor, target: Tensor) {
    let output = model.forward(&input);  // May contain neural gates
    let loss = mse_loss(&output, &target);

    loss.backward();  // Gradients flow through gates

    optimizer.step(&mut model.params());
}
```

---

## Performance Characteristics

| Metric | Training Mode | Inference Mode |
|--------|---------------|----------------|
| Gate overhead | ~5% vs discrete | 0% (identical to if) |
| Memory | +20% for gradients | Standard |
| Binary size | +15% for autograd | Within 2x of C |
| Throughput | 80% of inference | 100% baseline |

---

## Best Practices

### When to Use Neural Gates

**Good candidates:**
- Thresholds that could be learned (retry limits, confidence cutoffs)
- Routing decisions between strategies
- Feature selection paths

**Poor candidates:**
- Safety-critical branches (use contracts instead)
- Simple, well-understood logic
- Hot loops where any overhead matters

### Training Tips

1. Start with high temperature, anneal slowly
2. Use contracts for safety guarantees
3. Monitor gate weights during training
4. Prune aggressively before deployment

---

## See Also

- [Real-Time Learning](15-real-time-learning.md) - Online adaptation
- [Cognitive Hive AI](09-cognitive-hive.md) - SLM integration
- [The Anima](12-anima.md) - Cognitive agents
- [RELEASE-0.6.0.md](../RELEASE-0.6.0.md) - Full release notes
