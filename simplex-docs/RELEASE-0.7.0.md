# Simplex v0.7.0 Release Notes

**Release Date:** 2026-01-09
**Codename:** Real-Time Learning

---

## Overview

Simplex v0.7.0 introduces the `simplex-learning` library, enabling AI specialists to learn and adapt during runtime without requiring offline batch training. This completes the vision of truly adaptive cognitive agents that improve through interaction.

---

## Major Features

### simplex-learning Library

A complete real-time continuous learning framework:

```simplex
use simplex_learning::{OnlineLearner, StreamingAdam, SafeFallback};

// Create an online learner
let learner = OnlineLearner::new(model_params)
    .optimizer(StreamingAdam::new(0.001))
    .constraint(MaxLatency(10.0))
    .fallback(SafeFallback::with_default(default_output));

// Learn from each interaction
for (input, feedback) in interactions {
    let output = learner.forward(&input);
    learner.learn(&feedback);  // Adapts in real-time
}
```

### Tensor Operations with Autograd

Full tensor library with automatic differentiation:

```simplex
use simplex_learning::tensor::{Tensor, ops};

let x = Tensor::randn(&[32, 128]).requires_grad_();
let w = Tensor::randn(&[128, 64]).requires_grad_();

let y = ops::matmul(&x, &w);
let loss = ops::mse_loss(&y, &target);

// Backpropagate
loss.backward();

// Gradients available
let grad_w = w.grad();
```

**Supported Operations:**
- Matrix multiplication (including batch matmul with broadcasting)
- Element-wise: add, sub, mul, div
- Activations: relu, sigmoid, tanh, softmax, gelu
- Loss functions: mse, cross_entropy, huber
- Reductions: sum, mean, max, min

### Streaming Optimizers

Optimizers designed for online learning:

```simplex
// Streaming SGD with momentum
let sgd = StreamingSGD::new(0.01)
    .momentum(0.9)
    .weight_decay(0.0001)
    .max_grad_norm(1.0);  // Automatic gradient clipping

// Streaming Adam with gradient accumulation
let adam = StreamingAdam::new(0.001)
    .betas(0.9, 0.999)
    .accumulation_steps(4);  // Mini-batch accumulation

// AdamW with decoupled weight decay
let adamw = AdamW::new(0.001)
    .weight_decay(0.01);
```

**Gradient Clipping:**
```simplex
// Clip by norm (scales all gradients proportionally)
clip_grad_norm(&mut params, 1.0);

// Clip by value (clamps each gradient independently)
clip_grad_value(&mut params, 0.5);
```

### Safety Constraints and Fallbacks

Ensure learning doesn't destabilize the system:

```simplex
// Define constraints
let constraints = ConstraintManager::new()
    .add_soft(MaxLatency("latency", 10.0, penalty_weight: 0.5))
    .add_hard(NoLossExplosion("loss", 100.0));

// Safe learner with fallback
let safe_learner = SafeLearner::new(learner, SafeFallback::with_default(safe_output))
    .with_validator(|output| output.is_valid())
    .max_failures(3);

// Process with safety checks
match safe_learner.try_process(&input, compute_fn) {
    Ok(output) => use_output(output),
    Err(SafetyError::NoFallbackAvailable { failures }) => {
        log_error(f"Failed after {failures} attempts");
    }
}
```

**Fallback Strategies:**
| Strategy | Description |
|----------|-------------|
| `Default` | Return predefined safe output |
| `LastGood` | Return last successful output |
| `Function` | Execute custom fallback logic |
| `Checkpoint` | Restore from saved state |
| `SkipUpdate` | Continue without learning |

### Federated Learning

Distributed learning across hives:

```simplex
use simplex_learning::distributed::{FederatedLearner, AggregationStrategy};

let federated = FederatedLearner::new(config, initial_params);

// Specialists submit updates
federated.submit_update(NodeUpdate {
    node_id: "specialist_1",
    params: local_params,
    sample_count: 100,
    validation_acc: 0.85,
});

// Aggregation happens automatically when min_nodes reached
let global_params = federated.global_params();
```

**Aggregation Strategies:**
- `FedAvg` - Simple averaging
- `WeightedAvg` - Weighted by sample count
- `PerformanceWeighted` - Weighted by validation accuracy
- `Median` - Byzantine-resilient median
- `TrimmedMean` - Trimmed mean for robustness
- `AttentionWeighted` - Similarity-weighted aggregation

### Knowledge Distillation

Transfer knowledge between specialists:

```simplex
use simplex_learning::distributed::{KnowledgeDistiller, SelfDistillation};

// Teacher-student distillation
let distiller = KnowledgeDistiller::new(config);
let loss = distiller.distillation_loss(&student_logits, &teacher_logits, hard_labels);

// Self-distillation with EMA teacher
let self_distill = SelfDistillation::new(config, params);
self_distill.update_ema(&current_params);  // Update EMA teacher
```

### Belief Conflict Resolution

Reconcile conflicting beliefs across distributed specialists:

```simplex
use simplex_learning::distributed::{HiveBeliefManager, ConflictResolution};

let manager = HiveBeliefManager::new(ConflictResolution::BayesianCombination);

// Specialists submit beliefs
manager.submit_belief(Belief::new("user_prefers_concise", 0.8, "specialist_1"));
manager.submit_belief(Belief::new("user_prefers_concise", 0.6, "specialist_2"));

// Get consensus
let consensus = manager.consensus("user_prefers_concise");
```

**Resolution Strategies:**
- `HighestConfidence` - Use most confident belief
- `MostRecent` - Use newest belief
- `MostEvidence` - Use belief with most supporting evidence
- `EvidenceWeighted` - Weighted average by evidence count
- `BayesianCombination` - Log-odds combination
- `SemanticWeighted` - Weighted by embedding similarity
- `MajorityVote` - Democratic voting

### Hive Learning Coordinator

Orchestrates all learning components:

```simplex
use simplex_learning::distributed::{HiveLearningCoordinator, HiveLearningConfig};

let config = HiveLearningConfigBuilder::new()
    .sync_interval(100)
    .checkpoint_interval(1000)
    .aggregation(AggregationStrategy::FedAvg)
    .belief_resolution(ConflictResolution::EvidenceWeighted)
    .enable_distillation(true)
    .build();

let coordinator = HiveLearningCoordinator::new(config, initial_params);

// Register specialists
coordinator.register_specialist("security", security_params);
coordinator.register_specialist("quality", quality_params);

// Submit gradients and beliefs
coordinator.submit_gradients("security", &gradients);
coordinator.submit_belief("security", belief);

// Automatic sync and aggregation
coordinator.step();
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    simplex-learning                         │
├─────────────┬─────────────┬──────────────┬─────────────────┤
│   tensor/   │   optim/    │   safety/    │  distributed/   │
├─────────────┼─────────────┼──────────────┼─────────────────┤
│ Tensor      │ StreamingSGD│ SafetyBounds │ FederatedLearner│
│ Shape       │ StreamingAdam│ Constraints │ KnowledgeDistill│
│ Autograd    │ AdamW       │ SafeFallback │ BeliefResolver  │
│ Ops         │ Schedulers  │ SafeLearner  │ HiveCoordinator │
└─────────────┴─────────────┴──────────────┴─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │    runtime/     │
                    │ OnlineLearner   │
                    │ Checkpoint      │
                    │ Metrics         │
                    └─────────────────┘
```

---

## Performance

| Operation | Throughput | Latency |
|-----------|------------|---------|
| Forward pass (128-dim) | 50K/sec | 20μs |
| Backward pass | 25K/sec | 40μs |
| Gradient clipping | 100K/sec | 10μs |
| FedAvg aggregation | 10K params/ms | <1ms |
| Checkpoint save | 1M params/sec | <10ms |

---

## Migration Guide

### From v0.6.x

1. **No breaking changes** - existing code compiles unchanged
2. **New dependency**: Add `simplex-learning` to Modulus.toml

```toml
[dependencies]
simplex-learning = { path = "../simplex-learning" }
```

### Integrating with Existing Specialists

```simplex
// Before: Static specialist
specialist SecurityAnalyzer {
    model: "simplex-cognitive-7b";

    fn analyze(code: String) -> Analysis {
        infer("Analyze: " + code)
    }
}

// After: Learning specialist
use simplex_learning::{OnlineLearner, SafeFallback};

specialist SecurityAnalyzer {
    model: "simplex-cognitive-7b";
    learner: OnlineLearner;

    fn init() {
        self.learner = OnlineLearner::new(self.params())
            .fallback(SafeFallback::with_default(Analysis::unknown()));
    }

    fn analyze(code: String) -> Analysis {
        let result = infer("Analyze: " + code);
        result
    }

    fn feedback(analysis: Analysis, correct: bool) {
        self.learner.learn(&FeedbackSignal::from(correct));
    }
}
```

---

## Module Summary

| Module | Files | Lines | Purpose |
|--------|-------|-------|---------|
| `tensor/` | 4 | ~2,500 | Tensor ops, autograd, shapes |
| `optim/` | 5 | ~1,500 | Optimizers, schedulers, streaming |
| `safety/` | 4 | ~1,200 | Constraints, bounds, fallbacks |
| `distributed/` | 6 | ~3,500 | Federated, distillation, beliefs |
| `runtime/` | 4 | ~1,500 | Learner, checkpoints, metrics |
| `memory/` | 5 | ~1,000 | Replay buffer, EWC, progressive |
| `calibration/` | 4 | ~800 | Temperature scaling, metrics |
| `feedback/` | 4 | ~900 | Signals, channels, attribution |
| **Total** | **57** | **~13,000** | |

---

## Known Limitations

1. GPU acceleration requires separate `simplex-learning-gpu` package (coming soon)
2. Maximum tensor rank of 8 dimensions
3. Federated learning simulates network communication (MPI/NCCL integration planned)
4. Differential privacy adds ~10% overhead when enabled

---

## What's Next

- v0.8.0: GPU acceleration for tensor operations
- v0.9.0: Distributed hive clustering with actual network communication
- v1.0.0: Production-ready release with full documentation

---

## Credits

The simplex-learning library was designed and implemented by Rod Higgins ([@senuamedia](https://github.com/senuamedia)).

Key influences:
- PyTorch for tensor/autograd design patterns
- Flower for federated learning concepts
- Hinton et al. for knowledge distillation techniques
