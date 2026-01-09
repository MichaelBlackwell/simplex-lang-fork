# Simplex Test Coverage

**Version:** 0.7.0
**Last Updated:** v0.7.0 Release

## Coverage Summary

| Category | Tests | Features Covered | Coverage |
|----------|-------|------------------|----------|
| Language | 42 | Core syntax, types, async, neural gates | High |
| Standard Library | 11 | Collections, I/O, HTTP | Medium |
| Runtime | 6 | Actors, async, networking | Medium |
| AI/Cognitive | 38 | Anima, Hive, SLM, learning | High |
| Learning | 23 | Tensor, optim, safety, distributed | High |
| Toolchain | 15 | Compiler, sxpm, neural IR | Medium |
| Integration | 12 | End-to-end scenarios, learning | High |
| **Total** | **147+** | - | - |

## Language Tests Coverage

### Basics (`tests/language/basics/`)

| Test File | Features Covered |
|-----------|------------------|
| `test_loops.sx` | `while`, `for`, `loop`, `break`, `continue` |
| `test_closures.sx` | Closure syntax, capture, higher-order functions |
| `test_enums.sx` | Enum definition, variants, pattern matching |
| `test_pattern_matching.sx` | `match` expressions, destructuring |
| `test_error_handling.sx` | `Result`, `?` operator, error propagation |

### Types (`tests/language/types/`)

| Test File | Features Covered |
|-----------|------------------|
| `test_generics.sx` | Generic functions, generic structs |
| `test_option_result.sx` | `Option<T>`, `Result<T, E>`, unwrapping |
| `test_associated_types.sx` | Trait associated types |
| `test_impl_trait.sx` | `impl Trait` return types |
| `test_references.sx` | References, borrowing |
| `test_turbofish.sx` | Turbofish syntax `::<T>` |

### Async (`tests/language/async/`)

| Test File | Features Covered |
|-----------|------------------|
| `test_async_await.sx` | `async fn`, `.await` syntax |
| `test_async_multi.sx` | Multiple concurrent tasks |

### Control Flow (`tests/language/control_flow/`)

| Test File | Features Covered |
|-----------|------------------|
| `test_if_let.sx` | `if let` pattern matching |
| `test_match_binding.sx` | Match with variable binding |

### Modules (`tests/language/modules/`)

| Test File | Features Covered |
|-----------|------------------|
| `test_imports.sx` | `use` statements, module imports |
| `test_visibility.sx` | `pub`, private by default |

### Neural (`tests/language/neural/`) - v0.6.0

| Test File | Features Covered |
|-----------|------------------|
| `test_neural_gate.sx` | `neural_gate` keyword, basic gates |
| `test_gumbel_softmax.sx` | Gumbel-Softmax differentiable sampling |
| `test_differentiable_control.sx` | Differentiable if/match |
| `test_contracts.sx` | `requires`, `ensures`, `fallback` |
| `test_hardware_targeting.sx` | `@cpu`, `@gpu`, `@npu` annotations |

## Standard Library Coverage

| Test File | Features Covered |
|-----------|------------------|
| `test_hashmap.sx` | HashMap creation, insert, get, remove, iterate |
| `test_hashset.sx` | HashSet operations |
| `test_vec.sx` | Vec creation, push, pop, indexing |
| `test_strings.sx` | String concat, split, contains, formatting |
| `test_option.sx` | Option methods: map, unwrap_or, is_some |
| `test_result.sx` | Result methods: map, map_err, unwrap_or |
| `test_json.sx` | JSON parse, stringify, object/array access |
| `test_http_client.sx` | HTTP requests, URL parsing, response handling |
| `test_cli_terminal.sx` | Colors, progress bars, spinners, tables |

## AI/Cognitive Coverage

### Anima (`tests/ai/anima/`)

| Test File | Features Covered |
|-----------|------------------|
| `test_anima_memory.sx` | Episodic, semantic, working memory |
| `test_anima_beliefs.sx` | Belief creation, revision, threshold |
| `test_anima_desires.sx` | Goals, priorities, intentions |
| `test_anima_hive_integration.sx` | Anima + HiveMnemonic integration |

### Hive (`tests/ai/hive/`)

| Test File | Features Covered |
|-----------|------------------|
| `test_hive_mnemonic.sx` | Shared consciousness, 50% belief threshold |
| `test_per_hive_slm.sx` | ONE shared SLM per hive verification |
| `test_specialist.sx` | Specialist creation, routing |
| `test_collective.sx` | Multi-specialist coordination |

### Inference (`tests/ai/inference/`)

| Test File | Features Covered |
|-----------|------------------|
| `test_memory_augmented_inference.sx` | Context flow Anima → Mnemonic → SLM |
| `test_slm_config.sx` | SLM configuration options |

### Memory (`tests/ai/memory/`)

| Test File | Features Covered |
|-----------|------------------|
| `test_persistence.sx` | Save/load memory state |
| `test_consolidation.sx` | Memory pruning, consolidation |
| `test_recall.sx` | Goal-directed recall |

## Toolchain Coverage

### Compiler (`tests/toolchain/parser/`, `tests/toolchain/codegen/`)

| Test File | Features Covered |
|-----------|------------------|
| `test_parser.sx` | Parser correctness for all syntax |
| `test_codegen.sx` | Code generation for all constructs |
| `test_type_checker.sx` | Type inference, type errors |

### Package Manager (`tests/toolchain/sxpm/`)

| Test File | Features Covered |
|-----------|------------------|
| `test_model_commands.sx` | `sxpm model list/install/remove/info` |
| `test_build.sx` | `sxpm build` command |
| `test_dependencies.sx` | Dependency resolution |

## Integration Coverage

| Test File | Scenario |
|-----------|----------|
| `test_multi_specialist_reasoning.sx` | 3-specialist code review hive |
| `test_knowledge_persistence.sx` | Memory save/load across sessions |
| `test_model_provisioning_workflow.sx` | Model discovery → inference |
| `test_todo_app.sx` | Real-world todo application |
| `test_data_pipeline.sx` | Data processing workflow |
| `test_e2e_online_learning.sx` | End-to-end real-time learning |
| `test_multi_hive_sync.sx` | Federated learning across hives |
| `test_learning_with_safety.sx` | Learning with safety constraints |

## Learning Library Coverage (v0.7.0)

### Tensor (`tests/learning/tensor/`)

| Test File | Features Covered |
|-----------|------------------|
| `test_tensor_ops.sx` | Create, reshape, slice, element-wise ops |
| `test_autograd.sx` | Backward pass, gradient computation |
| `test_batch_matmul.sx` | Batched matrix multiplication, broadcasting |
| `test_activations.sx` | ReLU, sigmoid, tanh, softmax, GELU |
| `test_loss_functions.sx` | MSE, cross-entropy, Huber loss |

### Optimizers (`tests/learning/optim/`)

| Test File | Features Covered |
|-----------|------------------|
| `test_streaming_sgd.sx` | Streaming SGD, momentum, weight decay |
| `test_streaming_adam.sx` | Streaming Adam, accumulation steps |
| `test_adamw.sx` | AdamW with decoupled weight decay |
| `test_gradient_clipping.sx` | Clip by norm, clip by value |
| `test_schedulers.sx` | Learning rate schedulers |

### Safety (`tests/learning/safety/`)

| Test File | Features Covered |
|-----------|------------------|
| `test_safety_bounds.sx` | Min/max bounds, NaN/Inf detection |
| `test_safe_fallback.sx` | Default, LastGood, Function, Checkpoint |
| `test_constraints.sx` | Soft/hard constraints, constraint manager |
| `test_safe_learner.sx` | SafeLearner with validation |

### Distributed (`tests/learning/distributed/`)

| Test File | Features Covered |
|-----------|------------------|
| `test_federated.sx` | FedAvg, WeightedAvg, Median, TrimmedMean |
| `test_distillation.sx` | Teacher-student, self-distillation |
| `test_beliefs.sx` | Belief conflict resolution strategies |
| `test_hive_coordinator.sx` | HiveLearningCoordinator orchestration |
| `test_gradient_sync.sx` | Ring all-reduce, tree all-reduce |

### Runtime (`tests/learning/runtime/`)

| Test File | Features Covered |
|-----------|------------------|
| `test_online_learner.sx` | OnlineLearner forward/learn loop |
| `test_checkpointing.sx` | Checkpoint save/restore |
| `test_metrics.sx` | Training metrics, logging |
| `test_experience_replay.sx` | Replay buffer, sampling |

## v0.7.0 New Test Coverage

Tests added specifically for v0.7.0 Real-Time Learning features:

### simplex-learning Library
- Complete tensor operations with autograd (23 tests)
- Streaming optimizers with gradient accumulation
- Safety constraints and fallback strategies
- Federated learning with 6 aggregation strategies
- Knowledge distillation (teacher-student, self-distillation)
- Belief conflict resolution (8 resolution strategies)
- Hive learning coordinator

### AI/Cognitive Learning Integration
- `tests/ai/learning/test_specialist_online_learning.sx`
  - Specialists that learn during inference
  - Feedback signal processing
  - Safe learning with fallbacks
- `tests/ai/learning/test_hive_federated_learning.sx`
  - Multi-specialist gradient aggregation
  - Belief synchronization
  - Distributed checkpointing

## v0.6.0 New Test Coverage

Tests added specifically for v0.6.0 Neural IR features:

### Neural Gates
- `tests/language/neural/test_neural_gate.sx`
  - `neural_gate` keyword syntax
  - Training vs inference mode
  - Gradient flow through gates

### Differentiable Control Flow
- `tests/language/neural/test_differentiable_control.sx`
  - Gumbel-Softmax sampling
  - Temperature annealing
  - Straight-through estimator

### Contract Logic
- `tests/language/neural/test_contracts.sx`
  - `requires` preconditions
  - `ensures` postconditions
  - `fallback` handlers
  - Confidence thresholds

### Toolchain Neural IR
- `tests/toolchain/neural_ir/test_training_mode.sx`
  - Differentiable compilation
  - Autograd integration
- `tests/toolchain/neural_ir/test_inference_mode.sx`
  - Discrete compilation
  - Dead path elimination

## v0.5.0 New Test Coverage

Tests added specifically for v0.5.0 SLM Provisioning features:

### SLM Provisioning
- `tests/toolchain/sxpm/test_model_commands.sx`
  - Model directory discovery
  - Builtin models enumeration
  - Model manifest parsing
  - Installation status checks
  - Model path resolution

### Per-Hive SLM Architecture
- `tests/ai/hive/test_per_hive_slm.sx`
  - Verifies ONE shared model per hive
  - Tests 5 specialists sharing same SLM
  - Memory efficiency validation
  - Specialist isolation with shared inference

### HiveMnemonic (Shared Consciousness)
- `tests/ai/hive/test_hive_mnemonic.sx`
  - 50% belief threshold
  - Semantic knowledge sharing
  - Episodic memory sharing
  - Cross-specialist visibility
  - Consolidation and pruning

### Memory-Augmented Inference
- `tests/ai/inference/test_memory_augmented_inference.sx`
  - Context flow: Anima → Mnemonic → SLM
  - `<context>` and `<hive>` tag verification
  - Full prompt construction

### Anima-Hive Integration
- `tests/ai/anima/test_anima_hive_integration.sx`
  - Personal (30%) vs shared (50%) thresholds
  - Contributing to shared memory
  - Context formatting

## Coverage Gaps

Areas with limited or no test coverage:

| Area | Status | Priority |
|------|--------|----------|
| Distributed actors | Stub only | High |
| Network protocols | Basic only | Medium |
| GPU acceleration | Not tested | Medium |
| Cross-platform | macOS only | Medium |
| MPI/NCCL integration | Simulated | Low |

## Running Coverage Report

```bash
# Generate coverage report
sxpm test --coverage

# Generate HTML coverage report
sxpm test --coverage --format html --output coverage/

# View coverage for specific category
sxpm test --coverage --category ai
sxpm test --coverage --category learning
```

## Coverage Goals

| Category | Current | Target |
|----------|---------|--------|
| Language | 90% | 95% |
| Standard Library | 70% | 90% |
| Runtime | 60% | 80% |
| AI/Cognitive | 85% | 95% |
| Learning | 80% | 95% |
| Toolchain | 70% | 85% |
| Integration | 80% | 90% |
