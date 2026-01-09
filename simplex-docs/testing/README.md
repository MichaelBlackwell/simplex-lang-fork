# Simplex Testing Documentation

**Version:** 0.7.0
**Status:** Active Development

This directory contains comprehensive documentation for the Simplex testing framework, including test organization, coverage reports, testing methodologies, and best practices.

## Documentation Index

| Document | Description |
|----------|-------------|
| [Framework Overview](framework.md) | Testing framework architecture and components |
| [Running Tests](running-tests.md) | How to execute and interpret test results |
| [Test Coverage](coverage.md) | Current test coverage by category |
| [Testing Methods](methods.md) | Testing patterns, conventions, and best practices |
| [Writing Tests](writing-tests.md) | Guide for writing new tests |

## Quick Start

Run all tests:
```bash
sxpm test
```

Run tests in a specific category:
```bash
sxpm test --category language
sxpm test --category ai
sxpm test --category integration
```

Run a single test:
```bash
sxc run tests/language/basics/test_loops.sx
```

## Test Categories

The Simplex test suite is organized into eight major categories:

```
tests/
├── language/          # Core language features (42 tests)
│   ├── basics/        # Loops, closures, enums, pattern matching
│   ├── async/         # Async/await, futures
│   ├── control_flow/  # If-let, match expressions
│   ├── functions/     # Function definitions, closures
│   ├── modules/       # Module system, imports
│   ├── types/         # Generics, traits, Option/Result
│   ├── actors/        # Actor model basics
│   └── neural/        # Neural gates, differentiable control flow (v0.6.0)
│
├── stdlib/            # Standard library (11 tests)
│   └── collections, strings, HTTP, JSON, CLI
│
├── runtime/           # Runtime system (6 tests)
│   └── actors, async, I/O, networking
│
├── ai/                # AI/Cognitive systems (38 tests)
│   ├── anima/         # Agent animation layer
│   ├── hive/          # Collective intelligence
│   ├── memory/        # Memory persistence
│   ├── inference/     # LLM inference
│   ├── specialists/   # Specialist providers
│   └── learning/      # Real-time learning (v0.7.0)
│
├── learning/          # simplex-learning library (23 tests)
│   ├── tensor/        # Tensor operations, autograd
│   ├── optim/         # Optimizers, schedulers
│   ├── safety/        # Constraints, fallbacks
│   ├── distributed/   # Federated, distillation, beliefs
│   └── runtime/       # Online learner, checkpoints
│
├── toolchain/         # Compiler toolchain (15 tests)
│   ├── parser/        # Parser validation
│   ├── codegen/       # Code generation
│   ├── neural_ir/     # Neural IR compilation (v0.6.0)
│   ├── sxpm/          # Package manager
│   └── verification/  # Build verification
│
├── observability/     # Monitoring and tracing
│
└── integration/       # End-to-end scenarios (12 tests)
    ├── workflows/     # Real-world workflows
    └── learning/      # Learning integration tests (v0.7.0)
```

## Test Statistics (v0.7.0)

| Category | Test Files | Status |
|----------|------------|--------|
| Language | 42 | Active |
| Standard Library | 11 | Active |
| Runtime | 6 | Active |
| AI/Cognitive | 38 | Active |
| Learning | 23 | Active |
| Toolchain | 15 | Active |
| Integration | 12 | Active |
| **Total** | **147+** | - |

## v0.7.0 Test Additions

New tests added for v0.7.0 Real-Time Learning features:

### Learning Library (`tests/learning/`)
- **Tensor Operations**: `tests/learning/tensor/test_tensor_ops.sx`
- **Autograd**: `tests/learning/tensor/test_autograd.sx`
- **Batch MatMul**: `tests/learning/tensor/test_batch_matmul.sx`
- **Streaming SGD**: `tests/learning/optim/test_streaming_sgd.sx`
- **Streaming Adam**: `tests/learning/optim/test_streaming_adam.sx`
- **AdamW**: `tests/learning/optim/test_adamw.sx`
- **Gradient Clipping**: `tests/learning/optim/test_gradient_clipping.sx`
- **Safety Bounds**: `tests/learning/safety/test_safety_bounds.sx`
- **Safe Fallback**: `tests/learning/safety/test_safe_fallback.sx`
- **Constraints**: `tests/learning/safety/test_constraints.sx`
- **Federated Learning**: `tests/learning/distributed/test_federated.sx`
- **Knowledge Distillation**: `tests/learning/distributed/test_distillation.sx`
- **Belief Resolution**: `tests/learning/distributed/test_beliefs.sx`
- **Hive Coordinator**: `tests/learning/distributed/test_hive_coordinator.sx`
- **Online Learner**: `tests/learning/runtime/test_online_learner.sx`
- **Checkpointing**: `tests/learning/runtime/test_checkpointing.sx`

### AI/Cognitive (`tests/ai/learning/`)
- **Specialist Learning**: `tests/ai/learning/test_specialist_online_learning.sx`
- **Hive Learning**: `tests/ai/learning/test_hive_federated_learning.sx`
- **Adaptive Inference**: `tests/ai/learning/test_adaptive_inference.sx`

### Integration (`tests/integration/learning/`)
- **End-to-End Learning**: `tests/integration/learning/test_e2e_online_learning.sx`
- **Multi-Hive Sync**: `tests/integration/learning/test_multi_hive_sync.sx`
- **Learning with Safety**: `tests/integration/learning/test_learning_with_safety.sx`

## v0.6.0 Test Additions

New tests added for v0.6.0 Neural IR features:

### Language/Neural (`tests/language/neural/`)
- **Neural Gates**: `tests/language/neural/test_neural_gate.sx`
- **Gumbel-Softmax**: `tests/language/neural/test_gumbel_softmax.sx`
- **Differentiable Control**: `tests/language/neural/test_differentiable_control.sx`
- **Contract Logic**: `tests/language/neural/test_contracts.sx`
- **Hardware Targeting**: `tests/language/neural/test_hardware_targeting.sx`

### Toolchain/Neural IR (`tests/toolchain/neural_ir/`)
- **Training Mode**: `tests/toolchain/neural_ir/test_training_mode.sx`
- **Inference Mode**: `tests/toolchain/neural_ir/test_inference_mode.sx`

## v0.5.0 Test Additions

New tests added for v0.5.0 features:

- **SLM Provisioning**: `tests/toolchain/sxpm/test_model_commands.sx`
- **HiveMnemonic**: `tests/ai/hive/test_hive_mnemonic.sx`
- **Per-Hive SLM**: `tests/ai/hive/test_per_hive_slm.sx`
- **Memory-Augmented Inference**: `tests/ai/inference/test_memory_augmented_inference.sx`
- **Anima Integration**: `tests/ai/anima/test_anima_hive_integration.sx`
- **HTTP Client**: `tests/stdlib/test_http_client.sx`
- **Terminal/CLI**: `tests/stdlib/test_cli_terminal.sx`
- **Multi-Specialist Reasoning**: `tests/integration/test_multi_specialist_reasoning.sx`
- **Knowledge Persistence**: `tests/integration/test_knowledge_persistence.sx`
- **Model Provisioning Workflow**: `tests/integration/test_model_provisioning_workflow.sx`

## See Also

- [Simplex Specification](../spec/README.md)
- [Getting Started Guide](../guides/getting-started.md)
- [Release Notes v0.7.0](../RELEASE-0.7.0.md)
- [Release Notes v0.6.0](../RELEASE-0.6.0.md)
- [Release Notes v0.5.0](../RELEASE-0.5.0.md)
