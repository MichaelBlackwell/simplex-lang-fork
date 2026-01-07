# Simplex Testing Documentation

**Version:** 0.5.0
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

The Simplex test suite is organized into seven major categories:

```
tests/
├── language/          # Core language features (37 tests)
│   ├── basics/        # Loops, closures, enums, pattern matching
│   ├── async/         # Async/await, futures
│   ├── control_flow/  # If-let, match expressions
│   ├── functions/     # Function definitions, closures
│   ├── modules/       # Module system, imports
│   ├── types/         # Generics, traits, Option/Result
│   └── actors/        # Actor model basics
│
├── stdlib/            # Standard library (11 tests)
│   └── collections, strings, HTTP, JSON, CLI
│
├── runtime/           # Runtime system (6 tests)
│   └── actors, async, I/O, networking
│
├── ai/                # AI/Cognitive systems (29 tests)
│   ├── anima/         # Agent animation layer
│   ├── hive/          # Collective intelligence
│   ├── memory/        # Memory persistence
│   ├── inference/     # LLM inference
│   └── specialists/   # Specialist providers
│
├── toolchain/         # Compiler toolchain (13 tests)
│   ├── parser/        # Parser validation
│   ├── codegen/       # Code generation
│   ├── sxpm/          # Package manager
│   └── verification/  # Build verification
│
├── observability/     # Monitoring and tracing
│
└── integration/       # End-to-end scenarios (7 tests)
    └── Real-world workflows
```

## Test Statistics (v0.5.0)

| Category | Test Files | Status |
|----------|------------|--------|
| Language | 37 | Active |
| Standard Library | 11 | Active |
| Runtime | 6 | Active |
| AI/Cognitive | 29 | Active |
| Toolchain | 13 | Active |
| Integration | 7 | Active |
| **Total** | **103+** | - |

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
- [Release Notes](../RELEASE-0.5.0.md)
