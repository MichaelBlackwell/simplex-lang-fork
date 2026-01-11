# Simplex Testing Framework

**Version:** 0.9.0

## Overview

The Simplex testing framework is a native compile-and-run system built entirely on the Simplex toolchain. Tests are written in Simplex (`.sx` files) and executed using `sxc` (compiler) and the `run_tests.sh` test runner.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Test Runner (run_tests.sh)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Test File   │ -> │  Compiler    │ -> │  Executable  │       │
│  │   (.sx)      │    │    (sxc)     │    │              │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                 │               │
│                                                 v               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Result     │ <- │   Runtime    │ <- │   Execute    │       │
│  │  PASS/FAIL   │    │              │    │              │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Test Runner (run_tests.sh)

The primary test runner is `./tests/run_tests.sh`, which provides:

- **Category filtering**: Run tests by directory (neural, learning, stdlib, etc.)
- **Type filtering**: Run tests by prefix (unit_, spec_, integ_, e2e_)
- **Colored output**: Visual indicators for test types and results
- **Static analysis warnings**: Separate tracking of contract violations
- **Summary statistics**: Pass/fail counts and warnings

### Basic Usage

```bash
# Run all tests
./tests/run_tests.sh

# Run specific category
./tests/run_tests.sh neural
./tests/run_tests.sh stdlib
./tests/run_tests.sh learning

# Filter by test type
./tests/run_tests.sh all unit    # Only unit tests
./tests/run_tests.sh all spec    # Only spec tests
./tests/run_tests.sh all integ   # Only integration tests
./tests/run_tests.sh all e2e     # Only end-to-end tests

# Combine category and type
./tests/run_tests.sh stdlib unit
./tests/run_tests.sh neural spec

# Show help
./tests/run_tests.sh --help
```

## Toolchain Components

### 1. Simplex Compiler (`sxc`)

The native Simplex compiler that translates source to executable binaries.

**Commands:**
```bash
# Compile and run a single test
sxc run tests/stdlib/unit_hashmap.sx

# Compile only
sxc compile tests/stdlib/unit_hashmap.sx

# Check syntax only
sxc check tests/stdlib/unit_hashmap.sx
```

### 2. Simplex Package Manager (`sxpm`)

Manages dependencies and builds.

**Commands:**
```bash
# Build project
sxpm build

# Run a specific test file
sxpm test tests/stdlib/unit_hashmap.sx
```

## Naming Convention (v0.9.0)

All tests follow a consistent naming convention:

| Prefix | Type | Description | Color |
|--------|------|-------------|-------|
| `unit_` | Unit | Tests individual functions/types in isolation | Blue |
| `spec_` | Specification | Tests language specification compliance | Cyan |
| `integ_` | Integration | Tests integration between components | Magenta |
| `e2e_` | End-to-End | Tests complete workflows | Yellow |

### Examples

```
unit_crypto.sx        # Unit test for crypto module
spec_generics.sx      # Spec test for generic types
integ_networking.sx   # Integration test for networking
e2e_data_processor.sx # End-to-end data processing workflow
```

## Test Categories

### Language Tests (`tests/language/`)

Validate core language features:

| Subdirectory | Purpose |
|--------------|---------|
| `actors/` | Actor definitions, message handling |
| `async/` | Async/await syntax, futures, suspension |
| `basics/` | Loops, closures, enums, pattern matching, error handling |
| `closures/` | Closure capture, nested closures |
| `control/` | If-let expressions, match with bindings |
| `functions/` | Function definitions, closures, higher-order functions |
| `modules/` | Module system, public/private, imports |
| `traits/` | Trait definitions, associated types, impl blocks |
| `types/` | Generics, Option/Result, type aliases |

### Types Tests (`tests/types/`)

Focused type system tests including Phase 36 variants:
- Generic types and turbofish syntax
- Associated types and impl trait
- Pattern matching and destructuring
- Option and Result types

### Neural Tests (`tests/neural/`)

Neural IR and differentiable execution tests:
- Neural gates with Gumbel-Softmax
- Contract logic (requires/ensures/fallback)
- Hardware targeting annotations
- Structural pruning

### Standard Library Tests (`tests/stdlib/`)

Validate standard library functionality:
- Collections (HashMap, HashSet, Vec)
- String manipulation
- Option/Result types
- Crypto functions
- HTTP client
- Terminal/CLI features
- Regular expressions

### Runtime Tests (`tests/runtime/`)

Validate runtime behavior:
- Actor spawning and message passing
- Async execution and scheduling
- I/O operations
- Network communication
- Distribution

### AI/Cognitive Tests (`tests/ai/`)

Validate AI system components:

| Subdirectory | Purpose |
|--------------|---------|
| `anima/` | Personal agent memory, beliefs, desires |
| `hive/` | Shared consciousness, HiveMnemonic |
| `memory/` | Persistence, consolidation, recall |
| `inference/` | Memory-augmented LLM inference |
| `specialists/` | Specialist creation and routing |
| `orchestration/` | Multi-agent coordination |
| `tools/` | Tool calling |

### Learning Tests (`tests/learning/`)

Automatic differentiation and dual numbers:
- Dual number arithmetic
- Derivative propagation
- Transcendental function derivatives

### Toolchain Tests (`tests/toolchain/`)

Validate compiler and build tools:

| Subdirectory | Purpose |
|--------------|---------|
| `parser/` | Parser correctness |
| `codegen/` | Code generation |
| `sxpm/` | Package manager commands |
| `verification/` | Build verification |

### Integration Tests (`tests/integration/`)

End-to-end scenario tests:
- Multi-specialist reasoning
- Knowledge persistence across sessions
- Model provisioning workflows
- Real-world application scenarios

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Test passed |
| 1 | Test failed (assertion failure) |
| Non-zero | Compile error or runtime error |

## Test Discovery

The test runner automatically discovers tests:

1. Scans specified category directory (or all if not specified)
2. Finds all `.sx` files matching naming convention
3. Groups by test type (unit_, spec_, integ_, e2e_)
4. Executes and reports results

## Project Configuration

Tests are configured in `Modulus.toml`:

```toml
[package]
name = "simplex"
version = "0.9.0"

[test]
directory = "tests"
pattern = "*.sx"
parallel = false
timeout = 60

[test.categories]
language = "tests/language"
types = "tests/types"
neural = "tests/neural"
stdlib = "tests/stdlib"
ai = "tests/ai"
learning = "tests/learning"
toolchain = "tests/toolchain"
runtime = "tests/runtime"
integration = "tests/integration"
basics = "tests/basics"
async = "tests/async"
actors = "tests/actors"
observability = "tests/observability"
```

## Integration with CI/CD

The test runner returns non-zero exit code if any test fails:

```yaml
# Example CI configuration
test:
  script:
    - ./tests/run_tests.sh
```

## Test Output Colors

| Color | Meaning |
|-------|---------|
| GREEN | Test passed |
| RED | Test failed |
| BLUE | Unit test |
| CYAN | Spec test |
| MAGENTA | Integration test |
| YELLOW | E2E test / Category header |
