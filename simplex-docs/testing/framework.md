# Simplex Testing Framework

**Version:** 0.5.0

## Overview

The Simplex testing framework is a native compile-and-run system built entirely on the Simplex toolchain. Tests are written in Simplex (`.sx` files) and executed using `sxc` (compiler) and `sxpm` (package manager).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Test Runner (sxpm test)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Test File   │ -> │  Compiler    │ -> │  Executable  │       │
│  │   (.sx)      │    │    (sxc)     │    │   (.sxb)     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                 │               │
│                                                 v               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Result     │ <- │   Runtime    │ <- │   Execute    │       │
│  │  PASS/FAIL   │    │    (spx)     │    │              │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Toolchain Components

### 1. Simplex Compiler (`sxc`)

The native Simplex compiler that translates source to executable bytecode.

**Commands:**
```bash
# Compile a file
sxc compile test_file.sx

# Compile and run
sxc run test_file.sx

# Check syntax only
sxc check test_file.sx
```

### 2. Simplex Package Manager (`sxpm`)

Manages dependencies, builds, and testing.

**Test Commands:**
```bash
# Run all tests
sxpm test

# Run specific category
sxpm test --category language
sxpm test --category ai
sxpm test --category integration

# Run with verbose output
sxpm test --verbose

# Run specific test file
sxpm test tests/stdlib/test_hashmap.sx
```

### 3. Simplex Runner (`spx`)

Executes compiled Simplex bytecode.

**Usage:**
```bash
# Run compiled bytecode
spx test_file.sxb

# Run with debug output
spx --debug test_file.sxb
```

## Test Categories

### Language Tests (`tests/language/`)

Validate core language features:

| Subdirectory | Purpose |
|--------------|---------|
| `basics/` | Loops, closures, enums, pattern matching, error handling |
| `async/` | Async/await syntax, futures, suspension |
| `control_flow/` | If-let expressions, match with bindings |
| `functions/` | Function definitions, closures, higher-order functions |
| `modules/` | Module system, public/private, imports |
| `types/` | Generics, traits, Option/Result, associated types |
| `actors/` | Actor definitions, message handling |

### Standard Library Tests (`tests/stdlib/`)

Validate standard library functionality:
- Collections (HashMap, HashSet, Vec)
- String manipulation
- Option/Result types
- JSON parsing
- HTTP client
- Terminal/CLI features

### Runtime Tests (`tests/runtime/`)

Validate runtime behavior:
- Actor spawning and message passing
- Async execution and scheduling
- I/O operations
- Network communication

### AI/Cognitive Tests (`tests/ai/`)

Validate AI system components:

| Subdirectory | Purpose |
|--------------|---------|
| `anima/` | Personal agent memory, beliefs, desires |
| `hive/` | Shared consciousness, HiveMnemonic |
| `memory/` | Persistence, consolidation, recall |
| `inference/` | Memory-augmented LLM inference |
| `specialists/` | Specialist creation and routing |

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

The `sxpm test` command automatically discovers tests:

1. Scans `tests/` directory recursively
2. Finds all `.sx` files matching `test_*.sx` pattern
3. Groups by category (subdirectory)
4. Executes in order: Language → StdLib → Runtime → AI → Toolchain → Integration

## Project Configuration

Tests are configured in `Modulus.toml`:

```toml
[package]
name = "simplex"
version = "0.5.0"

[test]
directory = "tests"
pattern = "test_*.sx"
parallel = false
timeout = 60

[test.categories]
language = "tests/language"
stdlib = "tests/stdlib"
runtime = "tests/runtime"
ai = "tests/ai"
toolchain = "tests/toolchain"
integration = "tests/integration"
```

## Integration with CI/CD

The test runner returns non-zero exit code if any test fails:

```yaml
# Example CI configuration
test:
  script:
    - sxpm test
```
