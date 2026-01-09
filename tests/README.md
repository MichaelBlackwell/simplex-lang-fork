# Simplex Test Suite

This directory contains the test suite for the Simplex programming language.

## Running Tests

```bash
# Run all tests
./run_tests.sh

# Run specific category
./run_tests.sh neural      # Neural IR tests only
./run_tests.sh language    # Language feature tests
./run_tests.sh stdlib      # Standard library tests
./run_tests.sh runtime     # Runtime tests
./run_tests.sh ai          # AI/Cognitive tests
./run_tests.sh integration # Integration tests
./run_tests.sh toolchain   # Toolchain tests
```

## Directory Structure

```
tests/
├── neural/              # Neural IR and Neural Gates (TASK-001)
│   ├── gates/           # Neural gate tests (9 tests)
│   │   ├── comprehensive.sx    # Full neural gate test suite (6 subtests)
│   │   ├── basic_inference.sx  # Basic inference mode
│   │   ├── training_mode.sx    # Training vs inference modes
│   │   ├── gradient_tape.sx    # Gradient tape operations
│   │   ├── contracts.sx        # Contract clauses (requires/ensures)
│   │   ├── hardware_annotation.sx # @cpu/@gpu/@npu annotations
│   │   ├── multiple_gates.sx   # Multiple gate definitions
│   │   ├── minimal.sx          # Minimal gate test
│   │   └── float_negation.sx   # Negative float literal handling
│   ├── contracts/       # Static contract analysis (2 tests)
│   │   ├── static_analysis.sx  # Valid contracts (no warnings)
│   │   └── static_violation.sx # Contract violation detection
│   └── pruning/         # Structural pruning (1 test)
│       └── weight_magnitude.sx # Weight magnitude pruning API
│
├── language/            # Core language feature tests
│   ├── basics/          # Basic language features
│   │   ├── closure.sx       # Closures with nested capture
│   │   ├── enum.sx          # Enum definitions and matching
│   │   ├── for_loop.sx      # For loop iteration
│   │   ├── match.sx         # Pattern matching
│   │   ├── timing.sx        # Timing operations
│   │   └── try_operator.sx  # ? operator for Result types
│   ├── types/           # Type system tests
│   │   ├── generics.sx          # Generic functions
│   │   ├── associated_types.sx  # Associated types in traits
│   │   ├── generic_methods.sx   # Generic methods on structs
│   │   ├── turbofish.sx         # Turbofish syntax ::<>
│   │   ├── if_let.sx            # if-let pattern matching
│   │   ├── if_let_simple.sx     # Simple if-let cases
│   │   ├── impl_trait.sx        # impl Trait syntax
│   │   ├── match_binding.sx     # Match with variable binding
│   │   ├── match_patterns.sx    # Complex match patterns
│   │   ├── option_result.sx     # Option<T> and Result<T,E>
│   │   └── references.sx        # Reference types
│   ├── traits/          # Trait tests
│   │   ├── trait_self_ref.sx    # Self references in traits
│   │   ├── test_impl_trait.sx   # impl Trait tests
│   │   └── test_assoc_types.sx  # Associated type tests
│   ├── async/           # Async/await tests
│   │   ├── basic.sx         # Basic async functions
│   │   ├── closures.sx      # Async closures
│   │   └── multi_await.sx   # Multi-await state machines
│   ├── actors/          # Actor model tests
│   ├── control/         # Control flow tests
│   ├── functions/       # Function tests
│   └── modules/         # Module system tests
│
├── stdlib/              # Standard library tests
│   ├── collections/     # Vec, HashMap, etc.
│   ├── io/              # I/O operations
│   ├── string/          # String operations
│   └── ...
│
├── runtime/             # Runtime tests
│   ├── actors/          # Actor runtime tests
│   ├── async/           # Async runtime tests
│   ├── io/              # I/O runtime tests
│   ├── networking/      # Networking tests
│   └── distribution/    # Distributed runtime tests
│
├── ai/                  # AI/Cognitive tests
│   ├── specialist/      # AI specialist tests
│   ├── hive/            # Cognitive hive tests
│   ├── anima/           # Anima (cognitive soul) tests
│   └── memory/          # Semantic memory tests
│
├── toolchain/           # Toolchain tests
│   ├── compiler/        # Compiler tests
│   └── tools/           # CLI tool tests
│
├── observability/       # Observability tests
│   └── metrics/         # Metrics and tracing tests
│
└── integration/         # Integration tests
    └── ...              # End-to-end scenarios
```

## Writing Tests

Tests are `.sx` files that:
1. Return `0` on success, non-zero on failure
2. Can use `println()` or `print()` for output
3. Should be self-contained

### Example: Basic Test

```simplex
fn main() -> i64 {
    let x: i64 = 42;
    if x == 42 {
        println("PASS: value is correct");
        return 0;
    }
    println("FAIL: unexpected value");
    1
}
```

### Example: Neural Gate Test

```simplex
neural_gate threshold(x: f64) -> bool {
    x > 0.5
}

fn main() -> i64 {
    neural_set_training_mode(0);  // Inference mode

    let result: f64 = threshold(0.8);
    if result > 0.5 {
        println("TEST PASSED");
        return 0;
    }

    println("TEST FAILED");
    1
}
```

### Example: Comprehensive Test with Multiple Cases

```simplex
fn test_case_1() -> i64 {
    // Test implementation
    0  // Return 0 on success
}

fn test_case_2() -> i64 {
    // Test implementation
    0
}

fn main() -> i64 {
    print("Running test suite...");

    if test_case_1() != 0 {
        print("FAIL: test_case_1");
        return 1;
    }
    print("  PASS: test_case_1");

    if test_case_2() != 0 {
        print("FAIL: test_case_2");
        return 1;
    }
    print("  PASS: test_case_2");

    print("All tests passed!");
    0
}
```

## Test Categories

| Category | Description | Tests |
|----------|-------------|-------|
| neural | Neural IR, gates, contracts, pruning | 12 |
| language | Core language features (types, async, closures) | ~30 |
| stdlib | Standard library | ~20 |
| runtime | Runtime systems (actors, async, networking) | ~15 |
| ai | AI/Cognitive features (specialists, hives) | ~20 |
| toolchain | Compiler and tools | ~10 |
| integration | End-to-end scenarios | ~7 |

## Test Output

The test runner shows colored output:
- **GREEN (PASS)**: Test succeeded
- **RED (FAIL)**: Test failed (compile, link, or runtime)
- **YELLOW**: Category headers and warnings

Static analysis warnings (e.g., contract violations) are counted separately.

## Adding New Tests

1. Create a `.sx` file in the appropriate directory
2. Implement `fn main() -> i64` that returns 0 on success
3. Run `./run_tests.sh <category>` to verify

Tests are automatically discovered by the test runner.
