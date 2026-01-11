# Simplex Test Suite

This directory contains the test suite for the Simplex programming language.

## Running Tests

```bash
# Run all tests
./run_tests.sh

# Run specific category
./run_tests.sh neural      # Neural IR tests
./run_tests.sh language    # Language feature tests
./run_tests.sh types       # Type system tests
./run_tests.sh stdlib      # Standard library tests
./run_tests.sh runtime     # Runtime tests
./run_tests.sh ai          # AI/Cognitive tests
./run_tests.sh integration # Integration tests
./run_tests.sh toolchain   # Toolchain tests
./run_tests.sh basics      # Basic language tests
./run_tests.sh async       # Async/await tests
./run_tests.sh actors      # Actor model tests
./run_tests.sh learning    # Automatic differentiation tests
./run_tests.sh observability # Observability tests

# Filter by test type
./run_tests.sh all unit    # Run only unit tests
./run_tests.sh all spec    # Run only spec tests
./run_tests.sh all integ   # Run only integration tests
./run_tests.sh all e2e     # Run only end-to-end tests

# Combine category and type
./run_tests.sh stdlib unit     # Only stdlib unit tests
./run_tests.sh neural spec     # Only neural spec tests
./run_tests.sh toolchain integ # Only toolchain integration tests

# Show help
./run_tests.sh --help
```

## Naming Convention

All tests follow a consistent naming convention based on test type:

| Prefix | Type | Description | Color |
|--------|------|-------------|-------|
| `unit_` | Unit | Tests individual functions/types in isolation | Blue |
| `spec_` | Specification | Tests language specification compliance | Cyan |
| `integ_` | Integration | Tests integration between components | Magenta |
| `e2e_` | End-to-End | Tests complete workflows | Yellow |

### Examples

```
unit_crypto.sx       # Unit test for crypto module
spec_generics.sx     # Spec test for generic types
integ_networking.sx  # Integration test for networking
e2e_data_processor.sx # End-to-end data processing workflow
```

## Directory Structure

```
tests/
├── README.md
├── run_tests.sh
│
├── language/                    # Core language features (40 tests)
│   ├── actors/
│   │   └── spec_actor_basic.sx
│   ├── async/
│   │   ├── spec_async_basic.sx
│   │   ├── spec_async_closures.sx
│   │   └── spec_async_multi_await.sx
│   ├── basics/
│   │   ├── spec_closure.sx
│   │   ├── spec_enum.sx
│   │   ├── spec_for_loop.sx
│   │   ├── spec_match.sx
│   │   ├── spec_timing.sx
│   │   └── spec_try_operator.sx
│   ├── closures/
│   │   └── spec_nested_capture.sx
│   ├── control/
│   │   ├── spec_if_let.sx
│   │   ├── spec_if_let_simple.sx
│   │   ├── spec_match_bind.sx
│   │   └── spec_match_simple.sx
│   ├── functions/
│   │   ├── spec_closures.sx
│   │   ├── spec_generic_methods.sx
│   │   └── spec_turbofish.sx
│   ├── modules/
│   │   ├── spec_import.sx
│   │   └── spec_mathlib.sx
│   ├── traits/
│   │   ├── spec_assoc_types.sx
│   │   ├── spec_impl_trait.sx
│   │   ├── spec_trait_ref.sx
│   │   └── spec_trait_self_ref.sx
│   └── types/
│       ├── spec_associated_types.sx
│       ├── spec_generics.sx
│       ├── spec_if_let.sx
│       ├── spec_impl_trait.sx
│       ├── spec_match_binding.sx
│       ├── spec_option_result.sx
│       ├── spec_references.sx
│       └── spec_turbofish.sx
│
├── types/                       # Type system tests (24 tests)
│   ├── spec_associated_types.sx
│   ├── spec_associated_types_p36.sx
│   ├── spec_generic_methods.sx
│   ├── spec_generic_methods_p36.sx
│   ├── spec_generics.sx
│   ├── spec_if_let.sx
│   ├── spec_if_let_p36.sx
│   ├── spec_if_let_simple.sx
│   ├── spec_if_let_simple_p36.sx
│   ├── spec_impl_trait.sx
│   ├── spec_impl_trait_p36.sx
│   ├── spec_match_binding.sx
│   ├── spec_match_binding_p36.sx
│   ├── spec_match_patterns.sx
│   ├── spec_match_patterns_p36.sx
│   ├── spec_option_result.sx
│   ├── spec_option_result_p36.sx
│   ├── spec_references.sx
│   ├── spec_references_p36.sx
│   ├── spec_trait_self_ref.sx
│   ├── spec_trait_self_ref_p36.sx
│   ├── spec_turbofish.sx
│   ├── spec_turbofish_p36.sx
│   └── spec_type_alias.sx
│
├── neural/                      # Neural IR and gates (16 tests)
│   ├── contracts/
│   │   ├── spec_contracts.sx
│   │   ├── spec_static_analysis.sx
│   │   └── spec_static_violation.sx
│   ├── gates/
│   │   ├── spec_basic_inference.sx
│   │   ├── spec_comprehensive.sx
│   │   ├── spec_contracts.sx
│   │   ├── spec_float_negation.sx
│   │   ├── spec_gradient_tape.sx
│   │   ├── spec_hardware_annotation.sx
│   │   ├── spec_minimal.sx
│   │   ├── spec_multiple_gates.sx
│   │   └── spec_training_mode.sx
│   ├── pruning/
│   │   ├── spec_pruning.sx
│   │   └── spec_weight_magnitude.sx
│   ├── spec_hardware_aware.sx
│   └── spec_superposition.sx
│
├── stdlib/                      # Standard library tests (16 tests)
│   ├── unit_assert.sx
│   ├── unit_cli.sx
│   ├── unit_cli_terminal.sx
│   ├── unit_crypto.sx
│   ├── unit_hashmap.sx
│   ├── unit_hashset.sx
│   ├── unit_iterator.sx
│   ├── unit_log.sx
│   ├── unit_manifest.sx
│   ├── unit_option.sx
│   ├── unit_regex.sx
│   ├── unit_result.sx
│   ├── unit_semver.sx
│   ├── unit_string.sx
│   ├── unit_vec.sx
│   └── integ_http_client.sx
│
├── runtime/                     # Runtime system tests (5 tests)
│   ├── actors/
│   │   └── integ_actor_runtime.sx
│   ├── async/
│   │   └── integ_async_runtime.sx
│   ├── distribution/
│   │   └── integ_distribution.sx
│   ├── io/
│   │   └── integ_io.sx
│   └── networking/
│       └── integ_networking.sx
│
├── ai/                          # AI/Cognitive tests (17 tests)
│   ├── anima/
│   │   ├── integ_anima.sx
│   │   ├── integ_anima_hive.sx
│   │   ├── integ_native_ai.sx
│   │   ├── integ_native_hive.sx
│   │   └── unit_native_simple.sx
│   ├── hive/
│   │   ├── integ_hive_mnemonic.sx
│   │   └── integ_per_hive_slm.sx
│   ├── inference/
│   │   └── integ_memory_augmented.sx
│   ├── memory/
│   │   ├── integ_anima_bdi.sx
│   │   ├── integ_anima_persist.sx
│   │   ├── integ_cognitive.sx
│   │   └── unit_anima_memory.sx
│   ├── orchestration/
│   │   ├── integ_cognitive.sx
│   │   └── integ_orchestration.sx
│   ├── specialists/
│   │   └── unit_specialist.sx
│   ├── tools/
│   │   └── unit_tools.sx
│   ├── integ_infer_standalone.sx
│   └── unit_io.sx
│
├── toolchain/                   # Compiler toolchain tests (14 tests)
│   ├── codegen/
│   │   ├── unit_codegen.sx
│   │   ├── unit_compiler_types.sx
│   │   └── unit_phase35_codegen.sx
│   ├── parser/
│   │   ├── unit_parser.sx
│   │   └── unit_phase35_parser.sx
│   ├── sxpm/
│   │   └── unit_model_commands.sx
│   ├── verification/
│   │   ├── integ_verification_main.sx
│   │   ├── integ_verification_suite.sx
│   │   ├── unit_audit_additions.sx
│   │   ├── unit_failures.sx
│   │   └── unit_phase35_verification.sx
│   ├── integ_phase35_toolchain.sx
│   ├── integ_toolchain_main.sx
│   ├── unit_advanced_features.sx
│   ├── unit_phase34_advanced.sx
│   └── unit_phase35_runtime.sx
│
├── integration/                 # End-to-end tests (7 tests)
│   ├── e2e_config_parser.sx
│   ├── e2e_data_processor.sx
│   ├── e2e_knowledge_persistence.sx
│   ├── e2e_model_provisioning.sx
│   ├── e2e_multi_specialist.sx
│   ├── e2e_todo_list.sx
│   └── e2e_word_counter.sx
│
├── basics/                      # Basic language tests (6 tests)
│   ├── spec_closure.sx
│   ├── spec_enum.sx
│   ├── spec_for_loop.sx
│   ├── spec_match.sx
│   ├── spec_timing.sx
│   └── spec_try_operator.sx
│
├── async/                       # Async/await tests (3 tests)
│   ├── spec_async_basic.sx
│   ├── spec_async_closures.sx
│   └── spec_async_multi_await.sx
│
├── actors/                      # Actor model tests (1 test)
│   └── spec_actor_basic.sx
│
├── learning/                    # Automatic differentiation (3 tests)
│   ├── unit_debug_power.sx
│   ├── unit_dual_numbers.sx
│   └── unit_dual_simple.sx
│
└── observability/               # Metrics and tracing (1 test)
    └── integ_observability.sx
```

## Test Categories Summary

| Category | Tests | Description |
|----------|-------|-------------|
| language | 40 | Core language features (types, async, closures, traits) |
| types | 24 | Type system tests (generics, associated types, patterns) |
| neural | 16 | Neural IR, gates, contracts, pruning |
| stdlib | 16 | Standard library (collections, crypto, cli, regex) |
| ai | 17 | AI/Cognitive framework (anima, hive, memory, specialists) |
| toolchain | 14 | Compiler toolchain (parser, codegen, verification) |
| integration | 7 | End-to-end workflow tests |
| basics | 6 | Basic language constructs |
| runtime | 5 | Runtime systems (actors, async, networking) |
| async | 3 | Async/await features |
| learning | 3 | Automatic differentiation (dual numbers) |
| actors | 1 | Actor model |
| observability | 1 | Metrics and tracing |
| **Total** | **156** | |

## Test Type Distribution

| Type | Count | Description |
|------|-------|-------------|
| `spec_` | 90 | Language specification compliance tests |
| `unit_` | 35 | Isolated function/module tests |
| `integ_` | 24 | Integration tests |
| `e2e_` | 7 | End-to-end workflow tests |

## Writing Tests

Tests are `.sx` files that:
1. Follow the naming convention (`unit_`, `spec_`, `integ_`, or `e2e_` prefix)
2. Return `0` on success, non-zero on failure
3. Can use `println()` or `print()` for output
4. Should be self-contained

### Example: Unit Test

```simplex
// unit_example.sx - Tests a single function in isolation

fn test_addition() -> i64 {
    let result = 2 + 2;
    if result == 4 {
        println("PASS: addition works");
        return 0;
    }
    println("FAIL: addition broken");
    1
}

fn main() -> i64 {
    test_addition()
}
```

### Example: Spec Test (Language Feature)

```simplex
// spec_generics.sx - Tests generic type specification

fn identity<T>(x: T) -> T {
    x
}

fn main() -> i64 {
    let a: i64 = identity::<i64>(42);
    let b: f64 = identity::<f64>(3.14);

    if a == 42 && b > 3.0 {
        println("PASS: generics work correctly");
        return 0;
    }
    println("FAIL: generic instantiation failed");
    1
}
```

### Example: Integration Test

```simplex
// integ_networking.sx - Tests multiple networking components together

fn test_http_client() -> i64 {
    // Test HTTP client with server integration
    0
}

fn test_websocket() -> i64 {
    // Test WebSocket protocol
    0
}

fn main() -> i64 {
    var failures: i64 = 0;
    failures = failures + test_http_client();
    failures = failures + test_websocket();

    if failures == 0 {
        println("All networking tests passed!");
    }
    failures
}
```

### Example: End-to-End Test

```simplex
// e2e_data_processor.sx - Complete data processing workflow

fn main() -> i64 {
    // 1. Load data
    let data = load_csv("input.csv");

    // 2. Process data
    let processed = transform(data);

    // 3. Validate output
    if validate(processed) {
        println("PASS: data pipeline works");
        return 0;
    }
    println("FAIL: data pipeline error");
    1
}
```

## Test Output

The test runner shows colored output:
- **GREEN (PASS)**: Test succeeded
- **RED (FAIL)**: Test failed (compile, link, or runtime)
- **YELLOW**: Category headers and e2e tests
- **CYAN**: Spec tests
- **BLUE**: Unit tests
- **MAGENTA**: Integration tests

Static analysis warnings (e.g., contract violations) are counted separately.

## Adding New Tests

1. Choose the appropriate directory based on what you're testing
2. Choose the appropriate prefix based on test type:
   - `unit_` for isolated function/module tests
   - `spec_` for language feature compliance
   - `integ_` for component integration
   - `e2e_` for complete workflows
3. Create a `.sx` file with `fn main() -> i64` that returns 0 on success
4. Run `./run_tests.sh <category>` to verify

Tests are automatically discovered by the test runner.

## Phase Variants

Some tests have `_p36` suffix indicating Phase 36 enhancements to the language. These test newer features while maintaining compatibility with the base tests.

```
spec_generics.sx      # Base generics test
spec_generics_p36.sx  # Phase 36 enhanced generics test
```
