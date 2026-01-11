# Simplex Test Coverage

**Version:** 0.9.0
**Last Updated:** v0.9.0 Release

## Coverage Summary

| Category | Tests | Features Covered | Coverage |
|----------|-------|------------------|----------|
| Language | 40 | Core syntax, types, async, closures, traits | High |
| Types | 24 | Generics, associated types, patterns | High |
| Neural | 16 | Neural gates, contracts, pruning | High |
| Standard Library | 16 | Collections, I/O, HTTP, crypto | Medium |
| AI/Cognitive | 17 | Anima, Hive, SLM, memory | High |
| Toolchain | 14 | Compiler, sxpm, verification | Medium |
| Integration | 7 | End-to-end scenarios | High |
| Basics | 6 | Core language constructs | High |
| Runtime | 5 | Actors, async, networking | Medium |
| Async | 3 | Async/await features | High |
| Learning | 3 | Dual numbers, AD | High |
| Actors | 1 | Actor model | Medium |
| Observability | 1 | Metrics and tracing | Medium |
| **Total** | **156** | - | - |

## Test Type Distribution

| Type | Count | Description |
|------|-------|-------------|
| `spec_` | 90 | Language specification compliance |
| `unit_` | 35 | Isolated function/module tests |
| `integ_` | 24 | Component integration tests |
| `e2e_` | 7 | End-to-end workflow tests |

## Language Tests Coverage

### Basics (`tests/language/basics/`)

| Test File | Features Covered |
|-----------|------------------|
| `spec_closure.sx` | Closure syntax, capture, higher-order functions |
| `spec_enum.sx` | Enum definition, variants, pattern matching |
| `spec_for_loop.sx` | `for` loops, iterators |
| `spec_match.sx` | `match` expressions, destructuring |
| `spec_timing.sx` | Time-related operations |
| `spec_try_operator.sx` | `?` operator, error propagation |

### Types (`tests/language/types/`)

| Test File | Features Covered |
|-----------|------------------|
| `spec_generics.sx` | Generic functions, generic structs |
| `spec_references.sx` | References, borrowing |
| `spec_core_language.sx` | Core type system features |
| `spec_references_alt.sx` | Alternative reference patterns |

### Async (`tests/language/async/`)

| Test File | Features Covered |
|-----------|------------------|
| `spec_async_basic.sx` | `async fn`, `.await` syntax |
| `spec_async_closures.sx` | Async closures |
| `spec_async_multi_await.sx` | Multiple concurrent awaits |

### Actors (`tests/language/actors/`)

| Test File | Features Covered |
|-----------|------------------|
| `spec_actor_basic.sx` | Actor definitions, message handling |

### Traits (`tests/language/traits/`)

| Test File | Features Covered |
|-----------|------------------|
| `spec_assoc_types.sx` | Associated types |
| `spec_impl_trait.sx` | `impl Trait` return types |
| `spec_trait_ref.sx` | Trait references |
| `spec_trait_self_ref.sx` | Self-referential traits |

## Types Tests Coverage (`tests/types/`)

| Test File | Features Covered |
|-----------|------------------|
| `spec_generics.sx` | Generic type instantiation |
| `spec_associated_types.sx` | Trait associated types |
| `spec_option_result.sx` | `Option<T>`, `Result<T, E>`, unwrapping |
| `spec_impl_trait.sx` | `impl Trait` return types |
| `spec_references.sx` | Reference semantics |
| `spec_turbofish.sx` | Turbofish syntax `::<T>` |
| `spec_if_let.sx` | `if let` pattern matching |
| `spec_if_let_simple.sx` | Simple if-let patterns |
| `spec_match_binding.sx` | Match with variable binding |
| `spec_match_patterns.sx` | Complex match patterns |
| `spec_generic_methods.sx` | Generic methods on types |
| `spec_trait_self_ref.sx` | Self-referential trait bounds |
| `spec_type_alias.sx` | Type aliases |
| `spec_*_p36.sx` | Phase 36 enhanced versions |

## Neural Tests Coverage (`tests/neural/`)

### Gates (`tests/neural/gates/`)

| Test File | Features Covered |
|-----------|------------------|
| `spec_minimal.sx` | Minimal neural gate |
| `spec_basic_inference.sx` | Basic gate inference |
| `spec_comprehensive.sx` | Comprehensive gate testing |
| `spec_contracts.sx` | Gate contracts |
| `spec_float_negation.sx` | Float handling in gates |
| `spec_gradient_tape.sx` | Gradient recording |
| `spec_hardware_annotation.sx` | `@cpu`, `@gpu`, `@npu` |
| `spec_multiple_gates.sx` | Multiple gates interaction |
| `spec_training_mode.sx` | Training vs inference mode |

### Contracts (`tests/neural/contracts/`)

| Test File | Features Covered |
|-----------|------------------|
| `spec_contracts.sx` | `requires`, `ensures`, `fallback` |
| `spec_static_analysis.sx` | Static contract checking |
| `spec_static_violation.sx` | Contract violation detection |

### Pruning (`tests/neural/pruning/`)

| Test File | Features Covered |
|-----------|------------------|
| `spec_pruning.sx` | Structural pruning |
| `spec_weight_magnitude.sx` | Weight-based pruning |

### Other Neural Tests

| Test File | Features Covered |
|-----------|------------------|
| `spec_hardware_aware.sx` | Hardware-aware compilation |
| `spec_superposition.sx` | Superposition states |

## Standard Library Coverage (`tests/stdlib/`)

| Test File | Features Covered |
|-----------|------------------|
| `unit_hashmap.sx` | HashMap creation, insert, get, remove, iterate |
| `unit_hashset.sx` | HashSet operations |
| `unit_vec.sx` | Vec creation, push, pop, indexing |
| `unit_string.sx` | String concat, split, contains, formatting |
| `unit_option.sx` | Option methods: map, unwrap_or, is_some |
| `unit_result.sx` | Result methods: map, map_err, unwrap_or |
| `unit_crypto.sx` | Cryptographic functions |
| `unit_iterator.sx` | Iterator traits and methods |
| `unit_regex.sx` | Regular expression matching |
| `unit_cli.sx` | Command-line argument parsing |
| `unit_cli_terminal.sx` | Colors, progress bars, spinners, tables |
| `unit_log.sx` | Logging functionality |
| `unit_assert.sx` | Assertion macros |
| `unit_manifest.sx` | Modulus.toml parsing |
| `unit_semver.sx` | Semantic versioning |
| `integ_http_client.sx` | HTTP requests, URL parsing, response handling |

## AI/Cognitive Coverage (`tests/ai/`)

### Anima (`tests/ai/anima/`)

| Test File | Features Covered |
|-----------|------------------|
| `integ_anima.sx` | Full Anima integration |
| `integ_anima_hive.sx` | Anima + Hive integration |
| `integ_native_ai.sx` | Native AI integration |
| `integ_native_hive.sx` | Native Hive integration |
| `unit_native_simple.sx` | Simple native AI test |

### Hive (`tests/ai/hive/`)

| Test File | Features Covered |
|-----------|------------------|
| `integ_hive_mnemonic.sx` | Shared consciousness, 50% belief threshold |
| `integ_per_hive_slm.sx` | ONE shared SLM per hive verification |

### Inference (`tests/ai/inference/`)

| Test File | Features Covered |
|-----------|------------------|
| `integ_memory_augmented.sx` | Context flow Anima → Mnemonic → SLM |

### Memory (`tests/ai/memory/`)

| Test File | Features Covered |
|-----------|------------------|
| `unit_anima_memory.sx` | Episodic, semantic, working memory |
| `integ_anima_bdi.sx` | Beliefs, Desires, Intentions |
| `integ_anima_persist.sx` | Memory persistence |
| `integ_cognitive.sx` | Cognitive integration |

### Orchestration (`tests/ai/orchestration/`)

| Test File | Features Covered |
|-----------|------------------|
| `integ_orchestration.sx` | Multi-agent orchestration |
| `integ_cognitive.sx` | Cognitive orchestration |

### Specialists (`tests/ai/specialists/`)

| Test File | Features Covered |
|-----------|------------------|
| `unit_specialist.sx` | Specialist creation, routing |

### Tools (`tests/ai/tools/`)

| Test File | Features Covered |
|-----------|------------------|
| `unit_tools.sx` | Tool calling |

## Learning Coverage (`tests/learning/`)

| Test File | Features Covered |
|-----------|------------------|
| `unit_dual_numbers.sx` | Dual number type, arithmetic, derivatives |
| `unit_dual_simple.sx` | Simple dual number operations |
| `unit_debug_power.sx` | Power function derivatives |

### Features Tested

- **Dual Number Arithmetic**: Addition, subtraction, multiplication, division
- **Derivative Propagation**: Chain rule application
- **Transcendental Functions**: sin, cos, exp, ln derivatives
- **Multi-dimensional**: `multidual<N>` for gradients

## Toolchain Coverage (`tests/toolchain/`)

### Parser (`tests/toolchain/parser/`)

| Test File | Features Covered |
|-----------|------------------|
| `unit_parser.sx` | Parser correctness for all syntax |
| `unit_phase35_parser.sx` | Phase 35 parser enhancements |

### Codegen (`tests/toolchain/codegen/`)

| Test File | Features Covered |
|-----------|------------------|
| `unit_codegen.sx` | Code generation for all constructs |
| `unit_compiler_types.sx` | Compiler type handling |
| `unit_phase35_codegen.sx` | Phase 35 codegen enhancements |

### Package Manager (`tests/toolchain/sxpm/`)

| Test File | Features Covered |
|-----------|------------------|
| `unit_model_commands.sx` | `sxpm model list/install/remove/info` |

### Verification (`tests/toolchain/verification/`)

| Test File | Features Covered |
|-----------|------------------|
| `integ_verification_main.sx` | Main verification suite |
| `integ_verification_suite.sx` | Full verification |
| `unit_failures.sx` | Failure handling |
| `unit_audit_additions.sx` | Audit features |
| `unit_phase35_verification.sx` | Phase 35 verification |

## Integration Coverage (`tests/integration/`)

| Test File | Scenario |
|-----------|----------|
| `e2e_multi_specialist.sx` | Multi-specialist reasoning hive |
| `e2e_knowledge_persistence.sx` | Memory save/load across sessions |
| `e2e_model_provisioning.sx` | Model discovery → inference |
| `e2e_todo_list.sx` | Real-world todo application |
| `e2e_data_processor.sx` | Data processing workflow |
| `e2e_word_counter.sx` | Word counting application |
| `e2e_config_parser.sx` | Configuration file parsing |

## Runtime Coverage (`tests/runtime/`)

| Test File | Features Covered |
|-----------|------------------|
| `integ_actor_runtime.sx` | Actor spawning, message passing |
| `integ_async_runtime.sx` | Async execution, scheduling |
| `integ_distribution.sx` | Distributed execution |
| `integ_io.sx` | I/O operations |
| `integ_networking.sx` | Network communication |

## v0.9.0 New Test Coverage

### Test Suite Restructure

The v0.9.0 release completely reorganized the test suite:

- **New naming convention**: `unit_`, `spec_`, `integ_`, `e2e_` prefixes
- **156 total tests**: Organized across 13 categories
- **New test runner**: `./tests/run_tests.sh` with category/type filtering

### Learning Tests (v0.8.0/v0.9.0)

Tests for dual numbers and automatic differentiation:

- `unit_dual_numbers.sx` - Complete dual number API
- `unit_dual_simple.sx` - Basic AD operations
- `unit_debug_power.sx` - Power function derivatives

## v0.8.0 New Test Coverage

Tests added for dual numbers:

- Dual number type construction
- Arithmetic with derivative propagation
- Transcendental function derivatives
- Multi-dimensional gradients (`multidual<N>`)

## v0.7.0 New Test Coverage

Tests added for Real-Time Learning:

- simplex-learning library (tensor ops, autograd)
- Streaming optimizers (SGD, Adam, AdamW)
- Safety constraints and fallbacks
- Federated learning and distillation

## Coverage Gaps

Areas with limited or no test coverage:

| Area | Status | Priority |
|------|--------|----------|
| Distributed actors | Basic only | High |
| GPU acceleration | Not tested | Medium |
| Cross-platform | macOS primarily | Medium |
| MPI/NCCL integration | Simulated | Low |

## Running Coverage Report

```bash
# Run all tests
./tests/run_tests.sh

# Run specific category
./tests/run_tests.sh learning
./tests/run_tests.sh neural

# Filter by type
./tests/run_tests.sh all unit
```

## Coverage Goals

| Category | Current | Target |
|----------|---------|--------|
| Language | 90% | 95% |
| Types | 85% | 95% |
| Neural | 80% | 90% |
| Standard Library | 70% | 90% |
| AI/Cognitive | 85% | 95% |
| Learning | 80% | 95% |
| Toolchain | 70% | 85% |
| Runtime | 60% | 80% |
| Integration | 80% | 90% |
