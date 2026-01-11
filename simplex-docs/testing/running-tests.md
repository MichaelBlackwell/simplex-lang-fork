# Running Simplex Tests

**Version:** 0.9.0

## Quick Start

### Run All Tests

```bash
./tests/run_tests.sh
```

### Expected Output

```
================================================================================
                         Simplex Language Test Suite
================================================================================

Running tests in: language

  language/actors:
    [SPEC] spec_actor_basic                                              PASS

  language/async:
    [SPEC] spec_async_basic                                              PASS
    [SPEC] spec_async_closures                                           PASS
    ...

Running tests in: stdlib

  stdlib:
    [UNIT] unit_hashmap                                                  PASS
    [UNIT] unit_crypto                                                   PASS
    [UNIT] unit_string                                                   PASS
    ...

Running tests in: learning

  learning:
    [UNIT] unit_dual_numbers                                             PASS
    [UNIT] unit_dual_simple                                              PASS
    ...

================================================================================
                              Test Results Summary
================================================================================
  Passed:   156
  Failed:   0
  Warnings: 0 (static analysis)
  Total:    156 tests (100% pass rate)
================================================================================
```

## Running by Category

```bash
# Language tests
./tests/run_tests.sh language

# Type system tests
./tests/run_tests.sh types

# Neural IR tests
./tests/run_tests.sh neural

# Standard library tests
./tests/run_tests.sh stdlib

# AI/Cognitive tests
./tests/run_tests.sh ai

# Learning/AD tests
./tests/run_tests.sh learning

# Toolchain tests
./tests/run_tests.sh toolchain

# Runtime tests
./tests/run_tests.sh runtime

# Integration tests
./tests/run_tests.sh integration

# Basic language tests
./tests/run_tests.sh basics

# Async tests
./tests/run_tests.sh async

# Actor tests
./tests/run_tests.sh actors

# Observability tests
./tests/run_tests.sh observability
```

## Running by Test Type

Filter tests by their naming prefix:

```bash
# Only unit tests (unit_*)
./tests/run_tests.sh all unit

# Only spec tests (spec_*)
./tests/run_tests.sh all spec

# Only integration tests (integ_*)
./tests/run_tests.sh all integ

# Only end-to-end tests (e2e_*)
./tests/run_tests.sh all e2e
```

## Combining Category and Type

```bash
# Only stdlib unit tests
./tests/run_tests.sh stdlib unit

# Only neural spec tests
./tests/run_tests.sh neural spec

# Only toolchain integration tests
./tests/run_tests.sh toolchain integ
```

## Running Individual Tests

### Using sxc (Compiler)

```bash
# Compile and run a single test
sxc run tests/stdlib/unit_hashmap.sx

# Compile only (produces executable)
sxc compile tests/stdlib/unit_hashmap.sx

# Check syntax without running
sxc check tests/stdlib/unit_hashmap.sx
```

### Direct Execution

```bash
# Run the test runner for a single file
./tests/run_tests.sh stdlib | grep hashmap
```

## Verbose Output

The test runner shows detailed output for each test. To see even more detail:

```bash
# Run single test directly for full output
sxc run tests/stdlib/unit_hashmap.sx
```

Example verbose output:
```
=== HashMap Test ===

--- Test 1: Create HashMap ---
PASS: HashMap created

--- Test 2: Insert and Get ---
PASS: Value retrieved correctly

--- Test 3: Contains Key ---
PASS: Contains key works

=== HashMap Test Complete! ===
```

## Test Output Colors

The test runner uses colors to indicate test types and results:

| Color | Meaning |
|-------|---------|
| GREEN | Test passed (PASS) |
| RED | Test failed (FAIL) |
| BLUE | Unit test (`[UNIT]`) |
| CYAN | Spec test (`[SPEC]`) |
| MAGENTA | Integration test (`[INTEG]`) |
| YELLOW | E2E test (`[E2E]`) / Category headers |

## Debugging Failed Tests

### Step 1: Run the Specific Test

```bash
sxc run tests/path/to/failing_test.sx
```

### Step 2: Check Compilation

```bash
# Syntax check only
sxc check tests/path/to/failing_test.sx

# Compile with debug info
sxc compile --debug tests/path/to/failing_test.sx
```

### Step 3: Inspect Generated Code

```bash
# Generate readable IR
sxc compile --emit-ir tests/path/to/failing_test.sx

# View generated assembly
sxc compile --emit-asm tests/path/to/failing_test.sx
```

## Test Result Interpretation

### Failure Types

| Result | Cause |
|--------|-------|
| `COMPILE FAIL` | Syntax error or type error in test |
| `LINK FAIL` | Linking error |
| `RUNTIME FAIL` | Runtime exception or panic |
| `FAIL` | Test assertions failed (exit code non-zero) |

### Static Analysis Warnings

The test runner tracks static analysis warnings (e.g., contract violations) separately:

```
================================================================================
                              Test Results Summary
================================================================================
  Passed:   156
  Failed:   0
  Warnings: 3 (static analysis)
  Total:    156 tests (100% pass rate)
================================================================================
```

Warnings do not cause test failure but are reported for attention.

## Continuous Integration

### CI Configuration

```yaml
# .simplex-ci.yml
name: Simplex Tests

stages:
  - test

test:
  script:
    - ./tests/run_tests.sh
  artifacts:
    reports:
      - test-results.xml
```

### Exit Codes for CI

The test runner returns:
- `0` if all tests pass
- Non-zero if any test fails

## Show Help

```bash
./tests/run_tests.sh --help
```

Output:
```
Simplex Test Runner

Usage: ./run_tests.sh [category] [type]

Categories:
  all           Run all test categories (default)
  language      Core language features
  types         Type system tests
  neural        Neural IR and gates
  stdlib        Standard library
  ai            AI/Cognitive tests
  learning      Automatic differentiation
  toolchain     Compiler toolchain
  runtime       Runtime systems
  integration   End-to-end tests
  basics        Basic language tests
  async         Async/await tests
  actors        Actor model tests
  observability Metrics and tracing

Types:
  unit          Run only unit tests (unit_*)
  spec          Run only spec tests (spec_*)
  integ         Run only integration tests (integ_*)
  e2e           Run only end-to-end tests (e2e_*)

Examples:
  ./run_tests.sh                  # Run all tests
  ./run_tests.sh neural           # Run neural category
  ./run_tests.sh all spec         # Run all spec tests
  ./run_tests.sh stdlib unit      # Run stdlib unit tests
```

## Troubleshooting

### "Test not found"

```bash
# List all discovered tests
ls tests/category/*.sx

# Verify test file exists
ls tests/path/to/test.sx
```

### "Compilation failed"

```bash
# Get detailed error
sxc check tests/path/to/failing_test.sx

# Check for missing imports
sxc compile --verbose tests/path/to/failing_test.sx
```

### "Test hangs"

If a test appears to hang:
1. Check for infinite loops in test code
2. Check for blocking I/O without timeout
3. Run with timeout (Ctrl+C to cancel)

### "Inconsistent results"

If tests pass sometimes and fail others:
1. Check for uninitialized variables
2. Check for race conditions in async tests
3. Run tests individually to isolate issues

```bash
sxc run tests/path/to/flaky_test.sx
```
