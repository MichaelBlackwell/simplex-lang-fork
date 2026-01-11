# Writing Simplex Tests

**Version:** 0.9.0

## Getting Started

### 1. Choose the Right Location and Prefix

Place tests in the appropriate category directory with the correct prefix:

| Test Type | Prefix | Directory | Example |
|-----------|--------|-----------|---------|
| Unit | `unit_` | `tests/stdlib/` | `tests/stdlib/unit_hashmap.sx` |
| Specification | `spec_` | `tests/language/types/` | `tests/language/types/spec_generics.sx` |
| Integration | `integ_` | `tests/runtime/` | `tests/runtime/integ_networking.sx` |
| End-to-End | `e2e_` | `tests/integration/` | `tests/integration/e2e_data_processor.sx` |

### Category Guidelines

| Category | Test Types | Description |
|----------|------------|-------------|
| `tests/language/` | `spec_` | Core language features |
| `tests/types/` | `spec_` | Type system tests |
| `tests/neural/` | `spec_` | Neural IR tests |
| `tests/stdlib/` | `unit_`, `integ_` | Standard library tests |
| `tests/ai/` | `unit_`, `integ_` | AI/Cognitive tests |
| `tests/learning/` | `unit_` | Dual numbers and AD tests |
| `tests/toolchain/` | `unit_`, `integ_` | Compiler toolchain tests |
| `tests/runtime/` | `integ_` | Runtime system tests |
| `tests/integration/` | `e2e_` | End-to-end scenarios |

### 2. Basic Template

```simplex
// Test file: unit_my_feature.sx
// Description: Tests the my_feature functionality
// v0.9.0

// External declarations
fn println(s: i64);
fn print(s: i64);
fn print_i64(n: i64);
fn string_from(s: i64) -> i64;

fn main() -> i64 {
    println("=== My Feature Test ===");
    println("");

    // Test 1
    println("--- Test 1: Basic functionality ---");
    // Your test code here
    println("PASS: Basic functionality works");
    println("");

    // Test 2
    println("--- Test 2: Edge cases ---");
    // Your test code here
    println("PASS: Edge cases handled");
    println("");

    println("=== My Feature Test Complete! ===");
    0  // Return 0 for success
}
```

## Step-by-Step Guide

### Step 1: Declare External Functions

Declare any runtime functions your test needs:

```simplex
// I/O
fn println(s: i64);
fn print(s: i64);
fn print_i64(n: i64);
fn print_string(s: i64);

// Strings
fn string_from(s: i64) -> i64;
fn string_eq(a: i64, b: i64) -> i64;
fn string_contains(s: i64, substr: i64) -> i64;

// Collections (if needed)
fn vec_new() -> i64;
fn vec_push(v: i64, item: i64);
fn vec_len(v: i64) -> i64;
fn vec_close(v: i64);

// AI (if needed)
fn anima_memory_new(capacity: i64) -> i64;
fn anima_memory_close(mem: i64);

// Dual numbers (if needed)
fn dual_variable(val: f64) -> i64;
fn dual_val(x: i64) -> f64;
fn dual_der(x: i64) -> f64;
```

### Step 2: Write Test Cases

Each test case should:
1. Set up the test state
2. Perform the action
3. Verify the result
4. Clean up resources

```simplex
fn main() -> i64 {
    println("=== HashMap Test ===");
    println("");

    // --- Test 1: Create and insert ---
    println("--- Test 1: Create and insert ---");

    // Setup
    let map: i64 = hashmap_new();

    // Action
    hashmap_insert(map, string_from("key1"), string_from("value1"));

    // Verify
    let value: i64 = hashmap_get(map, string_from("key1"));
    if string_eq(value, string_from("value1")) != 1 {
        println("FAIL: Value mismatch");
        hashmap_close(map);
        return 1;
    }

    println("PASS: Insert and retrieve works");

    // Cleanup
    hashmap_close(map);
    println("");

    println("=== HashMap Test Complete! ===");
    0
}
```

### Step 3: Handle Failures Properly

Return non-zero on failure:

```simplex
fn main() -> i64 {
    let result: i64 = do_something();

    if result != expected {
        println("FAIL: Unexpected result");
        print("  Expected: ");
        print_i64(expected);
        println("");
        print("  Got: ");
        print_i64(result);
        println("");
        return 1;  // IMPORTANT: Return non-zero!
    }

    println("PASS: Result matches expected");
    0
}
```

### Step 4: Clean Up Resources

Always release resources, even on failure:

```simplex
fn main() -> i64 {
    let resource1: i64 = create_resource1();
    let resource2: i64 = create_resource2();

    // Test code...
    if something_failed {
        println("FAIL: Something failed");
        // Still clean up!
        close_resource(resource1);
        close_resource(resource2);
        return 1;
    }

    // Success path cleanup
    close_resource(resource1);
    close_resource(resource2);
    0
}
```

## Writing Different Test Types

### Unit Tests (`unit_*.sx`)

Test individual functions or modules in isolation:

```simplex
// unit_crypto.sx - Tests crypto module functions

fn main() -> i64 {
    println("=== Crypto Unit Tests ===");

    // Test SHA256
    println("--- Test: SHA256 hash ---");
    let input: i64 = string_from("hello");
    let hash: i64 = sha256(input);

    if string_len(hash) != 64 {
        println("FAIL: hash length incorrect");
        return 1;
    }
    println("PASS: SHA256 produces correct length hash");

    println("=== Crypto Unit Tests Complete! ===");
    0
}
```

### Spec Tests (`spec_*.sx`)

Test language specification compliance:

```simplex
// spec_generics.sx - Tests generic type specification

fn identity<T>(x: T) -> T {
    x
}

fn pair<A, B>(a: A, b: B) -> (A, B) {
    (a, b)
}

fn main() -> i64 {
    println("=== Generics Spec Test ===");

    // Test single generic
    println("--- Test: Single generic parameter ---");
    let a: i64 = identity::<i64>(42);
    if a != 42 {
        println("FAIL: identity<i64>");
        return 1;
    }
    println("PASS: identity<i64>");

    // Test multiple generics
    println("--- Test: Multiple generic parameters ---");
    let p = pair::<i64, f64>(1, 2.0);
    // Verify pair works
    println("PASS: pair<i64, f64>");

    println("=== Generics Spec Test Complete! ===");
    0
}
```

### Integration Tests (`integ_*.sx`)

Test multiple components working together:

```simplex
// integ_anima_hive.sx - Tests Anima + Hive integration

fn main() -> i64 {
    println("=== Anima-Hive Integration Test ===");

    // Create Anima
    let anima: i64 = anima_memory_new(10);
    if anima == 0 {
        println("FAIL: Anima creation");
        return 1;
    }
    println("PASS: Anima created");

    // Create HiveMnemonic
    let mnemonic: i64 = hive_mnemonic_new(100, 500, 50);
    if mnemonic == 0 {
        println("FAIL: Mnemonic creation");
        anima_memory_close(anima);
        return 1;
    }
    println("PASS: HiveMnemonic created");

    // Test interaction
    anima_learn(anima, string_from("test knowledge"), 0.8, string_from("self"));
    hive_mnemonic_learn(mnemonic, string_from("shared knowledge"), 0.6);
    println("PASS: Knowledge sharing works");

    // Cleanup
    anima_memory_close(anima);
    hive_mnemonic_close(mnemonic);

    println("=== Integration Test Complete! ===");
    0
}
```

### End-to-End Tests (`e2e_*.sx`)

Test complete workflows:

```simplex
// e2e_todo_list.sx - Complete todo application test

fn main() -> i64 {
    println("=== Todo List E2E Test ===");

    // Step 1: Initialize
    println("--- Step 1: Initialize ---");
    let list: i64 = todo_list_new();
    if list == 0 {
        println("FAIL: initialization");
        return 1;
    }
    println("PASS: initialization");

    // Step 2: Add items
    println("--- Step 2: Add items ---");
    todo_add(list, string_from("Task 1"));
    todo_add(list, string_from("Task 2"));
    todo_add(list, string_from("Task 3"));

    if todo_count(list) != 3 {
        println("FAIL: item count");
        todo_list_close(list);
        return 1;
    }
    println("PASS: add items");

    // Step 3: Complete items
    println("--- Step 3: Complete items ---");
    todo_complete(list, 0);
    if todo_completed_count(list) != 1 {
        println("FAIL: completed count");
        todo_list_close(list);
        return 1;
    }
    println("PASS: complete items");

    // Step 4: Save and load
    println("--- Step 4: Persistence ---");
    todo_save(list, string_from("test.todo"));
    let loaded: i64 = todo_load(string_from("test.todo"));
    if todo_count(loaded) != 3 {
        println("FAIL: persistence");
        todo_list_close(list);
        todo_list_close(loaded);
        return 1;
    }
    println("PASS: persistence");

    // Cleanup
    todo_list_close(list);
    todo_list_close(loaded);

    println("=== E2E Test Complete! ===");
    0
}
```

## Testing Dual Numbers (v0.8.0)

```simplex
// unit_dual_numbers.sx - Tests dual number operations

fn main() -> i64 {
    println("=== Dual Numbers Test ===");

    // Test 1: Basic arithmetic
    println("--- Test 1: Basic arithmetic ---");
    let x: i64 = dual_variable(3.0);    // x = 3, dx = 1
    let c: i64 = dual_constant(2.0);    // c = 2, dc = 0
    let y: i64 = dual_add(dual_mul(x, x), dual_mul(c, x));  // y = x^2 + 2x

    let val: f64 = dual_val(y);  // At x=3: y = 9 + 6 = 15
    let der: f64 = dual_der(y);  // dy/dx = 2x + 2 = 8

    if val < 14.9 || val > 15.1 {
        println("FAIL: value incorrect");
        return 1;
    }
    println("PASS: value correct");

    if der < 7.9 || der > 8.1 {
        println("FAIL: derivative incorrect");
        return 1;
    }
    println("PASS: derivative correct");

    // Test 2: Transcendental functions
    println("--- Test 2: Transcendental functions ---");
    let z: i64 = dual_sin(x);
    // sin(3) ≈ 0.1411, cos(3) ≈ -0.99 (derivative)

    let sin_val: f64 = dual_val(z);
    let sin_der: f64 = dual_der(z);

    if sin_val < 0.1 || sin_val > 0.2 {
        println("FAIL: sin value");
        return 1;
    }
    println("PASS: sin value");

    if sin_der > -0.9 || sin_der < -1.0 {
        println("FAIL: sin derivative (cos)");
        return 1;
    }
    println("PASS: sin derivative");

    println("=== Dual Numbers Test Complete! ===");
    0
}
```

## Best Practices

### 1. Use Correct Prefix

Choose the right prefix based on what you're testing:

```simplex
// GOOD: Appropriate prefixes
unit_hashmap.sx       // Tests HashMap in isolation
spec_generics.sx      // Tests language specification
integ_networking.sx   // Tests component integration
e2e_workflow.sx       // Tests complete workflow

// BAD: Wrong prefixes
test_hashmap.sx       // Old naming, unclear type
hashmap_test.sx       // Wrong format
```

### 2. One Concept Per Test

```simplex
// GOOD: Focused test
fn main() -> i64 {
    println("--- Test: HashMap insert ---");
    let map: i64 = hashmap_new();
    hashmap_insert(map, key, value);
    // Verify insert worked
    hashmap_close(map);
    0
}

// BAD: Testing too many things
fn main() -> i64 {
    // Tests insert, get, remove, iterate all at once
    // Hard to know what failed
}
```

### 3. Clear Test Names

```simplex
// GOOD: Descriptive
println("--- Test: HashMap returns null for missing key ---");

// BAD: Vague
println("--- Test 1 ---");
```

### 4. Test Edge Cases

```simplex
fn main() -> i64 {
    println("=== Edge Cases Test ===");

    // Empty input
    println("--- Test: Empty string ---");
    let empty: i64 = string_from("");
    // ...

    // Null/zero values
    println("--- Test: Null handle ---");
    let result: i64 = operation_on_null(0);
    // ...

    // Boundary values
    println("--- Test: Maximum capacity ---");
    let large: i64 = create_with_capacity(10000);
    // ...

    0
}
```

### 5. Provide Context on Failure

```simplex
if result != expected {
    println("FAIL: Memory count mismatch");
    print("  Expected at least: ");
    print_i64(expected);
    println("");
    print("  Actual count: ");
    print_i64(result);
    println("");
    print("  Test context: After adding 3 memories");
    println("");
    return 1;
}
```

## Running Your Test

```bash
# Run the test directly
sxc run tests/category/unit_my_feature.sx

# Run via test runner (finds by category)
./tests/run_tests.sh category

# Run all unit tests
./tests/run_tests.sh all unit
```

## Checklist Before Committing

- [ ] Test file uses correct prefix (`unit_`, `spec_`, `integ_`, or `e2e_`)
- [ ] Test is in correct category directory
- [ ] All external functions are declared
- [ ] Test returns 0 on success, non-zero on failure
- [ ] Resources are cleaned up (even on failure)
- [ ] Test output includes PASS/FAIL indicators
- [ ] Edge cases are covered
- [ ] Test runs successfully with `sxc run`
- [ ] All other tests still pass with `./tests/run_tests.sh`
