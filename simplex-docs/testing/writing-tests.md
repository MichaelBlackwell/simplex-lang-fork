# Writing Simplex Tests

**Version:** 0.5.0

## Getting Started

### 1. Choose the Right Location

Place tests in the appropriate category directory:

| Test Type | Directory | Example |
|-----------|-----------|---------|
| Language feature | `tests/language/{feature}/` | `tests/language/types/test_generics.sx` |
| Standard library | `tests/stdlib/` | `tests/stdlib/test_hashmap.sx` |
| Runtime behavior | `tests/runtime/` | `tests/runtime/test_actors.sx` |
| AI/Cognitive | `tests/ai/{component}/` | `tests/ai/hive/test_mnemonic.sx` |
| Toolchain | `tests/toolchain/{tool}/` | `tests/toolchain/sxpm/test_build.sx` |
| End-to-end | `tests/integration/` | `tests/integration/test_workflow.sx` |

### 2. Name Your Test File

Use the `test_` prefix:

```
test_feature_name.sx
test_component_behavior.sx
test_scenario_description.sx
```

### 3. Basic Template

```simplex
// Test file: test_my_feature.sx
// Description: Tests the my_feature functionality
// v0.5.0

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

## Best Practices

### 1. One Concept Per Test

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

### 2. Clear Test Names

```simplex
// GOOD: Descriptive
println("--- Test: HashMap returns null for missing key ---");

// BAD: Vague
println("--- Test 1 ---");
```

### 3. Test Edge Cases

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

### 4. Provide Context on Failure

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

### 5. Group Related Tests

```simplex
fn main() -> i64 {
    println("=== HiveMnemonic Test ===");
    println("");

    // Section 1: Creation
    println("### Section 1: Creation ###");
    println("");
    // Creation tests...

    // Section 2: Learning
    println("### Section 2: Learning ###");
    println("");
    // Learning tests...

    // Section 3: Recall
    println("### Section 3: Recall ###");
    println("");
    // Recall tests...

    println("=== HiveMnemonic Test Complete! ===");
    0
}
```

## Testing AI Components

### Testing Anima

```simplex
fn main() -> i64 {
    println("=== Anima Test ===");

    // Create Anima with capacity
    let anima: i64 = anima_memory_new(10);
    if anima == 0 {
        println("FAIL: Could not create Anima");
        return 1;
    }
    println("PASS: Anima created");

    // Test episodic memory
    let mem_id: i64 = anima_remember(anima, string_from("Test memory"), 0.8);
    if mem_id == 0 {
        println("FAIL: Could not add memory");
        anima_memory_close(anima);
        return 1;
    }
    println("PASS: Memory added");

    // Verify count
    let count: i64 = anima_episodic_count(anima);
    if count != 1 {
        println("FAIL: Episodic count mismatch");
        anima_memory_close(anima);
        return 1;
    }
    println("PASS: Episodic count correct");

    // Cleanup
    anima_memory_close(anima);
    println("=== Anima Test Complete! ===");
    0
}
```

### Testing HiveMnemonic

```simplex
fn main() -> i64 {
    println("=== HiveMnemonic Test ===");

    // Create with 50% belief threshold
    let mnemonic: i64 = hive_mnemonic_new(100, 500, 50);
    if mnemonic == 0 {
        println("FAIL: Could not create HiveMnemonic");
        return 1;
    }

    // Test belief threshold
    // Beliefs below 50% should be filtered
    hive_mnemonic_believe(mnemonic, string_from("Weak belief"), 0.3);
    hive_mnemonic_believe(mnemonic, string_from("Strong belief"), 0.7);

    let belief_count: i64 = hive_mnemonic_belief_count(mnemonic);
    if belief_count != 1 {
        println("FAIL: Belief threshold not applied");
        hive_mnemonic_close(mnemonic);
        return 1;
    }
    println("PASS: 50% belief threshold works");

    hive_mnemonic_close(mnemonic);
    0
}
```

### Testing Per-Hive SLM

```simplex
fn main() -> i64 {
    println("=== Per-Hive SLM Test ===");

    // Create shared infrastructure
    let mnemonic: i64 = hive_mnemonic_new(100, 500, 50);
    let hive_slm: i64 = hive_slm_new(
        string_from("TestHive"),
        string_from("simplex-cognitive-7b"),
        mnemonic
    );

    // Create multiple specialists
    let anima1: i64 = anima_memory_new(10);
    let anima2: i64 = anima_memory_new(10);
    let anima3: i64 = anima_memory_new(10);

    let spec1: i64 = specialist_create(string_from("Spec1"), hive_slm, anima1);
    let spec2: i64 = specialist_create(string_from("Spec2"), hive_slm, anima2);
    let spec3: i64 = specialist_create(string_from("Spec3"), hive_slm, anima3);

    // Verify all share same SLM
    let slm1: i64 = specialist_get_hive_slm(spec1);
    let slm2: i64 = specialist_get_hive_slm(spec2);
    let slm3: i64 = specialist_get_hive_slm(spec3);

    if slm1 != slm2 || slm2 != slm3 {
        println("FAIL: Specialists have different SLMs");
        // Cleanup...
        return 1;
    }
    println("PASS: All specialists share same Hive SLM");

    // Cleanup
    specialist_close(spec1);
    specialist_close(spec2);
    specialist_close(spec3);
    anima_memory_close(anima1);
    anima_memory_close(anima2);
    anima_memory_close(anima3);
    hive_slm_close(hive_slm);
    hive_mnemonic_close(mnemonic);

    0
}
```

## Running Your Test

```bash
# Compile and run
sxc run tests/category/test_my_feature.sx

# Or use sxpm
sxpm test tests/category/test_my_feature.sx

# Run all tests to verify no regressions
sxpm test
```

## Checklist Before Committing

- [ ] Test file starts with `test_` prefix
- [ ] Test is in correct category directory
- [ ] All external functions are declared
- [ ] Test returns 0 on success, non-zero on failure
- [ ] Resources are cleaned up (even on failure)
- [ ] Test output includes PASS/FAIL indicators
- [ ] Edge cases are covered
- [ ] Test runs successfully with `sxc run`
- [ ] All other tests still pass with `sxpm test`
