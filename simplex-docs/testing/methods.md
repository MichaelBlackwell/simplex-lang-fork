# Simplex Testing Methods

**Version:** 0.9.0

## Test Naming Convention

All tests follow a consistent naming convention based on test type:

| Prefix | Type | Description | Use For |
|--------|------|-------------|---------|
| `unit_` | Unit | Tests individual functions/types in isolation | Module tests, function tests |
| `spec_` | Specification | Tests language specification compliance | Language features, syntax |
| `integ_` | Integration | Tests integration between components | Multi-component tests |
| `e2e_` | End-to-End | Tests complete workflows | Full application scenarios |

### Examples

```
unit_hashmap.sx       # Unit test for HashMap module
spec_generics.sx      # Spec test for generic types
integ_networking.sx   # Integration test for networking
e2e_data_processor.sx # End-to-end data processing workflow
```

## Test Structure

### Basic Test Structure

Every Simplex test follows this structure:

```simplex
// Test file: unit_feature.sx
// Description of what this test validates

// External declarations for runtime functions
fn println(s: i64);
fn print(s: i64);
fn string_from(s: i64) -> i64;

fn main() -> i64 {
    println("=== Feature Test ===");

    // Test 1
    println("--- Test 1: Description ---");
    // ... test code ...
    println("PASS: Test 1");

    // Test 2
    println("--- Test 2: Description ---");
    // ... test code ...
    println("PASS: Test 2");

    println("=== Feature Test Complete! ===");
    0  // Return 0 for success
}
```

### Multi-Test Structure

For comprehensive feature testing:

```simplex
fn test_create() -> i64 {
    // Test creation
    // Return 0 on success, 1 on failure
    0
}

fn test_insert() -> i64 {
    // Test insertion
    0
}

fn test_retrieve() -> i64 {
    // Test retrieval
    0
}

fn main() -> i64 {
    var failures: i64 = 0;

    println("=== Comprehensive Test ===");

    if test_create() != 0 {
        println("FAIL: test_create");
        failures = failures + 1;
    } else {
        println("PASS: test_create");
    }

    if test_insert() != 0 {
        println("FAIL: test_insert");
        failures = failures + 1;
    } else {
        println("PASS: test_insert");
    }

    if test_retrieve() != 0 {
        println("FAIL: test_retrieve");
        failures = failures + 1;
    } else {
        println("PASS: test_retrieve");
    }

    println("");
    if failures == 0 {
        println("All tests passed!");
    } else {
        print("Tests failed: ");
        print_i64(failures);
        println("");
    }

    failures  // Return number of failures
}
```

## Assertion Patterns

### Simple Equality

```simplex
let result: i64 = compute_value();
if result != expected {
    println("FAIL: result mismatch");
    return 1;
}
println("PASS: result matches expected");
```

### String Comparison

```simplex
fn string_eq(a: i64, b: i64) -> i64;

let result: i64 = get_string();
if string_eq(result, string_from("expected")) != 1 {
    println("FAIL: string mismatch");
    return 1;
}
println("PASS: strings match");
```

### Contains Check

```simplex
fn string_contains(s: i64, substr: i64) -> i64;

let output: i64 = get_output();
if string_contains(output, string_from("expected")) != 1 {
    println("FAIL: output missing expected content");
    return 1;
}
println("PASS: output contains expected content");
```

### Null Check

```simplex
let handle: i64 = create_resource();
if handle == 0 {
    println("FAIL: resource creation returned null");
    return 1;
}
println("PASS: resource created successfully");
```

### Range Check

```simplex
let count: i64 = get_count();
if count < 1 || count > 100 {
    println("FAIL: count out of expected range");
    return 1;
}
println("PASS: count within expected range");
```

## External Function Declarations

Tests declare external functions that are provided by the Simplex runtime:

### I/O Functions

```simplex
fn print(s: i64);
fn println(s: i64);
fn print_i64(n: i64);
fn print_f64(n: f64);
fn print_string(s: i64);
```

### String Functions

```simplex
fn string_from(s: i64) -> i64;
fn string_eq(a: i64, b: i64) -> i64;
fn string_contains(s: i64, substr: i64) -> i64;
fn string_concat(a: i64, b: i64) -> i64;
fn string_len(s: i64) -> i64;
```

### Collection Functions

```simplex
// HashMap
fn hashmap_new() -> i64;
fn hashmap_insert(map: i64, key: i64, value: i64);
fn hashmap_get(map: i64, key: i64) -> i64;
fn hashmap_contains(map: i64, key: i64) -> i64;
fn hashmap_remove(map: i64, key: i64);
fn hashmap_close(map: i64);

// Vec
fn vec_new() -> i64;
fn vec_push(v: i64, item: i64);
fn vec_pop(v: i64) -> i64;
fn vec_get(v: i64, index: i64) -> i64;
fn vec_len(v: i64) -> i64;
fn vec_close(v: i64);
```

### AI/Cognitive Functions

```simplex
// Anima
fn anima_memory_new(capacity: i64) -> i64;
fn anima_remember(mem: i64, content: i64, importance: f64) -> i64;
fn anima_learn(mem: i64, content: i64, confidence: f64, source: i64) -> i64;
fn anima_believe(mem: i64, content: i64, confidence: f64, evidence: i64) -> i64;
fn anima_memory_close(mem: i64);

// HiveMnemonic
fn hive_mnemonic_new(episodic_cap: i64, semantic_cap: i64, belief_threshold: i64) -> i64;
fn hive_mnemonic_learn(mnemonic: i64, content: i64, confidence: f64) -> i64;
fn hive_mnemonic_close(mnemonic: i64);

// Hive SLM
fn hive_slm_new(name: i64, model: i64, mnemonic: i64) -> i64;
fn hive_slm_close(slm: i64);
```

### Dual Number Functions (v0.8.0)

```simplex
// Dual number construction
fn dual_new(val: f64, der: f64) -> i64;
fn dual_constant(val: f64) -> i64;
fn dual_variable(val: f64) -> i64;

// Dual number operations
fn dual_add(a: i64, b: i64) -> i64;
fn dual_mul(a: i64, b: i64) -> i64;
fn dual_sin(x: i64) -> i64;
fn dual_cos(x: i64) -> i64;
fn dual_exp(x: i64) -> i64;
fn dual_ln(x: i64) -> i64;

// Access value and derivative
fn dual_val(x: i64) -> f64;
fn dual_der(x: i64) -> f64;
```

## Test Categories

### Unit Tests (`unit_*.sx`)

Test individual functions or components in isolation:

```simplex
// unit_string_concat.sx
fn test_string_concat() -> i64 {
    let a: i64 = string_from("Hello, ");
    let b: i64 = string_from("World!");
    let result: i64 = string_concat(a, b);

    if string_eq(result, string_from("Hello, World!")) != 1 {
        return 1;
    }
    0
}

fn main() -> i64 {
    if test_string_concat() != 0 {
        println("FAIL: string concat");
        return 1;
    }
    println("PASS: string concat");
    0
}
```

### Spec Tests (`spec_*.sx`)

Test language specification compliance:

```simplex
// spec_generics.sx
fn identity<T>(x: T) -> T {
    x
}

fn main() -> i64 {
    let a: i64 = identity::<i64>(42);
    let b: f64 = identity::<f64>(3.14);

    if a != 42 {
        println("FAIL: generic i64");
        return 1;
    }
    println("PASS: generic i64");

    if b < 3.0 {
        println("FAIL: generic f64");
        return 1;
    }
    println("PASS: generic f64");

    0
}
```

### Integration Tests (`integ_*.sx`)

Test multiple components working together:

```simplex
// integ_specialist_anima.sx
fn main() -> i64 {
    println("=== Specialist with Anima Integration ===");

    // Create Anima
    let anima: i64 = anima_memory_new(10);
    anima_learn(anima, string_from("I am a specialist"), 0.9, string_from("self"));

    // Create HiveMnemonic
    let mnemonic: i64 = hive_mnemonic_new(100, 500, 50);

    // Create Hive SLM
    let hive_slm: i64 = hive_slm_new(
        string_from("TestHive"),
        string_from("simplex-cognitive-7b"),
        mnemonic
    );

    // Create Specialist
    let spec: i64 = specialist_create(string_from("TestSpec"), hive_slm, anima);

    if spec == 0 {
        println("FAIL: specialist creation");
        return 1;
    }
    println("PASS: specialist creation");

    // Cleanup
    specialist_close(spec);
    anima_memory_close(anima);
    hive_slm_close(hive_slm);
    hive_mnemonic_close(mnemonic);

    println("=== Integration Test Complete! ===");
    0
}
```

### End-to-End Tests (`e2e_*.sx`)

Test complete workflows:

```simplex
// e2e_data_processor.sx
fn main() -> i64 {
    println("=== Data Processing Workflow ===");

    // Step 1: Load data
    let data: i64 = load_data("input.csv");
    if data == 0 {
        println("FAIL: load data");
        return 1;
    }
    println("PASS: load data");

    // Step 2: Process data
    let processed: i64 = transform(data);
    if processed == 0 {
        println("FAIL: transform");
        return 1;
    }
    println("PASS: transform");

    // Step 3: Validate output
    if validate(processed) != 1 {
        println("FAIL: validation");
        return 1;
    }
    println("PASS: validation");

    println("=== Workflow Complete! ===");
    0
}
```

## Resource Management

Always clean up resources at the end of tests:

```simplex
fn main() -> i64 {
    // Create resources
    let map: i64 = hashmap_new();
    let vec: i64 = vec_new();
    let anima: i64 = anima_memory_new(10);

    // ... test code ...

    // Cleanup (always do this!)
    hashmap_close(map);
    vec_close(vec);
    anima_memory_close(anima);

    0
}
```

## Test Output Conventions

### Section Headers

```simplex
println("=== Test Suite Name ===");           // Major section
println("### Phase Name ###");                 // Phase within test
println("--- Test Name ---");                  // Individual test
```

### Result Indicators

```simplex
println("PASS: Description of what passed");
println("FAIL: Description of what failed");
println("WARN: Warning message");
println("INFO: Informational message");
```

## Testing Async Code

```simplex
async fn test_async_operation() -> i64 {
    let result: i64 = some_async_function().await;
    if result != expected {
        return 1;
    }
    0
}

fn main() -> i64 {
    // Run async test
    let result: i64 = block_on(test_async_operation());
    if result != 0 {
        println("FAIL: async test");
        return 1;
    }
    println("PASS: async test");
    0
}
```

## Testing Actors

```simplex
actor TestActor {
    state: i64,

    fn new() -> TestActor {
        TestActor { state: 0 }
    }

    fn increment(&mut self) {
        self.state = self.state + 1;
    }

    fn get_state(&self) -> i64 {
        self.state
    }
}

fn main() -> i64 {
    let actor: TestActor = spawn TestActor::new();

    actor.increment();
    actor.increment();

    let state: i64 = actor.get_state();
    if state != 2 {
        println("FAIL: actor state incorrect");
        return 1;
    }

    println("PASS: actor test");
    0
}
```

## Testing Dual Numbers (v0.8.0)

```simplex
fn main() -> i64 {
    println("=== Dual Number Test ===");

    // Test basic derivative
    let x: i64 = dual_variable(3.0);
    let y: i64 = dual_mul(x, x);  // y = x^2

    let val: f64 = dual_val(y);
    let der: f64 = dual_der(y);

    // At x=3: y = 9, dy/dx = 6
    if val < 8.9 || val > 9.1 {
        println("FAIL: value incorrect");
        return 1;
    }
    println("PASS: value correct");

    if der < 5.9 || der > 6.1 {
        println("FAIL: derivative incorrect");
        return 1;
    }
    println("PASS: derivative correct");

    println("=== Dual Number Test Complete! ===");
    0
}
```

## Best Practices

1. **Use appropriate test type**: Choose `unit_`, `spec_`, `integ_`, or `e2e_` based on what you're testing
2. **One concept per test**: Keep tests focused on a single behavior
3. **Clear test names**: Use descriptive names that explain what's being tested
4. **Test edge cases**: Include boundary conditions and error cases
5. **Provide context on failure**: Print expected vs actual values
6. **Always clean up resources**: Even on failure paths
7. **Return 0 on success, non-zero on failure**: Required for test runner
