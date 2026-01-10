# BUG-001: infer() outside specialist context passes raw string instead of SxString*

**Status:** Open
**Severity:** High
**Component:** stage0.py compiler
**Discovered:** 2026-01-10
**Reporter:** Codex deployment debugging
**Fix release:** v0.7.1

## Summary

When `infer(prompt)` is called outside of a specialist context, the stage0.py compiler passes a raw C string constant `"default"` directly to `intrinsic_ai_infer()`, but the runtime function expects an `SxString*` struct. This causes a crash when the runtime attempts to access `model->data`.

## Steps to Reproduce

1. Write a Simplex program that uses `infer()` outside of a specialist:

```simplex
fn main() -> i64 {
    let response = infer("What is 2 + 2?");
    println(response);
    0
}
```

2. Compile with stage0.py:
```bash
python3 stage0.py test.sx
```

3. Link and run:
```bash
clang -O2 test.ll standalone_runtime.c -o test -lm -lssl -lcrypto -lsqlite3
./test
```

4. Program crashes with segmentation fault

## Root Cause Analysis

### Compiler Code (stage0.py lines 9364-9370)

```python
# Not in specialist context - use default model
model_label = self.add_string_constant("default")
result_ptr = self.new_temp()
self.emit(f'  {result_ptr} = call ptr @intrinsic_ai_infer(ptr {model_label}, ptr {prompt_ptr}, i64 70)')
```

This generates LLVM IR like:
```llvm
@.str.test.5 = private unnamed_addr constant [8 x i8] c"default\00"
...
%t104 = call ptr @intrinsic_ai_infer(ptr @.str.test.5, ptr %t103, i64 70)
```

The first argument `@.str.test.5` is a raw `[8 x i8]*` pointer to the C string "default".

### Runtime Code (standalone_runtime.c)

```c
SxString* intrinsic_ai_infer(SxString* model, SxString* prompt, int64_t temperature) {
    // ...
    const char* model_name = model && model->data ? model->data : "default";
    // ...
}
```

The function expects `model` to be an `SxString*` struct:
```c
typedef struct {
    char* data;    // offset 0
    int64_t len;   // offset 8
    int64_t cap;   // offset 16
} SxString;
```

When a raw string pointer like `"default"` is passed:
- `model` points to the ASCII bytes `d e f a u l t \0`
- `model->data` reads the first 8 bytes as a pointer: `0x746c756166656400` (garbage address)
- Dereferencing this garbage pointer causes a segmentation fault

### Contrast with Specialist Context

Inside a specialist context (lines 9336-9362), the model comes from `self.__model` which is properly allocated as an SxString* during specialist initialization. This path works correctly.

## Expected Behavior

`infer()` should work identically whether called inside or outside a specialist context. The compiler should wrap the "default" string constant in an SxString before passing it to `intrinsic_ai_infer`.

## Proposed Fix

Modify stage0.py lines 9364-9370 to use `intrinsic_string_new`:

```python
# Not in specialist context - use default model
model_const = self.add_string_constant("default")
model_ptr = self.new_temp()
self.emit(f'  {model_ptr} = call ptr @intrinsic_string_new(ptr {model_const})')
result_ptr = self.new_temp()
self.emit(f'  {result_ptr} = call ptr @intrinsic_ai_infer(ptr {model_ptr}, ptr {prompt_ptr}, i64 70)')
```

This would generate:
```llvm
@.str.test.5 = private unnamed_addr constant [8 x i8] c"default\00"
...
%model = call ptr @intrinsic_string_new(ptr @.str.test.5)
%t104 = call ptr @intrinsic_ai_infer(ptr %model, ptr %t103, i64 70)
```

The compiler already uses `intrinsic_string_new` in other places for the same purpose (see lines 5977, 6408, 6423, 6843).

## Impact

- Any Simplex program using `infer()` outside a specialist will crash
- This affects all users trying to use AI inference in regular functions
- Workaround: Only use `infer()` inside specialist actors

## Related Files

- `stage0.py` lines 9325-9370 (InferExpr handling)
- `standalone_runtime.c` (intrinsic_ai_infer implementation)
- Discovered while deploying Codex project which uses `infer()` in a regular function

## Test Coverage Gap

**No existing tests cover `infer()` outside a specialist context.**

All current tests in `tests/ai/` only use `infer()` inside specialist `receive` handlers:

| Test File | Usage |
|-----------|-------|
| `tests/ai/test_native_simple.sx` | `infer()` inside `specialist TestSpec` |
| `tests/ai/test_native_ai.sx` | `infer()` inside `specialist CodeAnalyzer`, `specialist Summarizer` |
| `tests/ai/test_native_hive.sx` | `infer()` inside `specialist Analyzer`, `specialist Writer` |
| `tests/ai/anima/test_native_*.sx` | All inside specialists |

This bug was never caught because the untested code path was never exercised.

## Required Test Addition

Add new test file `tests/ai/test_infer_standalone.sx`:

```simplex
// Test for infer() outside specialist context
// This tests the "default model" code path in stage0.py

fn test_infer_in_function() -> i64 {
    println("Testing infer() in regular function...");

    // This should work - infer() outside specialist uses "default" model
    let response = infer("What is 2 + 2?");

    if response != 0 {
        print("Response: ");
        print_string(response);
        println("");
        println("PASS: infer() works outside specialist");
        return 1;
    } else {
        println("FAIL: infer() returned null");
        return 0;
    }
}

fn test_infer_with_concat() -> i64 {
    println("Testing infer() with string concatenation...");

    let prompt = string_concat("Explain ", "briefly");
    let response = infer(prompt);

    if response != 0 {
        println("PASS: infer() works with dynamic prompt");
        return 1;
    } else {
        println("FAIL: infer() with concat returned null");
        return 0;
    }
}

fn main() -> i64 {
    println("=== Testing infer() Outside Specialist Context ===");
    println("");

    let passed = 0;
    let total = 2;

    passed = passed + test_infer_in_function();
    println("");
    passed = passed + test_infer_with_concat();

    println("");
    print("Results: ");
    print_i64(passed);
    print("/");
    print_i64(total);
    println(" tests passed");

    if passed == total {
        println("=== ALL TESTS PASSED ===");
        0
    } else {
        println("=== SOME TESTS FAILED ===");
        1
    }
}
```

This test should be added alongside the compiler fix to prevent regression.

## Notes

The same issue may exist in other places where string constants are passed to functions expecting SxString*. A comprehensive audit of `add_string_constant` usage with runtime functions may be warranted.
