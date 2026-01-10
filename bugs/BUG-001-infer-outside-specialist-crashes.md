# BUG-001: infer() outside specialist context passes raw string instead of SxString*

**Status:** Fixed
**Severity:** High
**Component:** stage0.py compiler
**Discovered:** 2026-01-10
**Reporter:** Codex deployment debugging
**Fixed in:** v0.7.1

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

### Compiler Code (stage0.py line 10371-10377, before fix)

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

Inside a specialist context, the model comes from `self.__model` which is properly allocated as an SxString* during specialist initialization. This path works correctly.

## Fix Applied

Modified `stage0.py` line 10371-10380 to wrap the string constant using `intrinsic_string_new`:

```python
# Not in specialist context - use default model
# BUG-001 fix: wrap raw string constant in SxString via intrinsic_string_new
model_const = self.add_string_constant("default")
model_ptr = self.new_temp()
self.emit(f'  {model_ptr} = call ptr @intrinsic_string_new(ptr {model_const})')
result_ptr = self.new_temp()
self.emit(f'  {result_ptr} = call ptr @intrinsic_ai_infer(ptr {model_ptr}, ptr {prompt_ptr}, i64 70)')
```

This now generates correct LLVM IR:
```llvm
@.str.test.1 = private unnamed_addr constant [8 x i8] c"default\00"
...
%t3 = call ptr @intrinsic_string_new(ptr @.str.test.1)
%t4 = call ptr @intrinsic_ai_infer(ptr %t3, ptr %t2, i64 70)
```

## Test Coverage Gap

**No existing tests covered `infer()` outside a specialist context.**

All current tests in `tests/ai/` only use `infer()` inside specialist `receive` handlers:

| Test File | Usage |
|-----------|-------|
| `tests/ai/test_native_simple.sx` | `infer()` inside `specialist TestSpec` |
| `tests/ai/test_native_ai.sx` | `infer()` inside `specialist CodeAnalyzer`, `specialist Summarizer` |
| `tests/ai/test_native_hive.sx` | `infer()` inside `specialist Analyzer`, `specialist Writer` |
| `tests/ai/anima/test_native_*.sx` | All inside specialists |

This bug was never caught because the untested code path was never exercised.

## Test Added

New test file `tests/ai/test_infer_standalone.sx` added to prevent regression:

```simplex
// Test for infer() outside specialist context
// This tests the "default model" code path in stage0.py
// Added as part of BUG-001 fix

fn test_infer_in_function() -> i64 {
    println("Testing infer() in regular function...");
    let response = infer("What is 2 + 2?");
    if response != 0 {
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
    let passed = test_infer_in_function() + test_infer_with_concat();
    if passed == 2 {
        println("=== ALL TESTS PASSED ===");
        0
    } else {
        println("=== SOME TESTS FAILED ===");
        1
    }
}
```

## Files Changed

| File | Change |
|------|--------|
| `stage0.py` | Fixed line 10371-10380: wrap "default" string in SxString before calling intrinsic_ai_infer |
| `tests/ai/test_infer_standalone.sx` | New test file for infer() outside specialist context |

## Verification

After fix, compiled test program generates correct IR:
```
%t3 = call ptr @intrinsic_string_new(ptr @.str.test_bug001.1)
%t4 = call ptr @intrinsic_ai_infer(ptr %t3, ptr %t2, i64 70)
```

The model parameter `%t3` is now a proper `SxString*` created by `intrinsic_string_new`.

## Related Files

- `stage0.py` lines 10331-10380 (InferExpr handling)
- `standalone_runtime.c` (intrinsic_ai_infer implementation)
- Discovered while deploying Codex project which uses `infer()` in a regular function

## Notes

The same issue may exist in other places where string constants are passed to functions expecting SxString*. A comprehensive audit of `add_string_constant` usage with runtime functions may be warranted.
