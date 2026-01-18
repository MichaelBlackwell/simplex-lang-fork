# Simplex Profiling Guide

This guide explains how to profile Simplex programs for CPU and memory analysis.

## Table of Contents

1. [CPU Profiling](#cpu-profiling)
   - [macOS (Instruments)](#macos-instruments)
   - [Linux (perf)](#linux-perf)
2. [Memory Profiling](#memory-profiling)
3. [Benchmarking](#benchmarking)

---

## CPU Profiling

Simplex supports CPU profiling through integration with system profilers. The key is building with frame pointers preserved.

### Building for Profiling

Use the `--profile` flag when building:

```bash
sxc build myprogram.sx --profile -o myprogram
```

This adds the following compiler flags:
- `-fno-omit-frame-pointer`: Preserves frame pointers for stack unwinding
- `-g`: Includes debug symbols for function names

### macOS (Instruments)

#### Using Time Profiler

1. Build with profiling support:
   ```bash
   sxc build myprogram.sx --profile -o myprogram
   ```

2. Run with Instruments Time Profiler:
   ```bash
   instruments -t 'Time Profiler' ./myprogram
   ```

3. Or use the Instruments GUI:
   - Open Instruments (Cmd+Space, type "Instruments")
   - Choose "Time Profiler"
   - Click the target dropdown and choose "Choose Target..."
   - Select your compiled program
   - Click Record

#### Using `sample` Command

For quick profiling without the GUI:

```bash
# Run program in background
./myprogram &
PID=$!

# Sample for 5 seconds
sample $PID 5 -f profile.txt

# View results
cat profile.txt
```

#### Using DTrace

For more detailed analysis:

```bash
# Profile user stacks for 10 seconds
sudo dtrace -n 'profile-997 /pid == $target/ { @[ustack()] = count(); }' \
    -c ./myprogram

# CPU usage by function
sudo dtrace -n 'profile-997 /execname == "myprogram"/ { @[ufunc(arg1)] = count(); }'
```

### Linux (perf)

#### Basic CPU Profiling

1. Build with profiling support:
   ```bash
   sxc build myprogram.sx --profile -o myprogram
   ```

2. Record performance data:
   ```bash
   perf record -g ./myprogram
   ```

3. View the report:
   ```bash
   perf report
   ```

#### Advanced perf Usage

```bash
# Record with call graph (DWARF)
perf record -g --call-graph dwarf ./myprogram

# Record specific events
perf record -e cycles,cache-misses -g ./myprogram

# Generate flame graph (requires FlameGraph tools)
perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg

# Top functions by CPU time
perf top -p $(pgrep myprogram)
```

#### perf stat for Overall Statistics

```bash
perf stat ./myprogram
```

This shows:
- CPU cycles
- Instructions executed
- Cache references/misses
- Branch predictions

---

## Memory Profiling

Simplex includes built-in memory allocation tracking for debugging and optimization.

### Enabling Memory Tracking

Use the `--memory-debug` flag when building:

```bash
sxc build myprogram.sx --memory-debug -o myprogram
```

Or combine with profiling:

```bash
sxc build myprogram.sx --profile --memory-debug -o myprogram
```

### Using Memory Tracking in Code

Add a call to `memory_report()` in your Simplex code:

```simplex
fn main() -> i64 {
    // Your program logic here
    let v = vec_new();
    for i in 0..1000 {
        vec_push(v, i);
    }

    // Print memory report before exiting
    memory_report();

    0
}
```

### Memory Report Output

The `memory_report()` function prints:

```
=============================================
          MEMORY ALLOCATION REPORT
=============================================

Summary:
  Total allocations:    1523
  Total frees:          1200
  Outstanding allocs:   323

  Current memory:       45678 bytes (44.61 KB)
  Peak memory:          67890 bytes (66.30 KB)
  Total allocated:      234567 bytes (229.07 KB)

Allocation Buckets (active):
  Small  (<=64B):       200 allocs, 8000 bytes
  Medium (<=1KB):       100 allocs, 25000 bytes
  Large  (<=64KB):      20 allocs, 10000 bytes
  Huge   (>64KB):       3 allocs, 2678 bytes

Top Allocation Sites (by current bytes):
   1. vec_push                        150 allocs     12000 bytes
   2. string_concat                    80 allocs      8000 bytes
   3. hashmap_insert                   50 allocs      6000 bytes

=============================================
```

### Programmatic Memory Access

You can also query memory stats programmatically:

```simplex
fn analyze_memory() -> i64 {
    let current = memory_current();    // Current allocated bytes
    let peak = memory_peak();          // Peak allocated bytes
    let count = memory_alloc_count();  // Total allocation count

    print("Current: ");
    print_i64(current);
    print(" bytes\n");

    0
}
```

### Using Valgrind (Linux)

For deeper memory analysis on Linux:

```bash
# Build without --memory-debug for Valgrind (they conflict)
sxc build myprogram.sx -o myprogram

# Memory leak detection
valgrind --leak-check=full ./myprogram

# Memory usage profiling
valgrind --tool=massif ./myprogram
ms_print massif.out.*
```

### Using AddressSanitizer

For detecting memory errors:

```bash
# Manual clang compilation with ASAN
sxc compile myprogram.sx
clang -fsanitize=address -g myprogram.ll runtime/standalone_runtime.c -o myprogram
./myprogram
```

---

## Benchmarking

Simplex includes a built-in benchmarking framework similar to Rust's `#[bench]` or Go's `testing.B`.

### Writing Benchmarks

Create a file with benchmark functions:

```simplex
// benchmarks.sx
use bench;

#[bench]
fn bench_vec_push(b: i64) -> i64 {
    bench_run(b, || {
        let v = vec_new();
        for i in 0..1000 {
            vec_push(v, i);
        }
        0
    });
    0
}

#[bench]
fn bench_string_concat(b: i64) -> i64 {
    bench_run(b, || {
        let s = string_from("");
        for i in 0..100 {
            s = string_concat(s, string_from("x"));
        }
        0
    });
    0
}

fn main() -> i64 {
    run_benchmarks();
    0
}
```

### Running Benchmarks

```bash
# Run benchmarks
sxc bench benchmarks.sx

# Output JSON format
sxc bench benchmarks.sx --json

# Save results for comparison
sxc bench benchmarks.sx --save baseline.json

# Compare with baseline
sxc bench benchmarks.sx --baseline baseline.json
```

### Benchmark Output

```
Running benchmarks...

bench_vec_push          1,234 ns/iter (+/- 45)
bench_string_concat     5,678 ns/iter (+/- 123)
bench_hashmap_get         456 ns/iter (+/- 12)
```

### JSON Output

```json
{
  "benchmarks": [
    {
      "name": "bench_vec_push",
      "iterations": 810045,
      "mean_ns": 1234,
      "stddev_ns": 45,
      "min_ns": 1180,
      "max_ns": 1450
    }
  ]
}
```

### Benchmark API

The benchmarking library provides:

```simplex
// Create a new bencher
let b = bencher_new(string_from("my_bench"));

// Run with automatic iteration count
bench_run(b, || { /* work */ 0 });

// Run specific number of iterations
bench_n(b, 10000, || { /* work */ 0 });

// Set throughput measurement
bencher_set_bytes(b, 1024);  // bytes per operation

// Get statistics
let mean = bench_mean_ns(b);
let stddev = bench_stddev_ns(b);
let min = bench_min_ns(b);
let max = bench_max_ns(b);
let throughput = bench_throughput_bps(b);

// Format results
let human = bench_format_result(b);
let json = bench_format_json(b);

// Prevent optimization
let result = black_box(compute_something());
```

### Best Practices

1. **Warmup**: The framework automatically runs warmup iterations
2. **Black box**: Use `black_box()` to prevent compiler optimizations
3. **Consistent state**: Reset state between iterations
4. **Minimize noise**: Close other applications during benchmarking
5. **Multiple runs**: Run benchmarks multiple times for consistency

---

## Tips and Tricks

### Combining CPU and Memory Profiling

```bash
# Build with both
sxc build myprogram.sx --profile --memory-debug -o myprogram

# Run and get memory report in output
./myprogram

# Then profile separately
perf record -g ./myprogram
perf report
```

### Profiling Specific Code Sections

Use timing functions to measure specific sections:

```simplex
fn profile_section() -> i64 {
    let start = get_time_ns();

    // Code to profile
    expensive_operation();

    let end = get_time_ns();
    let elapsed = end - start;

    print("Elapsed: ");
    print_i64(elapsed / 1000000);
    print(" ms\n");

    0
}
```

### Comparing Performance

```bash
# Baseline
sxc bench mybench.sx --save baseline.json

# Make changes...

# Compare
sxc bench mybench.sx --baseline baseline.json
```

Output:
```
=== Comparison with Baseline ===

bench_vec_push                -15% (faster)
bench_string_concat           +5% (slower)
bench_hashmap_get             no change
```
