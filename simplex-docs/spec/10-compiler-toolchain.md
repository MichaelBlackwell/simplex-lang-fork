# Simplex Compiler Toolchain

**Version 0.9.0**

This document describes the Simplex compiler toolchain, which is **self-hosted** and compiles to native binaries via LLVM.

---

## Overview

The Simplex toolchain consists of five main components:

| Component | Binary | Version | Description |
|-----------|--------|---------|-------------|
| **sxc** | `sxc` | v0.9.0 | Simplex Compiler - compiles `.sx` source to native executables |
| **sxpm** | `sxpm` | v0.9.0 | Package manager with dependency resolution |
| **cursus** | `cursus` | v0.9.0 | Bytecode VM with garbage collection |
| **sxdoc** | `sxdoc` | v0.9.0 | Documentation generator |
| **sxlsp** | `sxlsp` | v0.9.0 | Language Server Protocol implementation |

All components are written in **Simplex** and compile to native binaries.

---

## Self-Hosting Architecture

### Bootstrap Process

Simplex uses a multi-stage bootstrap similar to GCC, Go, and Rust:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BOOTSTRAP PROCESS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Stage 0 (Python)        Stage 1 (Native)        Stage 2 (Self-Hosted)      │
│  ┌──────────────┐       ┌──────────────┐       ┌──────────────┐             │
│  │   stage0.py  │ ────► │ sxc-compile  │ ────► │ sxc-compile  │             │
│  │  (bootstrap) │       │   (native)   │       │  (verified)  │             │
│  └──────────────┘       └──────────────┘       └──────────────┘             │
│         │                      │                      │                      │
│         │ compiles             │ compiles             │                      │
│         ▼                      ▼                      ▼                      │
│  codegen.sx              codegen.sx              codegen.sx                 │
│  lexer.sx                lexer.sx                (identical output)         │
│  parser.sx               parser.sx                                          │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Stage 0: Python bootstrap compiler (generates LLVM IR)                │  │
│  │ Stage 1: Native compiler built by Stage 0                             │  │
│  │ Stage 2: Native compiler built by Stage 1 (verifies self-hosting)     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Compilation Pipeline

```
Source (.sx)
     │
     ▼
┌─────────┐
│  Lexer  │  Tokenization (lexer.sx)
└────┬────┘
     │
     ▼
┌─────────┐
│ Parser  │  AST construction (parser.sx)
└────┬────┘
     │
     ▼
┌─────────┐
│ Codegen │  LLVM IR generation (codegen.sx)
└────┬────┘
     │
     ▼
┌─────────┐
│  Clang  │  Native code generation + linking
└────┬────┘
     │
     ▼
Native Binary
```

### File Structure

```
simplex-lang/
├── sxc                     # Compiler wrapper script (bash)
├── sxc-compile             # Native self-hosted compiler (387KB)
├── sxpm                    # Package manager
├── cursus                  # Bytecode VM (227KB)
├── sxdoc                   # Documentation generator (225KB)
├── sxlsp                   # Language server (220KB)
├── standalone_runtime.c    # C runtime with intrinsics
├── stage0.py               # Python bootstrap (for rebuilding)
│
├── compiler/bootstrap/     # Compiler source
│   ├── lexer.sx           # Lexer
│   ├── parser.sx          # Parser
│   └── codegen.sx         # Code generator
│
└── tools/                  # Tool source
    ├── cursus.sx          # Bytecode VM source
    ├── sxdoc.sx           # Doc generator source
    └── sxlsp.sx           # LSP source
```

---

## sxc - Simplex Compiler

### Usage

```bash
sxc <command> [options] <file>

Commands:
  build <file.sx> [-o output]    Compile to native executable
  compile <file.sx>              Compile to LLVM IR only
  run <file.sx>                  Compile and run immediately
  version                        Show version
  help                           Show help

Options:
  -o <output>         Output file path
  -O                  Enable optimizations
  -v, --verbose       Verbose output
  -g, --debug         Include debug information
```

### Examples

```bash
# Compile to native executable
sxc build hello.sx -o hello
./hello

# Compile and run immediately
sxc run hello.sx

# Compile to LLVM IR only
sxc compile hello.sx
# Produces: hello.ll
```

### Compilation Process

The `sxc` wrapper script:
1. Invokes `sxc-compile` to generate LLVM IR (`.ll` file)
2. Links with `standalone_runtime.c` using clang
3. Produces a native executable

```bash
# What happens internally:
./sxc-compile hello.sx          # Generates hello.ll
clang -O2 hello.ll standalone_runtime.c -o hello -lm
```

---

## cursus - Bytecode VM

### Usage

```bash
cursus [OPTIONS] <FILE.sxb>
cursus compile <FILE.sx> -o <FILE.sxb>

Options:
  --trace         Enable instruction tracing
  --stats         Show VM statistics
  --version       Show version
  -h, --help      Show help
```

### Features

- Stack-based bytecode interpreter
- Garbage collection
- String table management
- Call frame tracking
- Debug tracing mode

---

## sxdoc - Documentation Generator

### Usage

```bash
sxdoc [OPTIONS] <FILES...>

Options:
  --html          Generate HTML output (default)
  --markdown      Generate Markdown output
  -o <dir>        Output directory (default: ./docs)
  --version       Show version
  -h, --help      Show help
```

### Features

- Extracts `///` doc comments from source
- Generates HTML or Markdown documentation
- Supports functions, structs, enums, traits

---

## sxlsp - Language Server

### Usage

```bash
sxlsp [OPTIONS]

Options:
  --stdio         Use stdio for communication (default)
  --version       Show version
  -h, --help      Show help
```

### Features

- JSON-RPC over stdio
- Diagnostics (syntax errors)
- Hover information
- Go to definition (planned)
- Completion (planned)

---

## Runtime System

### standalone_runtime.c

The C runtime provides:

| Category | Functions |
|----------|-----------|
| **Memory** | `malloc`, `free`, `load_i64`, `store_i64`, `load_ptr`, `store_ptr` |
| **Strings** | `string_from`, `string_concat`, `string_slice`, `string_eq`, `string_len` |
| **Vectors** | `vec_new`, `vec_push`, `vec_get`, `vec_len`, `vec_set` |
| **I/O** | `print`, `println`, `read_file`, `write_file`, `read_line` |
| **Files** | `file_exists`, `file_size`, `mkdir`, `list_dir`, `remove_file` |
| **Process** | `get_args`, `get_env`, `exit_program`, `system_call` |
| **Time** | `time_now`, `time_sleep` |
| **Network** | `http_get`, `http_post`, `tcp_connect`, `tcp_listen` |

### Intrinsic Mapping

Simplex functions are mapped to C intrinsics:

```simplex
// Simplex code:
let s = string_from("hello");
println(s);

// Maps to C:
// intrinsic_string_new("hello")
// intrinsic_println(s)
```

---

## Building from Source

### Prerequisites

| Platform | Requirements |
|----------|-------------|
| **macOS** | Xcode Command Line Tools (includes clang), Python 3 |
| **Linux** | clang or gcc, Python 3 |
| **Windows** | Visual Studio Build Tools (includes clang-cl), Python 3 |

### Platform Support

The Simplex toolchain is **fully cross-platform** as of v0.9.0:

| Component | macOS | Linux | Windows |
|-----------|-------|-------|---------|
| `stage0.py` (bootstrap) | ✅ | ✅ | ✅ |
| `sxc` (compiler) | ✅ | ✅ | ✅ |
| `sxpm` (package manager) | ✅ | ✅ | ✅ |
| `cursus` (VM) | ✅ | ✅ | ✅ |
| `sxdoc` (docs) | ✅ | ✅ | ✅ |
| `sxlsp` (LSP) | ✅ | ✅ | ✅ |

The bootstrap compiler (`stage0.py`) automatically detects the platform and generates the appropriate LLVM target triple:
- **macOS**: `x86_64-apple-macosx<version>` or `aarch64-apple-macosx<version>`
- **Linux**: `x86_64-unknown-linux-gnu` or `aarch64-unknown-linux-gnu`
- **Windows**: `x86_64-pc-windows-msvc`

### Full Bootstrap

```bash
# Clone repository
git clone https://github.com/user/simplex-lang.git
cd simplex-lang

# Bootstrap from Python (only needed once)
# Works on macOS, Linux, and Windows
python3 stage0.py compiler/bootstrap/codegen.sx -o sxc-compile

# Self-host verification
./sxc build compiler/bootstrap/codegen.sx -o sxc-compile-stage2

# Build tools
./sxc build tools/cursus.sx -o cursus
./sxc build tools/sxdoc.sx -o sxdoc
./sxc build tools/sxlsp.sx -o sxlsp
```

### Platform-Specific Notes

#### macOS
```bash
# Ensure Xcode CLI tools are installed
xcode-select --install

# Bootstrap
python3 stage0.py compiler/bootstrap/codegen.sx -o sxc-compile
```

#### Linux
```bash
# Install clang (Debian/Ubuntu)
sudo apt install clang python3

# Install clang (Fedora/RHEL)
sudo dnf install clang python3

# Bootstrap
python3 stage0.py compiler/bootstrap/codegen.sx -o sxc-compile
```

#### Windows
```powershell
# Install Visual Studio Build Tools with C++ workload
# Or install LLVM/Clang directly

# Bootstrap (PowerShell)
python stage0.py compiler/bootstrap/codegen.sx -o sxc-compile.exe
```

### Quick Build (already bootstrapped)

```bash
# Just build tools
./sxc build tools/cursus.sx -o cursus
./sxc build tools/sxdoc.sx -o sxdoc
./sxc build tools/sxlsp.sx -o sxlsp
```

---

## Binary Sizes

| Binary | Size | Description |
|--------|------|-------------|
| `sxc` | 6.5KB | Wrapper script |
| `sxc-compile` | 387KB | Native compiler |
| `cursus` | 227KB | Bytecode VM |
| `sxdoc` | 225KB | Doc generator |
| `sxlsp` | 220KB | Language server |

Total toolchain: ~1MB of native binaries.

---

## Test Suite (v0.9.0)

The test suite has been completely reorganized with 156 tests across 13 categories.

### Directory Structure

```
tests/
├── language/           # Core language features (40 tests)
│   ├── actors/
│   ├── async/
│   ├── basics/
│   ├── closures/
│   ├── control/
│   ├── functions/
│   ├── modules/
│   ├── traits/
│   └── types/
├── types/              # Type system tests (24 tests)
├── neural/             # Neural IR and gates (16 tests)
├── stdlib/             # Standard library (16 tests)
├── ai/                 # AI/Cognitive tests (17 tests)
├── toolchain/          # Compiler toolchain (14 tests)
├── runtime/            # Runtime systems (5 tests)
├── integration/        # End-to-end tests (7 tests)
├── basics/             # Basic language tests (6 tests)
├── async/              # Async/await tests (3 tests)
├── learning/           # Automatic differentiation (3 tests)
├── actors/             # Actor model tests (1 test)
└── observability/      # Metrics and tracing (1 test)
```

### Naming Convention

| Prefix | Type | Description |
|--------|------|-------------|
| `unit_` | Unit | Tests individual functions/types in isolation |
| `spec_` | Specification | Tests language specification compliance |
| `integ_` | Integration | Tests integration between components |
| `e2e_` | End-to-End | Tests complete workflows |

### Running Tests

```bash
# Run all tests
./tests/run_tests.sh

# Run specific category
./tests/run_tests.sh neural
./tests/run_tests.sh learning
./tests/run_tests.sh ai

# Filter by test type
./tests/run_tests.sh all unit    # Only unit tests
./tests/run_tests.sh all spec    # Only spec tests
./tests/run_tests.sh all integ   # Only integration tests
./tests/run_tests.sh all e2e     # Only end-to-end tests

# Combine category and type
./tests/run_tests.sh stdlib unit
./tests/run_tests.sh neural spec
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2024-12 | Initial Python bootstrap |
| 0.2.0 | 2024-12 | Self-hosted compiler (Stage 1) |
| 0.3.0 | 2025-01 | Native binary compilation |
| 0.3.1 | 2025-01 | Fixed lookup_variant bug, all tools compiled |
| 0.8.0 | 2026-01 | Native dual numbers for automatic differentiation |
| 0.9.0 | 2026-01 | Self-learning annealing, test suite restructure, llama.cpp integration |

---

*The Simplex toolchain is self-hosted. After initial bootstrap, the Python compiler is no longer needed.*
