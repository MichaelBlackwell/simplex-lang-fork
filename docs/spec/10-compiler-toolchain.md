# Simplex Compiler Toolchain

**Version 0.2.0**

This document describes the complete Simplex compiler toolchain, which is written entirely in pure Simplex.

---

## Current Implementation Status

The Simplex compiler is currently in **bootstrap phase**, with the following status:

| Component | Status | Backend |
|-----------|--------|---------|
| Compiler (sxc) | Self-hosted | LLVM IR |
| Merge Tool | Complete | LLVM IR |
| Package Manager (spx) | Planned | - |
| Bytecode Runtime (cursus) | Planned | - |

### Current Architecture

The current implementation compiles to **LLVM IR** rather than bytecode:

```
Source (.sx) → Lexer → Parser → Codegen → LLVM IR (.ll) → Native Binary
```

The bytecode format (SXB) described below is the **target architecture** for future versions.

---

## Overview

The Simplex toolchain consists of three main components:

| Component | Binary | Description |
|-----------|--------|-------------|
| **sxc** | `sxc` | Simplex Compiler - compiles `.sx` source to `.sxb` bytecode |
| **spx** | `spx` | Simplex Package Manager - project and dependency management |
| **cursus** | `cursus` | Simplex Runtime - executes bytecode with actor system |

All three components are written in **100% pure Simplex** with zero external dependencies.

---

## Self-Hosting Architecture

### Bootstrap Process

Like GCC, Go, and Rust, Simplex uses a multi-stage bootstrap:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BOOTSTRAP PROCESS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Bootstrap (minimal)     Stage 1 (full)         Stage 2 (verification)      │
│  ┌──────────────┐       ┌──────────────┐       ┌──────────────┐             │
│  │     sxc      │ ────► │ stage1.sxb   │ ────► │ stage2.sxb   │             │
│  │  (minimal)   │       │ (full syntax)│       │ (identical)  │             │
│  └──────────────┘       └──────────────┘       └──────────────┘             │
│         │                      │                      │                      │
│         │ compiles             │ compiles             │                      │
│         ▼                      ▼                      ▼                      │
│  stage1/src/            src/sxc.sx              src/sxc.sx                  │
│  compiler.sx                                                                │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Bootstrap: Minimal compiler written in bootstrap-compatible syntax     │  │
│  │ Stage 1: Full Simplex compiler with complete syntax support            │  │
│  │ Stage 2: Self-hosted compiler (Stage 1 bytecode == Stage 2 bytecode)   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage 1 Compiler

The Stage 1 compiler is written in bootstrap-compatible Simplex (using only syntax
supported by the bootstrap compiler) but compiles to support the full Simplex language:

| Feature | Bootstrap | Stage 1 |
|---------|-----------|---------|
| `while` loops | No (use recursion) | Yes |
| Variable reassignment (`x = val`) | No | Yes |
| `trait` definitions | No | Yes |
| Complex pattern matching | No | Yes |
| `super::` module paths | No | Yes |
| Full module system | Partial | Yes |

The Stage 1 compiler source is located at `stage1/src/compiler.sx` and includes:
- Complete lexer with all Simplex tokens
- Full expression parser with precedence
- Bytecode code generator
- Support for all Simplex syntax features

### Zero Rust Dependencies After Bootstrap

Once bootstrapped:
- The Stage 0 Rust binary is **no longer needed**
- The compiler runs entirely on the Simplex runtime
- All compilation uses pure Simplex code
- Cross-compilation to Windows/Linux requires only the target runtime

### File Structure

```
bootstrap/
├── src/                          # Pure Simplex source (100%)
│   ├── sxc.sx                   # Compiler CLI
│   ├── spx.sx                   # Package manager CLI
│   ├── cursus.sx                # Runtime CLI
│   ├── lib.sx                   # Library exports
│   │
│   ├── lex/                     # Lexer
│   │   ├── mod.sx
│   │   ├── lexer.sx
│   │   ├── token.sx
│   │   └── span.sx
│   │
│   ├── parse/                   # Parser
│   │   ├── mod.sx
│   │   ├── parser.sx
│   │   ├── parser_expr.sx
│   │   ├── parser_helpers.sx
│   │   ├── ast.sx
│   │   └── prec.sx
│   │
│   ├── types/                   # Type System
│   │   ├── mod.sx
│   │   ├── types.sx
│   │   ├── checker.sx
│   │   ├── inference.sx
│   │   └── environment.sx
│   │
│   ├── codegen/                 # Code Generation
│   │   ├── mod.sx
│   │   ├── ir.sx
│   │   ├── lower.sx
│   │   ├── optimize.sx
│   │   └── emit.sx
│   │
│   ├── runtime/                 # Runtime System
│   │   ├── mod.sx
│   │   ├── vm.sx
│   │   ├── value.sx
│   │   ├── gc.sx
│   │   ├── actor.sx
│   │   └── channel.sx
│   │
│   └── std/                     # Standard Library
│       ├── mod.sx
│       ├── io.sx
│       ├── fs.sx
│       ├── collections.sx
│       ├── string.sx
│       ├── time.sx
│       ├── math.sx
│       ├── net.sx
│       ├── env.sx
│       ├── process.sx
│       └── fmt.sx
│
├── bootstrap/                   # Bootstrap binaries
│   └── bin/
│       └── sxc                  # Bootstrap compiler (minimal syntax support)
│
├── stage1/                      # Stage 1 compiler source
│   └── src/
│       ├── compiler.sx          # Self-contained stage1 compiler
│       ├── lexer.sx             # Lexer module
│       ├── ast.sx               # AST definitions
│       └── parser.sx            # Parser module
│
└── stage2/                      # Stage 2 output
    └── simplex.sxb              # Compiled by Stage 1 (verification)
```

---

## sxc - Simplex Compiler

### Usage

```bash
sxc <command> [options] [file]

Commands:
  build <file.sx>     Compile source to bytecode
  run <file.sx>       Compile and execute
  check <file.sx>     Type-check without compiling
  repl                Interactive REPL
  version             Show version
  help                Show help

Options:
  -o <output>         Output file path
  -O0                 No optimization
  -O1                 Basic optimization
  -O2                 Standard optimization (default)
  -O3                 Aggressive optimization
  -Os                 Optimize for size
  --emit-ir           Emit IR for debugging
```

### Examples

```bash
# Compile a file
sxc build hello.sx -o hello.sxb

# Compile and run
sxc run hello.sx

# Type-check only
sxc check mylib.sx

# Interactive REPL
sxc repl
```

### Compilation Pipeline

```
Source (.sx)
     │
     ▼
┌─────────┐
│  Lexer  │  Tokenization
└────┬────┘
     │
     ▼
┌─────────┐
│ Parser  │  AST construction (Pratt parsing)
└────┬────┘
     │
     ▼
┌─────────┐
│  Types  │  Hindley-Milner type inference
└────┬────┘
     │
     ▼
┌─────────┐
│  Lower  │  Convert to SSA IR
└────┬────┘
     │
     ▼
┌─────────┐
│Optimize │  DCE, constant folding, inlining
└────┬────┘
     │
     ▼
┌─────────┐
│  Emit   │  Bytecode generation
└────┬────┘
     │
     ▼
Bytecode (.sxb)
```

---

## spx - Simplex Package Manager

### Usage

```bash
spx <command> [options]

Commands:
  new <name>          Create new project
  init                Initialize in current directory
  build               Build project
  run                 Build and run
  test                Run tests
  check               Type-check project
  add <package>       Add dependency
  remove <package>    Remove dependency
  update              Update dependencies
  publish             Publish to registry
  search <query>      Search packages
  install             Install dependencies
  clean               Clean build artifacts
  fmt                 Format source files
  doc                 Generate documentation
```

### Project Manifest (Modulus.toml)

```toml
[package]
name = "myproject"
version = "0.1.0"
authors = ["Your Name <you@example.com>"]
description = "A Simplex project"
license = "MIT"
repository = "https://github.com/user/myproject"

[dependencies]
simplex-json = "1.0"
simplex-http = "0.5"

[dev-dependencies]
simplex-test = "1.0"

[build]
entry = "src/main.sx"
target = "bytecode"

[features]
default = ["std"]
std = []
async = []
```

---

## cursus - Simplex Runtime

### Usage

```bash
cursus <command> [options]

Commands:
  run <file.sxb>      Execute bytecode
  daemon              Run as background service
  cluster <config>    Start cluster node
  repl                Interactive runtime REPL

Options:
  --gc-threshold <n>  GC threshold in bytes (default: 1MB)
  --heap-size <n>     Max heap size
  --actors <n>        Max concurrent actors
  --port <port>       Cluster port
  --bind <addr>       Bind address
  --join <addr>       Join existing cluster
```

### Runtime Features

- **Bytecode Interpreter**: Executes SXB format
- **Garbage Collector**: Mark-and-sweep with configurable threshold
- **Actor System**: Supervision trees, mailboxes, fault tolerance
- **Checkpointing**: Save and restore actor state
- **Clustering**: Distributed execution across nodes

---

## Bytecode Format (SXB)

### Header

```
Offset  Size  Description
──────  ────  ───────────
0x00    3     Magic: "SXB" (0x53 0x58 0x42)
0x03    1     Null byte (0x00)
0x04    4     Version (little-endian u32)
0x08    4     String table count
0x0C    ...   String table entries
...     4     Global count
...     ...   Globals
...     4     Function count
...     ...   Functions
...     ...   Code section
```

### Opcode Categories

| Range | Category |
|-------|----------|
| 0x00-0x0F | Stack operations |
| 0x10-0x1F | Constants |
| 0x20-0x2F | Integer arithmetic |
| 0x28-0x2F | Float arithmetic |
| 0x30-0x3F | Bitwise operations |
| 0x40-0x4F | Comparisons |
| 0x48-0x4F | Logical operations |
| 0x50-0x5F | Control flow |
| 0x60-0x6F | Variables |
| 0x80-0x8F | Structs/Arrays |
| 0x90-0x9F | Actor operations |
| 0xA0-0xAF | Async operations |
| 0xF0-0xFF | Debug/Halt |

---

## Cross-Platform Support

### Building for Different Platforms

The pure Simplex toolchain can generate bytecode that runs on any platform with the cursus runtime:

```bash
# Build bytecode (platform-independent)
sxc build myapp.sx -o myapp.sxb

# Run on any platform with cursus
cursus run myapp.sxb
```

### Platform-Specific Runtimes

To run on different platforms, only the cursus runtime needs to be built for that platform:

| Platform | Runtime Binary |
|----------|---------------|
| macOS (ARM64) | `cursus-darwin-arm64` |
| macOS (x86_64) | `cursus-darwin-amd64` |
| Linux (x86_64) | `cursus-linux-amd64` |
| Linux (ARM64) | `cursus-linux-arm64` |
| Windows (x86_64) | `cursus-windows-amd64.exe` |

The bytecode is platform-independent; only the runtime differs.

---

## Line Counts

The pure Simplex implementation:

| Component | Files | Lines |
|-----------|-------|-------|
| Compiler (sxc) | 1 | ~350 |
| Package Manager (spx) | 1 | ~650 |
| Runtime (cursus) | 1 | ~600 |
| Lexer | 4 | ~500 |
| Parser | 6 | ~2,500 |
| Type System | 5 | ~2,500 |
| Code Generation | 5 | ~2,200 |
| Runtime System | 6 | ~1,800 |
| Standard Library | 11 | ~5,500 |
| **Total** | **40** | **~16,600** |

All written in pure Simplex with zero external dependencies.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2024-12-29 | Initial pure Simplex implementation |

---

*The Simplex toolchain is self-hosting and requires no external compilers or runtimes after initial bootstrap.*
