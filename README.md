# Simplex Programming Language

Simplex is a modern systems programming language designed for AI-native applications, featuring first-class support for actors, cognitive agents, and distributed computing.

## Features

- **Actor Model**: Built-in actors with message passing for concurrent programming
- **Cognitive Agents**: First-class AI specialist agents with inference capabilities
- **Self-Hosted**: Compiler written in Simplex itself (bootstrapped from Python)
- **LLVM Backend**: Compiles to optimized native code via LLVM IR
- **Rust-Inspired Syntax**: Familiar syntax with enums, traits, and pattern matching
- **Async/Await**: Native async support with state machine generation
- **Option/Result Types**: Safe error handling with `Option<T>` and `Result<T, E>`

## Installation

### Prerequisites

| Platform | Requirements |
|----------|-------------|
| macOS    | Xcode Command Line Tools, LLVM/Clang |
| Linux    | GCC or Clang, OpenSSL, SQLite3 |
| Windows  | MSVC or MinGW, OpenSSL, SQLite3 (coming soon) |

### Quick Install (macOS)

```bash
# Install dependencies
brew install llvm openssl sqlite3

# Clone the repository
git clone https://github.com/senuamedia/simplex-lang.git
cd simplex-lang

# Build the compiler
./build.sh
```

### Quick Install (Linux)

```bash
# Ubuntu/Debian
sudo apt-get install clang libssl-dev libsqlite3-dev

# Clone and build
git clone https://github.com/senuamedia/simplex-lang.git
cd simplex-lang
./build.sh
```

### Building from Source

The repository includes a pre-built `sxc` compiler. To rebuild from scratch:

```bash
# Run the build script (requires Python 3 and clang)
./build.sh

# Or manually:
# 1. Compile with the bootstrap compiler
python3 stage0.py hello.sx

# 2. Link with clang
clang -O2 hello.ll standalone_runtime.c -o hello -lm -lssl -lcrypto -lsqlite3
```

## Getting Started

### Hello World

Create a file `hello.sx`:

```simplex
fn main() {
    println("Hello, Simplex!");
}
```

Note: `main()` without a return type implicitly returns exit code 0. You can also use `fn main() -> i64` for explicit exit codes.

Compile and run:

```bash
# Build to executable
./sxc build hello.sx

# Run directly
./hello

# Or compile and run in one step
./sxc run hello.sx
```

### Variables and Types

```simplex
fn main() {
    let x: i64 = 42;
    let name: String = "Simplex";
    let pi: f64 = 3.14159;
    let active: bool = true;

    println(f"Hello {name}, x = {x}");
}
```

### Enums and Pattern Matching

```simplex
enum Option<T> {
    None,
    Some(T),
}

fn main() {
    let value: Option<i64> = Some(42);

    match value {
        Some(x) => println(f"Got value: {x}"),
        None => println("No value"),
    }
}
```

### Structs and Traits

```simplex
struct Point {
    x: i64,
    y: i64,
}

trait Printable {
    fn print(&self);
}

impl Printable for Point {
    fn print(&self) {
        println(f"Point({self.x}, {self.y})");
    }
}

fn main() {
    let p = Point { x: 10, y: 20 };
    p.print();
}
```

### Actors

```simplex
actor Counter {
    count: i64 = 0;

    receive {
        Increment => {
            self.count = self.count + 1;
        }
        GetCount => {
            self.count
        }
    }
}

fn main() -> i64 {
    let counter = spawn Counter {};
    send counter Increment;
    send counter Increment;
    ask counter GetCount
}
```

### Async/Await

```simplex
async fn fetch_data(url: String) -> String {
    let response = http_get(url).await;
    response.body
}

async fn main() {
    let data = fetch_data("https://api.example.com").await;
    println(data);
}
```

### AI Specialists

```simplex
specialist Summarizer {
    model: "gpt-4";
    temperature: 0.7;

    fn summarize(text: String) -> String {
        infer("Summarize the following text: " + text)
    }
}

fn main() {
    let summary = Summarizer::summarize("Long article text...");
    println(summary);
}
```

## Project Structure

```
simplex-lang/
├── compiler/
│   └── bootstrap/          # Self-hosted compiler source
│       ├── stage0.py       # Python bootstrap compiler
│       ├── lexer.sx        # Lexical analysis
│       ├── parser.sx       # Parsing (73KB)
│       ├── codegen.sx      # Code generation (166KB)
│       ├── main.sx         # Compiler entry point
│       ├── stdlib.sx       # Standard library
│       └── utils.sx        # Utility functions
├── runtime/
│   └── standalone_runtime.c  # C runtime library (471KB)
├── tools/
│   ├── sxc.sx              # CLI compiler wrapper
│   ├── sxpm.sx             # Package manager
│   ├── cursus.sx           # Bytecode VM
│   ├── sxdoc.sx            # Documentation generator
│   └── sxlsp.sx            # Language server protocol
├── tests/
│   ├── basics/             # Core language tests
│   ├── types/              # Type system tests
│   ├── async/              # Async/await tests
│   ├── actors/             # Actor model tests
│   ├── phase36/            # Phase 36 feature tests
│   ├── stdlib/             # Standard library tests
│   ├── language/           # Language feature tests
│   └── integration/        # Integration tests
├── examples/               # Example programs
└── docs/
    ├── spec/               # Language specification
    ├── tutorial/           # Learning tutorial
    └── guides/             # How-to guides
```

## Compiler Toolchain

- **sxc** (Compiler): v0.4.1
- **sxpm** (Package Manager): v0.1.6
- **cursus** (Bytecode VM): v0.1.6
- **sxdoc** (Documentation Generator): v0.1.6
- **sxlsp** (Language Server): v0.1.6

### v0.4.1 (2026-01-07)

**Module System Enhancements:**
- Inline module blocks: `mod name { ... }` syntax
- Re-exports: `pub use other::Thing` for API design
- Glob re-exports: `pub use other::*` for convenience
- Prelude auto-imports: common types/functions available without imports

**Build System:**
- Build cache with mtime tracking and hash-based invalidation
- Incremental compilation support via `needs_recompile()`
- Cache persistence in `.simplex/cache/meta.json`

**Dependency Resolution:**
- Diamond dependency detection with version conflict reporting
- Full semver constraint support: `^`, `~`, `>=`, `<=`, `>`, `<`, `=`
- Enhanced lock file support with `sxpm update`

### v0.4.0 (2026-01-07)

**Phase 2: Package Ecosystem Core**
- Module system with `use`, `mod`, visibility modifiers (`pub`)
- Relative paths: `use super::sibling`, `use self::sub`
- Dependency graph with topological sort and cycle detection
- Lock file generation for reproducible builds
- Standard package structure: `src/lib.sx` vs `src/main.sx` detection

**sxpm Commands:**
- `sxpm check` - Type check without building
- `sxpm update` - Regenerate lock file
- `sxpm clean` - Remove build artifacts
- Version range support in dependencies

**v0.3.5 Changes:**
- Added `sxpm` package manager with full dependency resolution
- Phase 1 stdlib: HashMap, HashSet, Vec extensions, String operations
- Phase 2: Package ecosystem with JSON manifest support
- Added json_parse_simple, json_object_new, json_array_new, json_keys
- Fixed codegen handlers for stdlib functions

**v0.3.4 Changes:**
- Added Rust-style closure syntax (`|| expr` and `|x| expr`) to parser
- Added `block_on(future)` runtime function for async execution
- Fixed test files for proper exit code handling
- Added explicit type annotations for method call resolution
- All 13 phase36 tests now pass

**v0.3.3 Changes:**
- Fixed github test suite failure log

**v0.3.2 Changes:**
- Added Linux epoll support for async I/O (was macOS kqueue only)
- Fixed OpenSSL/SQLite include paths in CI workflow
- Platform-compatible standalone runtime

**v0.3.1 Changes:**
- Fixed critical `lookup_variant()` bug in codegen.sx
- Compiler now uses fully self-hosted native binary (`sxc-compile`)
- All toolchain binaries compiled: cursus, sxdoc, sxlsp

### sxc Commands

```bash
sxc build <file.sx> [-o output]   # Compile to native executable
sxc compile <file.sx>             # Compile to LLVM IR (.ll)
sxc run <file.sx>                 # Compile and run immediately
sxc version                        # Show version info
sxc help                          # Show usage help
```

### sxpm Commands

```bash
sxpm new <name>           # Create a new package
sxpm init                 # Initialize package in current directory
sxpm build                # Build the current package
sxpm run                  # Build and run the current package
sxpm test                 # Run tests
sxpm add <package>        # Add a dependency
sxpm remove <package>     # Remove a dependency
sxpm install              # Install dependencies
```

## Language Features

### Implemented
- Functions, closures, and generics
- Structs, enums, and traits
- Pattern matching with guards
- Option and Result types
- Async/await with state machines
- Actor model with supervision
- f-strings for formatting
- Reference types (&T, *T)
- Turbofish syntax (::<T>)

### In Development
- Full trait bounds and where clauses
- Module system improvements
- Enhanced package management features (cursus)
- Full LSP protocol support (sxlsp)

## Running Tests

```bash
# Run a specific test
./sxc run tests/basics/for_loop.sx

# Build a test to executable
./sxc build tests/basics/for_loop.sx -o test_for
./test_for

# Run all tests
./run_tests.sh
```

## Documentation

- [Language Specification](docs/spec/)
- [Tutorial](docs/tutorial/)
- [Getting Started Guide](docs/guides/getting-started.md)
- [API Reference](docs/api/)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[MIT License](LICENSE)

## Links

- **Repository**: https://github.com/senuamedia/simplex-lang
- **Issues**: https://github.com/senuamedia/simplex-lang/issues
- **Documentation**: https://simplex-lang.org (coming soon / maybe)
