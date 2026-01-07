# Simplex Programming Language

**Version 0.5.0**

Simplex is a modern systems programming language designed for AI-native applications, featuring first-class support for actors, cognitive agents, and distributed computing.

## Features

- **Actor Model**: Built-in actors with message passing for concurrent programming
- **Cognitive Agents**: First-class AI specialist agents with inference capabilities
- **Per-Hive SLM**: Each cognitive hive provisions its own shared language model
- **HiveMnemonic**: Shared consciousness across specialists within a hive
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
sxc build hello.sx

# Run directly
./hello

# Or compile and run in one step
sxc run hello.sx
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

### Cognitive Hive with Shared SLM (v0.5.0)

```simplex
// Create a hive with shared consciousness
let mnemonic = HiveMnemonic::new(100, 500, 50);  // 50% belief threshold
mnemonic.learn("Team coding standards require Result types", 0.95);

// Create shared SLM for the hive
let hive_slm = HiveSLM::new("CodeReviewHive", "simplex-cognitive-7b", mnemonic);

// Create specialists that share the SLM
let security = Specialist::new("SecurityAnalyzer", hive_slm, security_anima);
let quality = Specialist::new("QualityReviewer", hive_slm, quality_anima);
let perf = Specialist::new("PerformanceOptimizer", hive_slm, perf_anima);

// All three specialists share ONE model instance
// Findings shared via HiveMnemonic are visible to all
security.contribute_to_mnemonic("SQL injection found in process_user_input");
// quality and perf can now see this finding in their context
```

### AI Specialists with Anima

```simplex
// Create personal memory for a specialist
let anima = Anima::new(10);  // 30% belief threshold
anima.learn("I specialize in code security", 0.9, "self");
anima.desire("Find security vulnerabilities", 0.95);

specialist SecurityAnalyzer {
    model: "simplex-cognitive-7b";
    anima: anima;

    fn analyze(code: String) -> String {
        // Context from anima + hive mnemonic prepended automatically
        infer("Analyze this code for security issues: " + code)
    }
}
```

## Project Structure

```
simplex-lang/
├── compiler/
│   └── bootstrap/          # Self-hosted compiler source
│       ├── stage0.py       # Python bootstrap compiler
│       ├── lexer.sx        # Lexical analysis
│       ├── parser.sx       # Parsing
│       ├── codegen.sx      # Code generation
│       ├── main.sx         # Compiler entry point
│       ├── stdlib.sx       # Standard library
│       └── utils.sx        # Utility functions
├── runtime/
│   └── standalone_runtime.c  # C runtime library
├── tools/
│   ├── sxc.sx              # CLI compiler wrapper
│   ├── sxpm.sx             # Package manager
│   ├── cursus.sx           # Bytecode VM
│   ├── sxdoc.sx            # Documentation generator
│   └── sxlsp.sx            # Language server protocol
├── tests/
│   ├── language/           # Language feature tests
│   ├── stdlib/             # Standard library tests
│   ├── runtime/            # Runtime tests
│   ├── ai/                 # AI/Cognitive tests
│   ├── toolchain/          # Toolchain tests
│   └── integration/        # End-to-end scenario tests
├── simplex-docs/
│   ├── spec/               # Language specification
│   ├── tutorial/           # Learning tutorial
│   ├── testing/            # Testing documentation
│   └── guides/             # How-to guides
└── tasks/                  # Development roadmap
```

## Compiler Toolchain

| Tool | Version | Description |
|------|---------|-------------|
| **sxc** | 0.5.0 | Simplex Compiler |
| **sxpm** | 0.5.0 | Package Manager with SLM provisioning |
| **cursus** | 0.5.0 | Bytecode Virtual Machine |
| **sxdoc** | 0.5.0 | Documentation Generator |
| **sxlsp** | 0.5.0 | Language Server Protocol |

## Release History

### v0.5.0 (2026-01-07) - SLM Provisioning & Cognitive Hives

**Per-Hive SLM Architecture:**
- Each cognitive hive provisions ONE shared SLM
- All specialists within a hive share the same model instance
- Memory-efficient: 10 specialists = 1 model, not 10
- Three-tier belief thresholds: 30% (Anima) → 50% (Hive) → 70% (Divine)

**HiveMnemonic (Shared Consciousness):**
- Collective memory shared across all specialists in a hive
- Episodic, semantic, and belief memory types
- Automatic context injection into inference prompts
- Cross-specialist knowledge sharing

**Built-in Models:**
- `simplex-cognitive-7b` (4.1 GB) - Full cognitive capabilities
- `simplex-cognitive-1b` (700 MB) - Lightweight alternative
- `simplex-mnemonic-embed` (134 MB) - Embedding model

**sxpm Model Commands:**
```bash
sxpm model list              # List available models
sxpm model install <name>    # Install a model
sxpm model remove <name>     # Remove a model
sxpm model info <name>       # Show model details
```

**New Standard Library Features:**
- HTTP client with TLS support
- Terminal colors, progress bars, spinners, tables
- Enhanced JSON parsing

**Testing Framework:**
- Comprehensive test coverage documentation
- End-to-end scenario tests
- AI/Cognitive component tests

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

### sxc Commands

```bash
sxc build <file.sx> [-o output]   # Compile to native executable
sxc compile <file.sx>             # Compile to LLVM IR (.ll)
sxc run <file.sx>                 # Compile and run immediately
sxc check <file.sx>               # Syntax check only
sxc version                       # Show version info
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
sxpm model list           # List available models
sxpm model install <name> # Install a model
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
- Anima cognitive agents
- HiveMnemonic shared consciousness
- Per-hive SLM provisioning

### In Development
- Full trait bounds and where clauses
- Module system improvements
- GPU acceleration for SLMs
- Distributed hive clustering

## Running Tests

```bash
# Run all tests
sxpm test

# Run specific category
sxpm test --category ai
sxpm test --category integration

# Run a specific test
sxc run tests/language/basics/test_loops.sx
```

## Documentation

- [Language Specification](simplex-docs/spec/)
- [Tutorial](simplex-docs/tutorial/)
- [Testing Documentation](simplex-docs/testing/)
- [Getting Started Guide](simplex-docs/guides/getting-started.md)
- [Release Notes](simplex-docs/RELEASE-0.5.0.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[MIT License](LICENSE)

## Links

- **Repository**: https://github.com/senuamedia/simplex-lang
- **Issues**: https://github.com/senuamedia/simplex-lang/issues
- **Documentation**: https://simplex-lang.org (coming soon)
