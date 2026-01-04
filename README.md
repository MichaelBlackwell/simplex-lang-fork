# Simplex Programming Language

Simplex is a modern systems programming language designed for AI-native applications, featuring first-class support for actors, cognitive hives, and autonomous agent orchestration.

## Features

- **Actor Model**: Built-in actors with message passing for concurrent programming
- **Cognitive Hives**: First-class AI specialist agents with inference capabilities
- **Self-Hosted**: Compiler written in Simplex itself
- **LLVM Backend**: Compiles to optimized native code via LLVM IR
- **Familiar Syntax**: Rust-inspired syntax for easy adoption

## Quick Start

### Prerequisites

- LLVM/Clang (for compiling LLVM IR)
- A C compiler (for the runtime)

### Building the Compiler

```bash
# Build the runtime
cd runtime
clang -c -O2 standalone_runtime.c -o standalone_runtime.o

# Compile the Simplex compiler components
cd ../compiler/bootstrap
# Each .sx file compiles to .ll via the bootstrap process
```

### Hello World

```simplex
fn main() -> i64 {
    println("Hello, Simplex!");
    0
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

### AI Specialists

```simplex
specialist Summarizer {
    model: "gpt-4";
    temperature: 0.7;

    fn summarize(text: String) -> String {
        infer("Summarize the following text: " + text)
    }
}
```

## Project Structure

```
simplex-lang/
├── compiler/
│   └── bootstrap/      # Self-hosted compiler
│       ├── lexer.sx    # Lexical analysis
│       ├── parser.sx   # Parsing
│       ├── codegen.sx  # Code generation
│       ├── main.sx     # Compiler entry point
│       └── merge.sx    # LLVM IR merge tool
├── runtime/            # Runtime library
├── docs/
│   ├── spec/          # Language specification
│   ├── tutorial/      # Learning tutorial
│   └── guides/        # How-to guides
├── examples/          # Example programs
└── tests/             # Test suite
```

## Documentation

- [Language Specification](docs/spec/)
- [Tutorial](docs/tutorial/)
- [Getting Started Guide](docs/guides/getting-started.md)

## Current Status

Simplex is in **bootstrap phase**. The compiler is self-hosted and compiles to LLVM IR. See [Compiler Toolchain](docs/spec/10-compiler-toolchain.md) for details.

## License

[License details to be added]

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Report issues: https://github.com/senuamedia/simplex-lang/issues
