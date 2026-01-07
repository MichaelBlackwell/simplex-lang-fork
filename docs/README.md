# Simplex Language Documentation

**Version 0.4.1**

Simplex (Latin for "simple") is a programming language designed for the AI era. It combines the fault-tolerance of Erlang, the memory safety of Rust, the distributed computing model of Ray, and the content-addressable code of Unison into a cohesive system built for intelligent, distributed workloads.

---

## Documentation Index

### Specification

| Document | Description |
|----------|-------------|
| [Overview and Philosophy](spec/01-overview.md) | Vision, goals, and core design philosophy |
| [Architecture](spec/02-architecture.md) | System architecture and component overview |
| [Design Decisions](spec/03-design-decisions.md) | Technical decisions with rationale |
| [Language Syntax](spec/04-language-syntax.md) | Complete syntax reference |
| [Virtual Machine](spec/05-virtual-machine.md) | SVM internals, bytecode, and instruction set |
| [Swarm Computing](spec/06-swarm-computing.md) | Distributed execution model |
| [AI Integration](spec/07-ai-integration.md) | AI primitives and patterns |
| [Cost Optimization](spec/08-cost-optimization.md) | Enterprise deployment on cheap compute |
| [Cognitive Hive AI](spec/09-cognitive-hive.md) | CHAI architecture for SLM orchestration |
| [Compiler Toolchain](spec/10-compiler-toolchain.md) | sxc, spx, cursus - 100% pure Simplex |
| [Standard Library](spec/11-standard-library.md) | Complete std API reference |
| [The Anima](spec/12-anima.md) | Cognitive soul - memory, beliefs, intentions |

### Tutorial

A step-by-step learning path for the Simplex language.

| Chapter | Topic | Description |
|---------|-------|-------------|
| [01](tutorial/01-variables-and-types.md) | Variables and Types | Primitives, type inference, mutability |
| [02](tutorial/02-functions.md) | Functions | Parameters, returns, closures, generics |
| [03](tutorial/03-control-flow.md) | Control Flow | Conditionals, loops, pattern matching |
| [04](tutorial/04-collections.md) | Collections | Lists, maps, sets, iteration |
| [05](tutorial/05-custom-types.md) | Custom Types | Structs, enums, traits, methods |
| [06](tutorial/06-error-handling.md) | Error Handling | Option, Result, the ? operator |
| [07](tutorial/07-actors.md) | Actors | State, messages, spawning |
| [08](tutorial/08-messages.md) | Message Passing | Send, ask, patterns |
| [09](tutorial/09-supervision.md) | Supervision | Fault tolerance, restart strategies |
| [10](tutorial/10-ai-basics.md) | AI Integration | Completion, classification, embeddings |
| [11](tutorial/11-capstone.md) | Capstone Project | Build a complete application |
| [12](tutorial/12-cognitive-hives.md) | Cognitive Hives | Building SLM swarms with CHAI |

Start the tutorial: [Tutorial Index](tutorial/README.md)

### Examples

| Document | Description |
|----------|-------------|
| [Document Pipeline](examples/document-pipeline.md) | Complete example: distributed document processing |

### Guides

| Document | Description |
|----------|-------------|
| [Getting Started](guides/getting-started.md) | Quick start guide |

---

## Quick Links

- **New to Simplex?** Start with the [Tutorial](tutorial/README.md) or [Overview](spec/01-overview.md)
- **Learning the language?** Follow the [12-chapter tutorial](tutorial/README.md)
- **Want syntax reference?** Jump to [Language Syntax](spec/04-language-syntax.md)
- **Building AI agents?** See [The Anima](spec/12-anima.md) and [Cognitive Hive AI](spec/09-cognitive-hive.md)
- **See complete code?** Check [Examples](examples/document-pipeline.md)
- **Planning deployment?** See [Cost Optimization](spec/08-cost-optimization.md)
- **Building the VM?** Read [Virtual Machine](spec/05-virtual-machine.md)
- **Self-hosting compiler?** See [Compiler Toolchain](spec/10-compiler-toolchain.md)
- **Standard library API?** Check [Standard Library](spec/11-standard-library.md)

---

## Design Principles

1. **AI-native**: AI operations are first-class language constructs
2. **Anima-centric**: Every AI agent has a cognitive soul (anima) with memory, beliefs, and intentions
3. **Hive-oriented**: Small language models collaborate as cognitive hives (CHAI)
4. **Distributed-first**: Programs naturally decompose across VM swarms
5. **Fault-tolerant**: Workers can die and resume transparently
6. **Cost-efficient**: Runs on the cheapest cloud compute (spot instances, ARM)
7. **Simple syntax**: Lightweight, readable code

---

## Key Features (v0.4.1)

### Language Core
- Static typing with inference
- Pattern matching with guards
- Generics and traits
- Closures and lambdas
- Async/await
- Option and Result types
- Module system with `use`, `mod`, `pub`

### Actor System
- Message passing actors
- Supervision trees
- Fault tolerance ("let it crash")
- Checkpointing and recovery

### AI/Cognitive (The Anima)
- `anima` keyword for cognitive agents
- Episodic, semantic, procedural, and working memory
- Belief system with revision
- BDI (Beliefs-Desires-Intentions) architecture
- Goal-directed memory recall
- Memory persistence and sharing

### AI Specialists & Hives
- Multi-provider support (Anthropic, OpenAI, Ollama)
- Streaming responses
- Tool/function calling
- Multi-actor orchestration (pipeline, parallel, consensus)
- Semantic routing

### Observability
- Metrics (counter, gauge, histogram)
- Distributed tracing
- Structured logging
- Prometheus export

### Toolchain
- `sxc` - Native compiler (LLVM backend)
- `sxpm` - Package manager
- `sxdoc` - Documentation generator
- `sxlsp` - Language server
- `cursus` - Build system

---

## Influences

Simplex draws from:

- **Erlang/OTP**: Actor model, supervision trees, "let it crash"
- **Rust**: Ownership-based memory management, zero-cost abstractions
- **Unison**: Content-addressed code, seamless distribution
- **Ray**: Distributed computing primitives, AI/ML optimization
- **Go**: Simple syntax, fast compilation
- **CHAI**: Cognitive Hive AI architecture for SLM orchestration

---

## Module Terminology

Simplex uses the following terminology for its module system:

| Term | Definition |
|------|------------|
| **modulus** | A single compilation unit (like Rust's crate) |
| **moduli** | Plural of modulus - multiple compilation units |
| **module** | A namespace within a modulus (like Rust's mod) |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2024-12 | Initial draft specification |
| 0.2.0 | 2024-12 | Added Cognitive Hive AI (CHAI) specification |
| 0.3.0 | 2025-01 | Self-hosted native compiler via LLVM |
| 0.3.1 | 2025-01 | Complete toolchain: sxc, cursus, sxdoc, sxlsp all compiled |
| 0.3.5 | 2025-01 | Added sxpm package manager, Phase 1 stdlib (HashMap, HashSet, String ops) |
| 0.4.0 | 2026-01 | The Anima: cognitive memory, BDI, tool calling, observability, multi-actor orchestration |
| 0.4.1 | 2026-01 | sxpm enhancements: inline modules, pub use re-exports, diamond dependency detection, build cache, lock files |

---

*Simplex is a work in progress. This specification is subject to change.*
