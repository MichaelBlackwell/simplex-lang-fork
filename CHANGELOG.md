# Changelog

All notable changes to Simplex will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.10.0] - 2026-01-18

### Highlights

**Developer Experience Release** - This release focuses on tooling and developer productivity, bringing a complete suite of development tools written in Simplex itself.

### Added

#### Developer Tools
- **sxfmt** - Code formatter with consistent style enforcement
  - 4-space indentation, opening brace on same line
  - `--check` mode for CI integration
  - `--diff` mode for previewing changes

- **sxlint** - Static analysis linter
  - Unused variable detection
  - Unreachable code detection
  - Style checks and naming conventions
  - Performance hints

- **sxlsp** - Language Server Protocol implementation
  - Go to definition
  - Hover documentation
  - Diagnostics integration
  - Works with VS Code, Neovim, and other LSP clients

- **sxdoc** - Documentation generator
  - Extracts doc comments from source
  - Generates markdown documentation
  - Index generation for API reference

- **sxpm** - Package manager
  - `sxpm init` - Initialize new projects
  - `sxpm add/remove` - Manage dependencies
  - `sxpm build/run` - Build and execute projects
  - DAG-based dependency resolution
  - simplex.toml configuration

- **cursus** - Bytecode virtual machine
  - Portable bytecode format (.sxb)
  - Instruction tracing for debugging
  - Runtime statistics

#### Compiler Improvements
- **sxc** tool rewritten in Simplex
  - Multi-file compilation support
  - Project configuration via simplex.toml
  - Cross-platform build support (macOS, Linux, Windows)
  - Architecture auto-detection (x86_64, arm64)

#### Standard Library
- **lib/platform.sx** - Platform detection utilities
- **lib/version.sx** - Version information
- **lib/llm.sx** - LLM integration primitives (preview)

#### IDE Support
- **tree-sitter-simplex** - Tree-sitter grammar for syntax highlighting
  - Highlight queries for editors
  - Fold and indent queries
  - Text objects for structural editing

#### Interactive Tools
- **playground/** - Web-based Simplex playground
  - Live compilation and execution
  - Syntax highlighting
  - Share code snippets

### Changed

- Bootstrap compiler now outputs LLVM IR only (linking requires clang)
- Standalone runtime bundled in releases (standalone_runtime.c)
- GitHub Actions CI updated to bundle runtime in release artifacts

### Fixed

- GitHub Issue #69: Standalone runtime now included in release artifacts
- sxfmt: Fixed tokenizer issues with `init` and `infer` keywords
- sxpm: Fixed infinite recursion in version lookup
- sxdoc: Fixed orphaned code from refactoring
- sxc: Fixed hardcoded x86_64 architecture detection

### Known Limitations

- Actor message dispatch codegen incomplete (spec_actor_basic fails)
- Async/await runtime not fully functional (spec_async_basic returns wrong exit code)
- tree-sitter grammar requires manual installation of tree-sitter-cli

---

## [0.9.0] - 2025-12-XX

### Added
- Self-learning annealing optimization
- Adaptive learning rate schedules

## [0.8.0] - 2025-11-XX

### Added
- Dual numbers for automatic differentiation
- Forward-mode AD support

## [0.7.0] - 2025-10-XX

### Added
- Real-time learning during inference
- Online training capabilities

## [0.6.0] - 2025-09-XX

### Added
- Neural IR for differentiable programs
- Learnable control flow
