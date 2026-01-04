# Contributing to Simplex

Thank you for your interest in contributing to Simplex!

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a feature branch
4. Make your changes
5. Submit a pull request

## Development Setup

### Prerequisites

- LLVM/Clang toolchain
- A C compiler (gcc or clang)

### Building

```bash
# Build the runtime
cd runtime
clang -c -O2 standalone_runtime.c -o standalone_runtime.o

# The compiler is self-hosted - see docs for bootstrap process
```

## Code Style

- Use 4-space indentation
- Keep lines under 100 characters
- Follow existing naming conventions
- Add comments for non-obvious code

## Testing

Before submitting a PR:

1. Ensure all existing tests pass
2. Add tests for new features
3. Test on your local machine

## Pull Request Guidelines

- One feature/fix per PR
- Clear description of changes
- Reference any related issues
- Keep commits focused and atomic

## Reporting Issues

- Check existing issues first
- Include reproduction steps
- Provide system information
- Include error messages if applicable

## License

By contributing, you agree that your contributions will be licensed under the project's license.
