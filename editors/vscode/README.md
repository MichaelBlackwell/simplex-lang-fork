# Simplex Language Support for VS Code

Syntax highlighting and language support for the [Simplex programming language](https://github.com/senuamedia/simplex-lang).

## Features

- Syntax highlighting for all Simplex constructs
- Code snippets for common patterns
- Bracket matching and auto-closing
- Comment toggling

### Language Constructs

- **Core**: functions, structs, enums, traits, impl blocks
- **Actors**: actor definitions, receive blocks, spawn/send/ask
- **Async**: async/await functions
- **AI/Cognitive**: specialists, hives, infer keyword
- **Control flow**: if/else, match, for, while, loops

## Installation

### From VSIX (Local)

```bash
cd editors/vscode
npm install
npm run package
code --install-extension simplex-lang-0.3.4.vsix
```

### Manual Installation

Copy the `editors/vscode` folder to your VS Code extensions directory:

- **macOS**: `~/.vscode/extensions/simplex-lang`
- **Linux**: `~/.vscode/extensions/simplex-lang`
- **Windows**: `%USERPROFILE%\.vscode\extensions\simplex-lang`

## Snippets

| Prefix | Description |
|--------|-------------|
| `fn` | Function definition |
| `main` | Main entry point |
| `async` | Async function |
| `struct` | Struct definition |
| `enum` | Enum definition |
| `trait` | Trait definition |
| `impl` | Impl block |
| `implt` | Impl trait for type |
| `actor` | Actor definition |
| `actori` | Actor with init block |
| `specialist` | AI specialist |
| `hive` | Cognitive hive |
| `match` | Match expression |
| `matcho` | Match on Option |
| `matchr` | Match on Result |
| `iflet` | If let pattern |
| `for` | For loop |
| `forr` | For range loop |
| `while` | While loop |
| `let` | Variable binding |
| `letm` | Mutable binding |
| `spawn` | Spawn actor |
| `send` | Send message |
| `ask` | Ask actor |
| `test` | Test function |
| `///` | Doc comment block |

## License

MIT
