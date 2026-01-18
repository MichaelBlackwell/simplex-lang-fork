# tree-sitter-simplex

Tree-sitter grammar for the [Simplex](https://github.com/senuamedia/simplex-lang) programming language.

Simplex is a systems programming language featuring:
- Rust-like syntax with `fn`, `let`, `struct`, `enum`, `impl`, `trait`
- Actor model concurrency with `actor`, `receive`, `spawn`, `send`
- AI-native features with `specialist`, `hive`, `infer`, `anima`
- Async/await support
- Pattern matching with `match` expressions

## Installation

### Prerequisites

- Node.js 16+ and npm
- Tree-sitter CLI (`npm install -g tree-sitter-cli`)
- A C compiler (gcc, clang, or MSVC)

### Building from Source

```bash
# Clone and navigate to the grammar directory
cd tree-sitter-simplex

# Install dependencies
npm install

# Generate the parser
npm run generate

# Build native bindings
npm run build

# Run tests
npm test
```

## Editor Integration

### Neovim

1. Install the grammar using nvim-treesitter:

```lua
-- In your init.lua or plugin config
local parser_config = require("nvim-treesitter.parsers").get_parser_configs()

parser_config.simplex = {
  install_info = {
    url = "https://github.com/senuamedia/simplex-lang",
    files = { "tree-sitter-simplex/src/parser.c" },
    branch = "main",
    subdirectory = "tree-sitter-simplex",
  },
  filetype = "simplex",
}
```

2. Add filetype detection:

```lua
-- In ~/.config/nvim/ftdetect/simplex.lua
vim.filetype.add({
  extension = {
    sx = "simplex",
  },
})
```

3. Copy the query files:

```bash
mkdir -p ~/.config/nvim/queries/simplex
cp queries/*.scm ~/.config/nvim/queries/simplex/
```

4. Install and configure:

```vim
:TSInstall simplex
```

### Helix

1. Add the grammar to `~/.config/helix/languages.toml`:

```toml
[[language]]
name = "simplex"
scope = "source.simplex"
injection-regex = "^simplex$"
file-types = ["sx"]
roots = ["Simplex.toml"]
comment-token = "//"
indent = { tab-width = 4, unit = "    " }
auto-format = true

[language.auto-pairs]
'(' = ')'
'{' = '}'
'[' = ']'
'"' = '"'
"'" = "'"

[[grammar]]
name = "simplex"
source = { git = "https://github.com/senuamedia/simplex-lang", rev = "main", subpath = "tree-sitter-simplex" }
```

2. Fetch and build grammars:

```bash
helix --grammar fetch
helix --grammar build
```

3. Copy query files to `~/.config/helix/runtime/queries/simplex/`.

### Zed

1. Create an extension or add to your Zed extensions directory.

2. Add language configuration to `languages.toml`:

```toml
[simplex]
name = "Simplex"
path_suffixes = ["sx"]
line_comments = ["// "]
block_comment = ["/*", "*/"]
```

### VS Code

For VS Code, you can use the `vscode-tree-sitter` extension or create a dedicated Simplex extension that bundles this grammar.

## Supported Language Features

### Core Syntax

- **Functions**: `fn name(params) -> Type { body }`
- **Variables**: `let x: Type = value;` and `var x = mutable_value;`
- **Structs**: `struct Name { field: Type }`
- **Enums**: `enum Name { Variant(Type), Unit }`
- **Traits**: `trait Name { fn method(&self); }`
- **Impl blocks**: `impl Trait for Type { ... }`
- **Type aliases**: `type Alias = ExistingType;`

### Control Flow

- **If expressions**: `if condition { } else { }`
- **Match expressions**: `match value { Pattern => result }`
- **For loops**: `for item in iterator { }`
- **While loops**: `while condition { }`
- **Loop expressions**: `loop { break value; }`

### Pattern Matching

- Literal patterns: `42`, `"string"`, `true`
- Tuple patterns: `(a, b, c)`
- Struct patterns: `Point { x, y }`
- Enum patterns: `Some(value)`, `None`
- Or patterns: `A | B | C`
- Wildcard: `_`
- Range patterns: `1..10`, `'a'..='z'`

### Types

- Primitives: `i8`-`i128`, `u8`-`u128`, `f32`, `f64`, `bool`, `char`
- References: `&T`, `&mut T`
- Pointers: `*const T`, `*mut T`
- Arrays: `[T; N]`
- Slices: `[T]`
- Tuples: `(A, B, C)`
- Functions: `fn(A, B) -> C`
- Generics: `Vec<T>`, `HashMap<K, V>`

### Concurrency (Actor Model)

```simplex
actor Counter {
    var count: i64 = 0;

    receive Increment() {
        self.count = self.count + 1;
    }

    receive Get() -> i64 {
        self.count
    }
}

fn main() {
    let counter = spawn Counter {};
    send(counter, Increment());
    let value = ask(counter, Get());
}
```

### Async/Await

```simplex
async fn fetch_data(url: String) -> Result<String, Error> {
    let response = http_get(url).await?;
    Ok(response.body)
}
```

### AI-Native Features

```simplex
specialist Analyzer {
    model: "gpt-4",
    temperature: 0.7,

    fn analyze(text: String) -> String {
        infer("Analyze: " + text)
    }
}

hive Pipeline {
    specialists: [Analyzer, Summarizer],
    strategy: consensus,
}
```

## Query Files

### highlights.scm

Defines syntax highlighting for all language constructs:
- Keywords (control flow, declarations, types)
- Operators and punctuation
- Literals (numbers, strings, booleans)
- Functions, types, and variables
- Comments and documentation
- Simplex-specific features (actors, specialists, hives)

### locals.scm

Defines local scope analysis for:
- Function and block scopes
- Variable definitions and references
- Pattern bindings in match expressions
- Import handling

### indents.scm

Defines automatic indentation rules for:
- Block-level constructs (functions, structs, etc.)
- Multi-line expressions
- Continuation indents
- Special alignment cases

## Development

### Testing

Create test files in `test/corpus/`:

```
================================================================================
Function Definition
================================================================================

fn hello() -> i64 {
    42
}

--------------------------------------------------------------------------------

(source_file
  (function_definition
    name: (identifier)
    (parameters)
    (type (primitive_type))
    (block
      (integer_literal))))
```

Run tests:

```bash
npm test
```

### Debugging

Parse a file and show the syntax tree:

```bash
tree-sitter parse path/to/file.sx
```

Highlight a file:

```bash
tree-sitter highlight path/to/file.sx
```

## Contributing

Contributions are welcome! Please ensure:

1. Grammar changes don't break existing tests
2. New features have corresponding test cases
3. Query files are updated for new syntax

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related Projects

- [Simplex Language](https://github.com/senuamedia/simplex-lang) - The Simplex programming language
- [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) - Parser generator tool
- [nvim-treesitter](https://github.com/nvim-treesitter/nvim-treesitter) - Neovim Tree-sitter integration
