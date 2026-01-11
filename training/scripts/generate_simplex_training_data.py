#!/usr/bin/env python3
"""
SimplexSpecialist Training Data Generator
==========================================
Generates training data from the Simplex codebase for fine-tuning a
code-specialized model to understand and generate Simplex code.

Training data types:
1. Code completion - complete partial Simplex code
2. Instruction to code - natural language to Simplex
3. Code explanation - explain what Simplex code does
4. Code translation - Rust/Python idioms to Simplex
5. Bug fixing - identify and fix issues in Simplex code
6. Spec compliance - generate code following language spec
"""

import os
import re
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class TrainingExample:
    """A single training example."""
    instruction: str
    input: str
    output: str
    category: str


class SimplexCodeExtractor:
    """Extract and parse Simplex code from the codebase."""

    def __init__(self, codebase_path: str):
        self.codebase_path = Path(codebase_path)
        self.sx_files: List[Path] = []
        self.functions: List[Dict] = []
        self.structs: List[Dict] = []
        self.traits: List[Dict] = []
        self.modules: List[Dict] = []

    def scan_codebase(self) -> None:
        """Scan for all .sx files."""
        self.sx_files = list(self.codebase_path.rglob("*.sx"))
        print(f"Found {len(self.sx_files)} Simplex files")

    def extract_functions(self, content: str, filepath: str) -> List[Dict]:
        """Extract function definitions from Simplex code."""
        functions = []
        # Match fn definitions with various patterns
        fn_pattern = r'(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)\s*(?:->\s*([^{]+))?\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}'

        for match in re.finditer(fn_pattern, content, re.MULTILINE | re.DOTALL):
            name, params, return_type, body = match.groups()
            # Get preceding comment/docstring
            start = match.start()
            preceding = content[:start].split('\n')[-5:]
            docstring = '\n'.join([l for l in preceding if l.strip().startswith('//')])

            functions.append({
                'name': name,
                'params': params.strip() if params else '',
                'return_type': return_type.strip() if return_type else 'void',
                'body': body.strip(),
                'docstring': docstring,
                'filepath': filepath,
                'full_text': match.group(0)
            })
        return functions

    def extract_structs(self, content: str, filepath: str) -> List[Dict]:
        """Extract struct definitions."""
        structs = []
        struct_pattern = r'(?:pub\s+)?struct\s+(\w+)\s*(?:<[^>]*>)?\s*\{([^}]*)\}'

        for match in re.finditer(struct_pattern, content, re.MULTILINE | re.DOTALL):
            name, fields = match.groups()
            structs.append({
                'name': name,
                'fields': fields.strip(),
                'filepath': filepath,
                'full_text': match.group(0)
            })
        return structs

    def extract_traits(self, content: str, filepath: str) -> List[Dict]:
        """Extract trait definitions."""
        traits = []
        trait_pattern = r'(?:pub\s+)?trait\s+(\w+)\s*(?:<[^>]*>)?\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}'

        for match in re.finditer(trait_pattern, content, re.MULTILINE | re.DOTALL):
            name, body = match.groups()
            traits.append({
                'name': name,
                'body': body.strip(),
                'filepath': filepath,
                'full_text': match.group(0)
            })
        return traits

    def extract_all(self) -> None:
        """Extract all code elements from the codebase."""
        self.scan_codebase()

        for sx_file in self.sx_files:
            try:
                content = sx_file.read_text(encoding='utf-8')
                rel_path = str(sx_file.relative_to(self.codebase_path))

                self.functions.extend(self.extract_functions(content, rel_path))
                self.structs.extend(self.extract_structs(content, rel_path))
                self.traits.extend(self.extract_traits(content, rel_path))

            except Exception as e:
                print(f"Error processing {sx_file}: {e}")

        print(f"Extracted: {len(self.functions)} functions, {len(self.structs)} structs, {len(self.traits)} traits")


class SimplexTrainingDataGenerator:
    """Generate training examples from extracted Simplex code."""

    SIMPLEX_KEYWORDS = [
        'fn', 'let', 'mut', 'const', 'struct', 'enum', 'trait', 'impl',
        'pub', 'use', 'mod', 'async', 'await', 'spawn', 'actor', 'hive',
        'belief', 'confidence', 'if', 'else', 'match', 'for', 'while',
        'loop', 'return', 'break', 'continue', 'self', 'Self', 'super',
        'where', 'type', 'as', 'in', 'ref', 'move', 'dyn', 'box'
    ]

    SIMPLEX_TYPES = [
        'i8', 'i16', 'i32', 'i64', 'i128', 'isize',
        'u8', 'u16', 'u32', 'u64', 'u128', 'usize',
        'f32', 'f64', 'bool', 'char', 'str', 'String',
        'Vec', 'HashMap', 'Option', 'Result', 'Box', 'Rc', 'Arc',
        'Future', 'Stream', 'Actor', 'Hive', 'Belief', 'Channel'
    ]

    def __init__(self, extractor: SimplexCodeExtractor, spec_path: Optional[str] = None):
        self.extractor = extractor
        self.spec_path = Path(spec_path) if spec_path else None
        self.spec_content = ""
        self.examples: List[TrainingExample] = []

        if self.spec_path and self.spec_path.exists():
            self._load_spec()

    def _load_spec(self) -> None:
        """Load language specification documents."""
        spec_files = list(self.spec_path.glob("*.md"))
        for sf in spec_files:
            try:
                self.spec_content += sf.read_text(encoding='utf-8') + "\n\n"
            except:
                pass
        print(f"Loaded {len(spec_files)} spec documents")

    def generate_code_completion_examples(self, count: int = 2000) -> List[TrainingExample]:
        """Generate code completion training examples."""
        examples = []
        functions = self.extractor.functions.copy()
        random.shuffle(functions)

        for fn in functions[:count]:
            if len(fn['body']) < 20:
                continue

            # Split function at various points
            full_code = fn['full_text']
            lines = full_code.split('\n')

            if len(lines) < 3:
                continue

            # Complete from signature
            split_point = random.randint(1, max(1, len(lines) - 2))
            partial = '\n'.join(lines[:split_point])
            completion = '\n'.join(lines[split_point:])

            examples.append(TrainingExample(
                instruction="Complete the following Simplex code:",
                input=partial,
                output=completion,
                category="code_completion"
            ))

            # Complete function body
            if fn['params'] or fn['return_type'] != 'void':
                signature = f"fn {fn['name']}({fn['params']})"
                if fn['return_type'] != 'void':
                    signature += f" -> {fn['return_type']}"

                examples.append(TrainingExample(
                    instruction=f"Implement the body for this Simplex function:",
                    input=signature + " {",
                    output=fn['body'] + "\n}",
                    category="code_completion"
                ))

        return examples

    def generate_instruction_to_code_examples(self, count: int = 1500) -> List[TrainingExample]:
        """Generate instruction-to-code training examples."""
        examples = []

        # Function generation
        for fn in self.extractor.functions[:count]:
            if fn['docstring']:
                # Use docstring as instruction
                instruction = fn['docstring'].replace('//', '').strip()
                examples.append(TrainingExample(
                    instruction=f"Write a Simplex function that: {instruction}",
                    input="",
                    output=fn['full_text'],
                    category="instruction_to_code"
                ))
            else:
                # Generate instruction from function name
                name_words = re.sub(r'([A-Z])', r' \1', fn['name']).replace('_', ' ').lower().strip()
                examples.append(TrainingExample(
                    instruction=f"Write a Simplex function called '{fn['name']}' that {name_words}",
                    input=f"Parameters: {fn['params']}\nReturn type: {fn['return_type']}",
                    output=fn['full_text'],
                    category="instruction_to_code"
                ))

        # Struct generation
        for struct in self.extractor.structs[:count // 3]:
            name_words = re.sub(r'([A-Z])', r' \1', struct['name']).lower().strip()
            examples.append(TrainingExample(
                instruction=f"Define a Simplex struct for {name_words}",
                input="",
                output=struct['full_text'],
                category="instruction_to_code"
            ))

        return examples

    def generate_code_explanation_examples(self, count: int = 1000) -> List[TrainingExample]:
        """Generate code explanation training examples."""
        examples = []

        for fn in self.extractor.functions[:count]:
            # Generate explanation based on code structure
            explanation = self._generate_explanation(fn)

            examples.append(TrainingExample(
                instruction="Explain what this Simplex code does:",
                input=fn['full_text'],
                output=explanation,
                category="code_explanation"
            ))

        return examples

    def _generate_explanation(self, fn: Dict) -> str:
        """Generate an explanation for a function."""
        parts = []

        # Function purpose from name
        name_words = re.sub(r'([A-Z])', r' \1', fn['name']).replace('_', ' ').lower().strip()
        parts.append(f"This function '{fn['name']}' {name_words}.")

        # Parameters
        if fn['params']:
            params = [p.strip() for p in fn['params'].split(',') if p.strip()]
            parts.append(f"It takes {len(params)} parameter(s): {fn['params']}.")

        # Return type
        if fn['return_type'] and fn['return_type'] != 'void':
            parts.append(f"It returns a value of type {fn['return_type']}.")

        # Async
        if 'async' in fn['full_text'][:50]:
            parts.append("This is an async function that can be awaited.")

        # Actor-related
        if 'actor' in fn['body'].lower() or 'spawn' in fn['body'].lower():
            parts.append("It involves actor spawning or message passing.")

        # Error handling
        if 'Result' in fn['return_type'] or '?' in fn['body']:
            parts.append("It uses Result-based error handling.")

        return ' '.join(parts)

    def generate_syntax_examples(self, count: int = 500) -> List[TrainingExample]:
        """Generate Simplex syntax learning examples."""
        examples = []

        # Basic syntax patterns
        syntax_patterns = [
            ("variable declaration", "let x: i32 = 42;", "Declares an immutable variable 'x' of type i32 with value 42"),
            ("mutable variable", "let mut count = 0;", "Declares a mutable variable 'count' initialized to 0"),
            ("function definition", "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}", "Defines a function 'add' that takes two i32 parameters and returns their sum"),
            ("struct definition", "struct Point {\n    x: f64,\n    y: f64,\n}", "Defines a struct 'Point' with x and y coordinates"),
            ("enum definition", "enum Status {\n    Active,\n    Inactive,\n    Pending(String),\n}", "Defines an enum 'Status' with variants including one with data"),
            ("pattern matching", "match value {\n    Some(x) => x,\n    None => 0,\n}", "Pattern matches on an Option, extracting the value or defaulting to 0"),
            ("async function", "async fn fetch_data() -> Result<Data, Error> {\n    let response = http::get(url).await?;\n    Ok(response.json())\n}", "Async function that fetches data with error handling"),
            ("actor spawn", "let actor = spawn!(MyActor::new());", "Spawns a new actor instance"),
            ("channel creation", "let (tx, rx) = channel::<Message>();", "Creates a typed channel for message passing"),
            ("belief declaration", "belief user_preference: Belief<Theme> = Belief::new(Theme::Dark, 0.8);", "Declares a belief with confidence level"),
            ("hive creation", "let hive = Hive::builder()\n    .with_specialists(specialists)\n    .build();", "Creates a cognitive hive with specialists"),
            ("trait implementation", "impl Display for MyType {\n    fn fmt(&self, f: &mut Formatter) -> Result {\n        write!(f, \"{}\")\n    }\n}", "Implements the Display trait for a custom type"),
            ("generic function", "fn process<T: Clone + Send>(item: T) -> T {\n    item.clone()\n}", "Generic function with trait bounds"),
            ("closure", "let square = |x: i32| x * x;", "Defines a closure that squares its input"),
            ("iterator chain", "items.iter()\n    .filter(|x| x.is_valid())\n    .map(|x| x.transform())\n    .collect()", "Chains iterator operations with filter, map, and collect"),
        ]

        for name, code, explanation in syntax_patterns:
            examples.append(TrainingExample(
                instruction=f"Show me how to write a {name} in Simplex:",
                input="",
                output=f"```simplex\n{code}\n```\n\n{explanation}",
                category="syntax"
            ))

            examples.append(TrainingExample(
                instruction=f"What is this Simplex syntax?",
                input=code,
                output=f"This is a {name}. {explanation}",
                category="syntax"
            ))

        return examples * (count // len(syntax_patterns) + 1)

    def generate_error_fixing_examples(self, count: int = 500) -> List[TrainingExample]:
        """Generate error fixing examples."""
        examples = []

        error_patterns = [
            # Missing semicolon
            ("let x = 5", "let x = 5;", "Missing semicolon at end of statement"),
            # Wrong type annotation
            ("let x: string = \"hello\";", "let x: String = \"hello\";", "Type should be 'String' not 'string'"),
            # Missing mut
            ("let x = 0;\nx = 1;", "let mut x = 0;\nx = 1;", "Variable must be declared 'mut' to be reassigned"),
            # Missing await
            ("let result = async_fn();", "let result = async_fn().await;", "Async function call must be awaited"),
            # Wrong return
            ("fn foo() -> i32 {\n    return\n}", "fn foo() -> i32 {\n    return 0;\n}", "Function must return a value of type i32"),
            # Missing type annotation
            ("fn add(a, b) -> i32 { a + b }", "fn add(a: i32, b: i32) -> i32 { a + b }", "Parameters need type annotations"),
        ]

        for buggy, fixed, explanation in error_patterns:
            examples.append(TrainingExample(
                instruction="Fix the error in this Simplex code:",
                input=buggy,
                output=f"Fixed code:\n```simplex\n{fixed}\n```\n\nIssue: {explanation}",
                category="error_fixing"
            ))

        return examples * (count // len(error_patterns) + 1)

    def generate_all(self, total_examples: int = 10000) -> List[TrainingExample]:
        """Generate all training examples."""
        print(f"Generating {total_examples} training examples...")

        # Proportional distribution
        completion = self.generate_code_completion_examples(int(total_examples * 0.3))
        instruction = self.generate_instruction_to_code_examples(int(total_examples * 0.25))
        explanation = self.generate_code_explanation_examples(int(total_examples * 0.2))
        syntax = self.generate_syntax_examples(int(total_examples * 0.15))
        fixing = self.generate_error_fixing_examples(int(total_examples * 0.1))

        self.examples = completion + instruction + explanation + syntax + fixing
        random.shuffle(self.examples)

        print(f"Generated {len(self.examples)} total examples:")
        print(f"  - Code completion: {len(completion)}")
        print(f"  - Instruction to code: {len(instruction)}")
        print(f"  - Code explanation: {len(explanation)}")
        print(f"  - Syntax examples: {len(syntax)}")
        print(f"  - Error fixing: {len(fixing)}")

        return self.examples

    def to_chat_format(self) -> List[Dict]:
        """Convert examples to chat format for training."""
        chat_examples = []

        for ex in self.examples:
            messages = [
                {"role": "system", "content": "You are SimplexSpecialist, an AI assistant specialized in the Simplex programming language. You help users write, understand, and debug Simplex code."}
            ]

            if ex.input:
                messages.append({"role": "user", "content": f"{ex.instruction}\n\n```simplex\n{ex.input}\n```"})
            else:
                messages.append({"role": "user", "content": ex.instruction})

            messages.append({"role": "assistant", "content": ex.output})

            chat_examples.append({
                "messages": messages,
                "category": ex.category
            })

        return chat_examples

    def save(self, output_path: str) -> None:
        """Save training data to JSON file."""
        chat_data = self.to_chat_format()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(chat_data)} examples to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate Simplex training data")
    parser.add_argument("--codebase", type=str, default="/Users/rod/code/simplex",
                        help="Path to Simplex codebase")
    parser.add_argument("--spec", type=str, default="/Users/rod/code/simplex/simplex-docs/spec",
                        help="Path to spec documents")
    parser.add_argument("--output", type=str, default="simplex_training_data.json",
                        help="Output file path")
    parser.add_argument("--examples", type=int, default=10000,
                        help="Number of training examples to generate")

    args = parser.parse_args()

    # Extract code
    extractor = SimplexCodeExtractor(args.codebase)
    extractor.extract_all()

    # Generate training data
    generator = SimplexTrainingDataGenerator(extractor, args.spec)
    generator.generate_all(args.examples)
    generator.save(args.output)


if __name__ == "__main__":
    main()
