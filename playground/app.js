// Simplex Playground - Client-side Application

const API_BASE = '/api';

// Global state
let editor = null;
let examples = [];

// Default code
const DEFAULT_CODE = `// Welcome to the Simplex Playground!
// Write your Simplex code here and click Run to execute it.

fn main() {
    println("Hello, Simplex!");

    // Variables and types
    let x: i64 = 42;
    let name: String = "World";

    println(f"The answer is {x}");
    println(f"Hello, {name}!");
}
`;

// Simplex language definition for Monaco Editor
const simplexLanguage = {
    defaultToken: '',
    tokenPostfix: '.sx',

    keywords: [
        'fn', 'let', 'var', 'const', 'type', 'struct', 'enum', 'trait', 'impl',
        'mod', 'module', 'use', 'pub', 'if', 'else', 'match', 'while', 'for',
        'in', 'loop', 'break', 'continue', 'return', 'yield', 'actor', 'receive',
        'spawn', 'send', 'ask', 'init', 'async', 'await', 'specialist', 'hive',
        'infer', 'where', 'as', 'mut', 'ref', 'move', 'dyn', 'extern', 'unsafe'
    ],

    typeKeywords: [
        'i8', 'i16', 'i32', 'i64', 'i128', 'isize',
        'u8', 'u16', 'u32', 'u64', 'u128', 'usize',
        'f32', 'f64', 'bool', 'char', 'str', 'String',
        'Vec', 'Option', 'Result', 'Box', 'Rc', 'Arc',
        'Cell', 'RefCell', 'Mutex', 'RwLock',
        'HashMap', 'HashSet', 'BTreeMap', 'BTreeSet',
        'Actor', 'ActorRef', 'Message', 'Future', 'Stream',
        'Specialist', 'Hive', 'Model', 'Embedding'
    ],

    constants: ['true', 'false', 'None', 'Some', 'Ok', 'Err'],

    builtins: [
        'println', 'print', 'format', 'panic', 'assert', 'assert_eq',
        'dbg', 'todo', 'unimplemented', 'malloc', 'free', 'sizeof',
        'transmute', 'drop'
    ],

    operators: [
        '=', '>', '<', '!', '~', '?', ':', '==', '<=', '>=', '!=',
        '&&', '||', '++', '--', '+', '-', '*', '/', '&', '|', '^', '%',
        '<<', '>>', '+=', '-=', '*=', '/=', '&=', '|=', '^=',
        '%=', '<<=', '>>=', '->', '=>', '::', '..', '..='
    ],

    symbols: /[=><!~?:&|+\-*\/\^%]+/,

    escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

    tokenizer: {
        root: [
            // Identifiers and keywords
            [/[a-z_$][\w$]*/, {
                cases: {
                    '@keywords': 'keyword',
                    '@typeKeywords': 'type',
                    '@constants': 'constant',
                    '@builtins': 'predefined',
                    '@default': 'identifier'
                }
            }],

            // Type identifiers (PascalCase)
            [/[A-Z][\w$]*/, 'type.identifier'],

            // Whitespace
            { include: '@whitespace' },

            // Delimiters and operators
            [/[{}()\[\]]/, '@brackets'],
            [/[<>](?!@symbols)/, '@brackets'],
            [/@symbols/, {
                cases: {
                    '@operators': 'operator',
                    '@default': ''
                }
            }],

            // Numbers
            [/\d*\.\d+([eE][\-+]?\d+)?/, 'number.float'],
            [/0[xX][0-9a-fA-F]+/, 'number.hex'],
            [/0[bB][01]+/, 'number.binary'],
            [/0[oO][0-7]+/, 'number.octal'],
            [/\d+/, 'number'],

            // Delimiter
            [/[;,.]/, 'delimiter'],

            // Strings
            [/f"([^"\\]|\\.)*$/, 'string.invalid'],
            [/f"/, 'string', '@fstring'],
            [/"([^"\\]|\\.)*$/, 'string.invalid'],
            [/"/, 'string', '@string'],
            [/'[^\\']'/, 'string'],
            [/'/, 'string.invalid']
        ],

        whitespace: [
            [/[ \t\r\n]+/, ''],
            [/\/\/\/.*$/, 'comment.doc'],
            [/\/\/.*$/, 'comment'],
            [/\/\*/, 'comment', '@comment']
        ],

        comment: [
            [/[^\/*]+/, 'comment'],
            [/\*\//, 'comment', '@pop'],
            [/[\/*]/, 'comment']
        ],

        string: [
            [/[^\\"]+/, 'string'],
            [/@escapes/, 'string.escape'],
            [/\\./, 'string.escape.invalid'],
            [/"/, 'string', '@pop']
        ],

        fstring: [
            [/[^\\"{]+/, 'string'],
            [/\{/, 'delimiter.bracket', '@fstringExpr'],
            [/@escapes/, 'string.escape'],
            [/\\./, 'string.escape.invalid'],
            [/"/, 'string', '@pop']
        ],

        fstringExpr: [
            [/[^}]+/, 'identifier'],
            [/\}/, 'delimiter.bracket', '@pop']
        ]
    }
};

// Simplex theme for Monaco
const simplexTheme = {
    base: 'vs-dark',
    inherit: true,
    rules: [
        { token: 'comment', foreground: '6c7086', fontStyle: 'italic' },
        { token: 'comment.doc', foreground: '89b4fa', fontStyle: 'italic' },
        { token: 'keyword', foreground: 'cba6f7' },
        { token: 'type', foreground: 'f9e2af' },
        { token: 'type.identifier', foreground: 'f9e2af' },
        { token: 'constant', foreground: 'fab387' },
        { token: 'predefined', foreground: '89dceb' },
        { token: 'identifier', foreground: 'cdd6f4' },
        { token: 'string', foreground: 'a6e3a1' },
        { token: 'string.escape', foreground: 'f2cdcd' },
        { token: 'number', foreground: 'fab387' },
        { token: 'number.float', foreground: 'fab387' },
        { token: 'number.hex', foreground: 'fab387' },
        { token: 'operator', foreground: '89dceb' },
        { token: 'delimiter', foreground: '9399b2' },
        { token: 'delimiter.bracket', foreground: 'cba6f7' }
    ],
    colors: {
        'editor.background': '#1e1e2e',
        'editor.foreground': '#cdd6f4',
        'editor.lineHighlightBackground': '#313244',
        'editor.selectionBackground': '#45475a',
        'editorCursor.foreground': '#f5e0dc',
        'editorLineNumber.foreground': '#6c7086',
        'editorLineNumber.activeForeground': '#cdd6f4',
        'editorIndentGuide.background': '#313244',
        'editorIndentGuide.activeBackground': '#45475a'
    }
};

// Initialize Monaco Editor
function initEditor() {
    require.config({ paths: { vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs' } });

    require(['vs/editor/editor.main'], function () {
        // Register Simplex language
        monaco.languages.register({ id: 'simplex' });
        monaco.languages.setMonarchTokensProvider('simplex', simplexLanguage);
        monaco.editor.defineTheme('simplex-dark', simplexTheme);

        // Create editor
        editor = monaco.editor.create(document.getElementById('editor'), {
            value: getInitialCode(),
            language: 'simplex',
            theme: 'simplex-dark',
            fontSize: 14,
            fontFamily: "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
            minimap: { enabled: false },
            scrollBeyondLastLine: false,
            automaticLayout: true,
            tabSize: 4,
            insertSpaces: true,
            wordWrap: 'on',
            lineNumbers: 'on',
            renderWhitespace: 'selection',
            bracketPairColorization: { enabled: true },
            padding: { top: 16, bottom: 16 }
        });

        // Keyboard shortcuts
        editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => {
            runCode();
        });

        editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
            // Prevent default save dialog
            shareCode();
        });

        // Update URL hash on code change (debounced)
        let updateTimeout;
        editor.onDidChangeModelContent(() => {
            clearTimeout(updateTimeout);
            updateTimeout = setTimeout(() => {
                updateUrlHash();
            }, 1000);
        });
    });
}

// Get initial code from URL or default
function getInitialCode() {
    const hash = window.location.hash;
    if (hash && hash.length > 1) {
        try {
            const encoded = hash.substring(1);
            const decoded = decodeURIComponent(atob(encoded));
            return decoded;
        } catch (e) {
            console.error('Failed to decode URL hash:', e);
        }
    }
    return DEFAULT_CODE;
}

// Update URL hash with current code
function updateUrlHash() {
    if (!editor) return;
    const code = editor.getValue();
    const encoded = btoa(encodeURIComponent(code));
    history.replaceState(null, '', '#' + encoded);
}

// Load examples from JSON
async function loadExamples() {
    try {
        const response = await fetch('examples.json');
        examples = await response.json();
        renderExamplesMenu();
    } catch (e) {
        console.error('Failed to load examples:', e);
        // Use fallback examples
        examples = getFallbackExamples();
        renderExamplesMenu();
    }
}

// Fallback examples if JSON fails to load
function getFallbackExamples() {
    return {
        categories: [
            {
                name: 'Basics',
                examples: [
                    {
                        title: 'Hello World',
                        description: 'The classic first program',
                        code: 'fn main() {\n    println("Hello, World!");\n}'
                    }
                ]
            }
        ]
    };
}

// Render examples dropdown menu
function renderExamplesMenu() {
    const menu = document.getElementById('examples-menu');
    menu.innerHTML = '';

    for (const category of examples.categories) {
        const categoryDiv = document.createElement('div');
        categoryDiv.className = 'dropdown-category';
        categoryDiv.textContent = category.name;
        menu.appendChild(categoryDiv);

        for (const example of category.examples) {
            const item = document.createElement('div');
            item.className = 'dropdown-item';
            item.innerHTML = `
                <div class="dropdown-item-title">${example.title}</div>
                <div class="dropdown-item-description">${example.description}</div>
            `;
            item.addEventListener('click', () => {
                loadExample(example);
                toggleExamplesMenu(false);
            });
            menu.appendChild(item);
        }
    }
}

// Load an example into the editor
function loadExample(example) {
    if (editor) {
        editor.setValue(example.code);
        clearOutput();
        setStatus('Ready');
    }
}

// Toggle examples dropdown
function toggleExamplesMenu(show) {
    const menu = document.getElementById('examples-menu');
    if (show === undefined) {
        menu.classList.toggle('show');
    } else if (show) {
        menu.classList.add('show');
    } else {
        menu.classList.remove('show');
    }
}

// Run code
async function runCode() {
    if (!editor) return;

    const code = editor.getValue();
    const runBtn = document.getElementById('run-btn');
    const output = document.getElementById('output');

    // Update UI
    runBtn.disabled = true;
    runBtn.innerHTML = '<div class="loading-spinner"></div> Running...';
    setStatus('Running...', 'running');
    output.innerHTML = '<span class="output-info">Compiling and running...</span>';

    const startTime = performance.now();

    try {
        const response = await fetch(`${API_BASE}/run`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ code })
        });

        const result = await response.json();
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

        if (result.success) {
            setStatus('Success', 'success');
            let outputHtml = '';

            if (result.stdout) {
                outputHtml += `<span class="output-stdout">${escapeHtml(result.stdout)}</span>`;
            }
            if (result.stderr) {
                outputHtml += `<span class="output-stderr">${escapeHtml(result.stderr)}</span>`;
            }
            if (!result.stdout && !result.stderr) {
                outputHtml = '<span class="output-placeholder">Program completed with no output</span>';
            }

            outputHtml += `<div class="output-timing">Completed in ${elapsed}s (exit code: ${result.exit_code})</div>`;
            output.innerHTML = outputHtml;
        } else {
            setStatus('Error', 'error');
            let outputHtml = '';

            if (result.compile_error) {
                outputHtml += `<span class="output-info">Compilation Error:</span>\n`;
                outputHtml += `<span class="output-stderr">${escapeHtml(result.compile_error)}</span>`;
            } else if (result.error) {
                outputHtml += `<span class="output-stderr">${escapeHtml(result.error)}</span>`;
            }

            if (result.stderr) {
                outputHtml += `<span class="output-stderr">${escapeHtml(result.stderr)}</span>`;
            }

            outputHtml += `<div class="output-timing">Failed after ${elapsed}s</div>`;
            output.innerHTML = outputHtml;
        }
    } catch (e) {
        setStatus('Error', 'error');
        output.innerHTML = `<span class="output-stderr">Failed to connect to server: ${escapeHtml(e.message)}</span>`;
    } finally {
        runBtn.disabled = false;
        runBtn.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                <path d="M8 5v14l11-7z"/>
            </svg>
            Run
        `;
    }
}

// Share code
function shareCode() {
    updateUrlHash();
    const url = window.location.href;
    document.getElementById('share-url').value = url;
    document.getElementById('share-modal').classList.add('show');
}

// Copy share URL to clipboard
async function copyShareUrl() {
    const urlInput = document.getElementById('share-url');
    try {
        await navigator.clipboard.writeText(urlInput.value);
        const btn = document.getElementById('copy-url-btn');
        btn.textContent = 'Copied!';
        setTimeout(() => {
            btn.textContent = 'Copy';
        }, 2000);
    } catch (e) {
        urlInput.select();
        document.execCommand('copy');
    }
}

// Clear output
function clearOutput() {
    document.getElementById('output').innerHTML =
        '<div class="output-placeholder">Run your code to see output here</div>';
}

// Set status indicator
function setStatus(text, className) {
    const status = document.getElementById('editor-status');
    status.textContent = text;
    status.className = 'status';
    if (className) {
        status.classList.add(className);
    }
}

// Escape HTML for safe display
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Format code (placeholder - would need server-side formatter)
function formatCode() {
    // For now, just a placeholder
    alert('Code formatting coming soon!');
}

// Initialize event listeners
function initEventListeners() {
    // Run button
    document.getElementById('run-btn').addEventListener('click', runCode);

    // Share button
    document.getElementById('share-btn').addEventListener('click', shareCode);

    // Format button
    document.getElementById('format-btn').addEventListener('click', formatCode);

    // Examples dropdown
    document.getElementById('examples-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        toggleExamplesMenu();
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', () => {
        toggleExamplesMenu(false);
    });

    // Clear output button
    document.getElementById('clear-output-btn').addEventListener('click', clearOutput);

    // Modal close
    document.querySelector('.modal-close').addEventListener('click', () => {
        document.getElementById('share-modal').classList.remove('show');
    });

    // Copy URL button
    document.getElementById('copy-url-btn').addEventListener('click', copyShareUrl);

    // Close modal on backdrop click
    document.getElementById('share-modal').addEventListener('click', (e) => {
        if (e.target.id === 'share-modal') {
            e.target.classList.remove('show');
        }
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Escape to close modal
        if (e.key === 'Escape') {
            document.getElementById('share-modal').classList.remove('show');
            toggleExamplesMenu(false);
        }
    });
}

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    initEditor();
    loadExamples();
    initEventListeners();
});
