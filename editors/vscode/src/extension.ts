/**
 * Simplex Language Extension for VS Code
 *
 * Provides full IDE support including:
 * - Language Server Protocol integration (go to definition, hover, completion, etc.)
 * - Syntax highlighting
 * - Code snippets
 * - Build and run tasks
 * - Debugging support
 *
 * Copyright (c) 2025-2026 Rod Higgins
 * Licensed under MIT - see LICENSE file
 */

import * as vscode from 'vscode';
import * as path from 'path';
import { spawn, ChildProcess } from 'child_process';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind,
    State
} from 'vscode-languageclient/node';

let client: LanguageClient | undefined;
let outputChannel: vscode.OutputChannel;
let statusBarItem: vscode.StatusBarItem;

/**
 * Extension activation entry point
 */
export async function activate(context: vscode.ExtensionContext): Promise<void> {
    outputChannel = vscode.window.createOutputChannel('Simplex');
    context.subscriptions.push(outputChannel);

    // Create status bar item
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    statusBarItem.text = '$(loading~spin) Simplex';
    statusBarItem.tooltip = 'Simplex Language Server';
    context.subscriptions.push(statusBarItem);

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('simplex.restartServer', restartLanguageServer),
        vscode.commands.registerCommand('simplex.showVersion', showVersion),
        vscode.commands.registerCommand('simplex.buildFile', buildCurrentFile),
        vscode.commands.registerCommand('simplex.runFile', runCurrentFile)
    );

    // Register task provider
    context.subscriptions.push(
        vscode.tasks.registerTaskProvider('simplex', new SimplexTaskProvider())
    );

    // Start language server
    const config = vscode.workspace.getConfiguration('simplex');
    if (config.get<boolean>('enable', true)) {
        await startLanguageServer(context);
    }

    outputChannel.appendLine('Simplex extension activated');
}

/**
 * Extension deactivation
 */
export async function deactivate(): Promise<void> {
    if (client) {
        await client.stop();
        client = undefined;
    }
}

/**
 * Start the Simplex language server
 */
async function startLanguageServer(context: vscode.ExtensionContext): Promise<void> {
    const config = vscode.workspace.getConfiguration('simplex');
    const lspPath = config.get<string>('lspPath', 'sxlsp');
    const lspArgs = config.get<string[]>('lspArgs', ['--stdio']);

    outputChannel.appendLine(`Starting language server: ${lspPath} ${lspArgs.join(' ')}`);
    statusBarItem.show();

    // Server options - spawn the language server process
    const serverOptions: ServerOptions = {
        command: lspPath,
        args: lspArgs,
        transport: TransportKind.stdio,
        options: {
            env: {
                ...process.env,
                // Add any environment variables needed by the server
            }
        }
    };

    // Client options
    const clientOptions: LanguageClientOptions = {
        documentSelector: [
            { scheme: 'file', language: 'simplex' },
            { scheme: 'untitled', language: 'simplex' }
        ],
        synchronize: {
            // Notify server about file changes to .sx files in the workspace
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.sx')
        },
        outputChannel: outputChannel,
        traceOutputChannel: outputChannel,
        initializationOptions: {
            enableHoverDocs: config.get<boolean>('enableHoverDocs', true),
            enableDiagnostics: config.get<boolean>('enableDiagnostics', true)
        },
        middleware: {
            // Custom middleware for handling LSP messages
            provideCompletionItem: async (document, position, context, token, next) => {
                const result = await next(document, position, context, token);
                return result;
            },
            provideHover: async (document, position, token, next) => {
                const result = await next(document, position, token);
                return result;
            }
        }
    };

    // Create and start the language client
    client = new LanguageClient(
        'simplex',
        'Simplex Language Server',
        serverOptions,
        clientOptions
    );

    // Handle client state changes
    client.onDidChangeState((event) => {
        switch (event.newState) {
            case State.Starting:
                statusBarItem.text = '$(loading~spin) Simplex';
                statusBarItem.tooltip = 'Simplex Language Server starting...';
                break;
            case State.Running:
                statusBarItem.text = '$(check) Simplex';
                statusBarItem.tooltip = 'Simplex Language Server running';
                statusBarItem.backgroundColor = undefined;
                break;
            case State.Stopped:
                statusBarItem.text = '$(error) Simplex';
                statusBarItem.tooltip = 'Simplex Language Server stopped';
                statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground');
                break;
        }
    });

    // Start the client
    try {
        await client.start();
        outputChannel.appendLine('Language server started successfully');
    } catch (error) {
        outputChannel.appendLine(`Failed to start language server: ${error}`);
        statusBarItem.text = '$(error) Simplex';
        statusBarItem.tooltip = `Language server failed: ${error}`;
        statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground');

        vscode.window.showErrorMessage(
            `Failed to start Simplex language server. Make sure 'sxlsp' is installed and in your PATH.`,
            'Open Settings'
        ).then((selection) => {
            if (selection === 'Open Settings') {
                vscode.commands.executeCommand('workbench.action.openSettings', 'simplex.lspPath');
            }
        });
    }

    context.subscriptions.push(client);
}

/**
 * Restart the language server
 */
async function restartLanguageServer(): Promise<void> {
    outputChannel.appendLine('Restarting language server...');

    if (client) {
        await client.stop();
        client = undefined;
    }

    const context = getExtensionContext();
    if (context) {
        await startLanguageServer(context);
    }
}

// Store extension context for restart
let extensionContext: vscode.ExtensionContext | undefined;

function getExtensionContext(): vscode.ExtensionContext | undefined {
    return extensionContext;
}

/**
 * Show version information
 */
async function showVersion(): Promise<void> {
    const pkg = require('../package.json');
    const message = `Simplex Extension v${pkg.version}`;

    if (client && client.state === State.Running) {
        // Try to get server version
        try {
            const serverInfo = await client.sendRequest('simplex/version');
            vscode.window.showInformationMessage(`${message}\nLanguage Server: ${serverInfo}`);
        } catch {
            vscode.window.showInformationMessage(message);
        }
    } else {
        vscode.window.showInformationMessage(message);
    }
}

/**
 * Build the current Simplex file
 */
async function buildCurrentFile(): Promise<void> {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'simplex') {
        vscode.window.showWarningMessage('No Simplex file is currently open');
        return;
    }

    // Save the file first
    await editor.document.save();

    const config = vscode.workspace.getConfiguration('simplex');
    const compilerPath = config.get<string>('compilerPath', 'sxc');
    const filePath = editor.document.uri.fsPath;
    const outputPath = filePath.replace(/\.sx$/, '');

    const task = new vscode.Task(
        { type: 'simplex', task: 'build', file: filePath },
        vscode.TaskScope.Workspace,
        'Build Simplex File',
        'simplex',
        new vscode.ShellExecution(`${compilerPath} "${filePath}" -o "${outputPath}"`),
        '$simplex'
    );

    vscode.tasks.executeTask(task);
}

/**
 * Run the current Simplex file
 */
async function runCurrentFile(): Promise<void> {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'simplex') {
        vscode.window.showWarningMessage('No Simplex file is currently open');
        return;
    }

    // Save and build first
    await editor.document.save();

    const config = vscode.workspace.getConfiguration('simplex');
    const compilerPath = config.get<string>('compilerPath', 'sxc');
    const filePath = editor.document.uri.fsPath;
    const outputPath = filePath.replace(/\.sx$/, '');

    const task = new vscode.Task(
        { type: 'simplex', task: 'run', file: filePath },
        vscode.TaskScope.Workspace,
        'Run Simplex File',
        'simplex',
        new vscode.ShellExecution(`${compilerPath} "${filePath}" -o "${outputPath}" && "${outputPath}"`),
        '$simplex'
    );

    vscode.tasks.executeTask(task);
}

/**
 * Task provider for Simplex build tasks
 */
class SimplexTaskProvider implements vscode.TaskProvider {
    private tasks: vscode.Task[] | undefined;

    public async provideTasks(): Promise<vscode.Task[]> {
        if (this.tasks) {
            return this.tasks;
        }

        const config = vscode.workspace.getConfiguration('simplex');
        const compilerPath = config.get<string>('compilerPath', 'sxc');

        this.tasks = [];

        // Add default tasks
        const buildTask = new vscode.Task(
            { type: 'simplex', task: 'build' },
            vscode.TaskScope.Workspace,
            'Build',
            'simplex',
            new vscode.ShellExecution(`${compilerPath} \${file}`),
            '$simplex'
        );
        buildTask.group = vscode.TaskGroup.Build;
        this.tasks.push(buildTask);

        const runTask = new vscode.Task(
            { type: 'simplex', task: 'run' },
            vscode.TaskScope.Workspace,
            'Run',
            'simplex',
            new vscode.ShellExecution(`${compilerPath} \${file} && ./\${fileBasenameNoExtension}`),
            '$simplex'
        );
        this.tasks.push(runTask);

        const testTask = new vscode.Task(
            { type: 'simplex', task: 'test' },
            vscode.TaskScope.Workspace,
            'Test',
            'simplex',
            new vscode.ShellExecution(`${compilerPath} --test \${workspaceFolder}`),
            '$simplex'
        );
        testTask.group = vscode.TaskGroup.Test;
        this.tasks.push(testTask);

        return this.tasks;
    }

    public resolveTask(task: vscode.Task): vscode.Task | undefined {
        const config = vscode.workspace.getConfiguration('simplex');
        const compilerPath = config.get<string>('compilerPath', 'sxc');

        const definition = task.definition as { type: string; task: string; file?: string };

        if (definition.type === 'simplex') {
            const file = definition.file || '${file}';
            let command: string;

            switch (definition.task) {
                case 'build':
                    command = `${compilerPath} "${file}"`;
                    break;
                case 'run':
                    const output = file.replace(/\.sx$/, '');
                    command = `${compilerPath} "${file}" -o "${output}" && "${output}"`;
                    break;
                case 'test':
                    command = `${compilerPath} --test "${file}"`;
                    break;
                default:
                    return undefined;
            }

            return new vscode.Task(
                definition,
                task.scope || vscode.TaskScope.Workspace,
                task.name,
                'simplex',
                new vscode.ShellExecution(command),
                '$simplex'
            );
        }

        return undefined;
    }
}

// Export for testing
export { SimplexTaskProvider };
