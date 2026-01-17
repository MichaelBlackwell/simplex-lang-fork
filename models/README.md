# Simplex Cognitive Models

This directory contains metadata and configuration for the Simplex cognitive model family.

## Model Tiers

| Directory | Model | Status | Purpose |
|-----------|-------|--------|---------|
| `qwen3-8b/` | Qwen3-8B | **INSTALLED** | Hive tier - primary specialist SLM |
| `qwen3-32b/` | Qwen3-32B | PENDING | Divine tier - cross-hive arbitration |
| `qwen3-1.7b/` | Qwen3-1.7B | PENDING | Edge tier - mobile/embedded |
| `nomic-embed/` | nomic-embed-text-v1.5 | PENDING | Semantic routing and memory |

## Model Weights Location

Actual model weights are managed by Ollama and stored in:
```
~/.ollama/models/
```

This directory contains:
- Model configuration files
- Custom Modelfiles for Simplex
- Fine-tuned LoRA adapters (future)

## Quick Start

```bash
# List installed models
ollama list

# Run Hive tier model
ollama run qwen3:8b "Your prompt here"

# Run with thinking mode (complex reasoning)
ollama run qwen3:8b "/think Solve this complex problem..."

# Run without thinking mode (fast inference)
ollama run qwen3:8b "/no_think Quick answer needed"
```

## API Usage

```bash
# Generate completion
curl http://localhost:11434/api/generate \
  -d '{"model": "qwen3:8b", "prompt": "Hello", "stream": false}'

# Chat completion
curl http://localhost:11434/api/chat \
  -d '{
    "model": "qwen3:8b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

## Model Specifications

### Qwen3-8B (Hive Tier)

- **Parameters**: 8.2B
- **Context**: 32K (128K with YaRN)
- **Quantization**: Q4_K_M (5.2GB)
- **License**: Apache 2.0
- **Capabilities**: completion, tools, thinking

## See Also

- [DECISION-001-base-models.md](../simplex-docs/decisions/DECISION-001-base-models.md) - Base model selection rationale
- [TASK-002-cognitive-models.md](../tasks/TASK-002-cognitive-models.md) - Full implementation plan
