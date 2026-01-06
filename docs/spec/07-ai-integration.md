# Simplex AI Integration

**Version 0.1.0**

AI capabilities are first-class primitives in Simplex, not bolted-on libraries.

---

## Design Principles

1. **AI as a primitive**: AI operations are language constructs, not library calls
2. **Type safety**: AI outputs are validated against expected types
3. **Batching**: Multiple AI calls are automatically batched for efficiency
4. **Streaming**: Long-running AI operations support incremental output
5. **Fallback**: Graceful degradation when AI services are unavailable

---

## Architecture

### Swarm-Local Inference

AI inference runs on dedicated nodes within the swarm:

```
    Worker VM                         Inference Pool
    +------------+                    +------------------+
    |            |  ai::complete()    |                  |
    |  Actor     +-------------------->  GPU Node        |
    |            |     (~1-5ms)       |  (Model in VRAM) |
    |            <--------------------+                  |
    +------------+     Response       +------------------+
```

**Why swarm-local?**

| Approach | Latency | Cost | Flexibility |
|----------|---------|------|-------------|
| Embedded in VM | <1ms | High (model per VM) | Low |
| **Swarm-local** | **1-5ms** | **Low (shared)** | **High** |
| External API | 50-500ms | Variable | High |

### Request Batching

The inference pool automatically batches requests:

```
    Request Flow:

    Worker A: ai::embed("text 1") ----+
                                      |
    Worker B: ai::embed("text 2") ----+---> Batcher ---> GPU
                                      |     (5-50ms     (batch of 8)
    Worker C: ai::embed("text 3") ----+      window)

    Result: 3 requests processed in 1 GPU batch
    Efficiency: 3x throughput vs sequential
```

---

## AI Module API

### Text Completion

```simplex
module ai

// Basic completion
pub async fn complete(prompt: String) -> String

// Completion with options
pub async fn complete(
    prompt: String,
    model: String = "default",
    temperature: f64 = 0.7,
    max_tokens: u64 = 1024,
    stop: List<String> = []
) -> String
```

**Usage:**

```simplex
// Simple
let response = await ai::complete("Explain quantum computing")

// With options
let creative = await ai::complete(
    "Write a poem about Rust",
    model: "creative",
    temperature: 0.9,
    max_tokens: 500
)

// With stop sequences
let answer = await ai::complete(
    "Q: What is 2+2?\nA:",
    stop: ["\n", "Q:"]
)
```

### Streaming Completion

```simplex
// Streaming for long responses
pub async fn stream(
    prompt: String,
    model: String = "default"
) -> Stream<String>
```

**Usage:**

```simplex
// Stream chunks as they arrive
for chunk in ai::stream("Write a long story") {
    print(chunk)  // Display incrementally
}

// Collect into string
let full_response = ai::stream(prompt).collect()
```

### Embeddings

```simplex
// Single embedding
pub fn embed(text: String) -> Vector<f64, 1536>

// Batch embeddings (more efficient)
pub fn embed_batch(texts: List<String>) -> List<Vector<f64, 1536>>
```

**Usage:**

```simplex
// Single text
let embedding = ai::embed("Hello world")

// Batch (automatically batched by runtime)
let embeddings = ai::embed_batch([
    "First document",
    "Second document",
    "Third document"
])

// Semantic similarity
let similarity = dot(embedding1, embedding2)
```

### Structured Extraction

```simplex
// Extract typed data from text
pub async fn extract<T>(
    input: String,
    schema: Schema<T> = Schema::infer<T>()
) -> Result<T, ExtractionError>
```

**Usage:**

```simplex
// Define expected type
type ContactInfo {
    name: String,
    email: Option<String>,
    phone: Option<String>,
    company: Option<String>
}

// Extract from unstructured text
let email_body = "Hi, I'm John from Acme Corp. Reach me at john@acme.com"
let contact = await ai::extract<ContactInfo>(email_body)

// Result:
// ContactInfo {
//     name: "John",
//     email: Some("john@acme.com"),
//     phone: None,
//     company: Some("Acme Corp")
// }
```

### Classification

```simplex
// Classify into enum categories
pub async fn classify<T: Enum>(
    input: String,
    categories: List<T> = T::variants()
) -> T
```

**Usage:**

```simplex
enum Sentiment { Positive, Negative, Neutral }

let review = "This product exceeded my expectations!"
let sentiment = await ai::classify<Sentiment>(review)
// Result: Sentiment::Positive

enum Priority { Low, Medium, High, Critical }

let ticket = "Production server is down, customers affected"
let priority = await ai::classify<Priority>(ticket)
// Result: Priority::Critical
```

### Similarity Search

```simplex
// Compute similarity between vectors
pub fn similarity(a: Vector, b: Vector) -> f64

// Find k nearest neighbors
pub fn nearest(
    query: Vector,
    candidates: List<Vector>,
    k: u64
) -> List<(usize, f64)>  // (index, similarity)
```

**Usage:**

```simplex
// Build search index
let documents = load_documents()
let embeddings = ai::embed_batch(documents.map(d => d.content))

// Search
let query = "How do I reset my password?"
let query_embedding = ai::embed(query)

let results = ai::nearest(query_embedding, embeddings, k: 5)

for (idx, score) in results {
    print("Score {score:.2}: {documents[idx].title}")
}
```

---

## Model Selection

### Tiered Models

Simplex supports multiple model tiers:

```simplex
// Fast model - low latency, lower capability
let quick = await ai::complete(prompt, model: "fast")

// Default model - balanced
let standard = await ai::complete(prompt, model: "default")

// Quality model - higher capability, higher latency
let quality = await ai::complete(prompt, model: "quality")
```

**Runtime routing:**

| Tier | Typical Model | Latency | Cost |
|------|---------------|---------|------|
| fast | Llama 7B | 10-50ms | $0.0001/req |
| default | Llama 70B | 50-200ms | $0.001/req |
| quality | Claude/GPT-4 (API) | 200-2000ms | $0.01/req |

### Model Configuration

```simplex
// Configure available models in deployment
config AI {
    models: {
        "fast": {
            provider: Local,
            model: "llama-7b-q8",
            max_batch: 32
        },
        "default": {
            provider: Local,
            model: "llama-70b-q4",
            max_batch: 8
        },
        "quality": {
            provider: Anthropic,
            model: "claude-3-opus",
            api_key: env("ANTHROPIC_API_KEY")
        }
    },

    embedding: {
        provider: Local,
        model: "bge-large",
        dimensions: 1536
    }
}
```

---

## AI Actor Pattern

For stateful AI interactions:

### Chatbot with Memory

```simplex
actor Chatbot {
    var history: List<Message> = []
    var system_prompt: String = "You are a helpful assistant."

    receive Chat(user_message: String) -> String {
        // Add user message to history
        history.push(Message::user(user_message))

        // Build prompt with history
        let prompt = build_prompt(system_prompt, history)

        // Get AI response
        let response = await ai::complete(prompt)

        // Add to history
        history.push(Message::assistant(response))
        checkpoint()  // Persist conversation

        response
    }

    receive SetSystemPrompt(prompt: String) {
        system_prompt = prompt
        history = []  // Reset on system prompt change
        checkpoint()
    }

    receive GetHistory -> List<Message> {
        history.clone()
    }

    receive Reset {
        history = []
        checkpoint()
    }
}
```

### RAG (Retrieval-Augmented Generation)

```simplex
actor RAGAssistant {
    var document_embeddings: Map<DocId, Vector> = {}
    var documents: Map<DocId, Document> = {}

    receive IndexDocument(doc: Document) {
        let embedding = ai::embed(doc.content)
        document_embeddings.insert(doc.id, embedding)
        documents.insert(doc.id, doc)
        checkpoint()
    }

    receive Query(question: String) -> String {
        // Find relevant documents
        let query_embedding = ai::embed(question)
        let relevant = ai::nearest(
            query_embedding,
            document_embeddings.values().collect(),
            k: 3
        )

        // Build context from relevant docs
        let context = relevant
            .map((idx, _) => documents.values()[idx].content)
            .join("\n\n")

        // Generate answer with context
        let prompt = """
            Context:
            {context}

            Question: {question}

            Answer based on the context above:
        """

        await ai::complete(prompt)
    }

    receive IndexBatch(docs: List<Document>) {
        // Batch embedding is more efficient
        let contents = docs.map(d => d.content)
        let embeddings = ai::embed_batch(contents)

        for (doc, embedding) in docs.zip(embeddings) {
            document_embeddings.insert(doc.id, embedding)
            documents.insert(doc.id, doc)
        }
        checkpoint()
    }
}
```

### AI Agent

```simplex
actor AIAgent {
    var tools: Map<String, Tool> = {}

    receive RegisterTool(name: String, tool: Tool) {
        tools.insert(name, tool)
    }

    receive Execute(task: String) -> Result<String, AgentError> {
        var context = task
        var iterations = 0
        let max_iterations = 10

        loop {
            iterations += 1
            if iterations > max_iterations {
                return Err(AgentError::MaxIterations)
            }

            // Ask AI what to do next
            let decision = await ai::extract<AgentDecision>("""
                Task: {task}

                Current context:
                {context}

                Available tools:
                {tools.keys().join(", ")}

                What should we do next? Either use a tool or provide final answer.
            """)

            match decision {
                AgentDecision::UseTool(name, args) => {
                    let tool = tools.get(name)?
                    let result = tool.execute(args)
                    context = "{context}\n\nTool {name} result: {result}"
                },
                AgentDecision::FinalAnswer(answer) => {
                    return Ok(answer)
                }
            }
        }
    }
}

type AgentDecision {
    UseTool(tool_name: String, arguments: Map<String, String>),
    FinalAnswer(answer: String)
}
```

---

## Error Handling

### Extraction Errors

```simplex
let result = await ai::extract<ContactInfo>(messy_text)

match result {
    Ok(contact) => process(contact),
    Err(ExtractionError::InvalidFormat(msg)) => {
        log::warn("Could not parse: {msg}")
        fallback_extraction(messy_text)
    },
    Err(ExtractionError::MissingRequired(field)) => {
        log::error("Missing required field: {field}")
        Err(AppError::InvalidInput)
    }
}
```

### Retry with Backoff

```simplex
async fn ai_with_retry<T>(
    operation: async fn() -> Result<T, AIError>,
    max_retries: u64
) -> Result<T, AIError> {
    var retries = 0

    loop {
        match await operation() {
            Ok(result) => return Ok(result),
            Err(AIError::RateLimited) if retries < max_retries => {
                retries += 1
                let delay = Duration::seconds(2.pow(retries))
                sleep(delay)
            },
            Err(e) => return Err(e)
        }
    }
}
```

### Fallback Models

```simplex
async fn complete_with_fallback(prompt: String) -> String {
    // Try quality model first
    match await ai::complete(prompt, model: "quality") {
        Ok(response) => response,
        Err(_) => {
            // Fall back to local model
            log::warn("Quality model unavailable, using local")
            await ai::complete(prompt, model: "fast")
        }
    }
}
```

---

## Performance Optimization

### Batch Operations

```simplex
// Bad: Sequential calls
for doc in documents {
    let embedding = ai::embed(doc.content)  // N round trips
    store(doc.id, embedding)
}

// Good: Batch call
let contents = documents.map(d => d.content)
let embeddings = ai::embed_batch(contents)  // 1 batch operation

for (doc, embedding) in documents.zip(embeddings) {
    store(doc.id, embedding)
}
```

### Parallel AI Operations

```simplex
// Run independent AI calls in parallel
let (summary, entities, sentiment) = await parallel(
    ai::complete("Summarize: {text}"),
    ai::extract<List<Entity>>(text),
    ai::classify<Sentiment>(text)
)
```

### Caching

```simplex
actor CachedAI {
    var cache: Map<String, String> = {}

    receive Complete(prompt: String) -> String {
        // Check cache
        let cache_key = hash(prompt)
        match cache.get(cache_key) {
            Some(cached) => cached,
            None => {
                let response = await ai::complete(prompt)
                cache.insert(cache_key, response.clone())
                response
            }
        }
    }
}
```

---

## Inference Pool Configuration

### GPU Allocation

```simplex
config InferencePool {
    // GPU nodes
    nodes: [
        { instance: "g4dn.xlarge", count: 2, spot: true },
        { instance: "g4dn.xlarge", count: 1, spot: false }  // On-demand fallback
    ],

    // Model loading
    models: {
        "llama-7b": { vram: "8GB", preload: true },
        "llama-70b": { vram: "40GB", preload: false },  // Load on demand
        "bge-large": { vram: "2GB", preload: true }
    },

    // Batching
    batching: {
        max_batch_size: 32,
        max_wait_ms: 50
    }
}
```

### Cost Monitoring

```bash
# Monitor AI costs
simplex ai costs --watch

# Output:
# AI Inference Costs (last hour)
# --------------------------------
# Model           Requests    Tokens      Cost
# llama-7b        12,456      1.2M        $0.12
# llama-70b       1,234       500K        $0.50
# claude-3        89          45K         $1.35
# embeddings      45,678      -           $0.05
# --------------------------------
# Total                                   $2.02
```

---

## Next Steps

- [Examples](../examples/document-pipeline.md): See AI in a complete program
- [Cost Optimization](08-cost-optimization.md): AI cost strategies
- [Language Syntax](04-language-syntax.md): Full syntax reference
