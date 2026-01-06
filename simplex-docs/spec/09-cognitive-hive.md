# Cognitive Hive AI Architecture

**The future of AI is not one giant mind, but a swarm of specialists.**

Simplex embraces the Cognitive Hive AI (CHAI) philosophy as a core architectural principle. Rather than relying on monolithic Large Language Models (LLMs), Simplex enables the orchestration of Small Language Models (SLMs) working as specialized agents within a coordinated hive.

---

## Philosophy

### The Problem with Monolithic LLMs

| Issue | Impact |
|-------|--------|
| Cost | $0.03-0.12 per 1K tokens, unpredictable at scale |
| Latency | 500-3000ms per request |
| Control | Black box, prompt engineering required |
| Privacy | Data sent to external APIs |
| Reliability | Rate limits, outages, deprecation |
| Specialization | Jack of all trades, master of none |

### The CHAI Alternative

A cognitive hive is a collection of specialized small models (7B-13B parameters) that:

- **Specialize**: Each model masters a narrow domain
- **Collaborate**: Models communicate through message passing
- **Scale**: Add specialists as needs grow
- **Fail gracefully**: One specialist down doesn't stop the hive
- **Cost pennies**: Run on commodity ARM instances

```
         ┌─────────────────────────────────────────┐
         │              COGNITIVE HIVE              │
         │                                          │
         │    ┌──────┐  ┌──────┐  ┌──────┐        │
         │    │ SLM  │  │ SLM  │  │ SLM  │        │
         │    │ Code │  │ Text │  │ Legal│  ...   │
         │    └──┬───┘  └──┬───┘  └──┬───┘        │
         │       │         │         │             │
         │    ───┴─────────┴─────────┴───         │
         │              Message Bus                │
         │    ────────────────┬──────────         │
         │                    │                    │
         │              ┌─────┴─────┐              │
         │              │  Router   │              │
         │              └─────┬─────┘              │
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                         Task Input
```

---

## Naming Conventions

Simplex suggests (but does not enforce) meaningful naming for models and specialists. Two traditions are offered:

### Elvish (Sindarin/Quenya)

Drawing from Tolkien's languages, Elvish names evoke the organic, interconnected nature of the hive. These names carry poetic weight and connect to themes of wisdom, craft, and natural order.

| Elvish | Meaning | Suggested Use |
|--------|---------|---------------|
| **Isto** | Knowledge (Q) | Knowledge retrieval specialist |
| **Penna** | One who tells (S) | Text generation, storytelling |
| **Hend** | Eye (S) | Vision, image analysis |
| **Lasta** | To listen (S) | Audio transcription, understanding |
| **Thind** | Grey (wise) (S) | General reasoning |
| **Curu** | Skill, craft (Q) | Code generation, technical tasks |
| **Golwen** | Wise one (S) | Decision making, arbitration |
| **Runya** | Red flame (Q) | Fast, reactive processing |
| **Silma** | Crystal light (Q) | Clarity, summarization |
| **Ithil** | Moon (S) | Background processing, reflection |
| **Anor** | Sun (S) | Primary/lead specialist |
| **Mellon** | Friend (S) | Conversational, assistant |
| **Hîr** | Lord/Master (S) | Orchestrator, router |
| **Edhel** | Elf (S) | Meta-learning, adaptation |
| **Thalion** | Steadfast (S) | Reliable, deterministic tasks |

**Example Usage:**
```simplex
specialist Curu {
    model: "codellama-7b",
    domain: "code generation and review"
}

specialist Silma {
    model: "mistral-7b-instruct",
    domain: "summarization and clarity"
}

hive Mellon {
    router: Hîr,
    specialists: [Curu, Silma, Penna, Isto]
}
```

### Latin

Latin names carry gravitas and precision, connecting to scientific and legal traditions. They communicate function with classical clarity.

| Latin | Meaning | Suggested Use |
|-------|---------|---------------|
| **Cogito** | I think | Reasoning, analysis |
| **Scribo** | I write | Text generation, composition |
| **Lego** | I read/gather | Document processing, extraction |
| **Video** | I see | Vision, image understanding |
| **Audio** | I hear | Speech, audio processing |
| **Sentio** | I perceive/feel | Sentiment, emotion analysis |
| **Memor** | Mindful | Memory, context management |
| **Index** | Pointer | Routing, classification |
| **Judex** | Judge | Decision making, arbitration |
| **Custos** | Guardian | Validation, safety checking |
| **Nexus** | Connection | Integration, API bridging |
| **Vertex** | Peak/turning point | Orchestration, coordination |
| **Faber** | Craftsman | Code, technical generation |
| **Medicus** | Healer | Error correction, repair |
| **Vigil** | Watchman | Monitoring, alerting |

**Example Usage:**
```simplex
specialist Cogito {
    model: "llama2-13b",
    domain: "logical reasoning and analysis"
}

specialist Lego {
    model: "ner-extraction-7b",
    domain: "entity and data extraction"
}

hive Vertex {
    router: Index,
    specialists: [Cogito, Lego, Scribo, Sentio]
}
```

### Choosing a Convention

| Consider Elvish When | Consider Latin When |
|---------------------|---------------------|
| Building creative/consumer products | Building enterprise/B2B systems |
| Team appreciates whimsy | Team prefers classical formality |
| Names should feel organic | Names should feel technical |
| You're a Tolkien fan | You're classically inclined |

Both conventions can coexist. The Elvish router `Hîr` can orchestrate Latin specialists, or vice versa. The system doesn't enforce naming—these are cultural suggestions for readability and meaning.

---

## Core Constructs

### Specialists

A specialist is an actor wrapping a small language model:

```simplex
specialist EntityExtractor {
    // Model configuration
    model: "ner-fine-tuned-7b",
    domain: "named entity extraction",

    // Resource constraints
    memory: 8.GB,
    compute: "gpu.small",  // or "cpu.medium"

    // Behavioral configuration
    temperature: 0.1,      // Low for deterministic extraction
    max_tokens: 500,

    // Message handlers
    receive Extract(text: String) -> List<Entity> {
        let raw = infer("Extract all named entities from: {text}")
        parse_entities(raw)
    }

    receive ExtractTyped(text: String, entity_types: List<EntityType>) -> List<Entity> {
        let types_str = entity_types.join(", ")
        let raw = infer("Extract {types_str} entities from: {text}")
        parse_entities(raw).filter(|e| entity_types.contains(e.type))
    }
}
```

### The `infer` Primitive

Within a specialist, `infer` calls the underlying model:

```simplex
specialist Summarizer {
    model: "mistral-7b-instruct",

    receive Summarize(text: String, style: SummaryStyle) -> String {
        let prompt = match style {
            SummaryStyle::Brief => "Summarize in one sentence: {text}",
            SummaryStyle::Detailed => "Provide a detailed summary: {text}",
            SummaryStyle::Bullets => "Summarize as bullet points: {text}"
        }

        infer(prompt)  // Calls the specialist's model
    }
}
```

**`infer` Options:**

```simplex
// Basic inference
let result = infer(prompt)

// With parameters
let result = infer(
    prompt,
    temperature: 0.7,
    max_tokens: 200,
    stop_sequences: ["\n\n", "END"]
)

// Streaming inference
for chunk in infer_stream(prompt) {
    emit(chunk)
}

// Structured output (parsed to type)
let data = infer_typed<Person>(prompt)
```

### Hives

A hive is a supervisor for specialists with routing intelligence:

```simplex
hive DocumentProcessor {
    // Specialists in this hive
    specialists: [
        Summarizer,
        EntityExtractor,
        SentimentAnalyzer,
        Translator,
        Classifier
    ],

    // How tasks are routed to specialists
    router: SemanticRouter(
        embedding_model: "all-minilm-l6-v2",
        fallback: Summarizer  // Default if no match
    ),

    // Supervision strategy for specialists
    strategy: OneForOne,
    max_restarts: 5,
    within: Duration::minutes(1),

    // Shared resources
    memory: SharedVectorStore(dimension: 384),
    context: ConversationBuffer(max_turns: 50)
}
```

### Spawning and Using Hives

```simplex
fn main() {
    // Spawn the hive (starts all specialists)
    let hive = spawn DocumentProcessor

    // Automatic routing - hive decides which specialist
    let result = ask(hive, Process("Summarize this legal document..."))

    // Direct specialist access
    let summary = ask(hive.Summarizer, Summarize(text, Brief))

    // Check which specialists are available
    let available = hive.specialists()
    print("Active specialists: {available}")
}
```

---

## Routing Strategies

### Semantic Router

Routes based on embedding similarity between task and specialist domains:

```simplex
router SemanticRouter {
    embedding_model: "all-minilm-l6-v2",
    threshold: 0.7,  // Minimum similarity to route
    fallback: GeneralAssistant,

    fn route(task: Task) -> Specialist {
        let task_embedding = embed(task.description)

        var best_match: Option<Specialist> = None
        var best_score: f64 = 0.0

        for specialist in hive.specialists {
            let score = cosine_similarity(task_embedding, specialist.domain_embedding)
            if score > best_score && score >= threshold {
                best_score = score
                best_match = Some(specialist)
            }
        }

        best_match.unwrap_or(fallback)
    }
}
```

### Rule Router

Routes based on explicit rules:

```simplex
router RuleRouter {
    rules: [
        Rule(pattern: r"summar|brief|tldr", specialist: Summarizer),
        Rule(pattern: r"translat|convert to", specialist: Translator),
        Rule(pattern: r"extract|find all|list the", specialist: EntityExtractor),
        Rule(pattern: r"sentiment|feeling|emotion", specialist: SentimentAnalyzer),
        Rule(pattern: r"code|function|bug|review", specialist: CodeReviewer)
    ],
    fallback: GeneralAssistant
}
```

### LLM Router

Uses a small, fast model to decide routing:

```simplex
router LLMRouter {
    model: "tinyllama-1b",  // Very small, very fast

    fn route(task: Task) -> Specialist {
        let specialists_desc = hive.specialists
            .map(|s| "{s.name}: {s.domain}")
            .join("\n")

        let choice = infer("""
            Given these specialists:
            {specialists_desc}

            Which specialist should handle: {task.description}

            Reply with only the specialist name.
        """)

        hive.get_specialist(choice.trim())
    }
}
```

### Cascade Router

Try specialists in order until one succeeds:

```simplex
router CascadeRouter {
    cascade: [
        (EntityExtractor, confidence_threshold: 0.9),
        (GeneralExtractor, confidence_threshold: 0.7),
        (Fallback, confidence_threshold: 0.0)
    ],

    fn route(task: Task) -> Response {
        for (specialist, threshold) in cascade {
            let result = ask(specialist, task)
            if result.confidence >= threshold {
                return result
            }
        }
        Err(NoConfidentResult)
    }
}
```

---

## Ensemble Patterns

### Parallel Ensemble

Ask multiple specialists simultaneously:

```simplex
fn analyze_document(doc: Document) -> Analysis {
    // All specialists work in parallel
    let (summary, entities, sentiment, topics) = await parallel(
        ask(hive.Summarizer, Summarize(doc.text)),
        ask(hive.EntityExtractor, Extract(doc.text)),
        ask(hive.SentimentAnalyzer, Analyze(doc.text)),
        ask(hive.TopicClassifier, Classify(doc.text))
    )

    Analysis { summary, entities, sentiment, topics }
}
```

### Voting Ensemble

Multiple specialists vote on a decision:

```simplex
fn classify_with_confidence(text: String) -> Classification {
    // Ask three different classifiers
    let votes = await parallel(
        ask(hive.ClassifierA, Classify(text)),
        ask(hive.ClassifierB, Classify(text)),
        ask(hive.ClassifierC, Classify(text))
    )

    // Count votes
    let vote_counts = votes.group_by(|v| v.label).map(|g| (g.key, g.count()))
    let (winner, count) = vote_counts.max_by(|(_, c)| c)

    Classification {
        label: winner,
        confidence: count as f64 / votes.len() as f64,
        dissenting: votes.filter(|v| v.label != winner)
    }
}
```

### Weighted Ensemble

Specialists have different weights based on expertise:

```simplex
fn legal_analysis(document: String) -> LegalOpinion {
    let opinions = await parallel(
        (ask(hive.LegalExpert, Analyze(document)), weight: 0.5),
        (ask(hive.ComplianceChecker, Check(document)), weight: 0.3),
        (ask(hive.RiskAnalyzer, Assess(document)), weight: 0.2)
    )

    weighted_consensus(opinions)
}
```

### Chain of Specialists

Sequential processing pipeline:

```simplex
fn deep_research(query: String) -> ResearchReport {
    // Step 1: Understand the query
    let parsed_query = ask(hive.QueryParser, Parse(query))

    // Step 2: Retrieve relevant information
    let sources = ask(hive.Retriever, Search(parsed_query))

    // Step 3: Analyze each source
    let analyses = sources.map(|s| ask(hive.Analyzer, Analyze(s)))

    // Step 4: Synthesize findings
    let synthesis = ask(hive.Synthesizer, Combine(analyses))

    // Step 5: Format final report
    ask(hive.ReportWriter, Format(synthesis))
}
```

---

## Consensus Mechanisms

When specialists disagree:

### Majority Vote

```simplex
fn majority_vote<T: Eq>(responses: List<T>) -> Option<T> {
    let counts = responses.group_by(|r| r).map(|g| (g.key, g.count()))
    let (winner, count) = counts.max_by(|(_, c)| c)?

    if count > responses.len() / 2 {
        Some(winner)
    } else {
        None  // No majority
    }
}
```

### Arbiter Resolution

```simplex
specialist Arbiter {
    model: "llama2-13b",  // Larger model for complex decisions
    domain: "conflict resolution and synthesis",

    receive Resolve(opinions: List<Opinion>) -> Decision {
        let opinions_text = opinions
            .enumerate()
            .map(|(i, o)| "Opinion {i}: {o.content} (confidence: {o.confidence})")
            .join("\n")

        infer("""
            Multiple specialists have provided different opinions:

            {opinions_text}

            Analyze the disagreements and provide a final decision with reasoning.
        """)
    }
}
```

### Confidence-Weighted

```simplex
fn confidence_weighted<T>(responses: List<(T, f64)>) -> T {
    // Weight by confidence score
    let weighted = responses
        .group_by(|(value, _)| value)
        .map(|g| (g.key, g.values.map(|(_, conf)| conf).sum()))

    weighted.max_by(|(_, weight)| weight).0
}
```

---

## Shared Memory

Specialists can share context through hive memory:

### Vector Store

```simplex
hive KnowledgeHive {
    memory: VectorStore(
        dimension: 384,
        index: HNSW(ef_construction: 200, m: 16),
        persistence: S3("s3://hive-memory/vectors")
    )
}

specialist Researcher {
    receive Research(query: String) -> Report {
        // Search shared memory
        let relevant = hive.memory.search(query, k: 10)

        let context = relevant.map(|r| r.content).join("\n")
        let report = infer("Given context:\n{context}\n\nAnswer: {query}")

        // Store findings back
        hive.memory.add(
            content: report,
            metadata: { source: "Researcher", query: query }
        )

        report
    }
}
```

### Conversation Context

```simplex
hive ConversationalHive {
    context: ConversationBuffer(
        max_turns: 100,
        summarize_after: 50  // Auto-summarize old context
    )
}

specialist Assistant {
    receive Chat(message: String) -> String {
        // Access shared conversation history
        let history = hive.context.recent(10)

        let response = infer("""
            Conversation history:
            {history}

            User: {message}
            Assistant:
        """)

        // Add to shared context
        hive.context.add(Role::User, message)
        hive.context.add(Role::Assistant, response)

        response
    }
}
```

### Working Memory

Short-term shared state:

```simplex
hive TaskHive {
    working_memory: SharedMap<String, Dynamic>(
        ttl: Duration::minutes(30)  // Auto-expire old entries
    )
}

specialist TaskParser {
    receive Parse(task: String) -> TaskPlan {
        let plan = infer("Break down this task: {task}")

        // Store in working memory for other specialists
        hive.working_memory.set("current_task", task)
        hive.working_memory.set("current_plan", plan)

        plan
    }
}

specialist TaskExecutor {
    receive Execute(step: String) -> Result {
        // Read from working memory
        let context = hive.working_memory.get("current_plan")

        infer("Given plan: {context}\n\nExecute step: {step}")
    }
}
```

---

## Dynamic Specialist Management

### On-Demand Spawning

```simplex
hive AdaptiveHive {
    // Core specialists always running
    specialists: [GeneralAssistant, Router],

    // Available but not running until needed
    available: [
        LegalAnalyzer,
        MedicalAssistant,
        CodeReviewer,
        FinancialAdvisor
    ],

    on_route_miss(task: Task) {
        // No specialist matched - check if we can spawn one
        for specialist_type in available {
            if specialist_type.can_handle(task) {
                let new_specialist = spawn specialist_type
                register(new_specialist)
                return route_to(new_specialist, task)
            }
        }

        // Fall back to general assistant
        route_to(GeneralAssistant, task)
    }
}
```

### Hibernation

Save resources by hibernating idle specialists:

```simplex
hive EfficientHive {
    idle_timeout: Duration::minutes(5),

    on_specialist_idle(specialist: Specialist, duration: Duration) {
        if duration >= idle_timeout {
            // Save state and stop
            checkpoint(specialist)
            hibernate(specialist)
            log::info("{specialist.name} hibernated after {duration} idle")
        }
    }

    on_route_to_hibernated(specialist: Specialist, task: Task) {
        // Wake up the specialist
        restore(specialist)
        route_to(specialist, task)
    }
}
```

### Auto-Scaling

```simplex
hive ScalableHive {
    min_instances: { Summarizer: 1, Extractor: 1 },
    max_instances: { Summarizer: 10, Extractor: 5 },

    on_queue_depth(specialist_type: Type, depth: i64) {
        let current = instances_of(specialist_type).count()
        let max = max_instances[specialist_type]

        if depth > 10 && current < max {
            spawn specialist_type
            log::info("Scaled up {specialist_type} to {current + 1}")
        }
    }

    on_low_utilization(specialist: Specialist, utilization: f64) {
        let current = instances_of(specialist.type).count()
        let min = min_instances[specialist.type]

        if utilization < 0.1 && current > min {
            stop(specialist)
            log::info("Scaled down {specialist.type} to {current - 1}")
        }
    }
}
```

---

## Model Registry

Specialists declare their capabilities for discovery:

```simplex
specialist CodeReviewer {
    model: "codellama-7b",

    // Capability declarations
    capabilities: [
        Capability::CodeReview,
        Capability::BugDetection,
        Capability::SecurityAudit
    ],

    // Supported languages
    languages: ["rust", "python", "javascript", "simplex"],

    // Quality metrics
    metrics: {
        avg_latency: Duration::milliseconds(150),
        accuracy: 0.94,
        throughput: 100  // requests per minute
    }
}

// Query the registry
fn find_code_reviewer(language: String) -> Option<Specialist> {
    hive.registry.find(|s|
        s.capabilities.contains(Capability::CodeReview) &&
        s.languages.contains(language)
    )
}
```

---

## Cost Analysis

### Running Costs

| Specialist Config | Instance | Model Size | Spot Cost/hr | Monthly |
|------------------|----------|------------|--------------|---------|
| CPU inference | t4g.medium | 7B | $0.008 | ~$6 |
| CPU inference | t4g.large | 13B | $0.016 | ~$12 |
| GPU inference | g4dn.xlarge | 7B | $0.16 | ~$115 |
| GPU inference | g4dn.xlarge | 13B | $0.16 | ~$115 |

### Reference Architecture

**Small hive (5 specialists):**
- 5x t4g.medium (7B models on CPU)
- 1x t4g.small (router + embedding)
- S3 for vector storage
- **Total: ~$35/month**

**Medium hive (10 specialists):**
- 8x t4g.medium (7B models)
- 2x t4g.large (13B models)
- 1x t4g.medium (router)
- S3 + ElastiCache
- **Total: ~$85/month**

**High-performance hive (10 specialists):**
- 10x g4dn.xlarge (GPU inference)
- 1x t4g.medium (router)
- S3 + ElastiCache
- **Total: ~$1,200/month**

### Comparison to LLM APIs

| Workload | CHAI Hive | GPT-4 API | Savings |
|----------|-----------|-----------|---------|
| 100K requests/month | ~$35 | ~$300 | 88% |
| 1M requests/month | ~$85 | ~$3,000 | 97% |
| 10M requests/month | ~$1,200 | ~$30,000 | 96% |

---

## Best Practices

### 1. Right-Size Your Specialists

```simplex
// Good: Focused specialist
specialist SentimentAnalyzer {
    model: "distilbert-sentiment",  // Small, focused model
    domain: "sentiment analysis"
}

// Avoid: Overloaded specialist
specialist DoEverything {
    model: "llama2-70b",  // Overkill for most tasks
    domain: "everything"
}
```

### 2. Use Appropriate Models

| Task Type | Recommended Model Size |
|-----------|----------------------|
| Classification | 1B-3B |
| Extraction | 3B-7B |
| Summarization | 7B |
| Generation | 7B-13B |
| Complex reasoning | 13B+ |

### 3. Cache Aggressively

```simplex
specialist CachedSummarizer {
    cache: LRUCache(max_size: 10000),

    receive Summarize(text: String) -> String {
        let cache_key = hash(text)

        if let Some(cached) = cache.get(cache_key) {
            return cached
        }

        let result = infer("Summarize: {text}")
        cache.set(cache_key, result)
        result
    }
}
```

### 4. Graceful Degradation

```simplex
hive ResilientHive {
    on_specialist_failure(specialist: Specialist, error: Error) {
        log::warn("{specialist.name} failed: {error}")

        // Try fallback
        if let Some(fallback) = find_fallback(specialist.domain) {
            route_to(fallback, current_task)
        } else {
            // Degrade gracefully
            respond_with_error("Service temporarily unavailable")
        }
    }
}
```

---

## Summary

| Concept | Purpose |
|---------|---------|
| `specialist` | Actor wrapping an SLM with defined capabilities |
| `hive` | Supervisor coordinating multiple specialists |
| `infer` | Call the specialist's underlying model |
| `router` | Direct tasks to appropriate specialists |
| `ensemble` | Combine multiple specialist outputs |
| `consensus` | Resolve disagreements between specialists |
| `memory` | Share context across specialists |

The Cognitive Hive architecture enables Simplex programs to leverage AI at scale with:
- **Cost efficiency**: Run on commodity hardware
- **Low latency**: Local inference, no API round-trips
- **High reliability**: Fault-tolerant specialist swarm
- **Flexibility**: Add, remove, upgrade specialists independently
- **Control**: Fine-tune models for your exact domain

---

*"Many small minds, working in concert, can surpass a single great one."*

---

*See also: [AI Integration](07-ai-integration.md) | [Swarm Computing](06-swarm-computing.md) | [Cost Optimization](08-cost-optimization.md)*
