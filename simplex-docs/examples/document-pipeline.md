# Example: Document Processing Pipeline

**Simplex Version 0.1.0**

A complete example demonstrating actors, supervision, AI integration, and fault tolerance.

---

## Overview

This example implements a distributed document processing system that:
- Ingests documents from multiple sources
- Extracts structured data using AI
- Stores results with fault tolerance
- Scales across a swarm

---

## Architecture

```
    Document Pipeline Architecture:

    +------------------------------------------------------------------+
    |                                                                  |
    |   External Sources                                               |
    |   (API, Files, Queues)                                          |
    |            |                                                     |
    |            v                                                     |
    |   +------------------+                                           |
    |   | DocumentIngester |  Single actor, receives all documents    |
    |   +--------+---------+                                           |
    |            |                                                     |
    |            v                                                     |
    |   +------------------+                                           |
    |   |  ProcessorPool   |  Routes to available processors          |
    |   +--------+---------+                                           |
    |            |                                                     |
    |      +-----+-----+                                               |
    |      |     |     |                                               |
    |      v     v     v                                               |
    |   +----+ +----+ +----+                                           |
    |   | P1 | | P2 | | P3 |  DocumentProcessor actors (parallel)     |
    |   +--+-+ +--+-+ +--+-+                                           |
    |      |     |     |                                               |
    |      +-----+-----+                                               |
    |            |                                                     |
    |            v                                                     |
    |   +------------------+                                           |
    |   | DocumentStorage  |  Persists results, enables search        |
    |   +------------------+                                           |
    |                                                                  |
    +------------------------------------------------------------------+
```

---

## Complete Source Code

```simplex
// file: document_pipeline.sx
//
// A distributed document processing pipeline using Simplex
// Demonstrates: actors, supervision, AI integration, fault tolerance

module document_pipeline

use std::time::{Duration, Instant}
use std::collections::{Map, Queue}
use ai

// ----------------------------------------------------------------------------
// Types
// ----------------------------------------------------------------------------

type DocumentId = String

type Document {
    id: DocumentId,
    source: String,
    content: String,
    received_at: Instant
}

type ProcessedDocument {
    id: DocumentId,
    original: Document,
    summary: String,
    entities: List<Entity>,
    sentiment: Sentiment,
    embedding: Vector<f64, 1536>,
    processed_at: Instant
}

type Entity {
    text: String,
    entity_type: EntityType,
    confidence: f64
}

enum EntityType {
    Person,
    Organization,
    Location,
    Date,
    Money,
    Other
}

enum Sentiment {
    Positive,
    Negative,
    Neutral
}

enum ProcessingError {
    AiUnavailable(String),
    InvalidDocument(String),
    StorageFailure(String)
}

// ----------------------------------------------------------------------------
// Document Ingestion Actor
// ----------------------------------------------------------------------------

actor DocumentIngester {
    var stats: IngestionStats = IngestionStats::new()

    receive Ingest(source: String, content: String) {
        let doc = Document {
            id: generate_id(),
            source: source,
            content: content,
            received_at: Instant::now()
        }

        stats.documents_received += 1

        // Forward to processor pool
        send(processor_pool, Process(doc))

        log::info("Ingested document {doc.id} from {source}")
    }

    receive GetStats -> IngestionStats {
        stats.clone()
    }
}

// ----------------------------------------------------------------------------
// Document Processor Actor
// ----------------------------------------------------------------------------

actor DocumentProcessor {
    var processing: Option<Document> = None
    var processed_count: u64 = 0

    receive Process(doc: Document) {
        processing = Some(doc.clone())
        checkpoint()  // Save state before AI calls

        // Parallel AI operations for efficiency
        let (summary, entities, sentiment, embedding) = await parallel(
            ai::complete("Summarize in 2 sentences: {doc.content}"),
            ai::extract<List<Entity>>(doc.content),
            ai::classify(doc.content, Sentiment::variants()),
            ai::embed(doc.content)
        )

        let processed = ProcessedDocument {
            id: doc.id,
            original: doc,
            summary: summary,
            entities: entities,
            sentiment: sentiment,
            embedding: embedding,
            processed_at: Instant::now()
        }

        // Send to storage
        send(storage, Store(processed))

        processing = None
        processed_count += 1
        checkpoint()

        log::info("Processed document {doc.id}")
    }

    on_resume() {
        // If we crashed while processing, retry
        match processing {
            Some(doc) => {
                log::warn("Resuming interrupted processing for {doc.id}")
                send(self, Process(doc))
            },
            None => {}
        }
    }
}

// ----------------------------------------------------------------------------
// Storage Actor
// ----------------------------------------------------------------------------

actor DocumentStorage {
    var documents: Map<DocumentId, ProcessedDocument> = {}
    var embeddings: Map<DocumentId, Vector<f64, 1536>> = {}

    receive Store(doc: ProcessedDocument) {
        documents.insert(doc.id.clone(), doc.clone())
        embeddings.insert(doc.id.clone(), doc.embedding.clone())
        checkpoint()

        log::info("Stored document {doc.id}")
    }

    receive Get(id: DocumentId) -> Option<ProcessedDocument> {
        documents.get(id)
    }

    receive Search(query: String, limit: u64) -> List<ProcessedDocument> {
        let query_embedding = ai::embed(query)

        let similarities: List<(DocumentId, f64)> = embeddings
            .iter()
            .map((id, emb) => (id, ai::similarity(query_embedding, emb)))
            .collect()

        similarities
            .sort_by(|(_, a), (_, b)| b.cmp(a))  // Descending by similarity
            .take(limit)
            .filter_map(|(id, _)| documents.get(id))
            .collect()
    }

    receive GetStats -> StorageStats {
        StorageStats {
            document_count: documents.len(),
            total_entities: documents.values().map(d => d.entities.len()).sum()
        }
    }
}

// ----------------------------------------------------------------------------
// Processor Pool (load balancing)
// ----------------------------------------------------------------------------

actor ProcessorPool {
    var processors: List<ActorRef<DocumentProcessor>> = []
    var next_processor: usize = 0

    init(size: u64) {
        for i in 0..size {
            let processor = spawn DocumentProcessor
            processors.push(processor)
        }
    }

    receive Process(doc: Document) {
        // Round-robin distribution
        let processor = processors[next_processor]
        next_processor = (next_processor + 1) % processors.len()

        send(processor, Process(doc))
    }
}

// ----------------------------------------------------------------------------
// Supervision Tree
// ----------------------------------------------------------------------------

supervisor DocumentPipeline {
    strategy: OneForOne,
    max_restarts: 10,
    within: Duration::minutes(1),

    children: [
        // Single ingester
        child(DocumentIngester,
              name: "ingester",
              restart: Always),

        // Pool of processors (scales with load)
        child(ProcessorPool(size: 4),
              name: "processor_pool",
              restart: Always),

        // Single storage actor (could be sharded for scale)
        child(DocumentStorage,
              name: "storage",
              restart: Always)
    ]
}

// ----------------------------------------------------------------------------
// API Actor (external interface)
// ----------------------------------------------------------------------------

actor PipelineAPI {
    var pipeline: ActorRef<DocumentPipeline>

    init() {
        pipeline = spawn DocumentPipeline
    }

    // Ingest a new document
    receive IngestDocument(source: String, content: String) -> DocumentId {
        let id = generate_id()
        send(pipeline.child("ingester"), Ingest(source, content))
        id
    }

    // Search processed documents
    receive SearchDocuments(query: String, limit: u64) -> List<ProcessedDocument> {
        ask(pipeline.child("storage"), Search(query, limit))
    }

    // Get a specific document
    receive GetDocument(id: DocumentId) -> Option<ProcessedDocument> {
        ask(pipeline.child("storage"), Get(id))
    }

    // Get pipeline statistics
    receive GetStats -> PipelineStats {
        let ingestion_stats = ask(pipeline.child("ingester"), GetStats)
        let storage_stats = ask(pipeline.child("storage"), GetStats)

        PipelineStats {
            ingestion: ingestion_stats,
            storage: storage_stats
        }
    }
}

// ----------------------------------------------------------------------------
// Main Entry Point
// ----------------------------------------------------------------------------

fn main() {
    // Start the pipeline
    let api = spawn PipelineAPI

    log::info("Document processing pipeline started")

    // Example: Ingest a document
    let doc_id = ask(api, IngestDocument(
        source: "email",
        content: "Meeting scheduled for next Tuesday with John Smith
                  from Acme Corp to discuss the $50,000 contract."
    ))

    log::info("Ingested document: {doc_id}")

    // Wait for processing
    sleep(Duration::seconds(2))

    // Search for related documents
    let results = ask(api, SearchDocuments(
        query: "contract meetings",
        limit: 5
    ))

    for doc in results {
        print("Found: {doc.id}")
        print("  Summary: {doc.summary}")
        print("  Sentiment: {doc.sentiment}")
        print("  Entities: {doc.entities.len()}")
    }

    // Get statistics
    let stats = ask(api, GetStats)
    print("Pipeline stats:")
    print("  Documents received: {stats.ingestion.documents_received}")
    print("  Documents stored: {stats.storage.document_count}")
}

// ----------------------------------------------------------------------------
// Supporting Types
// ----------------------------------------------------------------------------

type IngestionStats {
    documents_received: u64
}

impl IngestionStats {
    fn new() -> IngestionStats {
        IngestionStats { documents_received: 0 }
    }
}

type StorageStats {
    document_count: u64,
    total_entities: u64
}

type PipelineStats {
    ingestion: IngestionStats,
    storage: StorageStats
}

fn generate_id() -> DocumentId {
    uuid::v4().to_string()
}
```

---

## Key Patterns Demonstrated

### 1. Actor-Based Architecture

Each component is an isolated actor:
- **DocumentIngester**: Single entry point, tracks statistics
- **ProcessorPool**: Load balances across workers
- **DocumentProcessor**: Parallel AI processing with checkpointing
- **DocumentStorage**: Persistent storage with semantic search

### 2. Fault Tolerance

```simplex
actor DocumentProcessor {
    var processing: Option<Document> = None  // Track in-flight work

    receive Process(doc: Document) {
        processing = Some(doc.clone())
        checkpoint()  // Persist before risky operations

        // ... AI processing ...

        processing = None
        checkpoint()  // Clear after success
    }

    on_resume() {
        // Retry any interrupted work
        match processing {
            Some(doc) => send(self, Process(doc)),
            None => {}
        }
    }
}
```

### 3. Parallel AI Operations

```simplex
// Run 4 AI operations in parallel
let (summary, entities, sentiment, embedding) = await parallel(
    ai::complete("Summarize: {content}"),
    ai::extract<List<Entity>>(content),
    ai::classify(content, Sentiment::variants()),
    ai::embed(content)
)
```

### 4. Supervision Tree

```simplex
supervisor DocumentPipeline {
    strategy: OneForOne,      // Isolate failures
    max_restarts: 10,         // Circuit breaker
    within: Duration::minutes(1),

    children: [
        child(DocumentIngester, restart: Always),
        child(ProcessorPool(size: 4), restart: Always),
        child(DocumentStorage, restart: Always)
    ]
}
```

### 5. Semantic Search

```simplex
receive Search(query: String, limit: u64) -> List<ProcessedDocument> {
    let query_embedding = ai::embed(query)

    let similarities = embeddings
        .iter()
        .map((id, emb) => (id, ai::similarity(query_embedding, emb)))
        .collect()

    similarities
        .sort_by(|(_, a), (_, b)| b.cmp(a))
        .take(limit)
        .filter_map(|(id, _)| documents.get(id))
        .collect()
}
```

---

## Running the Example

### Local Development

```bash
# Compile
simplex build document_pipeline.sx

# Run locally (single node)
simplex run document_pipeline.sbc

# Output:
# [INFO] Document processing pipeline started
# [INFO] Ingested document: abc-123
# [INFO] Processed document: abc-123
# [INFO] Stored document: abc-123
# Found: abc-123
#   Summary: A meeting is scheduled for Tuesday with John Smith from Acme Corp regarding a $50,000 contract.
#   Sentiment: Neutral
#   Entities: 4
# Pipeline stats:
#   Documents received: 1
#   Documents stored: 1
```

### Distributed Deployment

```bash
# Deploy to AWS with 5 nodes
simplex swarm deploy document_pipeline.sbc \
    --cloud aws \
    --region us-east-1 \
    --nodes 5 \
    --spot-enabled

# Scale processors
simplex swarm scale processor_pool --replicas 10

# Monitor
simplex swarm status
simplex swarm logs --follow
```

### Cost Estimate

For 10,000 documents/day:

| Component | Specification | Daily Cost |
|-----------|--------------|------------|
| Compute | 5x t4g.micro (spot) | $0.30 |
| Coordination | 3x t4g.small | $1.21 |
| AI Inference | 1x g4dn.xlarge (spot) | $3.84 |
| Storage | S3 (~1GB) | $0.02 |
| **Total** | | **~$5.37/day** |

---

## Extending the Example

### Add HTTP API

```simplex
actor HttpServer {
    var api: ActorRef<PipelineAPI>

    init(api: ActorRef<PipelineAPI>) {
        self.api = api
    }

    receive HandleRequest(req: HttpRequest) -> HttpResponse {
        match (req.method, req.path) {
            ("POST", "/documents") => {
                let body = parse_json<IngestRequest>(req.body)
                let id = ask(api, IngestDocument(body.source, body.content))
                HttpResponse::json({"id": id})
            },
            ("GET", "/search") => {
                let query = req.query_param("q")
                let results = ask(api, SearchDocuments(query, limit: 10))
                HttpResponse::json(results)
            },
            _ => HttpResponse::not_found()
        }
    }
}
```

### Add Batch Processing

```simplex
actor BatchIngester {
    receive IngestBatch(docs: List<(String, String)>) {
        // Batch embedding is more efficient
        let embeddings = ai::embed_batch(docs.map((_, content) => content))

        for ((source, content), embedding) in docs.zip(embeddings) {
            let doc = Document {
                id: generate_id(),
                source: source,
                content: content,
                received_at: Instant::now()
            }
            // Process with pre-computed embedding
            send(processor_pool, ProcessWithEmbedding(doc, embedding))
        }
    }
}
```

---

## Next Steps

- [Language Syntax](../spec/04-language-syntax.md): Full syntax reference
- [AI Integration](../spec/07-ai-integration.md): More AI patterns
- [Cost Optimization](../spec/08-cost-optimization.md): Deployment costs
