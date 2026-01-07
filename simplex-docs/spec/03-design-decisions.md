# Simplex Design Decisions

**Version 0.1.0**

This document captures key technical decisions and their rationale.

---

## Execution Model: Actor-Based

**Decision**: Actor-based execution over dataflow.

**Rationale**: Actors provide a natural model for real-world entity interactions. Each actor:
- Encapsulates state
- Communicates via asynchronous messages
- Processes one message at a time (no internal concurrency)
- Can supervise child actors

This model maps well to distributed systems where network partitions and partial failures are normal.

**Comparison with alternatives**:

| Model | Strengths | Weaknesses |
|-------|-----------|------------|
| Actor (chosen) | Natural isolation, proven fault-tolerance, models real entities | Can be verbose for pure data transforms |
| Dataflow | Excellent for ETL/pipelines, automatic parallelism | Poor fit for stateful interactions |
| CSP (Go-style) | Simple channel semantics | Channels are location-bound, harder to distribute |

---

## AI Integration: Swarm-Local Inference Pool

**Decision**: AI capabilities exposed as language primitives, executed on dedicated inference nodes within the swarm.

**Rationale**:

External cloud APIs introduce unacceptable latency (50-500ms) for tight integration. Embedding models in every VM wastes memory and prevents resource sharing. A swarm-local inference pool provides:

- Low latency (~1-5ms local network)
- Efficient resource utilization (shared GPU nodes, batched requests)
- Model flexibility (swap models without redeploying VMs)
- Graceful degradation (fallback to external APIs at capacity)

**Architecture**:

```
    Worker VM                         Inference Pool
    +------------+                    +------------------+
    |            |  ai::complete()    |                  |
    |  Actor     +-------------------->  GPU Node        |
    |            |     (~1ms)         |  (Model in VRAM) |
    |            <--------------------+                  |
    +------------+     Response       +------------------+
```

**Language interface**:

```simplex
// Completion - runtime routes to appropriate model
let response = ai::complete(prompt)
let response = ai::complete(prompt, model: "claude-3", temperature: 0.7)

// Embeddings - automatically batched by runtime
let vector = ai::embed(text)
let vectors = ai::embed_batch(texts)

// Structured output - validated against type
let data = ai::extract<InvoiceData>(document)

// Streaming
for chunk in ai::stream(prompt) {
    process(chunk)
}
```

---

## State Handling: Immutable Messages, Mutable Actor State

**Decision**: Messages between actors are immutable and copied. State within an actor may be mutable.

**Rationale**: This hybrid approach provides:

- **Safety**: No shared mutable state between actors eliminates race conditions
- **Ergonomics**: Mutable local state feels natural to imperative programmers
- **Checkpointing**: Actor state is captured at message boundaries (always consistent)
- **Performance**: In-place mutation within actors avoids copying overhead

**Comparison**:

| Approach | Checkpointing | Parallelism | Ergonomics |
|----------|---------------|-------------|------------|
| Pure immutable | Trivial | Automatic | Unfamiliar to most |
| Pure mutable | Complex (delta tracking) | Requires locks | Familiar |
| Hybrid (chosen) | Clean (actor boundaries) | Automatic (actor isolation) | Natural |

**Semantics**:

```simplex
actor BankAccount {
    var balance: i64 = 0  // Mutable, but only this actor can access

    receive Deposit(amount: i64) {
        balance += amount  // Safe mutation - no other actor can see this
        checkpoint()       // Explicit checkpoint after important state change
    }

    receive Transfer(to: ActorRef<BankAccount>, amount: i64) {
        if balance >= amount {
            balance -= amount
            send(to, Deposit(amount))  // Message is immutable copy
        }
    }
}
```

---

## Memory Model: Ownership-Based (Rust-Style)

**Decision**: Ownership and borrowing for memory management, no garbage collector.

**Rationale**:

Garbage collection introduces unpredictable pauses problematic for:
- Real-time message processing
- Consistent checkpoint timing
- Predictable latency in distributed systems

Ownership provides:
- Deterministic deallocation
- Zero-cost abstractions
- Compile-time memory safety
- Clear semantics for distributed serialization

**Core rules**:

1. Every value has exactly one owner
2. When the owner goes out of scope, the value is dropped
3. Values can be borrowed (shared `&` or exclusive `&mut`)
4. Borrows must not outlive the owner

```simplex
fn process(data: String) {          // Takes ownership of data
    let processed = transform(data)  // data moved to transform
    // data is no longer valid here
    save(processed)                  // processed moved to save
}

fn analyze(data: &String) {         // Borrows data (read-only)
    // Can read data, cannot modify or move it
    let len = data.len()
}
```

---

## Type System: Static with Inference

**Decision**: Static typing with aggressive type inference.

**Rationale**:

Dynamic typing is unsafe for distributed systems where:
- Checkpoints must deserialize correctly on any node
- Messages must be understood by receiving actors
- Work can migrate between heterogeneous nodes
- AI output must be validated before use

Static typing provides compile-time guarantees critical for correctness. Type inference eliminates verbosity.

**Developer experience**:

```simplex
// What you write (looks dynamic):
let users = fetch_users()
let active = users.filter(u => u.active)
let names = active.map(u => u.name)

// What compiler sees (fully typed):
let users: List<User> = fetch_users()
let active: List<User> = users.filter((u: User) -> Bool => u.active)
let names: List<String> = active.map((u: User) -> String => u.name)
```

**Handling dynamic AI output**:

```simplex
// AI returns unstructured data - explicit boundary
let raw: Dynamic = ai::complete("Extract entities from: {text}")

// Must parse into known type with error handling
match parse<EntityList>(raw) {
    Ok(entities) => process(entities),
    Err(e) => log_and_retry(e)
}

// Or use typed AI call (runtime validates output)
let entities = ai::extract<EntityList>(text)  // Throws if invalid
```

---

## Content-Addressed Code

**Decision**: Functions identified by the hash of their implementation.

**Rationale**: Borrowed from Unison, this provides:

| Benefit | Description |
|---------|-------------|
| Perfect caching | Same hash = same function = reuse cached results |
| Trivial distribution | Send function hash, receiver fetches if needed |
| No dependency hell | No version conflicts, no "works on my machine" |
| Seamless migration | Actor state includes function hashes, resumes anywhere |

**Example**:

```simplex
// Function foo has hash 0x7f3a8b...
fn foo(x: i64) -> i64 {
    x * 2
}

// If you change the implementation, hash changes
fn foo(x: i64) -> i64 {
    x * 2 + 1  // Now hash is 0x9c2d1e...
}

// Callers reference by hash, not name
// Old callers still use 0x7f3a8b (old behavior)
// New callers use 0x9c2d1e (new behavior)
```

---

## Checkpoint Strategy

**Decision**: Explicit checkpointing at message boundaries with async persistence.

**Rationale**:

| Alternative | Problem |
|-------------|---------|
| Continuous checkpointing | Too expensive, high I/O |
| No checkpointing | Lose all progress on failure |
| Transaction log | Complex, requires replay |
| Explicit (chosen) | Developer controls durability vs performance |

**Implementation**:

```simplex
actor OrderProcessor {
    var orders: Map<OrderId, Order> = {}

    receive ProcessOrder(order: Order) {
        // Before risky operation - save state
        orders.insert(order.id, order)
        checkpoint()  // Durable: order won't be lost

        // Risky operation
        let result = charge_payment(order)

        // After success - update state
        match result {
            Ok(_) => {
                orders.get_mut(order.id).status = Paid
                checkpoint()  // Durable: payment recorded
            },
            Err(e) => {
                // Don't checkpoint - on restart, we'll retry
                log_error(e)
            }
        }
    }
}
```

---

## Summary Table

| Decision | Choice | Alternative Considered | Why Chosen |
|----------|--------|----------------------|------------|
| Execution model | Actor-based | Dataflow, CSP | Fault isolation, natural for stateful entities |
| AI integration | Swarm-local pool | Embedded, external API | Latency + resource efficiency |
| State handling | Immutable messages, mutable actors | Pure immutable, pure mutable | Safety + ergonomics balance |
| Memory model | Ownership | Garbage collection | Predictable latency, clear distribution semantics |
| Type system | Static with inference | Dynamic, gradual | Distributed correctness requires static types |
| Code addressing | Content-addressed | Name-based | Perfect caching, no dependency conflicts |
| Checkpointing | Explicit, async | Continuous, transactional | Developer control over durability/performance |

---

## Next Steps

- [Language Syntax](04-language-syntax.md): See these decisions in action
- [Virtual Machine](05-virtual-machine.md): Implementation details
- [Cost Optimization](08-cost-optimization.md): Deployment considerations
