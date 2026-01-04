# Simplex Architecture

**Version 0.2.0**

---

## Overview

Simplex programs flow through three stages: source code, bytecode, and distributed execution. This document describes the high-level architecture.

---

## System Architecture

```
                                SIMPLEX ARCHITECTURE

    +------------------------------------------------------------------+
    |                                                                  |
    |                        Simplex Source Code                       |
    |              (Lightweight syntax, actor-based, AI primitives)    |
    |                                                                  |
    +--------------------------------+---------------------------------+
                                     |
                                     | Compilation
                                     v
    +------------------------------------------------------------------+
    |                                                                  |
    |                    Simplex Bytecode (SBC)                        |
    |                                                                  |
    |   - Portable across all SVM implementations                      |
    |   - Content-addressed (functions identified by hash)             |
    |   - Includes continuation points for resumption                  |
    |   - Serializable for distribution across swarm                   |
    |                                                                  |
    +--------------------------------+---------------------------------+
                                     |
                                     | Execution
                                     v
    +------------------------------------------------------------------+
    |                                                                  |
    |                  Simplex Virtual Machine (SVM)                   |
    |                                                                  |
    |   +------------------+  +------------------+  +---------------+  |
    |   |     Actor        |  |      Actor       |  |    Actor      |  |
    |   |    (Worker)      |  |     (Worker)     |  |   (Worker)    |  |
    |   |                  |  |                  |  |               |  |
    |   | - Owns state     |  | - Owns state     |  | - Owns state  |  |
    |   | - Processes msgs |  | - Processes msgs |  | - Processes   |  |
    |   | - Can die/resume |  | - Can die/resume |  | - Can die     |  |
    |   +--------+---------+  +--------+---------+  +-------+-------+  |
    |            |                     |                    |          |
    |   +--------v---------------------v--------------------v-------+  |
    |   |                                                           |  |
    |   |              Message Bus / Work Queue                     |  |
    |   |         (Async, persistent, exactly-once delivery)        |  |
    |   |                                                           |  |
    |   +-----------------------------------------------------------+  |
    |                                                                  |
    |   +-----------------------------------------------------------+  |
    |   |                                                           |  |
    |   |                   Checkpoint Store                        |  |
    |   |        (Persistent actor state for resume on death)       |  |
    |   |                                                           |  |
    |   +-----------------------------------------------------------+  |
    |                                                                  |
    |   +-----------------------------------------------------------+  |
    |   |                                                           |  |
    |   |                  AI Inference Pool                        |  |
    |   |    (Shared GPU resources, batched inference, warm models) |  |
    |   |                                                           |  |
    |   +-----------------------------------------------------------+  |
    |                                                                  |
    +--------------------------------+---------------------------------+
                                     |
                                     | Swarm Protocol
                                     v
    +------------------------------------------------------------------+
    |                                                                  |
    |                        Swarm Network                             |
    |                                                                  |
    |   +-------------+    +-------------+    +-------------+          |
    |   |   SVM Node  |    |   SVM Node  |    |   SVM Node  |          |
    |   |   (Peer)    |<-->|   (Peer)    |<-->|   (Peer)    |          |
    |   +-------------+    +-------------+    +-------------+          |
    |                                                                  |
    |   - Work migration and load balancing                           |
    |   - Consensus for coordination                                  |
    |   - Shared checkpoint storage                                    |
    |   - Distributed AI inference pool                               |
    |                                                                  |
    +------------------------------------------------------------------+
```

---

## Component Overview

### Simplex Compiler

Transforms source code into portable bytecode.

| Responsibility | Description |
|----------------|-------------|
| Parsing | Source code to AST |
| Type checking | Static type inference and validation |
| Ownership analysis | Borrow checker (Rust-style) |
| Optimization | Dead code elimination, inlining |
| Code generation | AST to Simplex Bytecode (SBC) |
| Content addressing | Hash each function for caching |

### Simplex Bytecode (SBC)

Portable intermediate representation.

| Property | Description |
|----------|-------------|
| Stack-based | Simple execution model |
| Content-addressed | Functions identified by hash |
| Continuation points | Marked locations for checkpointing |
| Self-describing | Type information embedded for serialization |

### Simplex Virtual Machine (SVM)

Executes bytecode and manages actors.

| Component | Responsibility |
|-----------|----------------|
| Bytecode Executor | Stack-based execution, optional JIT |
| Actor Scheduler | Fair scheduling, work stealing |
| Memory Manager | Ownership tracking, region allocation |
| Checkpoint Manager | Incremental snapshots, async persistence |
| Message Router | Local/remote delivery, dead letters |
| Swarm Client | Peer discovery, work migration |

### Swarm Network

Coordinates multiple SVM nodes.

| Component | Responsibility |
|-----------|----------------|
| Coordinator | Raft-based consensus for cluster state |
| Work Distributor | Load balancing, actor placement |
| Checkpoint Store | Shared durable storage (S3/Blob) |
| AI Inference Pool | Shared GPU resources, request batching |

---

## Data Flow

### Message Processing

```
    1. Message arrives at SVM
           |
           v
    2. Router determines target actor
           |
           +--> Local actor: direct delivery
           |
           +--> Remote actor: forward to correct node
           |
           v
    3. Actor receives message
           |
           v
    4. Actor processes message (may checkpoint)
           |
           v
    5. Actor sends response/new messages
```

### Checkpoint Flow

```
    1. Actor calls checkpoint()
           |
           v
    2. SVM serializes actor state
           |
           v
    3. Write to local buffer (fast)
           |
           v
    4. Async upload to durable storage (S3)
           |
           v
    5. Acknowledge checkpoint complete
```

### Actor Migration

```
    1. Coordinator decides to migrate actor
           |
           v
    2. Source node pauses actor
           |
           v
    3. Final checkpoint captured
           |
           v
    4. Checkpoint transferred to target node
           |
           v
    5. Target node restores actor state
           |
           v
    6. Actor resumes on target node
           |
           v
    7. Routing tables updated
```

---

## Deployment Topologies

### Single Node (Development)

```
    +------------------+
    |       SVM        |
    |  +------------+  |
    |  | Actor Pool |  |
    |  +------------+  |
    |  | Local Store|  |
    |  +------------+  |
    +------------------+
```

### Small Cluster

```
    +-------------+    +-------------+    +-------------+
    |    SVM 1    |<-->|    SVM 2    |<-->|    SVM 3    |
    | (Coord+Work)|    |   (Work)    |    |   (Work)    |
    +-------------+    +-------------+    +-------------+
           |                 |                 |
           +-----------------+-----------------+
                             |
                    +--------v--------+
                    | Shared Storage  |
                    +-----------------+
```

### Enterprise Swarm

See [Cost Optimization](08-cost-optimization.md) for detailed enterprise architecture.

---

## Next Steps

- [Design Decisions](03-design-decisions.md): Why we made these choices
- [Virtual Machine](05-virtual-machine.md): SVM internals
- [Swarm Computing](06-swarm-computing.md): Distributed execution details
