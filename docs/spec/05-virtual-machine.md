# Simplex Virtual Machine

**Version 0.2.0**

The Simplex Virtual Machine (SVM) executes Simplex bytecode and manages distributed actor systems.

---

## Overview

The SVM is designed for:
- Lightweight deployment (runs on 512MB instances)
- Fault tolerance (actors checkpoint and resume)
- Distribution (actors migrate between nodes)
- AI integration (connects to inference pools)

---

## Components

```
    +------------------------------------------------------------------+
    |                     Simplex Virtual Machine                       |
    |                                                                  |
    |  +---------------------------+  +----------------------------+   |
    |  |     Bytecode Executor     |  |      Actor Scheduler       |   |
    |  |                           |  |                            |   |
    |  | - Stack-based execution   |  | - Fair scheduling          |   |
    |  | - JIT compilation (opt)   |  | - Priority queues          |   |
    |  | - Continuation support    |  | - Work stealing            |   |
    |  +---------------------------+  +----------------------------+   |
    |                                                                  |
    |  +---------------------------+  +----------------------------+   |
    |  |    Memory Manager         |  |    Checkpoint Manager      |   |
    |  |                           |  |                            |   |
    |  | - Ownership tracking      |  | - Incremental snapshots    |   |
    |  | - Region-based alloc      |  | - Async persistence        |   |
    |  | - Zero-copy messaging     |  | - Resume coordination      |   |
    |  +---------------------------+  +----------------------------+   |
    |                                                                  |
    |  +---------------------------+  +----------------------------+   |
    |  |    Message Router         |  |    Swarm Client            |   |
    |  |                           |  |                            |   |
    |  | - Local delivery          |  | - Peer discovery           |   |
    |  | - Remote forwarding       |  | - Work migration           |   |
    |  | - Dead letter handling    |  | - Consensus participation  |   |
    |  +---------------------------+  +----------------------------+   |
    |                                                                  |
    +------------------------------------------------------------------+
```

---

## Bytecode Format

Simplex bytecode (.sbc files) is a portable, serializable representation.

### File Structure

```
    +------------------------------------------------------------------+
    |                    Simplex Bytecode File (.sbc)                   |
    +------------------------------------------------------------------+
    | Header                                                           |
    |   - Magic number: 0x53504C58 ("SPLX")                           |
    |   - Version: u16                                                 |
    |   - Flags: u32                                                   |
    |   - Content hash: [u8; 32]                                       |
    +------------------------------------------------------------------+
    | Content Hash Table                                               |
    |   - Function hash -> offset mapping                              |
    |   - Enables content-addressed lookup                             |
    |   - Format: [(hash: [u8; 32], offset: u32, size: u32), ...]     |
    +------------------------------------------------------------------+
    | Type Definitions                                                 |
    |   - Struct layouts with field offsets                            |
    |   - Actor message type schemas                                   |
    |   - Serialization metadata                                       |
    +------------------------------------------------------------------+
    | Function Bodies                                                  |
    |   - Stack-based instructions                                     |
    |   - Continuation points marked with CHECKPOINT_POINT             |
    |   - Local variable table                                         |
    |   - Exception handling tables                                    |
    +------------------------------------------------------------------+
    | Constant Pool                                                    |
    |   - String literals (deduplicated)                               |
    |   - Numeric constants                                            |
    |   - Type metadata references                                     |
    +------------------------------------------------------------------+
    | Debug Info (optional)                                            |
    |   - Source file mappings                                         |
    |   - Line number tables                                           |
    |   - Local variable names                                         |
    +------------------------------------------------------------------+
```

### Content Addressing

Every function is identified by the SHA-256 hash of its bytecode:

```
    Function lookup:

    1. Caller has function hash: 0x7f3a8b...
    2. Check local cache for hash
       - Found: Use cached bytecode
       - Not found: Fetch from swarm/registry
    3. Execute function
    4. Cache for future use
```

This enables:
- Perfect caching (same hash = identical behavior)
- Lazy code loading (fetch only what's needed)
- Seamless actor migration (checkpoint contains hashes)

---

## Instruction Set

### Stack Operations

```
    PUSH_CONST <idx>      Push constant from pool onto stack
    PUSH_LOCAL <slot>     Push local variable onto stack
    STORE_LOCAL <slot>    Pop stack, store in local variable
    POP                   Discard top of stack
    DUP                   Duplicate top of stack
    SWAP                  Swap top two stack items
    ROT                   Rotate top three items (a b c -> b c a)
```

### Arithmetic

```
    ADD                   Pop two, push sum
    SUB                   Pop two, push difference
    MUL                   Pop two, push product
    DIV                   Pop two, push quotient
    MOD                   Pop two, push remainder
    NEG                   Negate top of stack

    // Floating point
    FADD, FSUB, FMUL, FDIV

    // Comparison (push Bool)
    EQ, NE, LT, LE, GT, GE
```

### Control Flow

```
    JUMP <offset>         Unconditional jump
    JUMP_IF <offset>      Jump if top is true (pops condition)
    JUMP_UNLESS <offset>  Jump if top is false (pops condition)

    CALL <hash>           Call function by content hash
    CALL_LOCAL <slot>     Call function reference in local
    RETURN                Return from function

    // Pattern matching
    MATCH_TAG <tag>       Check if top matches enum variant
    MATCH_TYPE <type>     Check if top matches type
```

### Actor Operations

```
    SPAWN <actor_hash>    Create new actor, push ActorRef
    SEND                  Pop (ref, msg), send async message
    ASK                   Pop (ref, msg), send and await response

    RECEIVE <handler>     Begin message handler block
    REPLY                 Pop value, send as response to current ask

    CHECKPOINT            Trigger state checkpoint
    CHECKPOINT_POINT      Mark valid checkpoint location

    SELF                  Push reference to current actor
```

### AI Operations

```
    AI_COMPLETE           Pop prompt, push completion string
    AI_COMPLETE_OPTS      Pop (prompt, options), push completion
    AI_EMBED              Pop text, push embedding vector
    AI_EMBED_BATCH        Pop texts list, push embeddings list
    AI_EXTRACT <type>     Pop input, push extracted typed value
    AI_STREAM_START       Begin streaming completion
    AI_STREAM_NEXT        Get next chunk (or None if done)
```

### Memory Operations

```
    ALLOC <type>          Allocate memory for type, push reference
    LOAD_FIELD <offset>   Pop reference, push field value
    STORE_FIELD <offset>  Pop (reference, value), store field

    // Ownership
    MOVE                  Transfer ownership (invalidates source)
    BORROW                Create shared borrow
    BORROW_MUT            Create exclusive borrow
    DROP                  Release ownership, deallocate

    // Collections
    LIST_NEW              Create empty list
    LIST_PUSH             Pop (list, item), push to list
    LIST_GET              Pop (list, index), push item
    MAP_NEW               Create empty map
    MAP_INSERT            Pop (map, key, value), insert
    MAP_GET               Pop (map, key), push Option<value>
```

### Example Bytecode

```simplex
// Source
fn add(a: i64, b: i64) -> i64 {
    a + b
}
```

```
// Bytecode
add:                      ; Hash: 0x7f3a8b...
    PUSH_LOCAL 0          ; Push 'a'
    PUSH_LOCAL 1          ; Push 'b'
    ADD                   ; Add them
    RETURN                ; Return result
```

```simplex
// Source
actor Counter {
    var count: i64 = 0

    receive Increment {
        count += 1
        checkpoint()
    }
}
```

```
// Bytecode
Counter::Increment:       ; Hash: 0x9c2d1e...
    RECEIVE Increment     ; Begin handler
    CHECKPOINT_POINT      ; Valid checkpoint location
    PUSH_LOCAL 0          ; Push self.count
    PUSH_CONST 0          ; Push 1
    ADD                   ; count + 1
    STORE_FIELD 0         ; Store to self.count
    CHECKPOINT            ; Persist state
    RETURN
```

---

## Actor Scheduler

### Scheduling Model

Each SVM instance runs one scheduler managing multiple actors:

```
    +----------------------------------------------------------+
    |                     Actor Scheduler                       |
    |                                                          |
    |   Run Queue (priority-ordered):                          |
    |   +--------------------------------------------------+   |
    |   | [Actor A: 3 msgs] [Actor B: 1 msg] [Actor C: 5] |   |
    |   +--------------------------------------------------+   |
    |                                                          |
    |   Waiting (blocked on async):                            |
    |   +--------------------------------------------------+   |
    |   | [Actor D: await AI] [Actor E: await ask]         |   |
    |   +--------------------------------------------------+   |
    |                                                          |
    |   Idle (no messages):                                    |
    |   +--------------------------------------------------+   |
    |   | [Actor F] [Actor G] [Actor H]                    |   |
    |   +--------------------------------------------------+   |
    |                                                          |
    +----------------------------------------------------------+
```

### Fair Scheduling

- Each actor processes one message before yielding
- Round-robin among actors with pending messages
- Priority boost for actors with response waiters
- Work stealing from overloaded actors

### Message Processing Cycle

```
    1. Select next actor from run queue
           |
           v
    2. Dequeue one message
           |
           v
    3. Execute message handler
           |
           +---> Handler completes: Actor returns to run queue (if more messages)
           |                        or idle queue (if no messages)
           |
           +---> Handler awaits: Actor moves to waiting queue
           |                     Will resume when await completes
           |
           +---> Handler checkpoints: State persisted, execution continues
           |
           +---> Handler crashes: Supervisor notified, actor restarted
```

---

## Memory Manager

### Ownership Tracking

The memory manager enforces ownership at runtime (complementing compile-time checks):

```
    Ownership Table:
    +------------------+------------------+------------------+
    | Address          | Owner            | Borrow State     |
    +------------------+------------------+------------------+
    | 0x1000           | Actor A, local 3 | None             |
    | 0x1008           | Actor A, local 5 | Borrowed (2)     |
    | 0x1010           | Actor B, local 1 | Mutable borrow   |
    +------------------+------------------+------------------+
```

### Region-Based Allocation

Each actor has its own memory region:
- Fast allocation within region
- Bulk deallocation when actor dies
- No cross-actor memory sharing (messages are copied)

```
    Actor Memory Regions:

    +------------------+    +------------------+    +------------------+
    | Actor A Region   |    | Actor B Region   |    | Actor C Region   |
    |                  |    |                  |    |                  |
    | [state]          |    | [state]          |    | [state]          |
    | [locals]         |    | [locals]         |    | [locals]         |
    | [heap objects]   |    | [heap objects]   |    | [heap objects]   |
    |                  |    |                  |    |                  |
    | Free: 45KB       |    | Free: 120KB      |    | Free: 8KB        |
    +------------------+    +------------------+    +------------------+
```

### Zero-Copy Messaging (When Possible)

For large messages within the same SVM:
1. Transfer ownership instead of copying
2. Source actor loses access to data
3. Target actor gains ownership
4. No memory copy required

---

## Checkpoint Manager

### Checkpoint Process

```
    1. Actor calls checkpoint()
           |
           v
    2. Mark checkpoint point in execution
           |
           v
    3. Serialize actor state
       - Current local variables
       - Actor fields (var declarations)
       - Pending message queue
       - Function hashes for code references
           |
           v
    4. Write to local buffer (synchronous, fast)
           |
           v
    5. Async upload to durable storage (S3/Blob)
           |
           v
    6. Confirm checkpoint complete
           |
           v
    7. Continue execution
```

### Checkpoint Format

```
    +------------------------------------------------------------------+
    |                     Actor Checkpoint                              |
    +------------------------------------------------------------------+
    | Header                                                           |
    |   - Actor ID: UUID                                               |
    |   - Actor type hash: [u8; 32]                                    |
    |   - Checkpoint sequence: u64                                     |
    |   - Timestamp: i64                                               |
    +------------------------------------------------------------------+
    | State Snapshot                                                   |
    |   - Field values (serialized)                                    |
    |   - Field type hashes for validation                            |
    +------------------------------------------------------------------+
    | Execution Context                                                |
    |   - Current handler hash                                         |
    |   - Instruction pointer                                          |
    |   - Local variable values                                        |
    |   - Call stack (for nested calls)                                |
    +------------------------------------------------------------------+
    | Message Queue                                                    |
    |   - Pending messages (serialized)                                |
    |   - Message type hashes                                          |
    +------------------------------------------------------------------+
    | Metadata                                                         |
    |   - Supervisor reference                                         |
    |   - Child actor references                                       |
    |   - Checkpoint size: u64                                         |
    |   - Checksum: [u8; 32]                                           |
    +------------------------------------------------------------------+
```

### Resume Process

```
    1. Supervisor requests actor resume
           |
           v
    2. Fetch latest checkpoint from storage
           |
           v
    3. Validate checkpoint integrity (checksum)
           |
           v
    4. Fetch required code by content hash
           |
           v
    5. Allocate actor memory region
           |
           v
    6. Deserialize state into region
           |
           v
    7. Restore execution context
           |
           v
    8. Replay pending messages
           |
           v
    9. Call on_resume() hook
           |
           v
    10. Actor continues normal operation
```

---

## Memory Budget

Target: Run on 512MB instances (t4g.nano, B1ls)

```
    Memory Budget Breakdown:

    +------------------------------------------+
    | Component              | Budget          |
    +------------------------------------------+
    | SVM Runtime            | 10-20 MB        |
    |   - Executor           |   5 MB          |
    |   - Scheduler          |   2 MB          |
    |   - Router             |   3 MB          |
    |   - Other              |   5 MB          |
    +------------------------------------------+
    | Bytecode Cache         | 10-50 MB        |
    |   - Hot functions      |  30 MB          |
    |   - LRU eviction       |                 |
    +------------------------------------------+
    | Actor Heap (total)     | 100-200 MB      |
    |   - Per actor          |   1-10 MB       |
    |   - 20-100 actors      |                 |
    +------------------------------------------+
    | Message Queues         | 10-50 MB        |
    |   - In-flight messages |                 |
    +------------------------------------------+
    | Network Buffers        | 10-20 MB        |
    +------------------------------------------+
    | Checkpoint Buffer      | 50-100 MB       |
    |   - Write buffer       |                 |
    |   - Compression buffer |                 |
    +------------------------------------------+
    | Headroom               | 100-150 MB      |
    +------------------------------------------+
    | TOTAL                  | ~400-500 MB     |
    +------------------------------------------+
```

### Memory-Constrained Mode

For nano instances, SVM can run in constrained mode:
- Interpreter only (no JIT)
- Smaller bytecode cache
- Fewer actors per instance
- More aggressive checkpoint offloading

---

## JIT Compilation (Optional)

On larger instances, SVM can JIT-compile hot bytecode:

```
    JIT Tiers:

    1. Interpreter (always)
       - All code starts here
       - Low memory overhead
       - Acceptable performance

    2. Baseline JIT (optional)
       - Triggered after N executions
       - Fast compilation
       - 2-5x speedup

    3. Optimizing JIT (optional)
       - Triggered for hot loops
       - Slower compilation
       - 10-20x speedup
```

JIT is disabled on:
- Nano instances (insufficient memory)
- Short-lived actors (not worth compilation cost)
- Infrequently called code

---

## Next Steps

- [Swarm Computing](06-swarm-computing.md): Multi-node operation
- [Cost Optimization](08-cost-optimization.md): Deployment sizing
- [Architecture](02-architecture.md): System overview
