# Simplex Swarm Computing

**Version 0.1.0**

A Simplex swarm is a collection of SVM nodes that cooperate to execute distributed programs.

---

## Overview

The swarm provides:
- Automatic work distribution
- Fault tolerance through redundancy
- Transparent actor migration
- Shared checkpoint storage
- Pooled AI inference

---

## Swarm Topology

```
    +------------------------------------------------------------------+
    |                         Simplex Swarm                            |
    |                                                                  |
    |                      +------------------+                        |
    |                      |  Coordinator     |                        |
    |                      |  (Raft Leader)   |                        |
    |                      +--------+---------+                        |
    |                               |                                  |
    |         +---------------------+---------------------+            |
    |         |                     |                     |            |
    |   +-----v------+       +------v-----+       +------v-----+      |
    |   | SVM Node 1 |       | SVM Node 2 |       | SVM Node 3 |      |
    |   | (Compute)  |<----->| (Compute)  |<----->| (Compute)  |      |
    |   +------------+       +------------+       +------------+      |
    |         |                     |                     |            |
    |         +---------------------+---------------------+            |
    |                               |                                  |
    |                      +--------v---------+                        |
    |                      |  Shared Storage  |                        |
    |                      |  (Checkpoints)   |                        |
    |                      +------------------+                        |
    |                                                                  |
    |   +----------------------------------------------------------+  |
    |   |                  AI Inference Pool                        |  |
    |   |  +-------------+  +-------------+  +-------------+        |  |
    |   |  | GPU Node 1  |  | GPU Node 2  |  | GPU Node 3  |        |  |
    |   |  | (Inference) |  | (Inference) |  | (Inference) |        |  |
    |   |  +-------------+  +-------------+  +-------------+        |  |
    |   +----------------------------------------------------------+  |
    |                                                                  |
    +------------------------------------------------------------------+
```

---

## Coordination

### Raft Consensus

The swarm uses Raft for coordination:
- Leader election among coordinator-eligible nodes
- Replicated log for cluster state
- Consistent view of actor placement

```
    Raft State Machine:

    +------------------------------------------+
    | Cluster State                            |
    +------------------------------------------+
    | Nodes:                                   |
    |   node-1: healthy, 45 actors, 60% CPU    |
    |   node-2: healthy, 52 actors, 75% CPU    |
    |   node-3: healthy, 38 actors, 45% CPU    |
    +------------------------------------------+
    | Actor Placements:                        |
    |   actor-a -> node-1                      |
    |   actor-b -> node-2                      |
    |   actor-c -> node-1                      |
    +------------------------------------------+
    | Routing Table:                           |
    |   actor-a: node-1:8080                   |
    |   actor-b: node-2:8080                   |
    +------------------------------------------+
```

### Coordinator Responsibilities

| Task | Description |
|------|-------------|
| Node membership | Track which nodes are in the swarm |
| Health monitoring | Detect node failures via heartbeats |
| Actor placement | Decide which node hosts each actor |
| Load balancing | Migrate actors from overloaded nodes |
| Checkpoint coordination | Track checkpoint locations |

---

## Actor Placement

### Placement Algorithm

When spawning an actor, the coordinator considers:

1. **Resource requirements**: CPU, memory, GPU
2. **Locality hints**: Colocate with related actors
3. **Load balancing**: Prefer less-loaded nodes
4. **Fault domains**: Spread replicas across failure zones

```simplex
// Placement hints (optional)
let processor = spawn OrderProcessor with {
    placement: Locality(database_actor),  // Near database
    resources: Resources(cpu: 2, memory: "1GB"),
    replicas: 3  // For high availability
}

// Default placement (automatic)
let worker = spawn Worker  // Coordinator decides
```

### Placement Strategies

```
    Strategy: BinPacking
    - Fill nodes to capacity before using new nodes
    - Minimizes number of active nodes
    - Good for cost optimization

    Strategy: Spread
    - Distribute evenly across all nodes
    - Maximizes fault tolerance
    - Good for high availability

    Strategy: Locality
    - Colocate actors that communicate frequently
    - Minimizes network latency
    - Good for performance
```

---

## Message Routing

### Local Delivery

When sender and receiver are on the same node:

```
    Actor A                    Actor B
    +--------+                 +--------+
    | send() |---------------->| inbox  |
    +--------+                 +--------+
         Direct memory transfer (zero-copy possible)
```

### Remote Delivery

When sender and receiver are on different nodes:

```
    Node 1                              Node 2
    +--------+                          +--------+
    | Actor A|                          | Actor B|
    | send() |                          | inbox  |
    +---+----+                          +---^----+
        |                                   |
        v                                   |
    +--------+      Network             +--------+
    | Router |------------------------->| Router |
    +--------+      (serialized msg)    +--------+
```

### Routing Table

Each node maintains a routing table:

```
    +------------------+------------------+------------------+
    | Actor ID         | Node             | Status           |
    +------------------+------------------+------------------+
    | actor-abc123     | node-1:8080      | Active           |
    | actor-def456     | node-2:8080      | Active           |
    | actor-ghi789     | node-1:8080      | Migrating        |
    +------------------+------------------+------------------+
```

### Message Delivery Guarantees

| Guarantee | Description |
|-----------|-------------|
| At-least-once | Messages may be delivered multiple times (actor must be idempotent) |
| Ordering | Messages from A to B arrive in send order |
| Persistence | Undelivered messages survive node failures |

---

## Work Migration

### When to Migrate

Migration triggers:
- Node overloaded (CPU > 80%)
- Node failing (health check failures)
- Spot instance termination warning
- Manual rebalancing request

### Migration Process

```
    Migration Flow:

    Node A (source)                  Node B (target)
    +------------------+             +------------------+
    |  Actor X         |             |                  |
    |  [running]       |             |                  |
    +--------+---------+             +------------------+
             |
             | 1. Coordinator initiates migration
             |
             v
    +------------------+             +------------------+
    |  Actor X         |             |                  |
    |  [pausing]       |             |                  |
    |  - drain inbox   |             |                  |
    |  - finish msg    |             |                  |
    +--------+---------+             +------------------+
             |
             | 2. Final checkpoint
             |
             v
    +------------------+             +------------------+
    |  Actor X         |  -------->  | Checkpoint       |
    |  [checkpointed]  |  (storage)  | received         |
    +--------+---------+             +--------+---------+
             |                                |
             | 3. Transfer complete           | 4. Restore
             |                                |
             v                                v
    +------------------+             +------------------+
    |                  |             |  Actor X         |
    |  (X removed)     |             |  [resuming]      |
    +------------------+             +--------+---------+
                                              |
                                              | 5. Update routing
                                              |
                                              v
                                     +------------------+
                                     |  Actor X         |
                                     |  [running]       |
                                     +------------------+
```

### Migration Timing

| Phase | Typical Duration |
|-------|------------------|
| Pause and drain | 10-100ms |
| Checkpoint | 50-500ms |
| Transfer | 100-1000ms |
| Restore | 50-200ms |
| Route update | 10-50ms |
| **Total** | **200ms - 2s** |

---

## Fault Recovery

### Failure Detection

```
    Heartbeat Protocol:

    Coordinator                    Worker Node
    +------------+                 +------------+
    |            |  heartbeat req  |            |
    |            |---------------->|            |
    |            |  heartbeat resp |            |
    |            |<----------------|            |
    +------------+                 +------------+

    Timeout: 3 missed heartbeats = node failure
    Heartbeat interval: 1 second
    Failure detection time: ~3-5 seconds
```

### Recovery Process

When a node fails:

```
    1. Coordinator detects failure (missed heartbeats)
           |
           v
    2. Mark node as failed
           |
           v
    3. For each actor on failed node:
           |
           +---> Retrieve last checkpoint from storage
           |
           +---> Select new host node
           |
           +---> Spawn actor on new node
           |
           +---> Restore from checkpoint
           |
           +---> Update routing table
           |
           +---> Replay unacknowledged messages
           |
           v
    4. Remove failed node from cluster
```

### Actor Lifecycle Hooks

```simplex
actor PaymentProcessor {
    var pending_payments: Map<PaymentId, Payment> = {}

    receive ProcessPayment(payment: Payment) {
        pending_payments.insert(payment.id, payment)
        checkpoint()  // Ensure payment is persisted

        let result = process_with_provider(payment)

        match result {
            Ok(_) => {
                pending_payments.remove(payment.id)
                checkpoint()
            },
            Err(e) => {
                // If we crash here, on resume we'll see pending payment
                log_error(e)
            }
        }
    }

    on_resume() {
        // Handle any payments that were in-flight when we crashed
        for (id, payment) in pending_payments {
            log::warn("Retrying payment {id} after recovery")
            send(self, ProcessPayment(payment))
        }
    }
}
```

---

## Spot Instance Handling

### Termination Warning

Cloud providers give advance warning before terminating spot instances:
- AWS: 2 minutes
- Azure: 30 seconds
- GCP: 30 seconds

### Graceful Shutdown

```simplex
// Runtime automatically handles spot termination
actor Worker {
    var state: WorkerState

    // Called by runtime on spot termination warning
    on_termination_warning() {
        log::warn("Spot termination in 2 minutes, checkpointing...")
        checkpoint()  // Immediate checkpoint
        drain()       // Stop accepting new work
        // Runtime will migrate actor to available node
    }
}
```

### Spot Architecture

```
    Recommended Spot/On-Demand Mix:

    +------------------------------------------------------------------+
    |                                                                  |
    |   Spot Instances (90% of compute)                                |
    |   +----------------------------------------------------------+   |
    |   | Worker nodes - can be terminated any time                |   |
    |   | - All stateless processing                               |   |
    |   | - Frequent checkpointing                                 |   |
    |   | - 60-90% cost savings                                    |   |
    |   +----------------------------------------------------------+   |
    |                                                                  |
    |   On-Demand Instances (10% of compute)                           |
    |   +----------------------------------------------------------+   |
    |   | Coordinator nodes - never terminated                     |   |
    |   | - Raft consensus                                         |   |
    |   | - Message queue persistence                              |   |
    |   | - Critical actors                                        |   |
    |   +----------------------------------------------------------+   |
    |                                                                  |
    +------------------------------------------------------------------+
```

---

## Swarm Scaling

### Scale Out

```
    1. Workload increases (queue depth rises)
           |
           v
    2. Autoscaler requests new nodes
           |
           v
    3. New SVM instances join swarm
           |
           v
    4. Coordinator adds to cluster
           |
           v
    5. Actors migrated to new nodes for balance
```

### Scale In

```
    1. Workload decreases (low utilization)
           |
           v
    2. Autoscaler marks nodes for removal
           |
           v
    3. Actors migrated off marked nodes
           |
           v
    4. Empty nodes leave swarm
           |
           v
    5. Instances terminated
```

### Scaling Configuration

```simplex
config AutoScaling {
    // Scale triggers
    scale_up: {
        metric: QueueDepth,
        threshold: 1000,
        cooldown: Duration::minutes(2)
    },

    scale_down: {
        metric: CpuUtilization,
        threshold: 20,  // Below 20%
        cooldown: Duration::minutes(10)
    },

    // Limits
    limits: {
        min_nodes: 3,
        max_nodes: 100,
        prefer_spot: true
    }
}
```

---

## Network Topology

### Same Availability Zone

For cost efficiency, all swarm nodes should be in the same AZ:

```
    +------------------------------------------------------------------+
    |                     Single Availability Zone                     |
    |                     (Free intra-AZ traffic)                      |
    |                                                                  |
    |   +-------------+  +-------------+  +-------------+              |
    |   | SVM Node    |  | SVM Node    |  | SVM Node    |              |
    |   |             |<->|            |<->|             |   $0.00/GB  |
    |   +-------------+  +-------------+  +-------------+              |
    |         |                |                |                      |
    |         +----------------+----------------+                      |
    |                          |                                       |
    |                  +-------v-------+                               |
    |                  | S3 (same AZ)  |                    $0.00/GB   |
    |                  +---------------+                               |
    |                                                                  |
    +------------------------------------------------------------------+
```

### Multi-AZ (High Availability)

For critical workloads requiring zone redundancy:

```
    +----------------------+    +----------------------+
    |    AZ 1              |    |    AZ 2              |
    |  +--------------+    |    |    +--------------+  |
    |  | SVM (coord)  |<---|----|--->| SVM (coord)  |  |
    |  +--------------+    |    |    +--------------+  |
    |  | SVM (worker) |    |    |    | SVM (worker) |  |
    |  +--------------+    |    |    +--------------+  |
    +----------------------+    +----------------------+
              |                           |
              +-----------+---------------+
                          |
                  +-------v-------+
                  | S3 (regional) |
                  +---------------+

    Cost: ~$0.01/GB cross-AZ traffic
```

---

## CLI Commands

```bash
# Deploy a swarm
simplex swarm create --name production --nodes 5

# Add nodes
simplex swarm scale --nodes 10

# Check status
simplex swarm status

# Output:
# Swarm: production
# Nodes: 10 (8 healthy, 2 starting)
# Actors: 1,234
# Messages/sec: 45,678
# Checkpoints/min: 890

# View actor distribution
simplex swarm actors

# Output:
# Node          Actors    CPU    Memory
# node-1        145       67%    512MB
# node-2        132       58%    480MB
# ...

# Migrate actor manually
simplex swarm migrate actor-abc123 --to node-5

# Drain a node (graceful shutdown)
simplex swarm drain node-3

# View logs
simplex swarm logs --follow
simplex swarm logs --node node-1 --actor actor-abc123
```

---

## Next Steps

- [AI Integration](07-ai-integration.md): Distributed AI inference
- [Cost Optimization](08-cost-optimization.md): Swarm deployment costs
- [Virtual Machine](05-virtual-machine.md): Single-node details
