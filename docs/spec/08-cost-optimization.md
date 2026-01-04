# Simplex Cost-Optimized Enterprise Deployment

**Version 0.2.0**

Simplex is designed to run enterprise workloads on the cheapest available cloud compute.

---

## Design Principles

1. **Embrace ephemeral compute**: Spot/preemptible instances are 60-90% cheaper
2. **Minimize memory footprint**: Smaller instances = lower cost
3. **Use object storage for persistence**: S3/Blob storage is 10-100x cheaper than block storage
4. **Batch AI inference**: Maximize GPU utilization, minimize idle time
5. **Colocate to avoid network egress**: Same-AZ communication is free

---

## Recommended Instance Types

### AWS EC2

| Tier | Instance | vCPU | RAM | On-Demand | Spot | Use Case |
|------|----------|------|-----|-----------|------|----------|
| Nano | t4g.nano | 2 | 0.5GB | $0.0042/hr | $0.0013/hr | Lightweight actors |
| Micro | t4g.micro | 2 | 1GB | $0.0084/hr | $0.0025/hr | Standard workers |
| Small | t4g.small | 2 | 2GB | $0.0168/hr | $0.0050/hr | Stateful actors |
| Compute | c7g.medium | 1 | 2GB | $0.0361/hr | $0.0108/hr | CPU-intensive |

**Key insight**: ARM-based Graviton instances (t4g, c7g) are 20-40% cheaper than x86 equivalents with better performance per dollar.

### Azure

| Tier | Instance | vCPU | RAM | Pay-as-you-go | Spot | Use Case |
|------|----------|------|-----|---------------|------|----------|
| Minimal | B1ls | 1 | 0.5GB | $0.0052/hr | $0.0010/hr | Nano workers |
| Basic | B1s | 1 | 1GB | $0.0104/hr | $0.0021/hr | Standard workers |
| Standard | B2s | 2 | 4GB | $0.0416/hr | $0.0083/hr | Stateful actors |

### Google Cloud

| Tier | Instance | vCPU | RAM | On-Demand | Preemptible | Use Case |
|------|----------|------|-----|-----------|-------------|----------|
| Micro | e2-micro | 0.25 | 1GB | $0.0084/hr | $0.0025/hr | Nano workers |
| Small | e2-small | 0.5 | 2GB | $0.0168/hr | $0.0050/hr | Standard workers |

---

## SVM Memory Footprint

To run on nano/micro instances, the SVM must be lightweight:

```
    Target SVM Memory Budget (512MB total):

    +------------------------------------------+
    | Component              | Budget          |
    +------------------------------------------+
    | SVM Runtime            | 10-20 MB        |
    | Bytecode Cache         | 10-50 MB        |
    | Actor Heap (per actor) | 1-10 MB         |
    | Message Queues         | 10-50 MB        |
    | Network Buffers        | 10-20 MB        |
    | Checkpoint Buffer      | 50-100 MB       |
    | Headroom               | ~250 MB         |
    +------------------------------------------+
```

**Implementation requirements**:
- Zero-copy message passing where possible
- Streaming checkpoints (don't buffer entire state)
- Aggressive memory pooling and reuse
- No JIT on nano instances (interpreter only)

---

## Spot Instance Architecture

Spot instances can be terminated with 2-minute warning (AWS) or 30-second warning (Azure/GCP). Simplex is designed for this:

```
    Spot-Tolerant Architecture:

    +------------------------------------------------------------------+
    |                                                                  |
    |   +------------------+     Spot instances (90% of fleet)         |
    |   | Worker (spot)    |     - Run all stateless processing        |
    |   +------------------+     - Checkpoint every N messages         |
    |   | Worker (spot)    |     - Resume on any available node        |
    |   +------------------+     - 2-min warning triggers checkpoint   |
    |   | Worker (spot)    |                                           |
    |   +--------+---------+                                           |
    |            |                                                     |
    |            v                                                     |
    |   +------------------+     On-demand instances (10% of fleet)    |
    |   | Coordinator      |     - Swarm coordination (Raft)           |
    |   | (on-demand)      |     - Message queue persistence           |
    |   +------------------+     - Never interrupted                   |
    |            |                                                     |
    |            v                                                     |
    |   +------------------+                                           |
    |   | S3 / Blob Store  |     Object storage (checkpoints)          |
    |   | (managed)        |     - Infinitely durable                  |
    |   +------------------+     - $0.023/GB/month                     |
    |                                                                  |
    +------------------------------------------------------------------+
```

**Spot interruption handling**:

```simplex
// Runtime automatically handles spot termination
actor Worker {
    var state: WorkerState

    // Called by runtime on spot termination warning
    on_termination_warning() {
        checkpoint()  // Immediate checkpoint
        drain()       // Stop accepting new work
        // Runtime migrates actor to available node
    }
}
```

---

## Storage Tiering Strategy

Different data has different access patterns and durability requirements:

```
    Storage Hierarchy:

    +------------------------------------------------------------------+
    |                                                                  |
    |   HOT (sub-millisecond access)                                   |
    |   +----------------------------------------------------------+   |
    |   | In-Memory (Actor State)                                  |   |
    |   | - Active working set                                     |   |
    |   | - Cost: Instance RAM ($$$)                               |   |
    |   | - Capacity: MB per actor                                 |   |
    |   +----------------------------------------------------------+   |
    |                                                                  |
    |   WARM (millisecond access)                                      |
    |   +----------------------------------------------------------+   |
    |   | Local NVMe / Instance Store                              |   |
    |   | - Recent checkpoints                                     |   |
    |   | - Message queue overflow                                 |   |
    |   | - Cost: Free with instance                               |   |
    |   | - Capacity: GB (ephemeral)                               |   |
    |   +----------------------------------------------------------+   |
    |                                                                  |
    |   COLD (10-100ms access)                                         |
    |   +----------------------------------------------------------+   |
    |   | Object Storage (S3 / Azure Blob / GCS)                   |   |
    |   | - Durable checkpoints                                    |   |
    |   | - Historical data                                        |   |
    |   | - Cost: $0.023/GB/month (S3 Standard)                    |   |
    |   | - Capacity: Unlimited                                    |   |
    |   +----------------------------------------------------------+   |
    |                                                                  |
    |   ARCHIVE (minutes-hours access)                                 |
    |   +----------------------------------------------------------+   |
    |   | S3 Glacier / Azure Archive                               |   |
    |   | - Old checkpoints for compliance                         |   |
    |   | - Cost: $0.004/GB/month                                  |   |
    |   | - Capacity: Unlimited                                    |   |
    |   +----------------------------------------------------------+   |
    |                                                                  |
    +------------------------------------------------------------------+
```

**Storage cost comparison (1TB/month)**:

| Storage Type | Monthly Cost | Access Latency | Use Case |
|--------------|--------------|----------------|----------|
| EBS gp3 | $80 | <1ms | Don't use |
| S3 Standard | $23 | 10-100ms | Active checkpoints |
| S3 Infrequent | $12.50 | 10-100ms | Older checkpoints |
| S3 Glacier | $4 | Minutes-hours | Compliance archive |

**Recommendation**: Never use block storage (EBS/Azure Disk) for checkpoints. Object storage is 3-10x cheaper and Simplex's checkpoint design doesn't need block semantics.

---

## Network Cost Optimization

Cloud network egress is expensive ($0.09/GB). Simplex minimizes this:

```
    Network Topology:

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
                               |
                               | Cross-region replication
                               | (async, for DR only)
                               v                            $0.02/GB
    +------------------------------------------------------------------+
    |                   Secondary Region (DR)                          |
    +------------------------------------------------------------------+
```

**Network design rules**:

1. All swarm nodes in same AZ (free traffic)
2. Checkpoint storage in same AZ
3. AI inference pool in same AZ
4. Cross-region only for disaster recovery (async)
5. Client traffic via CloudFront/CDN (reduced egress)

---

## AI Inference Cost Optimization

GPU instances are expensive. Simplex maximizes utilization:

```
    AI Inference Architecture:

    +------------------------------------------------------------------+
    |                     AI Inference Pool                            |
    |                                                                  |
    |   Strategy: Shared GPU nodes with request batching               |
    |                                                                  |
    |   +----------------------------------------------------------+   |
    |   |                    Request Batcher                       |   |
    |   |                                                          |   |
    |   |  - Collects requests for 5-50ms                         |   |
    |   |  - Batches by model and operation type                  |   |
    |   |  - Maximizes GPU throughput                             |   |
    |   +---------------------------+------------------------------+   |
    |                               |                                  |
    |         +---------------------+---------------------+            |
    |         v                     v                     v            |
    |   +------------+        +------------+        +------------+     |
    |   | GPU Node   |        | GPU Node   |        | GPU Node   |     |
    |   | g4dn.xlarge|        | g4dn.xlarge|        | g4dn.xlarge|     |
    |   | (spot)     |        | (spot)     |        | (on-demand)|     |
    |   +------------+        +------------+        +------------+     |
    |   $0.16/hr spot         $0.16/hr spot         $0.526/hr          |
    |                                                                  |
    |   Model loading strategy:                                        |
    |   - Keep hot models in VRAM                                      |
    |   - LRU eviction for cold models                                |
    |   - Preload based on traffic patterns                           |
    |                                                                  |
    +------------------------------------------------------------------+
```

**Cost-saving techniques**:

| Technique | Savings | Trade-off |
|-----------|---------|-----------|
| Spot GPU instances | 60-70% | 2-min termination risk |
| Request batching | 50-80% | 5-50ms added latency |
| Smaller models | 70-90% | Reduced capability |
| Quantization (INT8) | 50% memory | Slight accuracy loss |
| Shared inference pool | Variable | Network hop |

**Tiered AI strategy**:

```simplex
// Language-level model selection
let quick_answer = ai::complete(prompt, model: "fast")      // Small, cheap model
let quality_answer = ai::complete(prompt, model: "quality") // Large, expensive model

// Runtime routes based on model tier:
// "fast"    -> Llama 7B on shared GPU pool ($0.0001/request)
// "quality" -> Claude/GPT-4 via API ($0.01/request)
// "local"   -> On-device for edge deployment (free)
```

---

## Reference Architecture

### Enterprise Swarm (1000 actors, 10M messages/day)

```
    +------------------------------------------------------------------+
    |                                                                  |
    |   COMPUTE TIER                                                   |
    |   +----------------------------------------------------------+   |
    |   | 20x t4g.micro (spot) - $0.0025/hr each = $1.20/day       |   |
    |   | - 40 vCPU total                                          |   |
    |   | - 20GB RAM total                                         |   |
    |   | - ~50 actors per instance                                |   |
    |   +----------------------------------------------------------+   |
    |                                                                  |
    |   COORDINATION TIER                                              |
    |   +----------------------------------------------------------+   |
    |   | 3x t4g.small (on-demand) - $0.0168/hr each = $1.21/day   |   |
    |   | - Raft consensus (3-node quorum)                         |   |
    |   | - Message queue coordination                             |   |
    |   | - Never interrupted                                      |   |
    |   +----------------------------------------------------------+   |
    |                                                                  |
    |   AI INFERENCE TIER                                              |
    |   +----------------------------------------------------------+   |
    |   | 2x g4dn.xlarge (spot) - $0.16/hr each = $7.68/day        |   |
    |   | - Batched inference                                      |   |
    |   | - ~10,000 requests/hour capacity                         |   |
    |   +----------------------------------------------------------+   |
    |                                                                  |
    |   STORAGE TIER                                                   |
    |   +----------------------------------------------------------+   |
    |   | S3 Standard - 100GB checkpoints = $2.30/month            |   |
    |   | S3 Glacier - 1TB archive = $4/month                      |   |
    |   +----------------------------------------------------------+   |
    |                                                                  |
    +------------------------------------------------------------------+

    DAILY COST BREAKDOWN:
    +----------------------------------+
    | Component        | Daily Cost   |
    +----------------------------------+
    | Compute (spot)   | $1.20        |
    | Coordination     | $1.21        |
    | AI Inference     | $7.68        |
    | Storage          | $0.21        |
    | Network          | ~$0          |
    +----------------------------------+
    | TOTAL            | ~$10.30/day  |
    |                  | ~$310/month  |
    +----------------------------------+
```

### Cost Comparison

```
    Simplex vs Traditional Architecture:

    +----------------------------------+----------------------------------+
    | Traditional                      | Simplex                          |
    +----------------------------------+----------------------------------+
    | 20x m5.large        | $1,752/mo | 20x t4g.micro (spot) | $36/mo   |
    | RDS PostgreSQL      | $200/mo   | S3 checkpoints       | $6/mo    |
    | ElastiCache         | $150/mo   | (not needed)         | $0       |
    | Load Balancer       | $50/mo    | (built-in)           | $0       |
    | AI API calls        | $500/mo   | GPU pool (spot)      | $230/mo  |
    +----------------------------------+----------------------------------+
    | TOTAL               | ~$2,650/mo | TOTAL               | ~$310/mo |
    +----------------------------------+----------------------------------+

    SAVINGS: ~88% ($2,340/month)
```

---

## Checkpoint Configuration

```simplex
config Checkpointing {
    // Incremental checkpoints (only changed state)
    mode: Incremental,

    // Batch small checkpoints together
    min_checkpoint_size: "64KB",

    // Compress before storage
    compression: Zstd(level: 3),

    // Write to local NVMe first, async upload to S3
    write_strategy: WriteThrough {
        local_buffer: "100MB",
        upload_concurrency: 4
    },

    // Lifecycle rules
    retention: {
        hot: Duration::hours(24),      // S3 Standard
        warm: Duration::days(30),       // S3 Infrequent Access
        cold: Duration::days(365),      // S3 Glacier
        delete_after: Duration::years(7) // Compliance
    }
}
```

**Checkpoint cost targets**:

| Actor State Size | Checkpoint Frequency | S3 Cost/Month |
|------------------|---------------------|---------------|
| 1KB | Every message | $0.10 |
| 10KB | Every 10 messages | $0.10 |
| 100KB | Every 100 messages | $0.10 |
| 1MB | Every 1000 messages | $0.23 |

---

## Auto-Scaling Configuration

```simplex
config AutoScaling {
    // Scale up when queue depth exceeds threshold
    scale_up: {
        metric: QueueDepth,
        threshold: 1000,
        cooldown: Duration::minutes(2)
    },

    // Scale down aggressively to minimize cost
    scale_down: {
        metric: CpuUtilization,
        threshold: 20,  // Below 20% utilization
        cooldown: Duration::minutes(10)
    },

    // Cost constraints
    limits: {
        min_instances: 3,
        max_instances: 100,
        max_hourly_cost: "$5.00",
        prefer_spot: true,
        spot_fallback: OnDemand  // Fall back if no spot available
    }
}
```

---

## Deployment Commands

```bash
# Deploy cost-optimized swarm to AWS
simplex deploy --cloud aws \
    --region us-east-1 \
    --instance-type t4g.micro \
    --spot-enabled \
    --min-nodes 3 \
    --max-nodes 50 \
    --storage s3 \
    --ai-pool g4dn.xlarge:2

# Monitor costs in real-time
simplex costs --watch

# Output:
# Current hourly burn rate: $0.43/hr
# Projected monthly cost: $310
# Spot savings: $892 (74%)
# Recommendation: Scale down 2 idle nodes to save $0.05/hr

# Set cost alerts
simplex costs --alert --daily-max 15 --monthly-max 400

# View cost breakdown
simplex costs --breakdown

# Output:
# Cost Breakdown (last 24h)
# -------------------------
# Compute:     $1.20 (12%)
# Coordination: $1.21 (12%)
# AI Inference: $7.68 (74%)
# Storage:     $0.21 (2%)
# Network:     $0.02 (0%)
# -------------------------
# Total:       $10.32
```

---

## Cost Optimization Checklist

- [ ] Use ARM instances (t4g/c7g on AWS)
- [ ] Enable spot instances for workers (90% of fleet)
- [ ] Keep coordinators on-demand (10% of fleet)
- [ ] All nodes in single availability zone
- [ ] Use S3/Blob for checkpoints (not EBS)
- [ ] Enable checkpoint compression (Zstd)
- [ ] Batch AI requests (5-50ms window)
- [ ] Use tiered AI models (fast/default/quality)
- [ ] Enable aggressive scale-down
- [ ] Set cost alerts and budgets

---

## Next Steps

- [Swarm Computing](06-swarm-computing.md): Distributed architecture
- [AI Integration](07-ai-integration.md): AI cost strategies
- [Examples](../examples/document-pipeline.md): See deployment in action
