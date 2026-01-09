# TASK-004: Real-Time Continuous Learning

**Status**: In Progress - Phase 2 (Optimizer Testing)
**Priority**: Critical
**Created**: 2026-01-09
**Updated**: 2026-01-09
**Target Version**: 0.7.0
**Depends On**: TASK-001 (Neural IR) ✓ Complete

## Overview

Implement real-time continuous learning in pure Simplex, enabling programs to learn and improve during execution. This is the foundational capability that makes Simplex unique: **the boundary between code and model dissolves**.

In traditional systems:
```
Training Phase → Frozen Model → Inference Phase
     (offline)      (static)      (online)
```

In Simplex:
```
┌─────────────────────────────────────────────────┐
│           Continuous Learning Loop               │
│                                                  │
│   Execute → Observe → Update → Execute → ...    │
│                                                  │
│   Code learns. Models adapt. Both improve.      │
└─────────────────────────────────────────────────┘
```

---

## The Core Philosophy

### Code That Learns

```simplex
// Traditional: static function, never improves
fn classify(text: String) -> Category {
    if text.contains("urgent") { Category::High }
    else { Category::Normal }
}

// Simplex: neural gate that learns from feedback
neural_gate classify(text: String) -> Category
    @learning(rate: 0.001, feedback: user_correction)
{
    // Initial logic provides baseline behavior
    // Gate weights adjust based on user corrections
    // Over time, classification improves
    semantic_match(text, learned_patterns)
}
```

### The Learning Primitives

Simplex v0.5.0 already has the conceptual foundation:

| Existing Feature | Learning Enhancement |
|------------------|---------------------|
| Belief confidence (0.0-1.0) | Updates via gradient descent |
| Neural gates (TASK-001) | Weights train during execution |
| Anima memory | Learns from experiences |
| HiveMnemonic | Collective learning across specialists |
| Soft logic gates | Differentiable decision boundaries |

What's missing: the infrastructure to **train these in real-time**.

---

## Core Technical Challenges

### Challenge 1: Continuous Learning Infrastructure

**Problem**: Traditional ML uses batch training with full datasets. Real-time learning must:
- Process single examples as they arrive
- Update weights incrementally without catastrophic forgetting
- Maintain stability while adapting

**Solution**: Implement **Streaming Gradient Descent** with memory replay.

```simplex
// The @learning annotation enables real-time updates
neural_gate route_query(query: String) -> Specialist
    @learning(
        rate: 0.0001,              // Small learning rate for stability
        replay_buffer: 1000,       // Remember recent examples
        replay_ratio: 0.1,         // Mix 10% old examples with new
        feedback: routing_outcome  // Signal from downstream
    )
{
    semantic_route(query, specialist_embeddings)
}

// Feedback flows back when specialist performance is measured
fn handle_result(query: String, specialist: Specialist, quality: f64) {
    // This triggers weight update in route_query gate
    routing_outcome.signal(query, specialist, quality)
}
```

**Implementation Components**:
1. `ReplayBuffer<T>` - Circular buffer storing recent (input, output, feedback) tuples
2. `StreamingOptimizer` - Online variant of AdamW with momentum decay
3. `FeedbackChannel` - Type-safe channel for routing learning signals
4. `LearningScheduler` - Manages when updates occur (every N examples, on idle, etc.)

### Challenge 2: Catastrophic Forgetting Prevention

**Problem**: Neural networks forget old knowledge when learning new patterns. A classifier that learns "urgent" emails might forget how to classify "normal" ones.

**Solution**: Implement **Elastic Weight Consolidation (EWC)** and **Experience Replay**.

```simplex
neural_gate email_classifier(email: Email) -> Priority
    @learning(
        strategy: ElasticWeightConsolidation(
            importance_decay: 0.99,  // How fast old knowledge fades
            fisher_samples: 100,     // Samples for importance estimation
        ),
        replay: ExperienceReplay(
            buffer_size: 10000,
            sample_strategy: Stratified,  // Balance across categories
        )
    )
{
    classify_priority(email.features)
}
```

**Strategies**:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| EWC | Penalize changes to important weights | Stable categories |
| Replay | Mix old examples with new | Balanced learning |
| Progressive | Add capacity for new knowledge | Growing domains |
| Distillation | Transfer from frozen snapshot | Critical systems |

### Challenge 3: Confidence-Calibrated Learning

**Problem**: A model should "know what it doesn't know." High confidence on wrong answers is dangerous. Learning should improve calibration, not just accuracy.

**Solution**: **Calibration-Aware Training** with temperature scaling.

```simplex
// Beliefs are confidence-calibrated
neural_gate confident_answer(question: String) -> (Answer, f64)
    @learning(
        objective: CalibrationAware(
            accuracy_weight: 0.7,
            calibration_weight: 0.3,  // ECE penalty
            target_ece: 0.05,         // Expected Calibration Error < 5%
        )
    )
{
    let (answer, logits) = generate_answer(question)
    let confidence = calibrated_confidence(logits, learned_temperature)
    (answer, confidence)
}

// Temperature is learned to improve calibration
@learnable
var learned_temperature: f64 = 1.0
```

**Calibration Metrics** (computed online):
- **ECE**: Expected Calibration Error - average |confidence - accuracy| per bin
- **Brier Score**: Mean squared error of probability estimates
- **Reliability Diagram**: Visual calibration curve (logged for monitoring)

### Challenge 4: Feedback Signal Design

**Problem**: What should the learning signal be? In supervised learning, we have labels. In real-time systems, feedback is implicit, delayed, and noisy.

**Solution**: **Multi-Signal Feedback Fusion** with attribution.

```simplex
// Define feedback sources for a specialist
neural_gate code_reviewer(code: String) -> Review
    @learning(
        feedback: [
            // Explicit: User rates the review
            UserRating { weight: 1.0, delay: minutes(5) },

            // Implicit: User accepts/rejects suggestions
            SuggestionAdoption { weight: 0.5, delay: hours(1) },

            // Downstream: Code passes/fails tests after changes
            TestOutcome { weight: 0.3, delay: hours(24) },

            // Self-consistency: Similar code should get similar reviews
            ConsistencyReward { weight: 0.2 },
        ],
        attribution: TemporalCreditAssignment(
            discount: 0.95,  // Decay for delayed feedback
        )
    )
{
    review_code(code, learned_patterns)
}
```

**Feedback Types**:

| Type | Description | Latency | Reliability |
|------|-------------|---------|-------------|
| Explicit | User provides rating/correction | Immediate | High |
| Implicit | User behavior (accept, ignore, undo) | Short | Medium |
| Downstream | Effect on subsequent operations | Long | Variable |
| Self-supervised | Internal consistency signals | None | Low |

### Challenge 5: Safe Learning Boundaries

**Problem**: Unrestricted learning can cause system instability. A gate that learns to always return `true` breaks program logic.

**Solution**: **Constrained Learning** with safety bounds.

```simplex
// Gate has hard safety constraints
neural_gate memory_allocator(request: MemoryRequest) -> Allocation
    @learning(
        // Soft: optimizer respects these
        soft_constraints: [
            MaxLatency(ms: 100),
            MinThroughput(ops_per_sec: 1000),
        ],
        // Hard: compiler enforces these
        hard_constraints: [
            AlwaysAllocates,          // Must return valid allocation
            BoundedSize(max: 1.GB),   // No single allocation > 1GB
        ],
        // Fallback when constraints would be violated
        fallback: default_allocator,
    )
{
    optimized_allocation(request, learned_strategy)
}
```

**Constraint Enforcement**:
1. **Compile-time**: Static analysis proves hard constraints
2. **Runtime**: Monitor soft constraints, revert if violated
3. **Fallback**: Safe default behavior when learning fails

### Challenge 6: Distributed Learning Coordination

**Problem**: Multiple specialists in a hive learn independently. Their learning must be coordinated to avoid conflicts and enable collective improvement.

**Solution**: **Federated Hive Learning** with gradient aggregation.

```simplex
hive CodeAnalysisHive {
    specialists: [Linter, Reviewer, Optimizer],
    slm: "simplex-cognitive-7b",

    // Hive-level learning coordination
    learning: {
        // Specialists share gradient updates
        aggregation: FederatedAveraging(
            sync_interval: examples(100),  // Sync every 100 examples
            staleness_tolerance: 3,        // Allow 3 rounds lag
        ),

        // HiveMnemonic learns from all specialists
        mnemonic_learning: CollectiveMemory(
            contribution_threshold: 0.8,   // High-confidence only
            conflict_resolution: EvidenceWeighted,
        ),

        // Cross-specialist knowledge transfer
        distillation: CrossSpecialist(
            teacher_confidence: 0.9,
            student_temperature: 2.0,
        ),
    }
}
```

---

## Simplex Training Library Architecture

### Core Libraries (Pure Simplex)

```
simplex-learning/
├── src/
│   ├── lib.sx                      # Public API
│   │
│   ├── tensor/                     # Tensor operations
│   │   ├── mod.sx
│   │   ├── tensor.sx               # Core Tensor<T, Shape> type
│   │   ├── ops.sx                  # matmul, add, mul, softmax, etc.
│   │   ├── autograd.sx             # Gradient tracking (builds on Neural IR)
│   │   └── backends/
│   │       ├── cpu.sx              # SIMD-optimized CPU ops
│   │       └── gpu.sx              # GPU via Neural IR Phase 3
│   │
│   ├── optim/                      # Optimizers
│   │   ├── mod.sx
│   │   ├── sgd.sx                  # Stochastic Gradient Descent
│   │   ├── adam.sx                 # Adam / AdamW
│   │   ├── streaming.sx            # Online learning variants
│   │   └── scheduler.sx            # Learning rate schedulers
│   │
│   ├── memory/                     # Learning memory systems
│   │   ├── mod.sx
│   │   ├── replay_buffer.sx        # Experience replay
│   │   ├── ewc.sx                  # Elastic Weight Consolidation
│   │   └── progressive.sx          # Progressive network expansion
│   │
│   ├── feedback/                   # Feedback collection
│   │   ├── mod.sx
│   │   ├── channel.sx              # FeedbackChannel<T>
│   │   ├── attribution.sx          # Credit assignment
│   │   └── signals.sx              # Built-in signal types
│   │
│   ├── calibration/                # Confidence calibration
│   │   ├── mod.sx
│   │   ├── temperature.sx          # Temperature scaling
│   │   ├── metrics.sx              # ECE, Brier score
│   │   └── online.sx               # Streaming calibration
│   │
│   ├── safety/                     # Safe learning
│   │   ├── mod.sx
│   │   ├── constraints.sx          # Soft/hard constraints
│   │   ├── bounds.sx               # Weight clipping, gradient norms
│   │   └── fallback.sx             # Fallback execution
│   │
│   ├── distributed/                # Multi-specialist coordination
│   │   ├── mod.sx
│   │   ├── federated.sx            # Federated averaging
│   │   ├── distillation.sx         # Knowledge distillation
│   │   └── sync.sx                 # Gradient synchronization
│   │
│   └── runtime/                    # Learning runtime
│       ├── mod.sx
│       ├── learner.sx              # Main learning loop
│       ├── metrics.sx              # Online metric tracking
│       └── checkpoint.sx           # Save/restore learning state
│
├── examples/
│   ├── sentiment_learner.sx        # Sentiment that improves
│   ├── router_learner.sx           # Routing that adapts
│   ├── code_reviewer.sx            # Code review that learns
│   └── hive_learning.sx            # Distributed hive learning
│
└── tests/
    ├── tensor_tests.sx
    ├── optim_tests.sx
    ├── replay_tests.sx
    ├── calibration_tests.sx
    └── integration_tests.sx
```

### LOC Estimates

| Component | Estimated LOC | Complexity |
|-----------|---------------|------------|
| tensor/ | 3,000 | High (numerics, autograd) |
| optim/ | 1,500 | Medium |
| memory/ | 1,200 | Medium |
| feedback/ | 800 | Medium |
| calibration/ | 600 | Medium |
| safety/ | 700 | Medium |
| distributed/ | 1,500 | High |
| runtime/ | 1,000 | Medium |
| **Total** | **~10,300** | - |

### Key Type Definitions

```simplex
// Core tensor type with shape tracking
struct Tensor<T, const SHAPE: [usize]> {
    data: Vec<T>,
    grad: Option<Box<Tensor<T, SHAPE>>>,
    requires_grad: bool,
}

// Learning configuration for neural gates
struct LearningConfig {
    rate: f64,
    optimizer: Optimizer,
    replay: Option<ReplayConfig>,
    constraints: Vec<Constraint>,
    feedback: Vec<FeedbackSource>,
}

// Feedback channel for learning signals
struct FeedbackChannel<Input, Output, Signal> {
    buffer: ReplayBuffer<(Input, Output, Signal)>,
    attribution: AttributionStrategy,
    subscribers: Vec<NeuralGateRef>,
}

// Online calibration state
struct CalibrationState {
    temperature: f64,
    confidence_buckets: [OnlineAccumulator; 10],
    ece_running: f64,
    brier_running: f64,
    samples_seen: u64,
}

// Constraint for safe learning
enum Constraint {
    Soft(SoftConstraint),
    Hard(HardConstraint),
}

struct SoftConstraint {
    check: fn(metrics: &Metrics) -> bool,
    penalty: f64,
}

struct HardConstraint {
    check: fn(output: &Output) -> bool,
    fallback: fn(input: &Input) -> Output,
}
```

---

## Integration with Neural IR (TASK-001)

TASK-001 provides the foundation:

| Neural IR Feature | Learning Integration |
|-------------------|---------------------|
| `neural_gate` keyword | Add `@learning` annotation |
| Gumbel-Softmax | Temperature is learnable |
| Dual compilation | Training mode enables gradient flow |
| Contract logic | Constraints for safe learning |
| Hardware targeting | GPU acceleration for tensor ops |
| Structural pruning | Prune gates that don't contribute |

### Extended Neural Gate Syntax

```simplex
// Full learning-enabled neural gate
neural_gate smart_router(query: String) -> Specialist
    // Existing Neural IR contracts
    requires query.len() > 0
    ensures result.is_valid()
    fallback default_specialist

    // NEW: Learning configuration
    @learning(
        rate: 0.0001,
        optimizer: AdamW { beta1: 0.9, beta2: 0.999, weight_decay: 0.01 },
        replay: { buffer_size: 1000, sample_ratio: 0.1 },
        feedback: routing_feedback,
        calibration: { target_ece: 0.05 },
        constraints: [
            soft: LatencyBound(ms: 50),
            hard: AlwaysReturnsValid,
        ],
        checkpoint: every(minutes(10)),
    )

    // Hardware hint from Neural IR
    @gpu
{
    // Gate implementation
    semantic_route(query, specialist_embeddings)
}
```

### Compilation Flow

```
Source with @learning
        │
        ▼
┌─────────────────────────────────────┐
│        Learning Annotation Parser    │
│  - Extract learning config           │
│  - Validate constraints              │
│  - Generate feedback channel         │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│         Neural IR Compiler           │
│  - Differentiable code generation   │
│  - Gradient tracking insertion       │
│  - Hardware targeting                │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│       Learning Runtime Codegen       │
│  - Optimizer initialization          │
│  - Replay buffer setup               │
│  - Feedback subscription             │
│  - Metric collection hooks           │
└─────────────────────────────────────┘
        │
        ▼
    Executable with
    Real-Time Learning
```

---

## Implementation Phases

### Phase 1: Tensor Library Foundation (2 weeks)

**Goal**: Core tensor operations in pure Simplex

**Deliverables**:
1. `Tensor<T, Shape>` type with basic operations
2. Autograd integration via Neural IR gradient tracking
3. CPU backend with SIMD optimizations
4. Basic ops: matmul, add, mul, div, softmax, relu, tanh

**Success Criteria**:
- [ ] Matrix multiplication correct for all shapes
- [ ] Gradients flow through computation graph
- [ ] Performance within 3x of NumPy on CPU
- [ ] All ops have gradient implementations

### Phase 2: Online Optimizers (1 week)

**Goal**: Streaming-capable optimizers

**Deliverables**:
1. `StreamingSGD` - Online SGD with momentum
2. `StreamingAdam` - Online Adam variant
3. Learning rate schedulers (cosine, step, warmup)
4. Gradient clipping and normalization

**Success Criteria**:
- [ ] Optimizer converges on streaming data
- [ ] Memory usage bounded regardless of data size
- [ ] Matches batch optimizer on accumulated data

### Phase 3: Experience Replay & Memory (1 week)

**Goal**: Prevent catastrophic forgetting

**Deliverables**:
1. `ReplayBuffer<T>` with configurable size and sampling
2. Elastic Weight Consolidation implementation
3. Stratified sampling for class balance
4. Memory-efficient storage (compressed, deduplicated)

**Success Criteria**:
- [ ] Old knowledge retained after learning new patterns
- [ ] Replay doesn't dominate compute budget
- [ ] Buffer handles 100K+ examples efficiently

### Phase 4: Feedback System (1 week)

**Goal**: Type-safe feedback collection and attribution

**Deliverables**:
1. `FeedbackChannel<I, O, S>` type
2. Temporal credit assignment
3. Multi-signal fusion
4. Delayed feedback handling

**Success Criteria**:
- [ ] Feedback flows to correct gates
- [ ] Delayed signals attributed correctly
- [ ] Multiple feedback sources combine properly

### Phase 5: Confidence Calibration (1 week)

**Goal**: Models know what they don't know

**Deliverables**:
1. Online ECE computation
2. Temperature scaling (learnable)
3. Calibration-aware loss functions
4. Reliability diagram logging

**Success Criteria**:
- [ ] ECE < 0.05 achievable after calibration
- [ ] Temperature adapts to improve calibration
- [ ] Metrics update in real-time

### Phase 6: Safe Learning Constraints (1 week)

**Goal**: Learning cannot break program correctness

**Deliverables**:
1. Soft constraint penalty integration
2. Hard constraint enforcement with fallback
3. Weight bound enforcement
4. Gradient explosion prevention

**Success Criteria**:
- [ ] Hard constraints never violated
- [ ] Soft constraints guide but don't block
- [ ] Fallback executes when needed
- [ ] System remains stable under adversarial feedback

### Phase 7: Distributed Learning (2 weeks)

**Goal**: Hive-wide coordinated learning

**Deliverables**:
1. Federated averaging for gradient sharing
2. Cross-specialist knowledge distillation
3. Gradient synchronization protocol
4. Conflict resolution for shared beliefs

**Success Criteria**:
- [ ] Multiple specialists learn without conflict
- [ ] Hive knowledge improves collectively
- [ ] Sync overhead < 10% of compute

### Phase 8: Integration & Testing (2 weeks)

**Goal**: End-to-end real-time learning works

**Deliverables**:
1. `@learning` annotation fully functional
2. Comprehensive test suite
3. Example applications
4. Performance benchmarks
5. Documentation

**Success Criteria**:
- [ ] Gate learns from feedback in production-like scenario
- [ ] System runs for 24+ hours without degradation
- [ ] Latency impact < 20% vs non-learning
- [ ] Memory usage stable over time

---

## Test Strategy

### Unit Tests

```simplex
// tests/tensor_tests.sx
test "matmul gradient" {
    let a = Tensor::randn([2, 3]).requires_grad()
    let b = Tensor::randn([3, 4]).requires_grad()
    let c = a.matmul(b)
    let loss = c.sum()
    loss.backward()

    // Numerical gradient check
    let numerical_grad = numerical_gradient(|a| a.matmul(b).sum(), a)
    assert_close(a.grad.unwrap(), numerical_grad, rtol: 1e-5)
}

// tests/optim_tests.sx
test "streaming adam convergence" {
    let gate = simple_gate()
    let optimizer = StreamingAdam::new(gate.parameters(), lr: 0.01)

    for _ in 0..1000 {
        let (input, target) = generate_example()
        let output = gate.forward(input)
        let loss = mse_loss(output, target)
        loss.backward()
        optimizer.step()
    }

    assert(gate.accuracy() > 0.9)
}

// tests/replay_tests.sx
test "replay prevents forgetting" {
    let buffer = ReplayBuffer::new(1000)
    let gate = classifier_gate()

    // Train on class A
    for _ in 0..500 {
        let example = generate_class_a()
        buffer.add(example.clone())
        train_step(gate, example, &buffer)
    }
    let acc_a_after_a = evaluate(gate, class_a_test)

    // Train on class B (without replay, would forget A)
    for _ in 0..500 {
        let example = generate_class_b()
        buffer.add(example.clone())
        train_step(gate, example, &buffer)  // Replays A examples
    }

    let acc_a_after_b = evaluate(gate, class_a_test)
    assert(acc_a_after_b > 0.8 * acc_a_after_a, "Catastrophic forgetting!")
}
```

### Integration Tests

```simplex
// tests/integration_tests.sx
test "end-to-end learning loop" {
    // Create a learning-enabled specialist
    specialist LearningClassifier {
        model: "simplex-cognitive-1b",

        neural_gate classify(text: String) -> Category
            @learning(
                rate: 0.001,
                feedback: user_feedback,
            )
        {
            infer("Classify: " + text)
        }
    }

    let hive = spawn TestHive { specialists: [LearningClassifier] }
    let initial_acc = evaluate_accuracy(hive)

    // Simulate user feedback loop
    for example in test_stream.take(1000) {
        let prediction = ask(hive.LearningClassifier, Classify(example.text))
        if prediction != example.label {
            user_feedback.correct(example.text, example.label)
        }
    }

    let final_acc = evaluate_accuracy(hive)
    assert(final_acc > initial_acc + 0.1, "Learning should improve accuracy")
}

test "distributed hive learning" {
    hive LearningHive {
        specialists: [Specialist1, Specialist2, Specialist3],
        learning: { aggregation: FederatedAveraging(sync_interval: 10) }
    }

    let hive = spawn LearningHive

    // Train each specialist on different data
    parallel {
        train_specialist(hive.Specialist1, dataset_1),
        train_specialist(hive.Specialist2, dataset_2),
        train_specialist(hive.Specialist3, dataset_3),
    }

    // All specialists should have learned from shared gradients
    for specialist in [hive.Specialist1, hive.Specialist2, hive.Specialist3] {
        assert(evaluate(specialist, combined_test) > 0.7)
    }
}
```

### Stress Tests

```simplex
// tests/stress_tests.sx
test "24-hour stability" {
    let learner = spawn LearningSpecialist
    let metrics = MetricsCollector::new()

    // Run for 24 hours of simulated time
    for hour in 0..24 {
        for _ in 0..1000 {
            let example = generate_example()
            let result = ask(learner, Process(example))

            // Random feedback
            if random() < 0.1 {
                send_feedback(learner, example, result)
            }

            metrics.record(learner.memory_usage(), learner.latency())
        }

        // Check stability
        assert(metrics.memory_growth_rate() < 0.01, "Memory leak!")
        assert(metrics.latency_p99() < ms(100), "Latency degradation!")
    }
}

test "adversarial feedback resistance" {
    let learner = spawn ConstrainedLearner

    // Send adversarial feedback trying to break constraints
    for _ in 0..1000 {
        let malicious_feedback = generate_adversarial_feedback()
        send_feedback(learner, malicious_feedback)
    }

    // Constraints must still hold
    assert(learner.hard_constraints_satisfied())
    assert(learner.performance() > 0.5, "Should resist adversarial feedback")
}
```

---

## Comparison: Python Training vs Simplex Real-Time Learning

| Aspect | Python (TASK-003) | Simplex (TASK-004) |
|--------|-------------------|-------------------|
| **When** | Offline, before deployment | During execution |
| **Data** | Pre-collected datasets | Live production data |
| **Feedback** | Labels in dataset | User actions, downstream signals |
| **Frequency** | Train once, deploy | Continuous updates |
| **Scope** | Full model fine-tuning | Gate-level updates |
| **Integration** | Separate pipeline | Native language feature |
| **Result** | Frozen GGUF model | Ever-improving system |

### Complementary Roles

```
TASK-003: Python Batch Training
┌─────────────────────────────────────┐
│  - Initial model training            │
│  - Major capability additions        │
│  - New specialist creation           │
│  - Large-scale dataset training      │
│  - Produces base model weights       │
└─────────────────────────────────────┘
                    │
                    ▼
              Base Model
                    │
                    ▼
TASK-004: Simplex Real-Time Learning
┌─────────────────────────────────────┐
│  - Continuous improvement            │
│  - User preference adaptation        │
│  - Domain drift handling             │
│  - Personalization                   │
│  - Production optimization           │
└─────────────────────────────────────┘
```

---

## Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Learning latency overhead | < 20% | Practical for production |
| Memory overhead per gate | < 10 MB | Scalable to many gates |
| Gradient computation | < 50ms | Real-time feedback loop |
| Forgetting rate | < 10% per week | Maintain old knowledge |
| ECE after calibration | < 0.05 | Well-calibrated confidence |
| Time to measurable improvement | < 1 hour | Visible progress |

---

## Milestones

| Milestone | Date | Deliverable |
|-----------|------|-------------|
| **M1** | Week 2 | Tensor ops + autograd working |
| **M2** | Week 3 | Online optimizer converges |
| **M3** | Week 4 | Replay buffer prevents forgetting |
| **M4** | Week 5 | Feedback flows to gates |
| **M5** | Week 6 | Calibration improves ECE |
| **M6** | Week 7 | Constraints enforce safety |
| **M7** | Week 9 | Hive learning coordinates |
| **M8** | Week 11 | Full system integration |
| **M9** | Week 12 | Production-ready release |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Gradient instability | Medium | High | Gradient clipping, careful initialization |
| Memory leaks in replay | Medium | High | Bounded buffers, explicit eviction |
| Adversarial feedback | Low | High | Hard constraints, anomaly detection |
| Learning interferes with inference | Medium | High | Async updates, latency budgeting |
| Catastrophic forgetting | Medium | Medium | EWC, replay, progressive networks |
| Distributed sync overhead | Low | Medium | Async aggregation, staleness tolerance |

---

## Example: Complete Learning-Enabled Specialist

```simplex
// A complete example of a real-time learning specialist

specialist AdaptiveCodeReviewer {
    model: "simplex-cognitive-7b",

    // Personal learning state
    anima: {
        purpose: "Review code and learn from feedback",
        beliefs: { revision_threshold: 30 },
    },

    // Feedback channel for learning signals
    feedback: FeedbackChannel<CodeReview, UserReaction>,

    // Main review function with learning
    neural_gate review(code: String) -> CodeReview
        requires code.len() > 0
        ensures result.suggestions.all(|s| s.is_actionable())
        fallback basic_review(code)

        @learning(
            rate: 0.0001,
            optimizer: StreamingAdam { weight_decay: 0.01 },

            replay: {
                buffer_size: 5000,
                sample_ratio: 0.2,
                strategy: Stratified(by: |r| r.severity),
            },

            feedback: self.feedback,

            calibration: {
                target_ece: 0.05,
                temperature_lr: 0.001,
            },

            constraints: [
                soft: ReviewLength(min: 10, max: 1000),
                soft: SuggestionCount(min: 1, max: 10),
                hard: NoOffensiveLanguage,
            ],

            checkpoint: every(minutes(30)),
        )
    {
        // Review implementation
        let context = self.anima.recall_for("code review patterns")
        let review = infer("Review this code:\n" + code)

        // Learn from this interaction
        self.anima.remember("Reviewed code: " + code.summary())

        review
    }

    // Handle feedback from users
    receive UserFeedback(review_id: u64, reaction: UserReaction) {
        // Signal the learning system
        self.feedback.signal(review_id, reaction)

        // Update beliefs based on feedback
        match reaction {
            UserReaction::Accepted => {
                self.anima.strengthen_belief("Review style is effective", 0.05)
            }
            UserReaction::Rejected(reason) => {
                self.anima.learn("Review rejected: " + reason)
                self.anima.weaken_belief("Review style is effective", 0.1)
            }
            UserReaction::Partial(accepted, rejected) => {
                self.anima.learn("Partial acceptance: " + accepted.join(", "))
            }
        }
    }
}

// Usage
fn main() {
    let reviewer = spawn AdaptiveCodeReviewer

    // Review code
    let review = ask(reviewer, Review(code))
    print("Review: {review}")

    // User provides feedback (triggers learning)
    send(reviewer, UserFeedback(review.id, UserReaction::Accepted))

    // Over time, reviewer improves based on feedback patterns
}
```

---

## Summary

TASK-004 implements the core Simplex vision: **code that learns**.

| What | Why | How |
|------|-----|-----|
| Online learning | Continuous improvement | Streaming optimizers |
| Forgetting prevention | Maintain old knowledge | EWC + replay |
| Calibrated confidence | Know what you don't know | Temperature scaling |
| Feedback attribution | Learn from delayed signals | Temporal credit |
| Safe learning | Never break correctness | Hard/soft constraints |
| Distributed learning | Collective improvement | Federated aggregation |

The result is a system where the boundary between code and model dissolves. Neural gates learn from feedback. Belief systems update with evidence. Specialists improve through use.

**This is not just training infrastructure - this is the foundation of adaptive software.**

---

## Resources

- TASK-001: Neural IR (complete) - provides differentiable execution
- Anima spec: simplex-docs/spec/12-anima.md - belief and memory systems
- Cognitive Hive spec: simplex-docs/spec/09-cognitive-hive.md - specialist architecture
- Python training: training/scripts/ - reference implementations

---

## Progress

### Library Skeleton Complete (2026-01-09)

The `simplex-learning` library skeleton has been created with all modules:

| Module | Files | Status | Notes |
|--------|-------|--------|-------|
| tensor/ | mod.sx, tensor.sx, ops.sx, autograd.sx, backends/ | ✓ Skeleton | Core tensor type, basic ops, autograd structure |
| optim/ | mod.sx, sgd.sx, adam.sx, streaming.sx, scheduler.sx | ✓ Skeleton | StreamingSGD, StreamingAdam, schedulers |
| memory/ | mod.sx, replay_buffer.sx, ewc.sx, progressive.sx | ✓ Skeleton | ReplayBuffer, EWC, MAS, Progressive Networks |
| feedback/ | mod.sx, channel.sx, attribution.sx, signals.sx | ✓ Skeleton | FeedbackChannel, credit assignment |
| calibration/ | mod.sx, temperature.sx, metrics.sx, online.sx | ✓ Skeleton | ECE, Brier, temperature scaling |
| safety/ | mod.sx, constraints.sx, bounds.sx, fallback.sx | ✓ Skeleton | Soft/hard constraints, fallbacks |
| distributed/ | mod.sx, federated.sx, distillation.sx, sync.sx | ✓ Skeleton | FederatedLearner, KnowledgeDistiller |
| runtime/ | mod.sx, learner.sx, metrics.sx, checkpoint.sx | ✓ Skeleton | ContinuousLearner runtime |

**Examples created:**
- `basic_learner.sx` - Basic continuous learning loop
- `federated_learning.sx` - Multi-node federated learning
- `neural_gate.sx` - Neural gate with @learning annotation

**Tests created:**
- `test_tensor.sx`, `test_optimizer.sx`, `test_memory.sx`
- `test_calibration.sx`, `test_safety.sx`, `test_distributed.sx`
- `test_runtime.sx`

**Total LOC:** ~10,300 (as estimated)

### Phase 1: Tensor Implementation (2026-01-09)

Enhanced tensor module with full autograd support:

**Completed:**
- Tensor type with shape tracking, gradient storage, requires_grad flag
- Element-wise ops (add, sub, mul, div) with autograd integration
- Matrix operations (matmul, transpose) with gradient tracking
- Activation functions (relu, sigmoid, tanh, gelu, softmax) with backward pass
- Normalization (layer_norm) and loss functions (mse, cross_entropy, bce)
- ComputationGraph with topological sort for backward pass
- GradientTape for operation recording
- Backward functions for all major operations
- Gradient checking utilities

**Key additions to autograd.sx:**
- `next_tensor_id()` - unique ID generation for tensors
- `ComputationGraph` - full computation graph with topological sort
- `record_op()` - record operations for backward pass
- `is_grad_enabled()` / `set_grad_enabled()` - control gradient tracking
- `clear_graph()` - reset computation graph

### Phase Checklist

- [x] Library skeleton created
- [x] Phase 1: Tensor ops fully functional with autograd
- [ ] Phase 2: Optimizers tested and converging
- [ ] Phase 3: Replay buffer preventing forgetting
- [ ] Phase 4: Feedback attribution working
- [ ] Phase 5: Calibration achieving ECE < 0.05
- [ ] Phase 6: Constraints enforced
- [ ] Phase 7: Distributed learning coordinated
- [ ] Phase 8: End-to-end integration tested

---

## Next Steps

1. ~~Review and approve this specification~~ ✓
2. ~~Create simplex-learning library skeleton~~ ✓
3. ~~Implement Phase 1 (Tensor library with autograd)~~ ✓
4. **Implement Phase 2 (Optimizer testing)** ← Current
5. Integration testing with existing Neural IR

---

*"The program that learns is the program that survives."*
