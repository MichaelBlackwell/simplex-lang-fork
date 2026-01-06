# Getting Started with Simplex

**Version 0.1.0**

A quick introduction to writing and running Simplex programs.

---

## Installation

```bash
# macOS
brew install simplex

# Linux
curl -fsSL https://simplex-lang.org/install.sh | sh

# Windows
winget install simplex

# From source
git clone https://github.com/simplex-lang/simplex
cd simplex && cargo install --path .
```

Verify installation:

```bash
simplex --version
# simplex 0.1.0
```

---

## Hello World

Create `hello.sx`:

```simplex
fn main() {
    print("Hello, Simplex!")
}
```

Run it:

```bash
simplex run hello.sx
# Hello, Simplex!
```

---

## Your First Actor

Create `counter.sx`:

```simplex
actor Counter {
    var count: i64 = 0

    receive Increment {
        count += 1
        print("Count: {count}")
    }

    receive GetCount -> i64 {
        count
    }
}

fn main() {
    let counter = spawn Counter

    send(counter, Increment)
    send(counter, Increment)
    send(counter, Increment)

    sleep(Duration::milliseconds(100))

    let final_count = ask(counter, GetCount)
    print("Final count: {final_count}")
}
```

Run it:

```bash
simplex run counter.sx
# Count: 1
# Count: 2
# Count: 3
# Final count: 3
```

---

## Using AI

Create `ai_example.sx`:

```simplex
use ai

fn main() {
    // Text completion
    let response = await ai::complete("What is the capital of France?")
    print("Answer: {response}")

    // Classification
    enum Sentiment { Positive, Negative, Neutral }

    let review = "This product is amazing, best purchase ever!"
    let sentiment = await ai::classify<Sentiment>(review)
    print("Sentiment: {sentiment}")

    // Structured extraction
    type Person {
        name: String,
        age: Option<i64>
    }

    let text = "John is 30 years old"
    let person = await ai::extract<Person>(text)
    print("Extracted: {person.name}, age {person.age}")
}
```

Run it:

```bash
simplex run ai_example.sx
# Answer: The capital of France is Paris.
# Sentiment: Positive
# Extracted: John, age Some(30)
```

---

## Project Structure

```
my-project/
├── simplex.toml          # Project configuration
├── src/
│   ├── main.sx           # Entry point
│   ├── actors/
│   │   ├── worker.sx
│   │   └── coordinator.sx
│   └── types/
│       └── models.sx
└── tests/
    └── worker_test.sx
```

### simplex.toml

```toml
[package]
name = "my-project"
version = "0.1.0"

[dependencies]
# Add dependencies here

[ai]
default_model = "fast"

[deployment]
cloud = "aws"
region = "us-east-1"
spot_enabled = true
```

---

## Building and Running

```bash
# Run directly (compiles and runs)
simplex run src/main.sx

# Compile to bytecode
simplex build src/main.sx -o app.sbc

# Run bytecode
simplex run app.sbc

# Run tests
simplex test

# Format code
simplex fmt

# Check for errors without running
simplex check
```

---

## Deploying to a Swarm

### Local Development Swarm

```bash
# Start a local 3-node swarm
simplex swarm start --local --nodes 3

# Deploy your app
simplex swarm deploy app.sbc

# View status
simplex swarm status

# View logs
simplex swarm logs --follow

# Stop swarm
simplex swarm stop
```

### Cloud Deployment

```bash
# Deploy to AWS
simplex deploy --cloud aws \
    --region us-east-1 \
    --instance-type t4g.micro \
    --spot-enabled \
    --min-nodes 3 \
    --max-nodes 10

# Monitor
simplex swarm status
simplex costs --watch
```

---

## Key Concepts

### Actors

Actors are isolated units of computation:

```simplex
actor MyActor {
    var state: i64 = 0           // Mutable state (private)

    receive DoSomething {        // Fire-and-forget message
        state += 1
    }

    receive GetState -> i64 {    // Request-response message
        state
    }
}

// Spawn an actor
let actor = spawn MyActor

// Send async message
send(actor, DoSomething)

// Send and wait for response
let value = ask(actor, GetState)
```

### Checkpointing

Persist state for fault tolerance:

```simplex
actor ImportantActor {
    var data: Map<String, String> = {}

    receive Store(key: String, value: String) {
        data.insert(key, value)
        checkpoint()  // Persist state to durable storage
    }

    on_resume() {
        // Called after recovery from failure
        print("Resumed with {data.len()} items")
    }
}
```

### Supervision

Organize actors into fault-tolerant trees:

```simplex
supervisor MySystem {
    strategy: OneForOne,    // Restart only failed child
    max_restarts: 5,
    within: Duration::minutes(1),

    children: [
        child(Worker, restart: Always),
        child(Logger, restart: Transient)
    ]
}
```

### Ownership

Values have single owners (like Rust):

```simplex
fn process(data: String) {   // Takes ownership
    print(data)
}                            // data dropped here

fn analyze(data: &String) {  // Borrows (read-only)
    print(data.len())
}                            // borrow ends, data still valid

fn main() {
    let s = "hello"
    analyze(&s)    // Borrow
    process(s)     // Move ownership
    // s is no longer valid here
}
```

---

## Common Patterns

### Worker Pool

```simplex
actor WorkerPool {
    var workers: List<ActorRef<Worker>> = []

    init(size: u64) {
        for _ in 0..size {
            workers.push(spawn Worker)
        }
    }

    receive Submit(task: Task) {
        // Round-robin distribution
        let worker = workers[task.id % workers.len()]
        send(worker, Process(task))
    }
}
```

### Request-Response with Timeout

```simplex
let result = ask(actor, Request, timeout: Duration::seconds(5))

match result {
    Ok(response) => handle(response),
    Err(Timeout) => fallback()
}
```

### Parallel Operations

```simplex
// Run multiple operations in parallel
let (a, b, c) = await parallel(
    fetch_data(),
    compute_result(),
    ai::complete(prompt)
)
```

---

## Next Steps

1. **Read the spec**: [Overview](../spec/01-overview.md) and [Language Syntax](../spec/04-language-syntax.md)
2. **Explore examples**: [Document Pipeline](../examples/document-pipeline.md)
3. **Learn about AI**: [AI Integration](../spec/07-ai-integration.md)
4. **Plan deployment**: [Cost Optimization](../spec/08-cost-optimization.md)

---

## Getting Help

```bash
# Built-in help
simplex help
simplex help run
simplex help swarm

# Documentation
simplex docs  # Opens browser to docs
```

Report issues: https://github.com/simplex-lang/simplex/issues
