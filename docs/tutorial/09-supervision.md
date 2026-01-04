# Chapter 9: Supervision and Fault Tolerance

Things fail. Networks disconnect, databases time out, and code has bugs. Traditional programming tries to prevent all failures—an impossible task. Simplex takes a different approach: **let it crash, then recover automatically**.

---

## The "Let It Crash" Philosophy

Instead of defensive programming with try-catch everywhere:

```javascript
// Traditional approach: handle every possible error
try {
    data = fetchData();
    try {
        parsed = parseData(data);
        try {
            result = processData(parsed);
        } catch (processError) {
            handleProcessError(processError);
        }
    } catch (parseError) {
        handleParseError(parseError);
    }
} catch (fetchError) {
    handleFetchError(fetchError);
}
```

Simplex says: write the happy path, and let supervisors handle failures:

```simplex
actor DataProcessor {
    receive Process(id: String) {
        let data = fetch_data(id)    // Might fail
        let parsed = parse_data(data) // Might fail
        let result = process(parsed)  // Might fail
        save(result)
    }
    // If anything fails, actor crashes
    // Supervisor restarts it automatically
}
```

---

## What is a Supervisor?

A supervisor is a special actor that:
- Watches over child actors
- Detects when children crash
- Restarts them according to a strategy

```
    +-------------------+
    |    Supervisor     |
    +-------------------+
           |
    +------+------+------+
    |      |      |      |
    v      v      v      v
  +---+  +---+  +---+  +---+
  | A |  | B |  | C |  | D |
  +---+  +---+  +---+  +---+
  Child actors (workers)
```

If actor B crashes, the supervisor can restart it without affecting A, C, or D.

---

## Defining a Supervisor

```simplex
actor Worker {
    var id: i64

    init(worker_id: i64) {
        id = worker_id
        print("Worker {id} started")
    }

    receive DoWork(task: String) {
        print("Worker {id} doing: {task}")

        // Simulate occasional failure
        if task == "bad" {
            panic("Worker {id} crashed!")
        }
    }

    on_stop() {
        print("Worker {id} stopped")
    }
}

supervisor WorkerPool {
    strategy: OneForOne,     // Only restart the failed child
    max_restarts: 3,         // Max 3 restarts...
    within: Duration::seconds(60),  // ...within 60 seconds

    children: [
        child(Worker(1), restart: Always),
        child(Worker(2), restart: Always),
        child(Worker(3), restart: Always)
    ]
}

fn main() {
    let pool = spawn WorkerPool

    // Get references to workers
    let w1 = pool.child(0)
    let w2 = pool.child(1)

    send(w1, DoWork("good"))
    send(w2, DoWork("good"))
    send(w1, DoWork("bad"))  // This will crash worker 1
    send(w1, DoWork("good"))  // After restart, this works

    sleep(Duration::seconds(1))
}
```

Output:
```
Worker 1 started
Worker 2 started
Worker 3 started
Worker 1 doing: good
Worker 2 doing: good
Worker 1 doing: bad
Worker 1 stopped
Worker 1 started     <-- Automatic restart!
Worker 1 doing: good
```

---

## Restart Strategies

### OneForOne

Restart only the failed child:

```simplex
supervisor Pool {
    strategy: OneForOne,
    children: [
        child(Worker(1)),  // If this fails...
        child(Worker(2)),  // ...these continue running
        child(Worker(3))
    ]
}
```

Use when children are independent.

### OneForAll

Restart ALL children if ANY fails:

```simplex
supervisor Pipeline {
    strategy: OneForAll,
    children: [
        child(Fetcher),    // If any fails...
        child(Parser),     // ...all are restarted
        child(Writer)      // ...to ensure consistent state
    ]
}
```

Use when children depend on each other and must be in sync.

### RestForOne

Restart the failed child AND all children started after it:

```simplex
supervisor Chain {
    strategy: RestForOne,
    children: [
        child(Database),   // If this fails, restart all
        child(Cache),      // If this fails, restart Cache + Api
        child(Api)         // If this fails, only restart Api
    ]
}
```

Use when later children depend on earlier ones.

---

## Restart Policies

Control when children are restarted:

```simplex
supervisor System {
    children: [
        // Always restart, no matter how it stopped
        child(CriticalService, restart: Always),

        // Only restart if it crashed (not normal shutdown)
        child(Worker, restart: Transient),

        // Never restart
        child(OneTimeTask, restart: Never)
    ]
}
```

| Policy | Restart on Crash | Restart on Normal Exit |
|--------|------------------|------------------------|
| Always | Yes | Yes |
| Transient | Yes | No |
| Never | No | No |

---

## Rate Limiting Restarts

Prevent restart loops:

```simplex
supervisor Protected {
    strategy: OneForOne,
    max_restarts: 5,              // Maximum 5 restarts...
    within: Duration::minutes(1), // ...in 1 minute

    children: [
        child(FlakeyService)
    ]
}
```

If the child crashes more than 5 times in 1 minute, the supervisor itself fails. This prevents infinite restart loops from consuming resources.

---

## Supervision Trees

Supervisors can supervise other supervisors, creating a tree:

```simplex
supervisor TopLevel {
    strategy: OneForOne,
    children: [
        child(DatabaseSupervisor),
        child(WebSupervisor),
        child(WorkerSupervisor)
    ]
}

supervisor DatabaseSupervisor {
    strategy: OneForAll,  // DB components must be in sync
    children: [
        child(ConnectionPool),
        child(QueryCache)
    ]
}

supervisor WebSupervisor {
    strategy: OneForOne,  // Web handlers are independent
    children: [
        child(RequestHandler(1)),
        child(RequestHandler(2)),
        child(RequestHandler(3))
    ]
}

supervisor WorkerSupervisor {
    strategy: OneForOne,
    children: [
        child(BackgroundWorker(1)),
        child(BackgroundWorker(2))
    ]
}
```

```
                    TopLevel
                       |
        +--------------+--------------+
        |              |              |
    Database        Web           Workers
        |              |              |
    +---+---+      +--+--+--+      +--+--+
    |       |      |  |  |  |      |     |
  Pool   Cache   H1  H2  H3       W1    W2
```

If a web handler crashes, only it restarts. If the entire database supervisor crashes, all database components restart together.

---

## Checkpointing for Recovery

When an actor restarts, it loses its state. Use checkpointing to persist important data:

```simplex
actor OrderProcessor {
    var pending_orders: Map<String, Order> = {}

    receive ProcessOrder(order: Order) {
        // Save state before risky operation
        pending_orders.insert(order.id, order)
        checkpoint()  // Persist to durable storage

        // Do risky work
        let result = charge_payment(order)

        match result {
            Ok(_) => {
                pending_orders.remove(order.id)
                checkpoint()  // Update persisted state
            },
            Err(e) => {
                // If we crash here, order is still in pending_orders
                log::error("Payment failed: {e}")
            }
        }
    }

    on_resume() {
        // Called after restart - recover pending orders
        for (id, order) in pending_orders {
            print("Resuming order: {id}")
            send(self, ProcessOrder(order))
        }
    }
}
```

---

## Lifecycle Hooks

Actors have several lifecycle hooks:

```simplex
actor LifecycleDemo {
    on_start() {
        print("1. Actor starting")
        // Initialize resources
    }

    on_checkpoint() {
        print("2. Checkpointing state")
        // Called when checkpoint() is invoked
    }

    on_resume() {
        print("3. Resuming from checkpoint")
        // Called after restart from checkpoint
    }

    on_stop() {
        print("4. Actor stopping")
        // Clean up resources
    }
}
```

---

## Graceful Shutdown

Actors can be stopped gracefully:

```simplex
actor Server {
    var connections: List<Connection> = []

    receive Connect(conn: Connection) {
        connections.push_mut(conn)
    }

    on_stop() {
        // Graceful shutdown: close all connections
        for conn in connections {
            conn.close()
        }
        print("Server shut down gracefully")
    }
}

fn main() {
    let server = spawn Server

    // ... use server ...

    // Request graceful shutdown
    stop(server)
}
```

---

## Error Escalation

Sometimes an actor should tell its supervisor about a problem:

```simplex
actor Worker {
    receive DoWork(task: Task) -> Result<Output, WorkError> {
        match process(task) {
            Ok(output) => Ok(output),
            Err(e) if e.is_recoverable() => {
                // Try to handle it ourselves
                log::warn("Recoverable error: {e}")
                retry(task)
            },
            Err(e) => {
                // Can't handle - crash and let supervisor decide
                panic("Unrecoverable error: {e}")
            }
        }
    }
}
```

---

## Practical Example: Resilient Web Service

```simplex
// Worker that handles web requests
actor RequestHandler {
    var id: i64

    init(id: i64) {
        self.id = id
    }

    receive HandleRequest(req: Request) -> Response {
        log::info("Handler {id} processing: {req.path}")

        // These operations might fail
        let data = fetch_from_database(req)?
        let processed = transform(data)?
        let response = format_response(processed)?

        response
    }
}

// Connection pool
actor DatabasePool {
    var connections: List<Connection> = []

    init(size: i64) {
        for _ in 0..size {
            connections.push_mut(create_connection())
        }
    }

    receive Query(sql: String) -> Result<Rows, DbError> {
        let conn = connections.get_available()?
        conn.execute(sql)
    }
}

// Cache that might need to be rebuilt
actor Cache {
    var data: Map<String, String> = {}

    on_start() {
        // Rebuild cache on start
        data = load_cache_from_disk()
    }

    receive Get(key: String) -> Option<String> {
        data.get(key)
    }

    receive Set(key: String, value: String) {
        data.insert(key, value)
        checkpoint()  // Persist cache
    }
}

// Supervision tree
supervisor WebService {
    strategy: OneForOne,
    max_restarts: 10,
    within: Duration::minutes(1),

    children: [
        child(DatabasePool(5), restart: Always),
        child(Cache, restart: Always)
    ]
}

supervisor RequestHandlers {
    strategy: OneForOne,
    max_restarts: 100,
    within: Duration::minutes(1),

    children: [
        child(RequestHandler(1), restart: Always),
        child(RequestHandler(2), restart: Always),
        child(RequestHandler(3), restart: Always),
        child(RequestHandler(4), restart: Always)
    ]
}

supervisor Application {
    strategy: OneForAll,  // If core services fail, restart everything
    children: [
        child(WebService),
        child(RequestHandlers)
    ]
}

fn main() {
    let app = spawn Application
    print("Application started")
    // Runs forever, automatically recovering from failures
}
```

---

## Summary

| Concept | Purpose |
|---------|---------|
| Supervisor | Watches and restarts child actors |
| OneForOne | Restart only failed child |
| OneForAll | Restart all children |
| RestForOne | Restart failed child and successors |
| Always | Always restart |
| Transient | Restart only on crash |
| Never | Never restart |
| max_restarts/within | Rate limit restarts |
| checkpoint() | Persist state for recovery |
| on_resume() | Recover from checkpoint |

---

## Exercises

1. **Retry Service**: Create a supervisor that manages a "flaky" worker that fails 50% of the time. The supervisor should restart it up to 3 times.

2. **Pipeline Supervisor**: Create a supervisor for a data pipeline (Fetcher → Processor → Writer) using RestForOne strategy.

3. **Circuit Breaker**: Implement an actor that tracks failures. After 5 failures, it stops accepting requests for 30 seconds (circuit "open"), then tries again.

---

*Next: [Chapter 10: AI Integration →](10-ai-basics.md)*
