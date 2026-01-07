# Chapter 8: Message Passing

In the last chapter, we used `send` to fire-and-forget messages. But actors often need to communicate bidirectionally. This chapter covers the full range of message passing patterns.

---

## Send vs Ask

Simplex provides two ways to send messages:

| Pattern | Syntax | Behavior |
|---------|--------|----------|
| `send` | `send(actor, Message)` | Fire and forget, non-blocking |
| `ask` | `ask(actor, Message)` | Wait for response |

### Send: Fire and Forget

```simplex
actor Logger {
    receive Log(message: String) {
        print("[LOG] {message}")
    }
}

fn main() {
    let logger = spawn Logger

    // Send doesn't wait for the message to be processed
    send(logger, Log("Starting up"))
    send(logger, Log("Doing work"))
    send(logger, Log("Shutting down"))

    // Need to wait for processing
    sleep(Duration::milliseconds(100))
}
```

Use `send` when:
- You don't need a response
- Fire-and-forget is acceptable
- Maximum throughput is important

### Ask: Request-Response

```simplex
actor Calculator {
    receive Add(a: i64, b: i64) -> i64 {
        a + b
    }

    receive Multiply(a: i64, b: i64) -> i64 {
        a * b
    }
}

fn main() {
    let calc = spawn Calculator

    // Ask waits for and returns the response
    let sum = ask(calc, Add(5, 3))
    print("5 + 3 = {sum}")  // 5 + 3 = 8

    let product = ask(calc, Multiply(4, 7))
    print("4 × 7 = {product}")  // 4 × 7 = 28
}
```

Notice the `-> i64` return type in the message handler. This indicates the message expects a response.

Use `ask` when:
- You need the result to continue
- The operation is synchronous from your perspective
- Request-response semantics are needed

---

## Message Types

### Messages Without Data

```simplex
actor Door {
    var is_open: Bool = false

    receive Open {
        is_open = true
        print("Door opened")
    }

    receive Close {
        is_open = false
        print("Door closed")
    }

    receive IsOpen -> Bool {
        is_open
    }
}
```

### Messages With Data

```simplex
actor Greeter {
    receive Greet(name: String) {
        print("Hello, {name}!")
    }

    receive GreetMany(names: List<String>) {
        for name in names {
            print("Hello, {name}!")
        }
    }

    receive PersonalGreeting(name: String, title: String) -> String {
        "Good day, {title} {name}!"
    }
}

fn main() {
    let greeter = spawn Greeter

    send(greeter, Greet("Alice"))
    send(greeter, GreetMany(["Bob", "Charlie", "Diana"]))

    let greeting = ask(greeter, PersonalGreeting("Smith", "Dr."))
    print(greeting)  // Good day, Dr. Smith!
}
```

### Complex Message Types

Messages can carry any type:

```simplex
type Order {
    id: String,
    items: List<Item>,
    total: f64
}

type Item {
    name: String,
    quantity: i64,
    price: f64
}

actor OrderProcessor {
    receive ProcessOrder(order: Order) -> Result<Receipt, OrderError> {
        print("Processing order {order.id}")

        // Validate
        if order.items.is_empty() {
            return Err(OrderError::EmptyOrder)
        }

        // Process...
        Ok(Receipt {
            order_id: order.id,
            status: "completed"
        })
    }
}
```

---

## Timeouts

What if an actor doesn't respond? Use timeouts:

```simplex
actor SlowService {
    receive DoWork -> String {
        sleep(Duration::seconds(5))  // Slow!
        "Done"
    }
}

fn main() {
    let service = spawn SlowService

    // With timeout
    match ask(service, DoWork, timeout: Duration::seconds(2)) {
        Ok(result) => print("Got: {result}"),
        Err(Timeout) => print("Service took too long!")
    }
}
```

Output:
```
Service took too long!
```

Always use timeouts in production code to prevent indefinite blocking.

---

## Passing Actor References

Actors can send references to other actors:

```simplex
actor Worker {
    receive DoJob(job: String, reply_to: ActorRef<Manager>) {
        print("Working on: {job}")
        sleep(Duration::milliseconds(100))
        send(reply_to, JobComplete(job))
    }
}

actor Manager {
    receive StartJob(job: String, worker: ActorRef<Worker>) {
        print("Delegating job: {job}")
        send(worker, DoJob(job, self))
    }

    receive JobComplete(job: String) {
        print("Job completed: {job}")
    }
}

fn main() {
    let worker = spawn Worker
    let manager = spawn Manager

    send(manager, StartJob("Build report", worker))

    sleep(Duration::milliseconds(200))
}
```

Output:
```
Delegating job: Build report
Working on: Build report
Job completed: Build report
```

---

## Request-Response Patterns

### Simple Request-Response

```simplex
actor Database {
    var data: Map<String, String> = {}

    receive Get(key: String) -> Option<String> {
        data.get(key)
    }

    receive Set(key: String, value: String) {
        data.insert(key, value)
    }
}

fn main() {
    let db = spawn Database

    send(db, Set("name", "Alice"))
    send(db, Set("city", "Boston"))

    sleep(Duration::milliseconds(50))

    let name = ask(db, Get("name"))
    print("Name: {name}")  // Name: Some("Alice")
}
```

### Async with Callback

When you don't want to block:

```simplex
actor AsyncService {
    receive FetchData(query: String, callback: ActorRef<Callback>) {
        // Simulate async work
        let result = expensive_operation(query)
        send(callback, DataReady(result))
    }
}

actor Client {
    receive RequestData(service: ActorRef<AsyncService>, query: String) {
        print("Requesting data...")
        send(service, FetchData(query, self))
        // Continue doing other work immediately
        print("Request sent, doing other work...")
    }

    receive DataReady(data: String) {
        print("Data received: {data}")
    }
}
```

---

## Broadcasting

Send messages to multiple actors:

```simplex
actor Subscriber {
    var name: String

    init(name: String) {
        self.name = name
    }

    receive Notify(message: String) {
        print("[{name}] Received: {message}")
    }
}

actor Publisher {
    var subscribers: List<ActorRef<Subscriber>> = []

    receive Subscribe(subscriber: ActorRef<Subscriber>) {
        subscribers.push_mut(subscriber)
    }

    receive Publish(message: String) {
        for sub in subscribers {
            send(sub, Notify(message))
        }
    }
}

fn main() {
    let pub = spawn Publisher

    let sub1 = spawn Subscriber("Alice")
    let sub2 = spawn Subscriber("Bob")
    let sub3 = spawn Subscriber("Charlie")

    send(pub, Subscribe(sub1))
    send(pub, Subscribe(sub2))
    send(pub, Subscribe(sub3))

    send(pub, Publish("Hello everyone!"))

    sleep(Duration::milliseconds(100))
}
```

Output:
```
[Alice] Received: Hello everyone!
[Bob] Received: Hello everyone!
[Charlie] Received: Hello everyone!
```

---

## Message Queues

Each actor has an inbox (message queue). Messages wait in the queue until processed:

```simplex
actor SlowProcessor {
    receive Process(item: String) {
        print("Starting: {item}")
        sleep(Duration::seconds(1))  // Slow processing
        print("Finished: {item}")
    }
}

fn main() {
    let processor = spawn SlowProcessor

    // All messages go to queue immediately
    send(processor, Process("A"))
    send(processor, Process("B"))
    send(processor, Process("C"))

    print("All messages sent!")

    sleep(Duration::seconds(4))
}
```

Output:
```
All messages sent!
Starting: A
Finished: A
Starting: B
Finished: B
Starting: C
Finished: C
```

Messages are queued and processed one at a time.

---

## Worker Pool Pattern

Distribute work across multiple actors:

```simplex
actor Worker {
    var id: i64

    init(worker_id: i64) {
        id = worker_id
    }

    receive DoTask(task: String) -> String {
        print("Worker {id} processing: {task}")
        sleep(Duration::milliseconds(100))
        "Result from worker {id}: {task}"
    }
}

actor Pool {
    var workers: List<ActorRef<Worker>> = []
    var next: i64 = 0

    init(size: i64) {
        for i in 0..size {
            workers.push_mut(spawn Worker(i))
        }
    }

    receive Submit(task: String) -> String {
        // Round-robin distribution
        let worker = workers[next % workers.len()]
        next += 1

        ask(worker, DoTask(task))
    }
}

fn main() {
    let pool = spawn Pool(3)

    // Distribute tasks across workers
    for i in 0..6 {
        let result = ask(pool, Submit("Task {i}"))
        print(result)
    }
}
```

---

## Best Practices

### 1. Prefer Send Over Ask

`ask` blocks the caller. When possible, use `send` with callbacks:

```simplex
// Blocking (avoid when possible)
let result = ask(actor, Query)
process(result)

// Non-blocking (preferred for performance)
send(actor, QueryWithCallback(self))
// ... continue working ...
receive QueryResult(result) {
    process(result)
}
```

### 2. Keep Messages Small

```simplex
// Good: send reference or ID
send(processor, ProcessDocument(doc_id))

// Avoid: sending huge data in every message
send(processor, ProcessDocument(huge_document_data))
```

### 3. Always Use Timeouts

```simplex
// Dangerous: blocks forever if actor is dead
let result = ask(actor, Query)

// Safe: fails gracefully
match ask(actor, Query, timeout: Duration::seconds(5)) {
    Ok(result) => use(result),
    Err(Timeout) => handle_failure()
}
```

### 4. Design Messages Carefully

```simplex
// Good: clear, typed messages
receive CreateUser(name: String, email: String) -> Result<User, Error>

// Avoid: untyped, unclear messages
receive HandleRequest(data: Map<String, Dynamic>) -> Dynamic
```

---

## Summary

| Pattern | Use Case | Syntax |
|---------|----------|--------|
| Fire-and-forget | Notifications, logging | `send(actor, Msg)` |
| Request-response | Queries, operations | `ask(actor, Msg)` |
| With timeout | Production code | `ask(actor, Msg, timeout: dur)` |
| Callback | Async without blocking | `send(actor, MsgWithCallback(self))` |
| Broadcast | Pub/sub, notifications | Loop over actor refs |

---

## Exercises

1. **Chat Room**: Create a chat room actor that:
   - Accepts `Join(user: ActorRef<User>)` to add users
   - Accepts `Say(from: String, message: String)` to broadcast to all users
   - User actors receive `Message(from: String, text: String)`

2. **Request Counter**: Create an actor that tracks how many requests it has processed. Include `GetCount -> i64` to query the count.

3. **Load Balancer**: Create a load balancer actor that distributes `Work(data: String)` messages across multiple worker actors using round-robin.

---

## Answers

<details>
<summary>Click to reveal answers</summary>

**Exercise 1:**
```simplex
actor User {
    var name: String

    init(name: String) {
        self.name = name
    }

    receive Message(from: String, text: String) {
        print("[{name}] {from}: {text}")
    }
}

actor ChatRoom {
    var users: List<ActorRef<User>> = []

    receive Join(user: ActorRef<User>) {
        users.push_mut(user)
        print("User joined the room")
    }

    receive Say(from: String, message: String) {
        for user in users {
            send(user, Message(from, message))
        }
    }
}

fn main() {
    let room = spawn ChatRoom

    let alice = spawn User("Alice")
    let bob = spawn User("Bob")

    send(room, Join(alice))
    send(room, Join(bob))

    send(room, Say("Alice", "Hello everyone!"))
    send(room, Say("Bob", "Hi Alice!"))

    sleep(Duration::milliseconds(100))
}
```

**Exercise 2:**
```simplex
actor RequestCounter {
    var count: i64 = 0

    receive HandleRequest(data: String) {
        count += 1
        print("Handling request {count}: {data}")
    }

    receive GetCount -> i64 {
        count
    }
}

fn main() {
    let counter = spawn RequestCounter

    send(counter, HandleRequest("First"))
    send(counter, HandleRequest("Second"))
    send(counter, HandleRequest("Third"))

    sleep(Duration::milliseconds(100))

    let total = ask(counter, GetCount)
    print("Total requests: {total}")  // 3
}
```

**Exercise 3:**
```simplex
actor Worker {
    var id: i64

    init(id: i64) {
        self.id = id
    }

    receive Work(data: String) {
        print("Worker {id} processing: {data}")
    }
}

actor LoadBalancer {
    var workers: List<ActorRef<Worker>> = []
    var current: i64 = 0

    receive AddWorker(worker: ActorRef<Worker>) {
        workers.push_mut(worker)
    }

    receive Dispatch(data: String) {
        if workers.is_empty() {
            print("No workers available!")
            return
        }

        let worker = workers[current % workers.len()]
        current += 1
        send(worker, Work(data))
    }
}

fn main() {
    let lb = spawn LoadBalancer

    // Add workers
    for i in 0..3 {
        send(lb, AddWorker(spawn Worker(i)))
    }

    // Dispatch work
    for i in 0..9 {
        send(lb, Dispatch("Task {i}"))
    }

    sleep(Duration::milliseconds(100))
}
```

</details>

---

*Next: [Chapter 9: Supervision and Fault Tolerance →](09-supervision.md)*
