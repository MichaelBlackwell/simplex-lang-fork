# Chapter 7: Introduction to Actors

Now we enter Simplex's most distinctive feature: the **actor model**. Actors are isolated units of computation that communicate through messages. They're the foundation of fault-tolerant, distributed systems in Simplex.

---

## What is an Actor?

Think of an actor as a self-contained entity that:
- Has its own private state
- Communicates only through messages
- Processes one message at a time
- Can create other actors

In traditional programming, objects share memory and call each other's methods directly. In Simplex, actors are isolated—they can only interact through message passing.

```
    Traditional Objects:           Simplex Actors:

    +--------+                     +--------+
    | Object |----method call----->| Actor  |
    +--------+                     +--------+
         |                              |
         | shared state                 | message
         v                              v
    +--------+                     +--------+
    | Object |                     | Actor  |
    +--------+                     +--------+
```

---

## Your First Actor

Let's create a simple counter:

```simplex
actor Counter {
    // Private state
    var count: i64 = 0

    // Message handler: increment the counter
    receive Increment {
        count += 1
        print("Count is now: {count}")
    }
}

fn main() {
    // Spawn an actor
    let counter = spawn Counter

    // Send messages
    send(counter, Increment)
    send(counter, Increment)
    send(counter, Increment)

    // Give actor time to process
    sleep(Duration::milliseconds(100))
}
```

Output:
```
Count is now: 1
Count is now: 2
Count is now: 3
```

Let's break this down:
1. `actor Counter` defines an actor type
2. `var count` is private state only this actor can access
3. `receive Increment` handles messages of type `Increment`
4. `spawn Counter` creates a new actor instance
5. `send(counter, Increment)` sends a message

---

## Actor State

Each actor has its own isolated state:

```simplex
actor BankAccount {
    var balance: f64 = 0.0
    var owner: String = ""

    receive Initialize(name: String, initial_balance: f64) {
        owner = name
        balance = initial_balance
        print("Account created for {owner} with ${balance}")
    }

    receive Deposit(amount: f64) {
        balance += amount
        print("{owner} deposited ${amount}. New balance: ${balance}")
    }

    receive Withdraw(amount: f64) {
        if amount <= balance {
            balance -= amount
            print("{owner} withdrew ${amount}. New balance: ${balance}")
        } else {
            print("Insufficient funds for {owner}")
        }
    }
}

fn main() {
    let account = spawn BankAccount

    send(account, Initialize("Alice", 100.0))
    send(account, Deposit(50.0))
    send(account, Withdraw(30.0))
    send(account, Withdraw(200.0))

    sleep(Duration::milliseconds(100))
}
```

Output:
```
Account created for Alice with $100.0
Alice deposited $50.0. New balance: $150.0
Alice withdrew $30.0. New balance: $120.0
Insufficient funds for Alice
```

Notice: No other code can directly access `balance`. The only way to interact is through messages.

---

## Constructors

Use `init` for initialization logic:

```simplex
actor Timer {
    var name: String
    var started_at: Instant

    init(timer_name: String) {
        name = timer_name
        started_at = Instant::now()
        print("Timer '{name}' started")
    }

    receive Elapsed -> Duration {
        Instant::now().duration_since(started_at)
    }
}

fn main() {
    let timer = spawn Timer("MyTimer")

    sleep(Duration::seconds(2))

    // We'll learn about 'ask' in the next chapter
    let elapsed = ask(timer, Elapsed)
    print("Elapsed: {elapsed}")
}
```

---

## Multiple Message Types

Actors typically handle multiple message types:

```simplex
actor Calculator {
    var result: f64 = 0.0

    receive Clear {
        result = 0.0
        print("Cleared")
    }

    receive Add(n: f64) {
        result += n
        print("Added {n}, result: {result}")
    }

    receive Subtract(n: f64) {
        result -= n
        print("Subtracted {n}, result: {result}")
    }

    receive Multiply(n: f64) {
        result *= n
        print("Multiplied by {n}, result: {result}")
    }

    receive GetResult -> f64 {
        result
    }
}

fn main() {
    let calc = spawn Calculator

    send(calc, Add(10.0))
    send(calc, Multiply(3.0))
    send(calc, Subtract(5.0))

    sleep(Duration::milliseconds(100))
}
```

Output:
```
Added 10.0, result: 10.0
Multiplied by 3.0, result: 30.0
Subtracted 5.0, result: 25.0
```

---

## Message Ordering

Messages from the same sender to the same actor arrive in order:

```simplex
fn main() {
    let counter = spawn Counter

    // These will be processed in order: 1, 2, 3
    send(counter, Increment)  // 1
    send(counter, Increment)  // 2
    send(counter, Increment)  // 3
}
```

But messages from different actors may interleave:

```simplex
fn main() {
    let counter = spawn Counter

    // Start two actors that both send to counter
    let sender1 = spawn Sender(counter, "A")
    let sender2 = spawn Sender(counter, "B")

    // Messages from sender1 and sender2 may interleave
    // But all A's are in order, and all B's are in order
}
```

---

## One Message at a Time

An actor processes only one message at a time. This eliminates race conditions:

```simplex
actor SafeCounter {
    var count: i64 = 0

    receive Increment {
        // No race condition! Only one message runs at a time.
        let current = count
        // Even if we sleep here, no other message can interfere
        count = current + 1
    }
}
```

This is fundamentally different from threads with shared memory, where you'd need locks.

---

## Actor References

When you spawn an actor, you get a reference:

```simplex
fn main() {
    let counter: ActorRef<Counter> = spawn Counter

    // References can be stored and passed around
    let refs: List<ActorRef<Counter>> = [
        spawn Counter,
        spawn Counter,
        spawn Counter
    ]

    // Send to all
    for r in refs {
        send(r, Increment)
    }
}
```

References are:
- Lightweight (just an ID)
- Safe to copy and share
- Location-transparent (work the same locally or remotely)

---

## Self Reference

Actors can refer to themselves with `self`:

```simplex
actor Repeater {
    var count: i64 = 0

    receive Start(times: i64) {
        if times > 0 {
            count += 1
            print("Repeat {count}")
            send(self, Start(times - 1))  // Send message to self
        } else {
            print("Done after {count} repeats")
        }
    }
}

fn main() {
    let repeater = spawn Repeater
    send(repeater, Start(5))

    sleep(Duration::milliseconds(100))
}
```

Output:
```
Repeat 1
Repeat 2
Repeat 3
Repeat 4
Repeat 5
Done after 5 repeats
```

---

## Lifecycle Hooks

Actors have lifecycle events you can hook into:

```simplex
actor Service {
    var connections: i64 = 0

    on_start() {
        print("Service starting up...")
        // Initialize resources
    }

    on_stop() {
        print("Service shutting down...")
        // Clean up resources
    }

    receive Connect {
        connections += 1
        print("New connection. Total: {connections}")
    }
}

fn main() {
    let service = spawn Service
    send(service, Connect)
    send(service, Connect)

    sleep(Duration::milliseconds(100))
}
```

Output:
```
Service starting up...
New connection. Total: 1
New connection. Total: 2
```

---

## Why Actors?

You might wonder: why not just use objects and methods?

**Isolation**: Actors can't accidentally corrupt each other's state. Each actor is an island.

**Concurrency**: Actors naturally run concurrently. No need for manual thread management.

**Location transparency**: An actor reference works the same whether the actor is local or on another machine.

**Fault tolerance**: When an actor crashes, it's contained. Other actors keep running. (We'll explore this in the Supervision chapter.)

**Scalability**: Need more capacity? Spawn more actors. They distribute naturally.

---

## When to Use Actors

Use actors when you have:

| Scenario | Example |
|----------|---------|
| Independent entities | Users, sessions, devices |
| Long-running state | Shopping carts, game characters |
| Concurrent operations | Request handlers, workers |
| Distributed systems | Microservices, IoT devices |

Don't use actors for:
- Simple data transformations (use functions)
- Stateless calculations (use functions)
- Tightly coupled operations (use single actor)

---

## Summary

| Concept | Syntax | Purpose |
|---------|--------|---------|
| Define actor | `actor Name { }` | Create actor type |
| State | `var field: Type` | Private data |
| Constructor | `init(params) { }` | Initialize state |
| Message handler | `receive MsgType { }` | Handle messages |
| Spawn | `spawn ActorType` | Create instance |
| Send | `send(ref, Message)` | Send message |
| Self reference | `self` | Current actor |
| Lifecycle | `on_start()`, `on_stop()` | Setup/teardown |

---

## Exercises

1. **TodoList Actor**: Create an actor that manages a todo list with these messages:
   - `Add(task: String)` - add a task
   - `Remove(task: String)` - remove a task
   - `List` - print all tasks

2. **Temperature Sensor**: Create an actor that:
   - Stores temperature readings in a list
   - Has `Record(temp: f64)` to add a reading
   - Has `Average` that prints the average temperature

3. **Ping Pong**: Create two actor types, Ping and Pong. Ping sends "ping" to Pong, Pong responds with "pong" back to Ping. Make them volley 5 times.

---

## Answers

<details>
<summary>Click to reveal answers</summary>

**Exercise 1:**
```simplex
actor TodoList {
    var tasks: List<String> = []

    receive Add(task: String) {
        tasks.push_mut(task.clone())
        print("Added: {task}")
    }

    receive Remove(task: String) {
        tasks = tasks.filter(t => t != task)
        print("Removed: {task}")
    }

    receive List {
        print("Tasks:")
        for (i, task) in tasks.enumerate() {
            print("  {i + 1}. {task}")
        }
    }
}

fn main() {
    let todos = spawn TodoList

    send(todos, Add("Buy groceries"))
    send(todos, Add("Write code"))
    send(todos, Add("Exercise"))
    send(todos, List)
    send(todos, Remove("Write code"))
    send(todos, List)

    sleep(Duration::milliseconds(100))
}
```

**Exercise 2:**
```simplex
actor TemperatureSensor {
    var readings: List<f64> = []

    receive Record(temp: f64) {
        readings.push_mut(temp)
        print("Recorded: {temp}°C")
    }

    receive Average {
        if readings.is_empty() {
            print("No readings yet")
        } else {
            let sum = readings.fold(0.0, (acc, t) => acc + t)
            let avg = sum / readings.len().to_f64()
            print("Average: {avg}°C")
        }
    }
}

fn main() {
    let sensor = spawn TemperatureSensor

    send(sensor, Record(20.5))
    send(sensor, Record(22.0))
    send(sensor, Record(21.5))
    send(sensor, Average)

    sleep(Duration::milliseconds(100))
}
```

**Exercise 3:**
```simplex
actor Ping {
    var pong_ref: Option<ActorRef<Pong>> = None
    var volleys: i64 = 0

    receive SetPong(pong: ActorRef<Pong>) {
        pong_ref = Some(pong)
    }

    receive Start {
        volleys = 0
        if let Some(pong) = pong_ref {
            print("Ping!")
            volleys += 1
            send(pong, Ball(self))
        }
    }

    receive Ball(from: ActorRef<Pong>) {
        volleys += 1
        if volleys <= 5 {
            print("Ping!")
            send(from, Ball(self))
        } else {
            print("Game over!")
        }
    }
}

actor Pong {
    receive Ball(from: ActorRef<Ping>) {
        print("Pong!")
        send(from, Ball(self))
    }
}

fn main() {
    let ping = spawn Ping
    let pong = spawn Pong

    send(ping, SetPong(pong))
    send(ping, Start)

    sleep(Duration::milliseconds(100))
}
```

</details>

---

*Next: [Chapter 8: Message Passing →](08-messages.md)*
