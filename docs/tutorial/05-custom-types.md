# Chapter 5: Custom Types

So far we've used built-in types. Now let's create our own types to model the data in our programs.

---

## Structs

Structs group related data together.

### Defining Structs

```simplex
type User {
    name: String,
    email: String,
    age: i64
}

fn main() {
    let alice = User {
        name: "Alice",
        email: "alice@example.com",
        age: 30
    }

    print("Name: {alice.name}")
    print("Email: {alice.email}")
    print("Age: {alice.age}")
}
```

### Shorthand Initialization

When variable names match field names:

```simplex
fn main() {
    let name = "Bob"
    let email = "bob@example.com"
    let age = 25

    // Shorthand: don't repeat "name: name"
    let bob = User { name, email, age }

    print(bob.name)  // Bob
}
```

### Nested Structs

```simplex
type Address {
    street: String,
    city: String,
    country: String
}

type Person {
    name: String,
    address: Address
}

fn main() {
    let person = Person {
        name: "Charlie",
        address: Address {
            street: "123 Main St",
            city: "Springfield",
            country: "USA"
        }
    }

    print("{person.name} lives in {person.address.city}")
}
```

### Updating Structs

Structs are immutable, but you can create modified copies:

```simplex
fn main() {
    let alice = User {
        name: "Alice",
        email: "alice@old.com",
        age: 30
    }

    // Create new struct with one field changed
    let updated = User {
        email: "alice@new.com",
        ..alice  // Copy remaining fields
    }

    print(alice.email)    // alice@old.com (unchanged)
    print(updated.email)  // alice@new.com
}
```

---

## Methods

Add behavior to your types with `impl` blocks:

```simplex
type Rectangle {
    width: f64,
    height: f64
}

impl Rectangle {
    // Method: takes &self (reference to instance)
    fn area(&self) -> f64 {
        self.width * self.height
    }

    fn perimeter(&self) -> f64 {
        2.0 * (self.width + self.height)
    }

    fn is_square(&self) -> Bool {
        self.width == self.height
    }
}

fn main() {
    let rect = Rectangle { width: 10.0, height: 5.0 }

    print("Area: {rect.area()}")          // Area: 50.0
    print("Perimeter: {rect.perimeter()}") // Perimeter: 30.0
    print("Square? {rect.is_square()}")    // Square? false
}
```

### Associated Functions (Constructors)

Functions that don't take `self`:

```simplex
impl Rectangle {
    // Associated function (no self)
    fn new(width: f64, height: f64) -> Rectangle {
        Rectangle { width, height }
    }

    fn square(size: f64) -> Rectangle {
        Rectangle { width: size, height: size }
    }
}

fn main() {
    let rect = Rectangle::new(10.0, 5.0)
    let square = Rectangle::square(7.0)

    print(rect.area())    // 50.0
    print(square.area())  // 49.0
}
```

### Method Chaining

Return `self` for fluent interfaces:

```simplex
type StringBuilder {
    content: String
}

impl StringBuilder {
    fn new() -> StringBuilder {
        StringBuilder { content: "" }
    }

    fn append(&self, text: String) -> StringBuilder {
        StringBuilder {
            content: "{self.content}{text}"
        }
    }

    fn build(&self) -> String {
        self.content.clone()
    }
}

fn main() {
    let result = StringBuilder::new()
        .append("Hello")
        .append(" ")
        .append("World")
        .build()

    print(result)  // Hello World
}
```

---

## Enums

Enums define a type that can be one of several variants.

### Simple Enums

```simplex
enum Direction {
    North,
    South,
    East,
    West
}

fn describe(dir: Direction) -> String {
    match dir {
        Direction::North => "Going up",
        Direction::South => "Going down",
        Direction::East => "Going right",
        Direction::West => "Going left"
    }
}

fn main() {
    let dir = Direction::North
    print(describe(dir))  // Going up
}
```

### Enums with Data

Each variant can hold different data:

```simplex
enum Message {
    Quit,                       // No data
    Move { x: i64, y: i64 },   // Named fields
    Write(String),              // Single value
    ChangeColor(i64, i64, i64)  // Multiple values
}

fn process(msg: Message) {
    match msg {
        Message::Quit => print("Quitting"),
        Message::Move { x, y } => print("Moving to ({x}, {y})"),
        Message::Write(text) => print("Writing: {text}"),
        Message::ChangeColor(r, g, b) => print("Color: RGB({r},{g},{b})")
    }
}

fn main() {
    process(Message::Quit)
    process(Message::Move { x: 10, y: 20 })
    process(Message::Write("Hello"))
    process(Message::ChangeColor(255, 128, 0))
}
```

### The Option Enum

You've seen `Option` already—it's just an enum:

```simplex
// This is how Option is defined (built-in)
enum Option<T> {
    Some(T),
    None
}

fn divide(a: i64, b: i64) -> Option<i64> {
    if b == 0 {
        None
    } else {
        Some(a / b)
    }
}

fn main() {
    match divide(10, 2) {
        Some(result) => print("Result: {result}"),
        None => print("Cannot divide by zero")
    }
}
```

### The Result Enum

Similarly, `Result` is an enum for operations that can fail:

```simplex
// This is how Result is defined (built-in)
enum Result<T, E> {
    Ok(T),
    Err(E)
}

enum MathError {
    DivisionByZero,
    NegativeSquareRoot
}

fn safe_divide(a: f64, b: f64) -> Result<f64, MathError> {
    if b == 0.0 {
        Err(MathError::DivisionByZero)
    } else {
        Ok(a / b)
    }
}
```

### Enum Methods

Enums can have methods too:

```simplex
enum Status {
    Pending,
    Active,
    Completed,
    Failed(String)
}

impl Status {
    fn is_finished(&self) -> Bool {
        match self {
            Status::Completed => true,
            Status::Failed(_) => true,
            _ => false
        }
    }

    fn description(&self) -> String {
        match self {
            Status::Pending => "Waiting to start",
            Status::Active => "In progress",
            Status::Completed => "Done",
            Status::Failed(reason) => "Failed: {reason}"
        }
    }
}

fn main() {
    let status = Status::Failed("Network error")
    print(status.description())     // Failed: Network error
    print(status.is_finished())     // true
}
```

---

## Type Aliases

Give existing types new names:

```simplex
type UserId = String
type Score = i64
type UserScores = Map<UserId, Score>

fn get_top_scorer(scores: UserScores) -> Option<UserId> {
    var best: Option<(UserId, Score)> = None

    for (id, score) in scores {
        match best {
            None => best = Some((id, score)),
            Some((_, best_score)) if score > best_score => {
                best = Some((id, score))
            },
            _ => {}
        }
    }

    best.map((id, _) => id)
}
```

Type aliases improve readability but don't create new types—`UserId` is still `String`.

---

## Generic Types

Make types work with any data:

```simplex
type Pair<T> {
    first: T,
    second: T
}

impl<T> Pair<T> {
    fn new(first: T, second: T) -> Pair<T> {
        Pair { first, second }
    }

    fn swap(&self) -> Pair<T> {
        Pair {
            first: self.second.clone(),
            second: self.first.clone()
        }
    }
}

fn main() {
    let numbers = Pair::new(1, 2)
    let strings = Pair::new("hello", "world")

    print(numbers.first)   // 1
    print(strings.second)  // world
}
```

### Multiple Type Parameters

```simplex
type KeyValue<K, V> {
    key: K,
    value: V
}

fn main() {
    let entry = KeyValue {
        key: "name",
        value: 42
    }

    print("{entry.key}: {entry.value}")
}
```

---

## Traits (Interfaces)

Traits define shared behavior:

```simplex
trait Printable {
    fn to_display(&self) -> String
}

type Point {
    x: f64,
    y: f64
}

impl Printable for Point {
    fn to_display(&self) -> String {
        "({self.x}, {self.y})"
    }
}

type Circle {
    center: Point,
    radius: f64
}

impl Printable for Circle {
    fn to_display(&self) -> String {
        "Circle at {self.center.to_display()} with radius {self.radius}"
    }
}

fn print_item<T: Printable>(item: T) {
    print(item.to_display())
}

fn main() {
    let point = Point { x: 3.0, y: 4.0 }
    let circle = Circle {
        center: Point { x: 0.0, y: 0.0 },
        radius: 5.0
    }

    print_item(point)   // (3.0, 4.0)
    print_item(circle)  // Circle at (0.0, 0.0) with radius 5.0
}
```

---

## Putting It Together: A Real Example

Let's build a simple task management system:

```simplex
// Types
type TaskId = String

enum Priority {
    Low,
    Medium,
    High,
    Critical
}

enum Status {
    Todo,
    InProgress,
    Done,
    Cancelled
}

type Task {
    id: TaskId,
    title: String,
    priority: Priority,
    status: Status
}

type TaskList {
    tasks: Map<TaskId, Task>
}

// Methods
impl Priority {
    fn weight(&self) -> i64 {
        match self {
            Priority::Low => 1,
            Priority::Medium => 2,
            Priority::High => 3,
            Priority::Critical => 4
        }
    }
}

impl Task {
    fn new(title: String, priority: Priority) -> Task {
        Task {
            id: uuid::v4(),
            title,
            priority,
            status: Status::Todo
        }
    }

    fn start(&self) -> Task {
        Task { status: Status::InProgress, ..self }
    }

    fn complete(&self) -> Task {
        Task { status: Status::Done, ..self }
    }

    fn is_active(&self) -> Bool {
        match self.status {
            Status::Todo | Status::InProgress => true,
            _ => false
        }
    }
}

impl TaskList {
    fn new() -> TaskList {
        TaskList { tasks: {} }
    }

    fn add(&self, task: Task) -> TaskList {
        var new_tasks = self.tasks.clone()
        new_tasks.insert(task.id.clone(), task)
        TaskList { tasks: new_tasks }
    }

    fn active_tasks(&self) -> List<Task> {
        self.tasks.values()
            .filter(t => t.is_active())
            .sorted_by_key(t => -t.priority.weight())  // High priority first
    }

    fn count_by_status(&self) -> Map<String, i64> {
        var counts: Map<String, i64> = {}
        for task in self.tasks.values() {
            let status_str = match task.status {
                Status::Todo => "todo",
                Status::InProgress => "in_progress",
                Status::Done => "done",
                Status::Cancelled => "cancelled"
            }
            let current = counts.get_or(status_str, 0)
            counts.insert(status_str, current + 1)
        }
        counts
    }
}

// Usage
fn main() {
    let list = TaskList::new()
        .add(Task::new("Write docs", Priority::High))
        .add(Task::new("Fix bug", Priority::Critical))
        .add(Task::new("Add tests", Priority::Medium))

    print("Active tasks (by priority):")
    for task in list.active_tasks() {
        print("  [{task.priority}] {task.title}")
    }

    print("Counts: {list.count_by_status()}")
}
```

---

## Summary

| Concept | Syntax | Use Case |
|---------|--------|----------|
| Struct | `type Name { field: Type }` | Group related data |
| Enum | `enum Name { Variant }` | One of several options |
| Methods | `impl Type { fn method(&self) }` | Add behavior to types |
| Associated fn | `impl Type { fn new() -> Type }` | Constructors |
| Generic | `type Name<T> { field: T }` | Work with any type |
| Trait | `trait Name { fn method(&self) }` | Shared behavior |
| Type alias | `type Name = OtherType` | Readable names |

---

## Exercises

1. **Temperature**: Create a `Temperature` enum with `Celsius(f64)` and `Fahrenheit(f64)` variants. Add methods `to_celsius()` and `to_fahrenheit()` that convert between them.

2. **Shape Trait**: Create a `Shape` trait with `area()` and `perimeter()` methods. Implement it for `Circle` and `Rectangle` types.

3. **Linked List**: Implement a simple linked list using an enum:
   ```simplex
   enum List<T> {
       Empty,
       Node(T, Box<List<T>>)
   }
   ```
   Add methods for `push`, `len`, and `contains`.

---

## Answers

<details>
<summary>Click to reveal answers</summary>

**Exercise 1:**
```simplex
enum Temperature {
    Celsius(f64),
    Fahrenheit(f64)
}

impl Temperature {
    fn to_celsius(&self) -> Temperature {
        match self {
            Temperature::Celsius(c) => Temperature::Celsius(c),
            Temperature::Fahrenheit(f) => {
                Temperature::Celsius((f - 32.0) * 5.0 / 9.0)
            }
        }
    }

    fn to_fahrenheit(&self) -> Temperature {
        match self {
            Temperature::Fahrenheit(f) => Temperature::Fahrenheit(f),
            Temperature::Celsius(c) => {
                Temperature::Fahrenheit(c * 9.0 / 5.0 + 32.0)
            }
        }
    }

    fn value(&self) -> f64 {
        match self {
            Temperature::Celsius(v) => v,
            Temperature::Fahrenheit(v) => v
        }
    }
}

fn main() {
    let temp = Temperature::Celsius(100.0)
    let f = temp.to_fahrenheit()
    print("100°C = {f.value()}°F")  // 212.0
}
```

**Exercise 2:**
```simplex
trait Shape {
    fn area(&self) -> f64
    fn perimeter(&self) -> f64
}

type Circle {
    radius: f64
}

type Rectangle {
    width: f64,
    height: f64
}

impl Shape for Circle {
    fn area(&self) -> f64 {
        3.14159 * self.radius * self.radius
    }

    fn perimeter(&self) -> f64 {
        2.0 * 3.14159 * self.radius
    }
}

impl Shape for Rectangle {
    fn area(&self) -> f64 {
        self.width * self.height
    }

    fn perimeter(&self) -> f64 {
        2.0 * (self.width + self.height)
    }
}

fn main() {
    let circle = Circle { radius: 5.0 }
    let rect = Rectangle { width: 10.0, height: 5.0 }

    print("Circle area: {circle.area()}")
    print("Rectangle area: {rect.area()}")
}
```

**Exercise 3:**
```simplex
enum LinkedList<T> {
    Empty,
    Node(T, Box<LinkedList<T>>)
}

impl<T> LinkedList<T> {
    fn new() -> LinkedList<T> {
        LinkedList::Empty
    }

    fn push(&self, value: T) -> LinkedList<T> {
        LinkedList::Node(value, Box::new(self.clone()))
    }

    fn len(&self) -> i64 {
        match self {
            LinkedList::Empty => 0,
            LinkedList::Node(_, rest) => 1 + rest.len()
        }
    }

    fn contains(&self, target: T) -> Bool where T: Eq {
        match self {
            LinkedList::Empty => false,
            LinkedList::Node(value, rest) => {
                value == target || rest.contains(target)
            }
        }
    }
}

fn main() {
    let list = LinkedList::new()
        .push(3)
        .push(2)
        .push(1)

    print("Length: {list.len()}")        // 3
    print("Contains 2: {list.contains(2)}") // true
    print("Contains 5: {list.contains(5)}") // false
}
```

</details>

---

*Next: [Chapter 6: Error Handling →](06-error-handling.md)*
