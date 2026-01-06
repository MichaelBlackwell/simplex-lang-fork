# Chapter 2: Functions

Functions are the building blocks of Simplex programs. They let you organize code into reusable pieces, give names to operations, and break complex problems into manageable parts.

---

## Defining Functions

Here's the basic syntax:

```simplex
fn greet() {
    print("Hello!")
}

fn main() {
    greet()  // Call the function
    greet()  // Call it again
}
```

Output:
```
Hello!
Hello!
```

The `fn` keyword introduces a function. The parentheses `()` hold parameters (none in this case). The curly braces `{}` contain the function body.

---

## Parameters

Functions become useful when they accept input:

```simplex
fn greet(name: String) {
    print("Hello, {name}!")
}

fn main() {
    greet("Alice")
    greet("Bob")
}
```

Output:
```
Hello, Alice!
Hello, Bob!
```

Parameters must have type annotations. This ensures you always know what a function expects.

### Multiple Parameters

```simplex
fn introduce(name: String, age: i64) {
    print("{name} is {age} years old")
}

fn add(a: i64, b: i64) {
    print("{a} + {b} = {a + b}")
}

fn main() {
    introduce("Alice", 30)
    add(5, 3)
}
```

---

## Return Values

Functions can return values using `->` to specify the return type:

```simplex
fn square(n: i64) -> i64 {
    return n * n
}

fn main() {
    let result = square(5)
    print("5 squared is {result}")  // 5 squared is 25
}
```

### Implicit Returns

The last expression in a function is automatically returned (no `return` needed):

```simplex
fn square(n: i64) -> i64 {
    n * n  // This value is returned
}

fn add(a: i64, b: i64) -> i64 {
    a + b  // Returned implicitly
}

fn main() {
    print(square(4))  // 16
    print(add(2, 3))  // 5
}
```

This is the idiomatic style in Simplex. Use explicit `return` only for early returns.

---

## Early Returns

Use `return` when you need to exit a function early:

```simplex
fn divide(a: i64, b: i64) -> i64 {
    if b == 0 {
        print("Error: division by zero")
        return 0  // Early exit
    }
    a / b  // Normal return
}

fn main() {
    print(divide(10, 2))  // 5
    print(divide(10, 0))  // Error: division by zero, then 0
}
```

---

## Functions Without Return Values

When a function doesn't return anything meaningful, omit the return type:

```simplex
fn log_message(msg: String) {
    print("[LOG] {msg}")
}

fn main() {
    log_message("Starting up")
    log_message("Processing complete")
}
```

These functions implicitly return `()` (unit type).

---

## Named Arguments

For clarity, you can name arguments when calling functions:

```simplex
fn create_user(name: String, age: i64, active: Bool) {
    print("Creating user: {name}, {age}, active={active}")
}

fn main() {
    // Positional arguments
    create_user("Alice", 30, true)

    // Named arguments (clearer, especially with many parameters)
    create_user(name: "Bob", age: 25, active: false)

    // Mix (positional first, then named)
    create_user("Charlie", age: 35, active: true)
}
```

Named arguments are especially helpful when a function has multiple parameters of the same type.

---

## Default Parameter Values

Give parameters default values to make them optional:

```simplex
fn greet(name: String, greeting: String = "Hello") {
    print("{greeting}, {name}!")
}

fn main() {
    greet("Alice")                    // Hello, Alice!
    greet("Bob", greeting: "Hi")      // Hi, Bob!
    greet("Charlie", "Good morning")  // Good morning, Charlie!
}
```

Parameters with defaults must come after required parameters.

---

## Generic Functions

Functions can work with multiple types using generics:

```simplex
fn first<T>(items: List<T>) -> Option<T> {
    match items {
        [] => None,
        [head, ..] => Some(head)
    }
}

fn main() {
    let numbers = [1, 2, 3]
    let names = ["Alice", "Bob"]

    print(first(numbers))  // Some(1)
    print(first(names))    // Some("Alice")
}
```

The `<T>` declares a type parameter. The function works with any type `T`.

### Multiple Type Parameters

```simplex
fn pair<A, B>(first: A, second: B) -> (A, B) {
    (first, second)
}

fn main() {
    let p1 = pair(1, "hello")       // (i64, String)
    let p2 = pair("name", true)     // (String, Bool)
}
```

---

## Closures (Anonymous Functions)

Closures are functions without names, often used inline:

```simplex
fn main() {
    // Basic closure
    let add = (a: i64, b: i64) => a + b
    print(add(2, 3))  // 5

    // Single parameter (parentheses optional)
    let double = x => x * 2
    print(double(5))  // 10

    // Multi-line closure
    let process = (x: i64) => {
        let squared = x * x
        let result = squared + 1
        result
    }
    print(process(3))  // 10
}
```

### Closures with Collections

Closures shine when working with collections:

```simplex
fn main() {
    let numbers = [1, 2, 3, 4, 5]

    // Map: transform each element
    let doubled = numbers.map(x => x * 2)
    print(doubled)  // [2, 4, 6, 8, 10]

    // Filter: keep matching elements
    let evens = numbers.filter(x => x % 2 == 0)
    print(evens)  // [2, 4]

    // Find: get first match
    let first_big = numbers.find(x => x > 3)
    print(first_big)  // Some(4)
}
```

### Capturing Variables

Closures can use variables from their surrounding scope:

```simplex
fn main() {
    let multiplier = 10

    // Closure captures 'multiplier'
    let scale = x => x * multiplier

    print(scale(5))   // 50
    print(scale(3))   // 30
}
```

---

## Higher-Order Functions

Functions can take other functions as parameters:

```simplex
fn apply_twice(f: fn(i64) -> i64, x: i64) -> i64 {
    f(f(x))
}

fn double(n: i64) -> i64 {
    n * 2
}

fn main() {
    let result = apply_twice(double, 5)
    print(result)  // 20 (5 -> 10 -> 20)

    // With a closure
    let result2 = apply_twice(x => x + 1, 5)
    print(result2)  // 7 (5 -> 6 -> 7)
}
```

### Functions Returning Functions

```simplex
fn make_multiplier(factor: i64) -> fn(i64) -> i64 {
    x => x * factor
}

fn main() {
    let double = make_multiplier(2)
    let triple = make_multiplier(3)

    print(double(5))  // 10
    print(triple(5))  // 15
}
```

---

## Recursion

Functions can call themselves:

```simplex
fn factorial(n: i64) -> i64 {
    if n <= 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}

fn main() {
    print(factorial(5))  // 120 (5 * 4 * 3 * 2 * 1)
}
```

### Tail Recursion

Simplex optimizes tail-recursive functions (where the recursive call is the last operation):

```simplex
fn factorial_tail(n: i64, acc: i64 = 1) -> i64 {
    if n <= 1 {
        acc
    } else {
        factorial_tail(n - 1, n * acc)  // Tail call - optimized
    }
}

fn main() {
    print(factorial_tail(5))  // 120
}
```

---

## Method Syntax

Functions can be called with method syntax on their first argument:

```simplex
fn double(n: i64) -> i64 {
    n * 2
}

fn add(a: i64, b: i64) -> i64 {
    a + b
}

fn main() {
    // Regular call
    print(double(5))      // 10

    // Method syntax (first arg before the dot)
    print(5.double())     // 10
    print(5.add(3))       // 8
}
```

This is especially nice for chaining:

```simplex
fn main() {
    let result = 5
        .double()    // 10
        .add(3)      // 13
        .double()    // 26

    print(result)
}
```

---

## Ownership and Functions

When you pass a value to a function, ownership transfers (like Rust):

```simplex
fn consume(s: String) {
    print("Got: {s}")
}

fn main() {
    let name = "Alice"
    consume(name)

    // print(name)  // Error: name was moved into consume
}
```

To avoid transferring ownership, use borrowing:

```simplex
fn inspect(s: &String) {  // Borrows, doesn't take ownership
    print("Length: {s.len()}")
}

fn main() {
    let name = "Alice"
    inspect(&name)     // Lend the value

    print(name)        // Still valid!
}
```

We'll cover ownership in detail later. For now, just know:
- `fn f(x: T)` — takes ownership of x
- `fn f(x: &T)` — borrows x (read-only)
- `fn f(x: &mut T)` — borrows x (can modify)

---

## Summary

| Concept | Syntax | Example |
|---------|--------|---------|
| Define function | `fn name() {}` | `fn greet() { print("Hi") }` |
| Parameters | `fn name(param: Type)` | `fn greet(name: String)` |
| Return type | `fn name() -> Type` | `fn add(a: i64, b: i64) -> i64` |
| Implicit return | Last expression | `fn double(x: i64) -> i64 { x * 2 }` |
| Early return | `return value` | `return 0` |
| Default params | `param: Type = value` | `fn greet(name: String = "World")` |
| Generics | `fn name<T>()` | `fn first<T>(list: List<T>) -> T` |
| Closure | `(params) => expr` | `x => x * 2` |
| Higher-order | `fn(Type) -> Type` | `fn apply(f: fn(i64) -> i64)` |

---

## Exercises

1. **Circle Area**: Write a function `circle_area(radius: f64) -> f64` that returns the area of a circle (π × r²).

2. **Greeting Function**: Write a function `formal_greeting(first: String, last: String, title: String = "Mr.")` that returns a formal greeting like "Hello, Mr. John Doe".

3. **List Operations**: Using closures with `map`, `filter`, and `fold`:
   - Double all numbers in `[1, 2, 3, 4, 5]`
   - Keep only numbers greater than 10 from `[5, 15, 3, 20, 8]`
   - Sum all numbers in `[1, 2, 3, 4, 5]`

4. **Compose**: Write a function `compose` that takes two functions `f` and `g` and returns a new function that applies `f(g(x))`.

---

## Answers

<details>
<summary>Click to reveal answers</summary>

**Exercise 1:**
```simplex
const PI: f64 = 3.14159

fn circle_area(radius: f64) -> f64 {
    PI * radius * radius
}

fn main() {
    print("Area: {circle_area(5.0)}")  // Area: 78.53975
}
```

**Exercise 2:**
```simplex
fn formal_greeting(first: String, last: String, title: String = "Mr.") -> String {
    "Hello, {title} {first} {last}"
}

fn main() {
    print(formal_greeting("John", "Doe"))
    // Hello, Mr. John Doe

    print(formal_greeting("Jane", "Smith", title: "Dr."))
    // Hello, Dr. Jane Smith
}
```

**Exercise 3:**
```simplex
fn main() {
    // Double all numbers
    let doubled = [1, 2, 3, 4, 5].map(x => x * 2)
    print(doubled)  // [2, 4, 6, 8, 10]

    // Keep numbers > 10
    let big = [5, 15, 3, 20, 8].filter(x => x > 10)
    print(big)  // [15, 20]

    // Sum all numbers
    let sum = [1, 2, 3, 4, 5].fold(0, (acc, x) => acc + x)
    print(sum)  // 15
}
```

**Exercise 4:**
```simplex
fn compose<A, B, C>(f: fn(B) -> C, g: fn(A) -> B) -> fn(A) -> C {
    x => f(g(x))
}

fn main() {
    let double = x => x * 2
    let add_one = x => x + 1

    // (double ∘ add_one)(5) = double(add_one(5)) = double(6) = 12
    let double_after_add = compose(double, add_one)
    print(double_after_add(5))  // 12
}
```

</details>

---

*Next: [Chapter 3: Control Flow →](03-control-flow.md)*
