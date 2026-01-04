# Chapter 3: Control Flow

Control flow determines which code runs and in what order. In this chapter, you'll learn how to make decisions with conditionals, repeat actions with loops, and elegantly handle multiple cases with pattern matching.

---

## If Expressions

The `if` expression lets you run code conditionally:

```simplex
fn main() {
    let temperature = 25

    if temperature > 30 {
        print("It's hot!")
    }
}
```

### If-Else

Handle both cases:

```simplex
fn main() {
    let age = 20

    if age >= 18 {
        print("You can vote")
    } else {
        print("Too young to vote")
    }
}
```

### If-Else Chains

Multiple conditions:

```simplex
fn main() {
    let score = 85

    if score >= 90 {
        print("Grade: A")
    } else if score >= 80 {
        print("Grade: B")
    } else if score >= 70 {
        print("Grade: C")
    } else {
        print("Grade: F")
    }
}
```

### If as an Expression

Unlike many languages, `if` in Simplex is an **expression**—it returns a value:

```simplex
fn main() {
    let age = 20

    let status = if age >= 18 { "adult" } else { "minor" }

    print("Status: {status}")  // Status: adult
}
```

This replaces the ternary operator (`? :`) found in other languages. Both branches must return the same type.

```simplex
fn main() {
    let n = 7

    let parity = if n % 2 == 0 { "even" } else { "odd" }
    print("{n} is {parity}")  // 7 is odd
}
```

---

## Loops

### For Loops

Iterate over collections:

```simplex
fn main() {
    let names = ["Alice", "Bob", "Charlie"]

    for name in names {
        print("Hello, {name}!")
    }
}
```

Output:
```
Hello, Alice!
Hello, Bob!
Hello, Charlie!
```

### Range Loops

Iterate over a range of numbers:

```simplex
fn main() {
    // 0 to 4 (exclusive end)
    for i in 0..5 {
        print(i)
    }

    // 1 to 5 (inclusive end)
    for i in 1..=5 {
        print(i)
    }
}
```

### For with Index

Get both index and value:

```simplex
fn main() {
    let fruits = ["apple", "banana", "cherry"]

    for (index, fruit) in fruits.enumerate() {
        print("{index}: {fruit}")
    }
}
```

Output:
```
0: apple
1: banana
2: cherry
```

### While Loops

Repeat while a condition is true:

```simplex
fn main() {
    var count = 0

    while count < 5 {
        print("Count: {count}")
        count += 1
    }
}
```

### Loop (Infinite)

The `loop` keyword creates an infinite loop (useful when you'll break manually):

```simplex
fn main() {
    var attempts = 0

    loop {
        attempts += 1
        print("Attempt {attempts}")

        if attempts >= 3 {
            break
        }
    }

    print("Done after {attempts} attempts")
}
```

### Loop with Return Value

`loop` can return a value when you `break`:

```simplex
fn main() {
    var counter = 0

    let result = loop {
        counter += 1

        if counter == 10 {
            break counter * 2  // Returns 20
        }
    }

    print("Result: {result}")  // Result: 20
}
```

---

## Break and Continue

Control loop execution:

```simplex
fn main() {
    // Skip even numbers
    for i in 1..10 {
        if i % 2 == 0 {
            continue  // Skip to next iteration
        }
        print(i)  // Prints 1, 3, 5, 7, 9
    }

    // Stop at 5
    for i in 1..10 {
        if i > 5 {
            break  // Exit the loop
        }
        print(i)  // Prints 1, 2, 3, 4, 5
    }
}
```

---

## Pattern Matching

Pattern matching is one of Simplex's most powerful features. It's like a supercharged `switch` statement.

### Basic Match

```simplex
fn main() {
    let day = 3

    match day {
        1 => print("Monday"),
        2 => print("Tuesday"),
        3 => print("Wednesday"),
        4 => print("Thursday"),
        5 => print("Friday"),
        6 => print("Saturday"),
        7 => print("Sunday"),
        _ => print("Invalid day")  // _ matches anything
    }
}
```

### Match as Expression

`match` returns a value:

```simplex
fn main() {
    let day = 3

    let name = match day {
        1 => "Monday",
        2 => "Tuesday",
        3 => "Wednesday",
        4 => "Thursday",
        5 => "Friday",
        6 | 7 => "Weekend",  // Multiple patterns with |
        _ => "Invalid"
    }

    print("Day: {name}")  // Day: Wednesday
}
```

### Matching Ranges

```simplex
fn main() {
    let score = 85

    let grade = match score {
        90..=100 => "A",
        80..=89 => "B",
        70..=79 => "C",
        60..=69 => "D",
        _ => "F"
    }

    print("Grade: {grade}")
}
```

### Guards

Add conditions to patterns:

```simplex
fn main() {
    let number = -5

    match number {
        n if n > 0 => print("Positive: {n}"),
        n if n < 0 => print("Negative: {n}"),
        _ => print("Zero")
    }
}
```

### Destructuring

Extract values while matching:

```simplex
fn main() {
    let point = (3, 4)

    match point {
        (0, 0) => print("Origin"),
        (x, 0) => print("On x-axis at {x}"),
        (0, y) => print("On y-axis at {y}"),
        (x, y) => print("Point at ({x}, {y})")
    }
}
```

### Matching Enums

Pattern matching really shines with enums:

```simplex
enum Color {
    Red,
    Green,
    Blue,
    Custom(i64, i64, i64)
}

fn describe_color(color: Color) -> String {
    match color {
        Color::Red => "Pure red",
        Color::Green => "Pure green",
        Color::Blue => "Pure blue",
        Color::Custom(r, g, b) => "RGB({r}, {g}, {b})"
    }
}

fn main() {
    print(describe_color(Color::Red))
    print(describe_color(Color::Custom(255, 128, 0)))
}
```

### Matching Options

The `Option` type is commonly matched:

```simplex
fn main() {
    let maybe_number: Option<i64> = Some(42)

    match maybe_number {
        Some(n) => print("Got: {n}"),
        None => print("Nothing")
    }
}
```

---

## If Let

For simple cases where you only care about one pattern, use `if let`:

```simplex
fn main() {
    let maybe_name: Option<String> = Some("Alice")

    // Instead of full match:
    if let Some(name) = maybe_name {
        print("Hello, {name}!")
    }

    // With else:
    if let Some(name) = maybe_name {
        print("Hello, {name}!")
    } else {
        print("Hello, stranger!")
    }
}
```

---

## While Let

Combine `while` with pattern matching:

```simplex
fn main() {
    var stack = [1, 2, 3, 4, 5]

    while let Some(top) = stack.pop() {
        print("Popped: {top}")
    }

    print("Stack is empty")
}
```

---

## Exhaustiveness

The compiler ensures your matches cover all possibilities:

```simplex
enum Direction {
    North,
    South,
    East,
    West
}

fn go(dir: Direction) {
    match dir {
        Direction::North => print("Going north"),
        Direction::South => print("Going south"),
        // Error: non-exhaustive patterns
        // East and West not covered
    }
}
```

This catches bugs at compile time. You must handle every case, or use `_` as a catch-all.

---

## Combining Control Flow

These constructs combine naturally:

```simplex
fn process_numbers(numbers: List<i64>) {
    for num in numbers {
        let result = match num {
            n if n < 0 => "negative",
            0 => "zero",
            n if n % 2 == 0 => "even",
            _ => "odd"
        }

        if result == "negative" {
            continue  // Skip negative numbers
        }

        print("{num} is {result}")
    }
}

fn main() {
    process_numbers([-2, -1, 0, 1, 2, 3, 4, 5])
}
```

Output:
```
0 is zero
1 is odd
2 is even
3 is odd
4 is even
5 is odd
```

---

## Summary

| Concept | Syntax | Example |
|---------|--------|---------|
| If | `if cond { }` | `if x > 0 { print("positive") }` |
| If-else | `if cond { } else { }` | `if x > 0 { "pos" } else { "neg" }` |
| For loop | `for item in collection { }` | `for x in [1,2,3] { print(x) }` |
| Range | `start..end` | `for i in 0..10 { }` |
| While | `while cond { }` | `while x < 10 { x += 1 }` |
| Loop | `loop { break }` | `loop { if done { break } }` |
| Match | `match value { pattern => expr }` | `match x { 1 => "one", _ => "other" }` |
| If let | `if let pattern = value { }` | `if let Some(x) = opt { }` |

---

## Exercises

1. **FizzBuzz**: Print numbers 1 to 30. For multiples of 3, print "Fizz". For multiples of 5, print "Buzz". For multiples of both, print "FizzBuzz".

2. **Number Classifier**: Write a function that takes a number and returns:
   - "negative" if less than 0
   - "zero" if equal to 0
   - "small" if 1-10
   - "medium" if 11-100
   - "large" if > 100

3. **Sum Until**: Using a loop, sum numbers from 1 upward until the sum exceeds 100. Return the count of numbers summed.

4. **List Search**: Using a for loop with break, find the first number greater than 50 in a list.

---

## Answers

<details>
<summary>Click to reveal answers</summary>

**Exercise 1:**
```simplex
fn main() {
    for n in 1..=30 {
        match (n % 3, n % 5) {
            (0, 0) => print("FizzBuzz"),
            (0, _) => print("Fizz"),
            (_, 0) => print("Buzz"),
            _ => print(n)
        }
    }
}
```

**Exercise 2:**
```simplex
fn classify(n: i64) -> String {
    match n {
        x if x < 0 => "negative",
        0 => "zero",
        1..=10 => "small",
        11..=100 => "medium",
        _ => "large"
    }
}

fn main() {
    print(classify(-5))   // negative
    print(classify(0))    // zero
    print(classify(7))    // small
    print(classify(50))   // medium
    print(classify(200))  // large
}
```

**Exercise 3:**
```simplex
fn sum_until_100() -> i64 {
    var sum = 0
    var count = 0

    loop {
        count += 1
        sum += count

        if sum > 100 {
            break count
        }
    }
}

fn main() {
    let result = sum_until_100()
    print("Summed {result} numbers")  // Summed 14 numbers
}
```

**Exercise 4:**
```simplex
fn find_greater_than_50(numbers: List<i64>) -> Option<i64> {
    for n in numbers {
        if n > 50 {
            return Some(n)
        }
    }
    None
}

fn main() {
    let nums = [10, 25, 40, 55, 60, 30]
    match find_greater_than_50(nums) {
        Some(n) => print("Found: {n}"),  // Found: 55
        None => print("None found")
    }
}
```

</details>

---

*Next: [Chapter 4: Collections →](04-collections.md)*
