# Chapter 4: Collections

Collections let you store and work with groups of values. Simplex provides three main collection types: Lists, Maps, and Sets.

---

## Lists

A list is an ordered, growable sequence of values.

### Creating Lists

```simplex
fn main() {
    // Create with values
    let numbers = [1, 2, 3, 4, 5]
    let names = ["Alice", "Bob", "Charlie"]

    // Empty list (must specify type)
    let empty: List<i64> = []

    // Create with repeated value
    let zeros = List::repeat(0, 5)  // [0, 0, 0, 0, 0]
}
```

### Accessing Elements

```simplex
fn main() {
    let fruits = ["apple", "banana", "cherry"]

    // By index (0-based)
    print(fruits[0])  // apple
    print(fruits[2])  // cherry

    // Safe access (returns Option)
    print(fruits.get(1))   // Some("banana")
    print(fruits.get(99))  // None

    // First and last
    print(fruits.first())  // Some("apple")
    print(fruits.last())   // Some("cherry")
}
```

### Modifying Lists

Lists are immutable by default, but you can create modified copies or use mutable lists:

```simplex
fn main() {
    // Immutable operations (return new list)
    let nums = [1, 2, 3]
    let more = nums.push(4)        // [1, 2, 3, 4] - new list
    let combined = nums.concat([4, 5])  // [1, 2, 3, 4, 5]

    // Mutable list
    var items: List<String> = []
    items.push_mut("first")
    items.push_mut("second")
    print(items)  // ["first", "second"]

    // Remove last
    let removed = items.pop_mut()  // Some("second")
    print(items)  // ["first"]
}
```

### List Properties

```simplex
fn main() {
    let items = [1, 2, 3, 4, 5]

    print(items.len())       // 5
    print(items.is_empty())  // false

    let empty: List<i64> = []
    print(empty.is_empty())  // true
}
```

### Slicing

```simplex
fn main() {
    let numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    print(numbers[2..5])   // [2, 3, 4]
    print(numbers[..3])    // [0, 1, 2]
    print(numbers[7..])    // [7, 8, 9]
    print(numbers[..])     // Full copy
}
```

---

## Iteration and Transformation

Lists support powerful functional operations.

### Map

Transform each element:

```simplex
fn main() {
    let numbers = [1, 2, 3, 4, 5]

    // Double each number
    let doubled = numbers.map(x => x * 2)
    print(doubled)  // [2, 4, 6, 8, 10]

    // Convert to strings
    let strings = numbers.map(n => "Number: {n}")
    print(strings)  // ["Number: 1", "Number: 2", ...]
}
```

### Filter

Keep only matching elements:

```simplex
fn main() {
    let numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    // Keep even numbers
    let evens = numbers.filter(x => x % 2 == 0)
    print(evens)  // [2, 4, 6, 8, 10]

    // Keep numbers > 5
    let big = numbers.filter(x => x > 5)
    print(big)  // [6, 7, 8, 9, 10]
}
```

### Fold (Reduce)

Combine all elements into one value:

```simplex
fn main() {
    let numbers = [1, 2, 3, 4, 5]

    // Sum: start with 0, add each number
    let sum = numbers.fold(0, (acc, x) => acc + x)
    print(sum)  // 15

    // Product: start with 1, multiply each number
    let product = numbers.fold(1, (acc, x) => acc * x)
    print(product)  // 120

    // Build a string
    let text = numbers.fold("", (acc, x) => "{acc}{x},")
    print(text)  // "1,2,3,4,5,"
}
```

### Common Aggregations

```simplex
fn main() {
    let numbers = [3, 1, 4, 1, 5, 9, 2, 6]

    print(numbers.sum())    // 31
    print(numbers.min())    // Some(1)
    print(numbers.max())    // Some(9)
    print(numbers.count())  // 8
}
```

### Find

Get the first matching element:

```simplex
fn main() {
    let numbers = [1, 2, 3, 4, 5]

    let first_even = numbers.find(x => x % 2 == 0)
    print(first_even)  // Some(2)

    let first_big = numbers.find(x => x > 10)
    print(first_big)  // None
}
```

### Any and All

Check conditions across elements:

```simplex
fn main() {
    let numbers = [2, 4, 6, 8]

    print(numbers.all(x => x % 2 == 0))  // true (all even)
    print(numbers.any(x => x > 5))       // true (some > 5)
    print(numbers.any(x => x > 10))      // false (none > 10)
}
```

### Chaining Operations

Operations chain naturally:

```simplex
fn main() {
    let data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    let result = data
        .filter(x => x % 2 == 0)   // [2, 4, 6, 8, 10]
        .map(x => x * x)            // [4, 16, 36, 64, 100]
        .filter(x => x > 20)        // [36, 64, 100]
        .sum()                      // 200

    print(result)  // 200
}
```

---

## Maps

Maps store key-value pairs. Keys must be unique.

### Creating Maps

```simplex
fn main() {
    // Create with values
    let ages = {"Alice": 30, "Bob": 25, "Charlie": 35}

    // Empty map
    let empty: Map<String, i64> = {}

    // From list of tuples
    let pairs = [("a", 1), ("b", 2)]
    let from_pairs: Map<String, i64> = pairs.collect()
}
```

### Accessing Values

```simplex
fn main() {
    let scores = {"Alice": 95, "Bob": 87, "Charlie": 92}

    // Get (returns Option)
    print(scores.get("Alice"))    // Some(95)
    print(scores.get("Unknown"))  // None

    // Get with default
    print(scores.get_or("Alice", 0))    // 95
    print(scores.get_or("Unknown", 0))  // 0
}
```

### Modifying Maps

```simplex
fn main() {
    var users: Map<String, i64> = {}

    // Insert
    users.insert("Alice", 30)
    users.insert("Bob", 25)
    print(users)  // {"Alice": 30, "Bob": 25}

    // Update (insert with same key)
    users.insert("Alice", 31)
    print(users.get("Alice"))  // Some(31)

    // Remove
    let removed = users.remove("Bob")
    print(removed)  // Some(25)
    print(users)    // {"Alice": 31}
}
```

### Map Properties

```simplex
fn main() {
    let data = {"a": 1, "b": 2, "c": 3}

    print(data.len())          // 3
    print(data.is_empty())     // false
    print(data.contains_key("a"))  // true
    print(data.contains_key("z"))  // false
}
```

### Iterating Maps

```simplex
fn main() {
    let scores = {"Alice": 95, "Bob": 87, "Charlie": 92}

    // Iterate over key-value pairs
    for (name, score) in scores {
        print("{name}: {score}")
    }

    // Just keys
    for name in scores.keys() {
        print(name)
    }

    // Just values
    for score in scores.values() {
        print(score)
    }
}
```

### Map Transformations

```simplex
fn main() {
    let prices = {"apple": 1.50, "banana": 0.75, "cherry": 2.00}

    // Map values
    let doubled = prices.map_values(price => price * 2)
    // {"apple": 3.00, "banana": 1.50, "cherry": 4.00}

    // Filter entries
    let expensive = prices.filter((_, price) => price > 1.00)
    // {"apple": 1.50, "cherry": 2.00}
}
```

---

## Sets

Sets store unique values with fast lookup.

### Creating Sets

```simplex
fn main() {
    // Create with values
    let numbers = {1, 2, 3, 4, 5}

    // Empty set
    let empty: Set<String> = {}

    // From list (removes duplicates)
    let list = [1, 2, 2, 3, 3, 3]
    let unique: Set<i64> = list.collect()
    print(unique)  // {1, 2, 3}
}
```

### Set Operations

```simplex
fn main() {
    var tags: Set<String> = {}

    // Add elements
    tags.insert("rust")
    tags.insert("simplex")
    tags.insert("programming")

    // Check membership
    print(tags.contains("rust"))    // true
    print(tags.contains("python"))  // false

    // Remove
    tags.remove("rust")
    print(tags.contains("rust"))    // false
}
```

### Set Math

```simplex
fn main() {
    let a = {1, 2, 3, 4}
    let b = {3, 4, 5, 6}

    // Union: elements in either set
    print(a.union(b))        // {1, 2, 3, 4, 5, 6}

    // Intersection: elements in both sets
    print(a.intersection(b)) // {3, 4}

    // Difference: elements in a but not b
    print(a.difference(b))   // {1, 2}

    // Symmetric difference: elements in exactly one set
    print(a.symmetric_difference(b))  // {1, 2, 5, 6}
}
```

### Set Comparisons

```simplex
fn main() {
    let small = {1, 2}
    let large = {1, 2, 3, 4}

    print(small.is_subset(large))    // true
    print(large.is_superset(small))  // true
    print(small.is_disjoint({5, 6})) // true (no overlap)
}
```

---

## Tuples

Tuples are fixed-size collections of values with potentially different types.

```simplex
fn main() {
    // Create tuples
    let pair = (42, "hello")
    let triple = (1, "two", 3.0)

    // Access by position
    print(pair.0)    // 42
    print(pair.1)    // "hello"
    print(triple.2)  // 3.0

    // Destructure
    let (number, text) = pair
    print(number)  // 42
    print(text)    // "hello"
}
```

### Returning Multiple Values

Tuples are great for functions that return multiple values:

```simplex
fn divide_with_remainder(a: i64, b: i64) -> (i64, i64) {
    (a / b, a % b)
}

fn main() {
    let (quotient, remainder) = divide_with_remainder(17, 5)
    print("17 / 5 = {quotient} remainder {remainder}")
    // 17 / 5 = 3 remainder 2
}
```

---

## Collection Conversions

Convert between collection types:

```simplex
fn main() {
    // List to Set (removes duplicates)
    let list = [1, 2, 2, 3, 3, 3]
    let set: Set<i64> = list.iter().collect()
    print(set)  // {1, 2, 3}

    // Set to List
    let back_to_list: List<i64> = set.iter().collect()

    // List of pairs to Map
    let pairs = [("a", 1), ("b", 2)]
    let map: Map<String, i64> = pairs.iter().collect()

    // Map to List of pairs
    let map_pairs: List<(String, i64)> = map.iter().collect()
}
```

---

## Sorting

```simplex
fn main() {
    let numbers = [3, 1, 4, 1, 5, 9, 2, 6]

    // Sort ascending
    let sorted = numbers.sorted()
    print(sorted)  // [1, 1, 2, 3, 4, 5, 6, 9]

    // Sort descending
    let desc = numbers.sorted_by(|a, b| b.cmp(a))
    print(desc)  // [9, 6, 5, 4, 3, 2, 1, 1]

    // Sort by key
    let words = ["banana", "apple", "cherry"]
    let by_length = words.sorted_by_key(w => w.len())
    print(by_length)  // ["apple", "banana", "cherry"]
}
```

---

## Summary

| Collection | Syntax | Use Case |
|------------|--------|----------|
| List | `[1, 2, 3]` | Ordered sequence |
| Map | `{"a": 1}` | Key-value lookup |
| Set | `{1, 2, 3}` | Unique values, fast membership |
| Tuple | `(1, "two")` | Fixed group of mixed types |

| Operation | List | Map | Set |
|-----------|------|-----|-----|
| Create | `[1, 2, 3]` | `{"a": 1}` | `{1, 2, 3}` |
| Access | `list[0]` | `map.get("a")` | `set.contains(1)` |
| Add | `list.push(x)` | `map.insert(k, v)` | `set.insert(x)` |
| Remove | `list.pop()` | `map.remove(k)` | `set.remove(x)` |
| Length | `list.len()` | `map.len()` | `set.len()` |

---

## Exercises

1. **Unique Words**: Given a list of words, return a list of unique words (no duplicates).

2. **Word Frequency**: Count how many times each word appears in a list. Return a `Map<String, i64>`.

3. **Top Scorer**: Given a map of names to scores, find the name of the person with the highest score.

4. **List Difference**: Write a function that takes two lists and returns elements in the first list but not the second.

---

## Answers

<details>
<summary>Click to reveal answers</summary>

**Exercise 1:**
```simplex
fn unique_words(words: List<String>) -> List<String> {
    let set: Set<String> = words.iter().collect()
    set.iter().collect()
}

fn main() {
    let words = ["apple", "banana", "apple", "cherry", "banana"]
    print(unique_words(words))  // ["apple", "banana", "cherry"]
}
```

**Exercise 2:**
```simplex
fn word_frequency(words: List<String>) -> Map<String, i64> {
    var counts: Map<String, i64> = {}

    for word in words {
        let current = counts.get_or(word, 0)
        counts.insert(word, current + 1)
    }

    counts
}

fn main() {
    let words = ["apple", "banana", "apple", "cherry", "apple"]
    let freq = word_frequency(words)
    print(freq)  // {"apple": 3, "banana": 1, "cherry": 1}
}
```

**Exercise 3:**
```simplex
fn top_scorer(scores: Map<String, i64>) -> Option<String> {
    var best: Option<(String, i64)> = None

    for (name, score) in scores {
        match best {
            None => best = Some((name, score)),
            Some((_, best_score)) if score > best_score => {
                best = Some((name, score))
            },
            _ => {}
        }
    }

    best.map((name, _) => name)
}

fn main() {
    let scores = {"Alice": 85, "Bob": 92, "Charlie": 78}
    print(top_scorer(scores))  // Some("Bob")
}
```

**Exercise 4:**
```simplex
fn list_difference(a: List<i64>, b: List<i64>) -> List<i64> {
    let set_b: Set<i64> = b.iter().collect()
    a.filter(x => !set_b.contains(x))
}

fn main() {
    let a = [1, 2, 3, 4, 5]
    let b = [3, 4, 5, 6, 7]
    print(list_difference(a, b))  // [1, 2]
}
```

</details>

---

*Next: [Chapter 5: Custom Types →](05-custom-types.md)*
