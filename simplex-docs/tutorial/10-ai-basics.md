# Chapter 10: AI Integration

Simplex treats AI as a first-class citizen. AI operations are built into the language, not bolted-on libraries. In this chapter, you'll learn to use AI for text generation, classification, extraction, and search.

---

## Your First AI Call

```simplex
use ai

fn main() {
    let response = await ai::complete("What is the capital of France?")
    print(response)
}
```

Output:
```
The capital of France is Paris.
```

That's it. No API keys in code, no complex setup. The runtime handles connection to the AI inference pool.

---

## Text Completion

### Basic Completion

```simplex
use ai

fn main() {
    // Simple question
    let answer = await ai::complete("Explain quantum computing in one sentence.")
    print(answer)

    // Creative writing
    let story = await ai::complete("Write a haiku about programming.")
    print(story)
}
```

### With Options

```simplex
use ai

fn main() {
    let response = await ai::complete(
        "Write a product description for a smart water bottle.",
        model: "creative",      // Use creative model
        temperature: 0.9,       // More randomness
        max_tokens: 200         // Limit length
    )
    print(response)
}
```

### Streaming

For long responses, stream the output:

```simplex
use ai

fn main() {
    print("Generating story...")

    for chunk in ai::stream("Tell me a short story about a robot.") {
        print(chunk)  // Print each chunk as it arrives
    }
}
```

---

## Classification

Classify text into categories:

```simplex
use ai

enum Sentiment {
    Positive,
    Negative,
    Neutral
}

fn main() {
    let review = "This product exceeded my expectations! Best purchase ever."

    let sentiment = await ai::classify<Sentiment>(review)

    print("Sentiment: {sentiment}")  // Sentiment: Positive
}
```

### Multi-Category

```simplex
use ai

enum Category {
    Technology,
    Sports,
    Politics,
    Entertainment,
    Science,
    Business
}

fn categorize_headlines() {
    let headlines = [
        "Apple announces new iPhone",
        "Lakers win championship",
        "New study reveals black hole secrets"
    ]

    for headline in headlines {
        let category = await ai::classify<Category>(headline)
        print("{headline} -> {category}")
    }
}
```

Output:
```
Apple announces new iPhone -> Technology
Lakers win championship -> Sports
New study reveals black hole secrets -> Science
```

---

## Structured Extraction

Extract typed data from unstructured text:

```simplex
use ai

type Person {
    name: String,
    age: Option<i64>,
    occupation: Option<String>
}

fn main() {
    let text = "John Smith is a 35-year-old software engineer from Seattle."

    let person = await ai::extract<Person>(text)

    print("Name: {person.name}")           // Name: John Smith
    print("Age: {person.age}")             // Age: Some(35)
    print("Occupation: {person.occupation}") // Occupation: Some("software engineer")
}
```

### Complex Extraction

```simplex
use ai

type Address {
    street: String,
    city: String,
    state: String,
    zip: String
}

type ContactInfo {
    name: String,
    email: Option<String>,
    phone: Option<String>,
    address: Option<Address>
}

fn main() {
    let email = """
        Hi, I'm Sarah Johnson. You can reach me at sarah@email.com
        or call 555-123-4567. My office is at 123 Main Street,
        Boston, MA 02101.
    """

    let contact = await ai::extract<ContactInfo>(email)

    print("Extracted contact: {contact.name}")
    if let Some(addr) = contact.address {
        print("City: {addr.city}")
    }
}
```

### Extracting Lists

```simplex
use ai

type Task {
    description: String,
    priority: String,
    due_date: Option<String>
}

fn main() {
    let notes = """
        Meeting notes:
        - Finish the quarterly report by Friday (high priority)
        - Review John's PR (low priority)
        - Schedule team lunch for next week
        - Update documentation (medium priority, due Monday)
    """

    let tasks = await ai::extract<List<Task>>(notes)

    print("Found {tasks.len()} tasks:")
    for task in tasks {
        print("  [{task.priority}] {task.description}")
    }
}
```

---

## Embeddings

Embeddings convert text to vectors for similarity search:

```simplex
use ai

fn main() {
    // Single embedding
    let text = "The quick brown fox"
    let embedding = ai::embed(text)

    print("Dimensions: {embedding.len()}")  // Dimensions: 1536
}
```

### Batch Embeddings

More efficient for multiple texts:

```simplex
use ai

fn main() {
    let documents = [
        "Introduction to machine learning",
        "Advanced Python programming",
        "Web development with React"
    ]

    // One batch call instead of three separate calls
    let embeddings = ai::embed_batch(documents)

    print("Got {embeddings.len()} embeddings")
}
```

### Similarity Search

```simplex
use ai

fn main() {
    // Document corpus
    let docs = [
        "How to train a neural network",
        "Best practices for REST APIs",
        "Introduction to quantum computing",
        "Machine learning for beginners",
        "Building mobile apps with Flutter"
    ]

    // Embed all documents
    let doc_embeddings = ai::embed_batch(docs)

    // Search query
    let query = "I want to learn about AI"
    let query_embedding = ai::embed(query)

    // Find similar documents
    let results = ai::nearest(query_embedding, doc_embeddings, k: 2)

    print("Most relevant documents:")
    for (index, score) in results {
        print("  [{score:.2}] {docs[index]}")
    }
}
```

Output:
```
Most relevant documents:
  [0.92] How to train a neural network
  [0.87] Machine learning for beginners
```

---

## AI in Actors

Combine AI with the actor model:

### Chatbot Actor

```simplex
use ai

actor Chatbot {
    var history: List<Message> = []
    var system_prompt: String = "You are a helpful assistant."

    receive Chat(user_message: String) -> String {
        history.push_mut(Message::user(user_message))

        let prompt = format_conversation(system_prompt, history)
        let response = await ai::complete(prompt)

        history.push_mut(Message::assistant(response))
        checkpoint()  // Save conversation state

        response
    }

    receive SetPersonality(prompt: String) {
        system_prompt = prompt
        history = []  // Reset on personality change
        checkpoint()
    }

    receive ClearHistory {
        history = []
        checkpoint()
    }
}

fn format_conversation(system: String, messages: List<Message>) -> String {
    var prompt = "System: {system}\n\n"

    for msg in messages {
        match msg {
            Message::User(text) => prompt += "User: {text}\n",
            Message::Assistant(text) => prompt += "Assistant: {text}\n"
        }
    }

    prompt += "Assistant: "
    prompt
}

fn main() {
    let bot = spawn Chatbot

    send(bot, SetPersonality("You are a pirate. Respond in pirate speak."))

    let r1 = ask(bot, Chat("Hello, how are you?"))
    print("Bot: {r1}")

    let r2 = ask(bot, Chat("What's the weather like?"))
    print("Bot: {r2}")
}
```

### Document Analyzer

```simplex
use ai

type Analysis {
    summary: String,
    sentiment: Sentiment,
    topics: List<String>,
    key_entities: List<Entity>
}

actor DocumentAnalyzer {
    receive Analyze(content: String) -> Analysis {
        // Run AI operations in parallel for speed
        let (summary, sentiment, topics, entities) = await parallel(
            ai::complete("Summarize in 2 sentences: {content}"),
            ai::classify<Sentiment>(content),
            ai::extract<List<String>>("Extract main topics: {content}"),
            ai::extract<List<Entity>>(content)
        )

        Analysis {
            summary,
            sentiment,
            topics,
            key_entities: entities
        }
    }
}

fn main() {
    let analyzer = spawn DocumentAnalyzer

    let doc = """
        Apple Inc. announced today that CEO Tim Cook will present
        the company's new AI strategy at next month's developer
        conference in Cupertino. The initiative focuses on
        privacy-preserving machine learning...
    """

    let analysis = ask(analyzer, Analyze(doc))

    print("Summary: {analysis.summary}")
    print("Sentiment: {analysis.sentiment}")
    print("Topics: {analysis.topics}")
}
```

---

## RAG (Retrieval-Augmented Generation)

Combine search with generation:

```simplex
use ai

actor KnowledgeBase {
    var documents: List<Document> = []
    var embeddings: List<Vector<f64, 1536>> = []

    receive AddDocument(doc: Document) {
        let embedding = ai::embed(doc.content)
        documents.push_mut(doc)
        embeddings.push_mut(embedding)
        checkpoint()
    }

    receive Query(question: String) -> String {
        // 1. Find relevant documents
        let query_embedding = ai::embed(question)
        let matches = ai::nearest(query_embedding, embeddings, k: 3)

        // 2. Build context from matches
        var context = ""
        for (idx, _) in matches {
            context += documents[idx].content + "\n\n"
        }

        // 3. Generate answer using context
        let prompt = """
            Based on the following context, answer the question.

            Context:
            {context}

            Question: {question}

            Answer:
        """

        await ai::complete(prompt)
    }
}

fn main() {
    let kb = spawn KnowledgeBase

    // Add documents
    send(kb, AddDocument(Document {
        title: "Company Policy",
        content: "Employees get 20 days of PTO per year..."
    }))

    send(kb, AddDocument(Document {
        title: "Benefits Guide",
        content: "Health insurance covers 80% of medical expenses..."
    }))

    sleep(Duration::milliseconds(100))

    // Query
    let answer = ask(kb, Query("How many vacation days do I get?"))
    print(answer)
}
```

---

## Model Selection

Choose models based on your needs:

```simplex
use ai

fn main() {
    // Fast model - quick responses, lower quality
    let quick = await ai::complete(prompt, model: "fast")

    // Default model - balanced
    let standard = await ai::complete(prompt, model: "default")

    // Quality model - best results, slower
    let quality = await ai::complete(prompt, model: "quality")
}
```

| Model | Speed | Quality | Cost | Use Case |
|-------|-------|---------|------|----------|
| fast | ~50ms | Good | Low | Simple queries, high volume |
| default | ~200ms | Great | Medium | Most use cases |
| quality | ~1s | Best | High | Complex reasoning, important outputs |

---

## Error Handling

AI operations can fail:

```simplex
use ai

fn main() {
    match await ai::extract<ContactInfo>(messy_text) {
        Ok(contact) => print("Found: {contact.name}"),
        Err(ExtractionError::InvalidFormat) => {
            print("Could not parse the text")
        },
        Err(e) => print("AI error: {e}")
    }
}
```

### Retry Logic

```simplex
use ai

async fn ai_with_retry<T>(
    operation: async fn() -> Result<T, AiError>,
    max_attempts: i64
) -> Result<T, AiError> {
    var attempt = 0

    loop {
        attempt += 1

        match await operation() {
            Ok(result) => return Ok(result),
            Err(e) if attempt < max_attempts => {
                log::warn("Attempt {attempt} failed, retrying...")
                sleep(Duration::seconds(attempt))
            },
            Err(e) => return Err(e)
        }
    }
}
```

---

## Summary

| Function | Purpose | Example |
|----------|---------|---------|
| `ai::complete` | Generate text | `ai::complete("Write a poem")` |
| `ai::stream` | Stream text | `for chunk in ai::stream(prompt)` |
| `ai::classify<T>` | Classify into enum | `ai::classify<Sentiment>(text)` |
| `ai::extract<T>` | Extract structured data | `ai::extract<Person>(text)` |
| `ai::embed` | Get embedding vector | `ai::embed("text")` |
| `ai::embed_batch` | Batch embeddings | `ai::embed_batch(texts)` |
| `ai::nearest` | Find similar vectors | `ai::nearest(query, docs, k: 5)` |

---

## Exercises

1. **Sentiment Dashboard**: Create an actor that receives customer reviews and maintains a running count of positive, negative, and neutral sentiment.

2. **Auto-Tagger**: Build a system that automatically assigns tags to blog posts based on their content.

3. **FAQ Bot**: Create a simple FAQ bot that uses RAG to answer questions based on a knowledge base.

---

*Next: [Chapter 11: Building a Complete Project →](11-capstone.md)*
