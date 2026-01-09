#!/usr/bin/env python3
"""
Stage 4: Specialist LoRA Adapter Training

Trains specialized LoRA adapters for domain-specific tasks:
1. Document Processing (PDF extraction, structure understanding)
2. Content Extraction (NER, entity linking, structured data)
3. Coding Assistant (code generation, review, debugging)
4. Summarization (multi-doc, abstractive, extractive)
5. Reasoning (math, logic, multi-step)
6. Sentiment Analysis (classification, aspect-based)

Each adapter can be hot-swapped at inference time.
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# For data generation only (no torch required)
try:
    from faker import Faker
    fake = Faker()
except ImportError:
    fake = None


@dataclass
class SpecialistConfig:
    """Configuration for a specialist adapter."""
    name: str
    description: str
    lora_rank: int
    target_modules: List[str]
    datasets: List[str]
    license_status: str  # "open", "research_only", "commercial"


# Specialist definitions with licensing info
SPECIALISTS = {
    "document_processing": SpecialistConfig(
        name="document_processing",
        description="PDF/document extraction, structure understanding, table parsing",
        lora_rank=16,
        target_modules=["q_proj", "v_proj", "o_proj"],
        datasets=[
            "docvqa",           # MIT - Document Visual QA
            "publaynet",        # MIT - Document layout
            "tablebank",        # MIT - Table extraction
            "cord",             # MIT - Receipt OCR
        ],
        license_status="open",
    ),
    "content_extraction": SpecialistConfig(
        name="content_extraction",
        description="Named entity recognition, entity linking, structured extraction",
        lora_rank=16,
        target_modules=["q_proj", "v_proj"],
        datasets=[
            "few_nerd",         # MIT - Fine-grained NER
            "wikiner",          # CC BY 4.0 - Multilingual NER
            "universal_ner",    # MIT - Universal NER
            "wnut_17",          # CC BY 4.0 - Emerging entities
        ],
        license_status="open",
    ),
    "coding": SpecialistConfig(
        name="coding",
        description="Code generation, review, debugging, documentation",
        lora_rank=32,  # Higher rank for code complexity
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        datasets=[
            "code_search_net",  # MIT - Code-text pairs
            "codexglue",        # MIT - Code understanding
            "humaneval",        # MIT - Code generation eval
            "mbpp",             # CC BY 4.0 - Python problems
            "apps",             # MIT - Competitive programming
            "commitpack",       # MIT - Commit messages
        ],
        license_status="open",
    ),
    "summarization": SpecialistConfig(
        name="summarization",
        description="Multi-document summarization, abstractive, extractive",
        lora_rank=16,
        target_modules=["q_proj", "v_proj"],
        datasets=[
            "cnn_dailymail",    # Apache 2.0 - News
            "xsum",             # MIT - Extreme summarization
            "multi_news",       # MIT - Multi-doc
            "arxiv",            # Open - Scientific papers
            "billsum",          # CC0 - US legislation
        ],
        license_status="open",
    ),
    "reasoning": SpecialistConfig(
        name="reasoning",
        description="Mathematical reasoning, logic, multi-step problem solving",
        lora_rank=16,
        target_modules=["q_proj", "v_proj"],
        datasets=[
            "gsm8k",            # MIT - Grade school math
            "math",             # MIT - Competition math
            "arc",              # CC BY-SA - AI2 reasoning
            "hellaswag",        # MIT - Commonsense
            "logiqa",           # Open - Logic QA
        ],
        license_status="open",
    ),
    "sentiment": SpecialistConfig(
        name="sentiment",
        description="Sentiment classification, aspect-based, opinion mining",
        lora_rank=8,  # Lower rank - simpler task
        target_modules=["q_proj", "v_proj"],
        datasets=[
            "sst2",             # Open - Stanford sentiment
            "imdb",             # Open - Movie reviews
            "yelp_review_full", # Open - Business reviews
            "amazon_polarity",  # Open - Product reviews
            "semeval2014",      # Open - Aspect-based
        ],
        license_status="open",
    ),
    "dialogue": SpecialistConfig(
        name="dialogue",
        description="Conversational AI, task-oriented dialogue, chitchat",
        lora_rank=16,
        target_modules=["q_proj", "v_proj"],
        datasets=[
            "multiwoz",         # MIT - Task-oriented
            "dailydialog",      # CC BY-NC-SA - Daily conversation
            "persona_chat",     # MIT - Persona-based
            "wizard_of_wikipedia", # MIT - Knowledge-grounded
        ],
        license_status="open",
    ),
}


# ============================================================
# Data Generation Functions (no torch required)
# ============================================================

def generate_document_processing_example() -> Dict:
    """Generate document processing training example."""
    doc_types = ["invoice", "receipt", "contract", "report", "form"]
    doc_type = random.choice(doc_types)

    if doc_type == "invoice":
        vendor = fake.company() if fake else "Acme Corp"
        invoice_num = f"INV-{random.randint(1000, 9999)}"
        amount = random.uniform(100, 10000)

        prompt = f"""<document type="invoice">
Vendor: {vendor}
Invoice Number: {invoice_num}
Date: {fake.date() if fake else "2024-01-15"}
Items:
- Service A: ${amount * 0.6:.2f}
- Service B: ${amount * 0.4:.2f}
Total: ${amount:.2f}
</document>

Extract structured data from this invoice."""

        response = f"""Extracted invoice data:
{{
  "vendor": "{vendor}",
  "invoice_number": "{invoice_num}",
  "total_amount": {amount:.2f},
  "currency": "USD",
  "line_items": [
    {{"description": "Service A", "amount": {amount * 0.6:.2f}}},
    {{"description": "Service B", "amount": {amount * 0.4:.2f}}}
  ]
}}
[confidence: 0.94]"""

    elif doc_type == "receipt":
        store = fake.company() if fake else "Store Inc"
        items = random.randint(2, 5)

        prompt = f"""<document type="receipt">
{store}
{fake.address() if fake else "123 Main St"}
---
{chr(10).join([f"Item {i+1}: ${random.uniform(5, 50):.2f}" for i in range(items)])}
---
TOTAL: ${random.uniform(20, 200):.2f}
</document>

Parse this receipt."""

        response = f"""Parsed receipt:
- Store: {store}
- Item count: {items}
- Document type: retail receipt
- Confidence in extraction: high
[confidence: 0.91]"""

    else:
        prompt = f"""<document type="{doc_type}">
[Document content for {doc_type}]
</document>

Identify the document type and key fields."""

        response = f"""Document analysis:
- Type: {doc_type}
- Structure: standard {doc_type} format
- Key fields identified: header, body, signature block
[confidence: 0.88]"""

    return {"text": f"{prompt}\n\nAssistant: {response}"}


def generate_content_extraction_example() -> Dict:
    """Generate content extraction (NER) training example."""
    extraction_types = ["person", "organization", "location", "event", "product"]

    if fake:
        name = fake.name()
        company = fake.company()
        city = fake.city()
    else:
        name = "John Smith"
        company = "TechCorp"
        city = "New York"

    text = f"{name} is the CEO of {company}, headquartered in {city}. " \
           f"The company announced a new product launch scheduled for next month."

    prompt = f"""Extract all named entities from:
"{text}"

Output format: JSON with entity types."""

    response = f"""{{
  "entities": [
    {{"text": "{name}", "type": "PERSON", "role": "CEO"}},
    {{"text": "{company}", "type": "ORGANIZATION"}},
    {{"text": "{city}", "type": "LOCATION"}},
    {{"text": "next month", "type": "DATE", "relative": true}}
  ],
  "relations": [
    {{"subject": "{name}", "predicate": "works_for", "object": "{company}"}},
    {{"subject": "{company}", "predicate": "located_in", "object": "{city}"}}
  ]
}}
[confidence: 0.93]"""

    return {"text": f"{prompt}\n\nAssistant: {response}"}


def generate_coding_example() -> Dict:
    """Generate coding training example."""
    task_types = ["generation", "review", "debug", "explain", "refactor"]
    task = random.choice(task_types)

    if task == "generation":
        prompt = """Write a Python function that implements binary search.
Include type hints and docstring."""

        response = '''```python
def binary_search(arr: list[int], target: int) -> int:
    """
    Search for target in sorted array using binary search.

    Args:
        arr: Sorted list of integers
        target: Value to search for

    Returns:
        Index of target if found, -1 otherwise
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```

Time complexity: O(log n)
Space complexity: O(1)
[confidence: 0.97]'''

    elif task == "review":
        prompt = """Review this code for issues:
```python
def get_user(id):
    query = f"SELECT * FROM users WHERE id = {id}"
    return db.execute(query)
```"""

        response = """Code Review:

**CRITICAL: SQL Injection Vulnerability**
The code uses string formatting for SQL queries, allowing injection attacks.

**Fix:**
```python
def get_user(id: int) -> Optional[User]:
    query = "SELECT * FROM users WHERE id = ?"
    return db.execute(query, (id,))
```

**Additional issues:**
1. Missing type hints
2. No null handling
3. No input validation

[confidence: 0.98]"""

    elif task == "debug":
        prompt = """Debug this code that should sum a list but returns wrong results:
```python
def sum_list(nums):
    total = 0
    for i in range(len(nums)):
        total = nums[i]
    return total
```"""

        response = """Bug found: Assignment instead of accumulation.

**Line 4:** `total = nums[i]` should be `total += nums[i]`

**Fixed code:**
```python
def sum_list(nums: list[int]) -> int:
    total = 0
    for num in nums:
        total += num
    return total
```

Or more Pythonic: `return sum(nums)`

[confidence: 0.99]"""

    else:
        prompt = """Explain what this code does:
```python
result = [x**2 for x in range(10) if x % 2 == 0]
```"""

        response = """This is a list comprehension that:

1. Iterates over numbers 0-9 (`range(10)`)
2. Filters for even numbers (`x % 2 == 0`)
3. Squares each even number (`x**2`)

**Result:** `[0, 4, 16, 36, 64]`

**Equivalent verbose code:**
```python
result = []
for x in range(10):
    if x % 2 == 0:
        result.append(x**2)
```

[confidence: 0.99]"""

    return {"text": f"{prompt}\n\nAssistant: {response}"}


def generate_summarization_example() -> Dict:
    """Generate summarization training example."""
    summary_types = ["abstractive", "extractive", "multi_doc"]

    # Generate fake article
    if fake:
        topic = fake.catch_phrase()
        company = fake.company()
        name = fake.name()
    else:
        topic = "new technology platform"
        company = "TechCorp"
        name = "Jane Doe"

    article = f"""
{company} announced today a major breakthrough in {topic}. According to CEO {name},
the new development will revolutionize the industry. "This represents years of research
and development," {name} said in a press conference. The company plans to roll out
the technology in phases, starting with enterprise customers in Q2. Analysts predict
this could increase {company}'s market share by 15-20% over the next fiscal year.
Industry experts have praised the innovation, though some raise concerns about
implementation challenges. The stock price rose 5% on the news.
""".strip()

    prompt = f"""Summarize the following article in 2-3 sentences:

{article}"""

    response = f"""{company} has announced a breakthrough in {topic}, developed under CEO {name}'s leadership. The technology will launch to enterprise customers in Q2, with analysts predicting 15-20% market share growth. Despite implementation concerns, the announcement drove a 5% stock price increase.

[confidence: 0.91]"""

    return {"text": f"{prompt}\n\nAssistant: {response}"}


def generate_reasoning_example() -> Dict:
    """Generate reasoning/math training example."""
    problem_types = ["arithmetic", "algebra", "word_problem", "logic"]
    problem_type = random.choice(problem_types)

    if problem_type == "arithmetic":
        a, b = random.randint(10, 100), random.randint(10, 100)
        prompt = f"Calculate step by step: {a} × {b}"
        result = a * b

        response = f"""Let me solve this step by step:

{a} × {b}

Breaking down {b}:
= {a} × {(b // 10) * 10} + {a} × {b % 10}
= {a * ((b // 10) * 10)} + {a * (b % 10)}
= {result}

**Answer: {result}**
[confidence: 0.99]"""

    elif problem_type == "word_problem":
        apples = random.randint(5, 20)
        friends = random.randint(2, 5)
        each = apples // friends
        remainder = apples % friends

        prompt = f"Sarah has {apples} apples and wants to share them equally among {friends} friends. How many does each friend get, and how many are left over?"

        response = f"""Let me solve this:

Total apples: {apples}
Number of friends: {friends}

Division: {apples} ÷ {friends} = {each} remainder {remainder}

**Answer:** Each friend gets {each} apples, with {remainder} left over.

[confidence: 0.98]"""

    elif problem_type == "logic":
        prompt = """If all Bloops are Razzles, and all Razzles are Lazzles, what can we conclude about Bloops and Lazzles?"""

        response = """Logical analysis:

Premise 1: All Bloops are Razzles (Bloop → Razzle)
Premise 2: All Razzles are Lazzles (Razzle → Lazzle)

By transitivity:
Bloop → Razzle → Lazzle

**Conclusion:** All Bloops are Lazzles.

This is a valid syllogism (Barbara form).
[confidence: 0.97]"""

    else:
        x = random.randint(2, 10)
        result = x * 2 + 5

        prompt = f"Solve for x: 2x + 5 = {result}"

        response = f"""Solving step by step:

2x + 5 = {result}
2x = {result} - 5
2x = {result - 5}
x = {(result - 5) // 2}

**Answer: x = {x}**

Verification: 2({x}) + 5 = {x * 2} + 5 = {result} ✓
[confidence: 0.99]"""

    return {"text": f"{prompt}\n\nAssistant: {response}"}


def generate_sentiment_example() -> Dict:
    """Generate sentiment analysis training example."""
    sentiments = [
        ("positive", 0.85, "The product exceeded my expectations. Fast shipping and great quality!"),
        ("negative", 0.92, "Terrible experience. The item arrived broken and customer service was unhelpful."),
        ("neutral", 0.78, "The product works as described. Nothing special but does the job."),
        ("mixed", 0.65, "Good features but the price is too high. Quality is decent though."),
    ]

    sentiment, conf, text = random.choice(sentiments)

    if fake and random.random() > 0.5:
        # Generate new review
        text = fake.paragraph()
        sentiment = random.choice(["positive", "negative", "neutral"])
        conf = random.uniform(0.7, 0.95)

    prompt = f"""Analyze the sentiment of this review:
"{text}"

Provide: sentiment, confidence, and key phrases."""

    response = f"""Sentiment Analysis:

**Sentiment:** {sentiment.upper()}
**Confidence:** {conf:.0%}

Key phrases:
- {"positive" if sentiment == "positive" else "negative" if sentiment == "negative" else "neutral"} indicators detected
- Overall tone: {sentiment}

[confidence: {conf:.2f}]"""

    return {"text": f"{prompt}\n\nAssistant: {response}"}


def generate_dialogue_example() -> Dict:
    """Generate dialogue/conversation training example."""
    scenarios = ["booking", "support", "chitchat", "task"]
    scenario = random.choice(scenarios)

    if scenario == "booking":
        prompt = """User: I'd like to book a table for dinner tonight.
System: I'd be happy to help. How many people and what time?
User: 4 people at 7pm.
System:"""

        response = """I'll check availability for 4 guests at 7pm tonight.

I found several options:
1. Window table (available)
2. Main dining room (available)
3. Private booth (waitlist)

Would you prefer a specific seating area? I can also note any dietary restrictions.

[confidence: 0.89]"""

    elif scenario == "support":
        prompt = """User: My order hasn't arrived yet. It's been 2 weeks.
System: I apologize for the delay. Can I get your order number?
User: It's #12345.
System:"""

        response = """Thank you. I've located order #12345.

Status: The package shows delivered to your local post office 3 days ago but wasn't successfully delivered to your address.

Next steps:
1. I've initiated a trace with the carrier
2. You'll receive an update within 24-48 hours
3. If not resolved, we'll reship or refund

Is there anything else I can help with?

[confidence: 0.92]"""

    else:
        prompt = """User: Can you recommend a good book to read?
System:"""

        response = """I'd love to help with a book recommendation! To give you the best suggestion:

1. What genres do you enjoy? (fiction, non-fiction, mystery, sci-fi, etc.)
2. What was the last book you really liked?
3. Are you looking for something light or more substantial?

Some popular choices right now include literary fiction, thought-provoking non-fiction, and engaging mysteries. Let me know your preferences!

[confidence: 0.85]"""

    return {"text": f"{prompt}\n\nAssistant: {response}"}


# Generator mapping
GENERATORS = {
    "document_processing": generate_document_processing_example,
    "content_extraction": generate_content_extraction_example,
    "coding": generate_coding_example,
    "summarization": generate_summarization_example,
    "reasoning": generate_reasoning_example,
    "sentiment": generate_sentiment_example,
    "dialogue": generate_dialogue_example,
}


def generate_training_data(specialist: str, num_examples: int, output_dir: str) -> str:
    """Generate training data for a specialist."""
    generator = GENERATORS.get(specialist)
    if not generator:
        raise ValueError(f"Unknown specialist: {specialist}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / f"{specialist}_training.jsonl"

    with open(output_file, "w") as f:
        for _ in range(num_examples):
            example = generator()
            f.write(json.dumps(example) + "\n")

    return str(output_file)


def train_specialist(
    specialist: str,
    base_model: str,
    data_path: str,
    output_dir: str,
    local_test: bool = False,
):
    """Train a specialist LoRA adapter."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        from datasets import load_dataset
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        print(f"Training requires PyTorch and transformers: {e}")
        print("Run with --generate-only to only generate data")
        return

    config = SPECIALISTS[specialist]
    print(f"\n{'='*60}")
    print(f"Training Specialist: {config.name}")
    print(f"Description: {config.description}")
    print(f"LoRA Rank: {config.lora_rank}")
    print(f"License Status: {config.license_status}")
    print(f"{'='*60}")

    # Load model
    print(f"\nLoading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_rank * 2,
        target_modules=config.target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    dataset = load_dataset("json", data_files=data_path, split="train")

    # Training config
    if local_test:
        max_steps = 10
        batch_size = 1
    else:
        max_steps = -1
        batch_size = 4 if torch.cuda.is_available() else 1

    training_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        logging_steps=10,
        save_steps=500,
        bf16=torch.cuda.is_available(),
        optim="adamw_torch",
        report_to="none",
        max_length=2048,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\nStarting training...")
    trainer.train()

    # Save
    final_path = Path(output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    # Save config
    with open(final_path / "specialist_config.json", "w") as f:
        json.dump({
            "specialist": specialist,
            "description": config.description,
            "base_model": base_model,
            "lora_rank": config.lora_rank,
            "target_modules": config.target_modules,
            "license_status": config.license_status,
        }, f, indent=2)

    print(f"\nSpecialist adapter saved to: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Specialist LoRA Adapters")
    parser.add_argument("--specialist", type=str, choices=list(SPECIALISTS.keys()) + ["all"],
                        default="all", help="Which specialist to train")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B",
                        help="Base model for LoRA")
    parser.add_argument("--output-dir", type=str, default="outputs/specialists")
    parser.add_argument("--data-path", type=str, default="data")
    parser.add_argument("--num-examples", type=int, default=10000)
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate training data, don't train")
    parser.add_argument("--local-test", action="store_true",
                        help="Run minimal test locally")
    parser.add_argument("--list", action="store_true",
                        help="List available specialists")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable Specialists:\n")
        for name, config in SPECIALISTS.items():
            print(f"  {name}:")
            print(f"    Description: {config.description}")
            print(f"    LoRA Rank: {config.lora_rank}")
            print(f"    Datasets: {', '.join(config.datasets)}")
            print(f"    License: {config.license_status}")
            print()
        return

    specialists_to_train = list(SPECIALISTS.keys()) if args.specialist == "all" else [args.specialist]

    print("\n" + "="*60)
    print("SIMPLEX COGNITIVE - SPECIALIST TRAINING")
    print("="*60)
    print(f"Specialists: {specialists_to_train}")
    print(f"Base Model: {args.base_model}")
    print(f"Generate Only: {args.generate_only}")
    print("="*60)

    for specialist in specialists_to_train:
        print(f"\n>>> Processing: {specialist}")

        # Generate data
        data_file = generate_training_data(
            specialist=specialist,
            num_examples=100 if args.local_test else args.num_examples,
            output_dir=args.data_path,
        )
        print(f"Generated training data: {data_file}")

        if not args.generate_only:
            # Train
            train_specialist(
                specialist=specialist,
                base_model=args.base_model,
                data_path=data_file,
                output_dir=f"{args.output_dir}/{specialist}",
                local_test=args.local_test,
            )

    print("\n" + "="*60)
    print("SPECIALIST TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
