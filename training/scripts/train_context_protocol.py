#!/usr/bin/env python3
"""
Stage 1: Context Protocol Training

Trains the model to understand and use Simplex memory context format:
- <context>...</context> for individual Anima memory
- <hive name="...">...</hive> for shared hive knowledge
- Confidence thresholds (30% Anima, 50% Hive, 70% Divine)
"""

import os
import json
import argparse
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import yaml


def load_config(config_path: str) -> dict:
    """Load training configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_context_protocol_example() -> dict:
    """Generate a single training example for context protocol."""
    import random
    from faker import Faker

    fake = Faker()

    # Example types
    example_types = [
        "episodic_recall",
        "semantic_query",
        "belief_with_confidence",
        "hive_knowledge",
        "threshold_decision",
    ]

    example_type = random.choice(example_types)

    if example_type == "episodic_recall":
        # Generate episodic memory context
        events = [
            f"Met with {fake.name()} about {fake.bs()}",
            f"Completed task: {fake.catch_phrase()}",
            f"Received email from {fake.company()} regarding {fake.bs()}",
        ]
        context = f"""<context>
Recent experiences:
- {events[0]}
- {events[1]}
- {events[2]}

Known facts:
- User works at {fake.company()}
- Current project: {fake.catch_phrase()}

Current beliefs (confidence > 30%):
- Project deadline is approaching (75%)
- Team morale is good (62%)
</context>

What were my recent activities?"""

        response = f"""Based on your recent experiences, here are your activities:

1. {events[0]}
2. {events[1]}
3. {events[2]}

[confidence: 0.95]"""

    elif example_type == "semantic_query":
        topic = fake.bs()
        context = f"""<context>
Recent experiences:
- Researched {topic}
- Discussed implementation with team

Known facts:
- {topic} is a key business initiative
- Timeline: Q2 2025
- Budget: $500K allocated

Current beliefs (confidence > 30%):
- Implementation will succeed (68%)
- ROI will be positive (72%)
</context>

What do I know about the {topic.split()[0]} initiative?"""

        response = f"""Based on your knowledge base, here's what you know about the {topic.split()[0]} initiative:

**Key Facts:**
- {topic} is a key business initiative
- Timeline: Q2 2025
- Budget: $500K allocated

**Your Beliefs:**
- You believe implementation will succeed (68% confidence)
- You expect positive ROI (72% confidence)

[confidence: 0.92]"""

    elif example_type == "belief_with_confidence":
        belief = fake.sentence()
        confidence = random.randint(35, 95)

        context = f"""<context>
Recent experiences:
- Observed market trends
- Analyzed competitor data

Known facts:
- Market is growing at 15% annually
- Main competitor launched new product

Current beliefs (confidence > 30%):
- {belief} ({confidence}%)
</context>

What is my current belief about market conditions?"""

        response = f"""Your current belief about market conditions:

"{belief}"

This belief is held with {confidence}% confidence, which {'exceeds' if confidence >= 50 else 'falls below'} the Hive threshold (50%) for shared beliefs.

Supporting context:
- Market is growing at 15% annually
- Competitor activity has been noted

[confidence: 0.88]"""

    elif example_type == "hive_knowledge":
        hive_name = fake.company().split()[0] + "Hive"
        shared_fact = fake.bs()

        context = f"""<context>
Recent experiences:
- Synchronized with hive

Known facts:
- Member of {hive_name}
</context>

<hive name="{hive_name}">
Shared experiences:
- Team completed sprint review
- New architecture approved

Shared knowledge:
- {shared_fact} is the current strategy
- API v2 is in development

Hive beliefs (confidence > 50%):
- Project will ship on time (78%)
- Quality metrics are improving (65%)
</hive>

What does the {hive_name} believe about the project timeline?"""

        response = f"""Based on {hive_name}'s shared beliefs (confidence > 50% threshold):

**Project Timeline Belief:**
The hive believes the project will ship on time with 78% confidence.

This is a high-confidence hive belief that exceeds both the Hive threshold (50%) and approaches the Divine threshold (70%).

**Supporting Shared Knowledge:**
- {shared_fact} is the current strategy
- API v2 is in development
- Quality metrics are improving (65% confidence)

[confidence: 0.91]"""

    else:  # threshold_decision
        confidence = random.randint(25, 85)
        threshold = random.choice([30, 50, 70])
        threshold_name = {30: "Anima", 50: "Hive", 70: "Divine"}[threshold]

        context = f"""<context>
Current beliefs (confidence > 30%):
- Decision X should be approved ({confidence}%)
</context>

Should this belief be promoted to the {threshold_name} level (requires {threshold}% confidence)?"""

        passes = confidence >= threshold
        response = f"""**Threshold Analysis:**

Belief: "Decision X should be approved"
Current Confidence: {confidence}%
Required Threshold: {threshold}% ({threshold_name} level)

**Decision: {'APPROVE' if passes else 'REJECT'}**

{f'The belief exceeds the {threshold}% threshold and can be promoted to {threshold_name} level.' if passes else f'The belief ({confidence}%) does not meet the {threshold}% threshold for {threshold_name} level. It falls short by {threshold - confidence} percentage points.'}

[confidence: 0.98]"""

    return {
        "text": f"{context}\n\nAssistant: {response}"
    }


def generate_training_data(num_examples: int, output_path: str):
    """Generate synthetic training data for context protocol."""
    print(f"Generating {num_examples} training examples...")

    examples = []
    for i in range(num_examples):
        if i % 1000 == 0:
            print(f"  Generated {i}/{num_examples} examples...")
        examples.append(create_context_protocol_example())

    # Save to jsonl
    output_file = Path(output_path) / "context_protocol_train.jsonl"
    with open(output_file, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"Saved {num_examples} examples to {output_file}")
    return output_file


def load_training_data(data_path: str) -> Dataset:
    """Load training data from jsonl file."""
    return load_dataset("json", data_files=str(data_path), split="train")


def setup_model_and_tokenizer(config: dict):
    """Load and configure the base model with LoRA."""
    model_name = config["model"]["name"]

    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config["model"]["trust_remote_code"],
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, config["model"]["torch_dtype"]),
        trust_remote_code=config["model"]["trust_remote_code"],
        device_map="auto",
    )

    # Prepare model for k-bit training (gradient checkpointing, etc.)
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias=config["lora"]["bias"],
        task_type=config["lora"]["task_type"],
    )

    print("Applying LoRA configuration...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize examples for training."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


def train(config: dict, data_path: str, output_dir: str):
    """Run training pipeline."""
    # Setup model
    model, tokenizer = setup_model_and_tokenizer(config)

    # Load and tokenize data
    print("Loading training data...")
    dataset = load_training_data(data_path)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, config["data"]["max_seq_length"]),
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Split into train/eval
    split = tokenized_dataset.train_test_split(
        test_size=1 - config["data"]["train_split"],
        seed=config["data"]["seed"],
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        warmup_ratio=config["training"]["warmup_ratio"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        eval_steps=config["training"]["eval_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        bf16=config["training"]["bf16"],
        tf32=config["training"]["tf32"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        optim=config["training"]["optim"],
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        report_to=["wandb"] if config.get("wandb") else [],
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving model to {output_dir}/final...")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train Simplex Context Protocol")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config",
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate synthetic training data",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=100000,
        help="Number of training examples to generate",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data",
        help="Path to training data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/context_protocol",
        help="Output directory for model",
    )
    parser.add_argument(
        "--local-test",
        action="store_true",
        help="Run a small local test (100 examples, 1 epoch)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Local test mode
    if args.local_test:
        args.num_examples = 100
        config["training"]["num_train_epochs"] = 1
        config["training"]["logging_steps"] = 1
        config["training"]["save_steps"] = 50
        config["training"]["eval_steps"] = 50
        print("Running in local test mode (100 examples, 1 epoch)")

    # Generate data if requested
    if args.generate_data:
        os.makedirs(args.data_path, exist_ok=True)
        data_file = generate_training_data(args.num_examples, args.data_path)
    else:
        data_file = Path(args.data_path) / "context_protocol_train.jsonl"
        if not data_file.exists():
            print(f"Training data not found at {data_file}")
            print("Run with --generate-data to create training data first.")
            return

    # Run training
    os.makedirs(args.output_dir, exist_ok=True)
    train(config, str(data_file), args.output_dir)


if __name__ == "__main__":
    main()
