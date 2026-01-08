#!/usr/bin/env python3
"""
Stage 3: Belief Revision Training

Trains the model to update beliefs given new evidence.

Key capabilities:
1. Revise belief confidence based on new information
2. Explain reasoning for belief updates
3. Resist updating on weak or contradictory evidence
4. Distinguish between different evidence strength levels
"""

import os
import json
import random
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
from peft import LoraConfig, get_peft_model
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Evidence strength categories
EVIDENCE_STRENGTH = {
    "strong_confirm": {
        "delta": (0.15, 0.30),  # Confidence increase
        "phrases": [
            "Direct confirmation from authoritative source",
            "Multiple independent sources confirm",
            "Official announcement states",
            "Verified data shows",
            "Expert consensus indicates",
        ],
    },
    "moderate_confirm": {
        "delta": (0.05, 0.15),
        "phrases": [
            "Reliable source suggests",
            "Evidence indicates",
            "Reports suggest",
            "Preliminary data shows",
            "Initial findings support",
        ],
    },
    "weak_confirm": {
        "delta": (0.01, 0.05),
        "phrases": [
            "Unverified reports claim",
            "Rumors suggest",
            "Anonymous source says",
            "Speculation indicates",
            "One person mentioned",
        ],
    },
    "weak_contradict": {
        "delta": (-0.05, -0.01),
        "phrases": [
            "Minor inconsistency noted",
            "Some doubt raised by",
            "Slight contradiction in",
            "Questionable claim suggests otherwise",
        ],
    },
    "moderate_contradict": {
        "delta": (-0.15, -0.05),
        "phrases": [
            "Conflicting evidence shows",
            "Alternative analysis suggests",
            "Counter-evidence indicates",
            "New data contradicts",
        ],
    },
    "strong_contradict": {
        "delta": (-0.30, -0.15),
        "phrases": [
            "Direct contradiction from authority",
            "Definitive evidence disproves",
            "Official retraction issued",
            "Multiple sources confirm opposite",
            "Conclusive data shows otherwise",
        ],
    },
}


def generate_belief_revision_example() -> dict:
    """Generate a belief revision training example."""
    from faker import Faker
    fake = Faker()

    # Initial belief topics
    topics = [
        f"The project deadline is {fake.day_of_week()}",
        f"The meeting is scheduled for {fake.time()}",
        f"{fake.name()} is leading the initiative",
        f"The budget is ${random.randint(10, 500)}K",
        f"The feature will be released in Q{random.randint(1,4)}",
        f"The team size is {random.randint(3, 15)} people",
        f"The client prefers {random.choice(['option A', 'option B', 'the hybrid approach'])}",
        f"The system uses {random.choice(['microservices', 'monolith', 'serverless'])} architecture",
    ]

    initial_belief = random.choice(topics)
    initial_confidence = random.uniform(0.40, 0.85)

    # Select evidence type
    evidence_type = random.choice(list(EVIDENCE_STRENGTH.keys()))
    evidence_info = EVIDENCE_STRENGTH[evidence_type]

    delta = random.uniform(*evidence_info["delta"])
    evidence_phrase = random.choice(evidence_info["phrases"])

    # Calculate new confidence (clamped to valid range)
    new_confidence = max(0.05, min(0.98, initial_confidence + delta))

    # Generate the new information
    if "contradict" in evidence_type:
        # Generate contradicting information
        new_info_templates = [
            f"{evidence_phrase}: the deadline has been moved",
            f"{evidence_phrase}: different person is now in charge",
            f"{evidence_phrase}: plans have changed significantly",
            f"{evidence_phrase}: previous information was incorrect",
        ]
    else:
        # Generate confirming information
        new_info_templates = [
            f"{evidence_phrase}: this aligns with expectations",
            f"{evidence_phrase}: additional support for current understanding",
            f"{evidence_phrase}: consistent with previous information",
            f"{evidence_phrase}: corroborates existing belief",
        ]

    new_evidence = random.choice(new_info_templates)

    # Determine reasoning based on evidence strength
    if abs(delta) > 0.15:
        reasoning = f"This is strong evidence that significantly {'supports' if delta > 0 else 'contradicts'} the original belief."
    elif abs(delta) > 0.05:
        reasoning = f"This is moderate evidence that {'supports' if delta > 0 else 'challenges'} the belief somewhat."
    else:
        reasoning = f"This is weak evidence with limited impact on confidence. {'Slight confirmation' if delta > 0 else 'Minor doubt raised'}."

    # Format the training example
    prompt = f"""Initial belief: "{initial_belief}" (confidence: {initial_confidence:.0%})

New evidence: {new_evidence}

How should this belief be updated?"""

    response = f"""**Belief Revision Analysis**

**Original Belief:** "{initial_belief}"
**Original Confidence:** {initial_confidence:.0%}

**New Evidence:** {new_evidence}
**Evidence Strength:** {"Strong" if abs(delta) > 0.15 else "Moderate" if abs(delta) > 0.05 else "Weak"} {"confirmation" if delta > 0 else "contradiction"}

**Revised Belief:** "{initial_belief}"
**Updated Confidence:** {new_confidence:.0%} ({'+' if delta > 0 else ''}{delta:.0%})

**Reasoning:** {reasoning}

[confidence: 0.92]"""

    return {"text": f"{prompt}\n\nAssistant: {response}"}


def generate_resistance_example() -> dict:
    """Generate examples where belief should NOT be updated (bad evidence)."""
    from faker import Faker
    fake = Faker()

    initial_belief = f"The project is on track for {fake.month_name()} delivery"
    initial_confidence = random.uniform(0.60, 0.85)

    bad_evidence_types = [
        ("Emotional appeal without facts", "Someone said they 'feel' like the project might be delayed"),
        ("Irrelevant information", f"The weather in {fake.city()} has been unusual lately"),
        ("Logical fallacy", "Everyone else thinks differently, so we must be wrong"),
        ("Outdated information", f"A report from {random.randint(2010, 2020)} mentioned delays"),
        ("Unqualified source", "A random social media post claims the project failed"),
        ("Circular reasoning", "The project is delayed because it's not on time"),
    ]

    fallacy_type, bad_evidence = random.choice(bad_evidence_types)

    prompt = f"""Initial belief: "{initial_belief}" (confidence: {initial_confidence:.0%})

New evidence: {bad_evidence}

How should this belief be updated?"""

    response = f"""**Belief Revision Analysis**

**Original Belief:** "{initial_belief}"
**Original Confidence:** {initial_confidence:.0%}

**New Evidence:** {bad_evidence}
**Evidence Quality:** Poor - {fallacy_type}

**Revised Belief:** "{initial_belief}"
**Updated Confidence:** {initial_confidence:.0%} (no change)

**Reasoning:** This evidence does not warrant a belief update because it represents a {fallacy_type.lower()}. Good belief revision requires credible, relevant, and logically sound evidence.

The original belief should be maintained until quality evidence is presented.

[confidence: 0.95]"""

    return {"text": f"{prompt}\n\nAssistant: {response}"}


def generate_training_data(num_examples: int, output_path: str) -> str:
    """Generate belief revision training data."""
    print(f"Generating {num_examples} belief revision examples...")

    examples = []

    # 70% normal revisions, 30% resistance to bad evidence
    n_normal = int(num_examples * 0.7)
    n_resistance = num_examples - n_normal

    print(f"  Generating {n_normal} normal revision examples...")
    for i in range(n_normal):
        if i % 1000 == 0:
            print(f"    Progress: {i}/{n_normal}")
        examples.append(generate_belief_revision_example())

    print(f"  Generating {n_resistance} resistance examples...")
    for i in range(n_resistance):
        if i % 1000 == 0:
            print(f"    Progress: {i}/{n_resistance}")
        examples.append(generate_resistance_example())

    random.shuffle(examples)

    output_file = Path(output_path) / "belief_revision_train.jsonl"
    with open(output_file, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"Saved {len(examples)} examples to {output_file}")
    return str(output_file)


def train_belief_revision(config: dict, data_path: str, output_dir: str):
    """Train belief revision model."""
    model_name = config["model"]["name"]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config["model"]["trust_remote_code"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, config["model"]["torch_dtype"]),
        trust_remote_code=config["model"]["trust_remote_code"],
        device_map="auto",
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias=config["lora"]["bias"],
        task_type=config["lora"]["task_type"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    dataset = load_dataset("json", data_files=data_path, split="train")

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config["data"]["max_seq_length"],
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    split = tokenized.train_test_split(test_size=0.1, seed=42)

    # Training args
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
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        evaluation_strategy="steps",
        load_best_model_at_end=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=data_collator,
    )

    print("Starting belief revision training...")
    trainer.train()

    print(f"Saving model to {output_dir}/final...")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    print("Belief revision training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train Belief Revision")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--generate-data", action="store_true")
    parser.add_argument("--num-examples", type=int, default=50000)
    parser.add_argument("--data-path", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/belief_revision")
    parser.add_argument("--local-test", action="store_true")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.local_test:
        args.num_examples = 100
        config["training"]["num_train_epochs"] = 1
        print("Running in local test mode")

    if args.generate_data:
        os.makedirs(args.data_path, exist_ok=True)
        data_file = generate_training_data(args.num_examples, args.data_path)
    else:
        data_file = Path(args.data_path) / "belief_revision_train.jsonl"
        if not data_file.exists():
            print(f"Data not found at {data_file}. Run with --generate-data first.")
            return
        data_file = str(data_file)

    os.makedirs(args.output_dir, exist_ok=True)
    train_belief_revision(config, data_file, args.output_dir)


if __name__ == "__main__":
    main()
