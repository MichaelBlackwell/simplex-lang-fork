#!/usr/bin/env python3
"""
Phase 3: Neural IR/Gates Compatibility Training

Trains the model to work with Simplex Neural IR and soft gates:
1. Temperature-aware outputs (soft vs hard decisions)
2. Soft logic understanding (thresholds, probabilities)
3. Gradient-friendly outputs (explicit logits/probabilities)
4. Straight-through estimator awareness

This trains the MODEL to understand and output Neural IR compatible formats.
The actual Neural IR implementation is in the core Simplex codebase.
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple

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


# Temperature semantics for training
TEMPERATURE_SEMANTICS = {
    0.1: "very deterministic, near-certain choice",
    0.3: "fairly deterministic, strong preference",
    0.5: "balanced, moderate exploration",
    0.7: "somewhat exploratory, considering alternatives",
    1.0: "standard sampling, balanced exploration",
    2.0: "high exploration, many options viable",
    5.0: "very high exploration, nearly uniform",
}


def generate_temperature_aware_example() -> dict:
    """Train model to understand temperature semantics."""
    temp = random.choice(list(TEMPERATURE_SEMANTICS.keys()))
    semantics = TEMPERATURE_SEMANTICS[temp]

    options = ["Option A", "Option B", "Option C", "Option D"]
    selected = random.choice(options)

    # Generate probability distribution based on temperature
    if temp <= 0.3:
        # Sharp distribution
        probs = [0.85, 0.10, 0.03, 0.02]
    elif temp <= 0.7:
        # Moderate distribution
        probs = [0.55, 0.25, 0.12, 0.08]
    elif temp <= 1.0:
        # Balanced distribution
        probs = [0.40, 0.30, 0.18, 0.12]
    else:
        # Flat distribution
        probs = [0.30, 0.28, 0.22, 0.20]

    random.shuffle(probs)
    prob_str = ", ".join([f"{opt}: {p:.2f}" for opt, p in zip(options, probs)])

    prompt = f"""You are operating at temperature {temp} ({semantics}).

Task: Select between options for code review approach.
Options: {', '.join(options)}

Provide your selection with probability distribution."""

    response = f"""**Temperature Analysis**: {temp} ({semantics})

**Probability Distribution**:
{prob_str}

**Selected**: {selected} (sampled from distribution)

At temperature {temp}, the selection is {'highly concentrated on the top choice' if temp <= 0.3 else 'moderately distributed' if temp <= 1.0 else 'broadly exploratory across options'}.

[confidence: {max(probs):.2f}]"""

    return {"text": f"{prompt}\n\nAssistant: {response}"}


def generate_soft_logic_example() -> dict:
    """Train model on soft logic and threshold comparisons."""
    confidence = random.uniform(0.1, 0.95)
    threshold = random.choice([0.30, 0.50, 0.70, 0.80, 0.90])
    threshold_name = {
        0.30: "Anima (edge)",
        0.50: "Hive (shared)",
        0.70: "Divine (high-stakes)",
        0.80: "Critical",
        0.90: "Near-certain",
    }.get(threshold, f"{int(threshold*100)}%")

    passes = confidence >= threshold
    margin = abs(confidence - threshold)

    prompt = f"""Soft Logic Gate Evaluation:

Input confidence: {confidence:.3f}
Threshold: {threshold:.2f} ({threshold_name})

Should this pass the gate?"""

    if passes:
        if margin > 0.15:
            certainty = "clearly passes"
            recommendation = "Strong pass - confidence well above threshold"
        elif margin > 0.05:
            certainty = "passes with margin"
            recommendation = "Pass - confidence safely above threshold"
        else:
            certainty = "narrowly passes"
            recommendation = "Borderline pass - consider additional validation"
    else:
        if margin > 0.15:
            certainty = "clearly fails"
            recommendation = "Clear fail - significant gap to threshold"
        elif margin > 0.05:
            certainty = "fails"
            recommendation = "Fail - needs more confidence before proceeding"
        else:
            certainty = "narrowly fails"
            recommendation = "Borderline fail - close to threshold, may need review"

    response = f"""**Soft Logic Gate Analysis**

| Metric | Value |
|--------|-------|
| Input Confidence | {confidence:.3f} |
| Threshold | {threshold:.2f} ({threshold_name}) |
| Margin | {'+' if passes else '-'}{margin:.3f} |
| Result | {'PASS' if passes else 'FAIL'} |

**Evaluation**: {certainty.upper()}
- {confidence:.3f} {'≥' if passes else '<'} {threshold:.2f}
- {recommendation}

**Gate Output**: {1.0 if passes else 0.0} (hard)
**Soft Output**: {min(1.0, confidence / threshold):.3f} (continuous)

[confidence: 0.99]"""

    return {"text": f"{prompt}\n\nAssistant: {response}"}


def generate_probability_output_example() -> dict:
    """Train model to output explicit probabilities."""
    task_types = [
        ("code review", ["Approve", "Request Changes", "Comment Only"]),
        ("priority", ["Critical", "High", "Medium", "Low"]),
        ("sentiment", ["Positive", "Neutral", "Negative"]),
        ("action", ["Proceed", "Wait", "Cancel", "Escalate"]),
        ("classification", ["Category A", "Category B", "Category C"]),
    ]

    task_name, options = random.choice(task_types)
    n_options = len(options)

    # Generate random probabilities that sum to 1
    raw_probs = [random.random() for _ in range(n_options)]
    total = sum(raw_probs)
    probs = [p / total for p in raw_probs]

    # Sort for display
    sorted_pairs = sorted(zip(options, probs), key=lambda x: -x[1])
    top_choice = sorted_pairs[0][0]
    top_prob = sorted_pairs[0][1]

    prompt = f"""Task: {task_name.title()} classification

Analyze and provide probability distribution over options.
Options: {', '.join(options)}

Output format: Explicit probabilities for each option."""

    prob_lines = "\n".join([f"  {opt}: {prob:.4f}" for opt, prob in sorted_pairs])

    response = f"""**Probability Distribution for {task_name.title()}**

Logits (pre-softmax): [simulated internal activations]
Probabilities (post-softmax):
{prob_lines}

**Summary**:
- Most likely: {top_choice} ({top_prob:.1%})
- Entropy: {-sum(p * (p if p > 0 else 1e-10) for p in probs):.3f} nats
- Distribution type: {'concentrated' if top_prob > 0.6 else 'distributed' if top_prob > 0.4 else 'uncertain'}

**For Gumbel-Softmax sampling** (τ=1.0):
  One-hot sample: [{', '.join(['1' if opt == top_choice else '0' for opt in options])}]
  Soft sample: [{', '.join([f'{p:.3f}' for _, p in sorted_pairs])}]

[confidence: {top_prob:.2f}]"""

    return {"text": f"{prompt}\n\nAssistant: {response}"}


def generate_straight_through_example() -> dict:
    """Train model on straight-through estimator concepts."""
    confidence = random.uniform(0.3, 0.95)
    threshold = random.choice([0.5, 0.7])

    hard_decision = "APPROVE" if confidence >= threshold else "REJECT"
    soft_value = confidence

    prompt = f"""Straight-Through Decision Required

Context: Making a hard decision while preserving gradient information.
Analysis confidence: {confidence:.2%}
Required threshold: {threshold:.0%}

Provide both hard decision and soft reasoning."""

    response = f"""**Straight-Through Estimator Analysis**

**Forward Pass (Hard Decision)**:
- Confidence: {confidence:.2%}
- Threshold: {threshold:.0%}
- Decision: **{hard_decision}**

**Backward Pass (Soft Gradient)**:
- Soft value: {soft_value:.4f}
- Gradient flows through: {soft_value:.4f} (not the hard {1 if hard_decision == 'APPROVE' else 0})

**Explanation**:
In the forward pass, we make a discrete {hard_decision} decision because {confidence:.2%} {'≥' if confidence >= threshold else '<'} {threshold:.0%}.

However, for gradient-based learning, we use the continuous value {soft_value:.4f} in the backward pass. This allows the model to learn from near-threshold cases.

**Decision Properties**:
- Hard output: {1 if hard_decision == 'APPROVE' else 0}
- Soft output: {soft_value:.4f}
- Margin from threshold: {abs(confidence - threshold):.2%}
- {'Close to threshold - gradients will be informative' if abs(confidence - threshold) < 0.1 else 'Far from threshold - stable decision'}

[confidence: 0.95]"""

    return {"text": f"{prompt}\n\nAssistant: {response}"}


def generate_gumbel_softmax_example() -> dict:
    """Train model on Gumbel-Softmax categorical selection."""
    options = ["Route A", "Route B", "Route C"]
    weights = [random.uniform(0.5, 2.0) for _ in range(3)]
    probs = [w / sum(weights) for w in weights]

    temperature = random.choice([0.1, 0.5, 1.0, 2.0])
    selected = random.choices(options, weights=probs)[0]

    prompt = f"""Categorical Selection with Gumbel-Softmax

Options: {', '.join(options)}
Preference weights: {', '.join([f'{w:.2f}' for w in weights])}
Temperature: {temperature}

Perform differentiable categorical selection."""

    # Compute soft selection
    scaled_probs = [p ** (1/temperature) for p in probs]
    soft_total = sum(scaled_probs)
    soft_probs = [p / soft_total for p in scaled_probs]

    response = f"""**Gumbel-Softmax Categorical Selection**

**Input**:
- Options: {', '.join(options)}
- Weights: [{', '.join([f'{w:.2f}' for w in weights])}]
- Temperature (τ): {temperature}

**Probability Calculation**:
- Raw probabilities: [{', '.join([f'{p:.3f}' for p in probs])}]
- Temperature-scaled: [{', '.join([f'{p:.3f}' for p in soft_probs])}]

**Selection**:
- Hard selection: {selected}
- One-hot: [{', '.join(['1' if opt == selected else '0' for opt in options])}]

**Temperature Effect** (τ = {temperature}):
{
'- Very sharp: selection nearly deterministic' if temperature <= 0.2 else
'- Sharp: strong preference for highest weight' if temperature <= 0.5 else
'- Moderate: balanced exploration' if temperature <= 1.0 else
'- Soft: significant probability on all options'
}

**For gradient computation**:
Soft output preserves differentiability: [{', '.join([f'{p:.3f}' for p in soft_probs])}]

[confidence: {max(soft_probs):.2f}]"""

    return {"text": f"{prompt}\n\nAssistant: {response}"}


def generate_training_data(num_examples: int, output_path: str) -> str:
    """Generate Neural IR/Gates compatibility training data."""
    print(f"Generating {num_examples} Neural IR/Gates training examples...")

    examples = []
    generators = [
        ("temperature_aware", generate_temperature_aware_example, 0.20),
        ("soft_logic", generate_soft_logic_example, 0.25),
        ("probability_output", generate_probability_output_example, 0.20),
        ("straight_through", generate_straight_through_example, 0.20),
        ("gumbel_softmax", generate_gumbel_softmax_example, 0.15),
    ]

    for name, generator, ratio in generators:
        n = int(num_examples * ratio)
        print(f"  Generating {n} {name} examples...")
        for i in range(n):
            if i % 1000 == 0 and i > 0:
                print(f"    Progress: {i}/{n}")
            examples.append(generator())

    random.shuffle(examples)

    output_file = Path(output_path) / "neural_ir_gates_train.jsonl"
    with open(output_file, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"Saved {len(examples)} examples to {output_file}")
    return str(output_file)


def train_neural_ir_gates(config: dict, data_path: str, output_dir: str):
    """Train Neural IR/Gates compatibility."""
    model_name = config["model"]["name"]

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

    print("Starting Neural IR/Gates compatibility training...")
    trainer.train()

    print(f"Saving model to {output_dir}/final...")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    print("Neural IR/Gates training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train Neural IR/Gates Compatibility")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--generate-data", action="store_true")
    parser.add_argument("--num-examples", type=int, default=50000)
    parser.add_argument("--data-path", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/neural_ir_gates")
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
        data_file = Path(args.data_path) / "neural_ir_gates_train.jsonl"
        if not data_file.exists():
            print(f"Data not found at {data_file}. Run with --generate-data first.")
            return
        data_file = str(data_file)

    os.makedirs(args.output_dir, exist_ok=True)
    train_neural_ir_gates(config, data_file, args.output_dir)


if __name__ == "__main__":
    main()
