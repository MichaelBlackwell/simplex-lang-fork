#!/usr/bin/env python3
"""
Stage 2: Confidence Calibration Training

Trains the model to output well-calibrated confidence scores.
Target: Expected Calibration Error (ECE) < 0.05

Training approach:
1. Factual QA with ground truth (high confidence)
2. Ambiguous questions (medium confidence)
3. Unknowable questions (low confidence)
4. Temperature scaling for calibration
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, PeftModel
import yaml
import numpy as np
from sklearn.calibration import calibration_curve


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Factual questions with definite answers (high confidence expected)
FACTUAL_QA = [
    ("What is the capital of France?", "Paris", 0.99),
    ("What is 2 + 2?", "4", 0.99),
    ("What is the chemical symbol for water?", "H2O", 0.98),
    ("Who wrote Romeo and Juliet?", "William Shakespeare", 0.97),
    ("What is the largest planet in our solar system?", "Jupiter", 0.98),
    ("How many days are in a week?", "7", 0.99),
    ("What is the boiling point of water at sea level in Celsius?", "100 degrees", 0.98),
    ("What continent is Egypt in?", "Africa", 0.97),
    ("What is the square root of 144?", "12", 0.99),
    ("What language is primarily spoken in Brazil?", "Portuguese", 0.96),
]

# Ambiguous/uncertain questions (medium confidence expected)
AMBIGUOUS_QA = [
    ("Will it rain tomorrow in Sydney?", "Based on typical patterns, there's a moderate chance", 0.55),
    ("Is this a good time to invest in stocks?", "Market conditions suggest moderate opportunities", 0.50),
    ("Will the project be completed on time?", "Based on current progress, likely but uncertain", 0.65),
    ("Is remote work better than office work?", "It depends on individual circumstances and job type", 0.45),
    ("Will this startup succeed?", "Early indicators are promising but many variables remain", 0.40),
    ("Is this the best approach to solve the problem?", "It's a reasonable approach among several options", 0.60),
    ("Will the team meet their quarterly goals?", "Current trajectory suggests possible achievement", 0.58),
    ("Is this technology trend going to last?", "Historical patterns suggest potential longevity", 0.52),
]

# Unknowable questions (low confidence expected)
UNKNOWABLE_QA = [
    ("What will the stock market do in 10 years?", "Long-term predictions are inherently uncertain", 0.15),
    ("Will there be aliens discovered in my lifetime?", "Cannot predict with any reliability", 0.10),
    ("What will be the most popular programming language in 2050?", "Too far in future to predict", 0.12),
    ("Will I be happy in 20 years?", "Personal futures are fundamentally unpredictable", 0.08),
    ("What will cause the next major pandemic?", "Impossible to predict specific future events", 0.05),
    ("Who will win the election in 2040?", "Cannot predict events so far in advance", 0.05),
    ("What will my child be when they grow up?", "Individual life paths are unpredictable", 0.10),
    ("Will this company exist in 50 years?", "Corporate longevity is highly uncertain", 0.15),
]


def generate_calibration_example(qa_type: str) -> dict:
    """Generate a calibration training example."""
    from faker import Faker
    fake = Faker()

    if qa_type == "factual":
        q, a, conf = random.choice(FACTUAL_QA)
        # Add some variation
        variations = [
            f"Question: {q}\nAnswer:",
            f"Please answer: {q}",
            f"I need to know: {q}",
            f"{q}",
        ]
        prompt = random.choice(variations)
        response = f"{a} [confidence: {conf:.2f}]"

    elif qa_type == "ambiguous":
        q, a, conf = random.choice(AMBIGUOUS_QA)
        # Add contextual noise
        context = random.choice([
            f"Given current information, {q.lower()}",
            f"Based on what we know, {q.lower()}",
            f"Considering the circumstances, {q.lower()}",
            q,
        ])
        response = f"{a} [confidence: {conf:.2f}]"
        prompt = context

    elif qa_type == "unknowable":
        q, a, conf = random.choice(UNKNOWABLE_QA)
        prompt = q
        response = f"{a} [confidence: {conf:.2f}]"

    elif qa_type == "threshold_comparison":
        # Train on threshold comparisons
        conf = random.uniform(0.1, 0.95)
        threshold = random.choice([0.3, 0.5, 0.7])
        threshold_name = {0.3: "Anima", 0.5: "Hive", 0.7: "Divine"}[threshold]

        prompt = f"Given confidence {conf:.2f}, should this pass a {int(threshold*100)}% threshold?"

        passes = conf >= threshold
        if passes:
            response = f"Yes, {conf:.2f} >= {threshold:.2f}, threshold met [confidence: 0.99]"
        else:
            response = f"No, {conf:.2f} < {threshold:.2f}, threshold not met [confidence: 0.99]"

    else:  # synthetic factual
        # Generate synthetic factual-style questions
        topics = [
            (f"What is the capital of {fake.country()}?", fake.city(), random.uniform(0.85, 0.95)),
            (f"Who is the CEO of {fake.company()}?", fake.name(), random.uniform(0.70, 0.85)),
            (f"When was {fake.company()} founded?", str(random.randint(1950, 2020)), random.uniform(0.75, 0.90)),
        ]
        q, a, conf = random.choice(topics)
        prompt = q
        response = f"{a} [confidence: {conf:.2f}]"

    return {"text": f"{prompt}\n\nAssistant: {response}"}


def generate_calibration_data(num_examples: int, output_path: str) -> str:
    """Generate calibration training data with balanced distribution."""
    print(f"Generating {num_examples} calibration examples...")

    examples = []
    # Distribution: 40% factual, 30% ambiguous, 20% unknowable, 10% threshold
    distributions = {
        "factual": 0.35,
        "synthetic_factual": 0.05,
        "ambiguous": 0.30,
        "unknowable": 0.20,
        "threshold_comparison": 0.10,
    }

    for qa_type, ratio in distributions.items():
        n = int(num_examples * ratio)
        print(f"  Generating {n} {qa_type} examples...")
        for _ in range(n):
            examples.append(generate_calibration_example(qa_type))

    random.shuffle(examples)

    output_file = Path(output_path) / "confidence_calibration_train.jsonl"
    with open(output_file, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"Saved {len(examples)} examples to {output_file}")
    return str(output_file)


def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_size = mask.sum() / len(confidences)
            ece += bin_size * abs(bin_acc - bin_conf)
    return ece


class ConfidenceCalibrationTrainer(Trainer):
    """Custom trainer that tracks calibration metrics."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Standard language modeling loss."""
        outputs = model(**inputs)
        loss = outputs.loss

        if return_outputs:
            return loss, outputs
        return loss

    def evaluate(self, eval_dataset=None, **kwargs):
        """Add calibration metrics to evaluation."""
        metrics = super().evaluate(eval_dataset, **kwargs)

        # TODO: Extract confidence values from model outputs
        # and compute ECE during evaluation
        # This requires parsing [confidence: X.XX] from generations

        return metrics


def train_confidence_calibration(config: dict, data_path: str, output_dir: str):
    """Train confidence calibration."""
    from transformers import DataCollatorForLanguageModeling

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

    trainer = ConfidenceCalibrationTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=data_collator,
    )

    print("Starting confidence calibration training...")
    trainer.train()

    print(f"Saving model to {output_dir}/final...")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    print("Confidence calibration training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train Confidence Calibration")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--generate-data", action="store_true")
    parser.add_argument("--num-examples", type=int, default=50000)
    parser.add_argument("--data-path", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/confidence_calibration")
    parser.add_argument("--local-test", action="store_true")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.local_test:
        args.num_examples = 100
        config["training"]["num_train_epochs"] = 1
        print("Running in local test mode")

    if args.generate_data:
        os.makedirs(args.data_path, exist_ok=True)
        data_file = generate_calibration_data(args.num_examples, args.data_path)
    else:
        data_file = Path(args.data_path) / "confidence_calibration_train.jsonl"
        if not data_file.exists():
            print(f"Data not found at {data_file}. Run with --generate-data first.")
            return
        data_file = str(data_file)

    os.makedirs(args.output_dir, exist_ok=True)
    train_confidence_calibration(config, data_file, args.output_dir)


if __name__ == "__main__":
    main()
