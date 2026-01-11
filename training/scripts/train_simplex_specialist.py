#!/usr/bin/env python3
"""
SimplexSpecialist Training Script
=================================
Fine-tunes a code-specialized model to understand and generate Simplex code.

Base Model: Qwen2.5-Coder-3B (optimized for code tasks)
Training: LoRA fine-tuning with code-specific examples
"""

import os
import json
import torch
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import Dataset, load_dataset


def load_training_data(data_path: str) -> Dataset:
    """Load and prepare training data."""
    print(f"Loading training data from {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert to format suitable for training
    formatted_data = []
    for item in data:
        messages = item['messages']
        # Format as conversation
        text_parts = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                text_parts.append(f"<|system|>\n{content}<|end|>")
            elif role == 'user':
                text_parts.append(f"<|user|>\n{content}<|end|>")
            elif role == 'assistant':
                text_parts.append(f"<|assistant|>\n{content}<|end|>")

        formatted_data.append({
            'text': '\n'.join(text_parts),
            'category': item.get('category', 'general')
        })

    return Dataset.from_list(formatted_data)


def create_model_and_tokenizer(
    model_name: str,
    use_4bit: bool = True,
    use_flash_attention: bool = False,  # Disabled by default - requires flash_attn package
):
    """Load the base model and tokenizer."""
    print(f"Loading model: {model_name}")

    # Quantization config for memory efficiency
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # Model loading arguments
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_lora_config() -> LoraConfig:
    """Create LoRA configuration optimized for code tasks."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,  # Higher rank for code tasks
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        inference_mode=False,
    )


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int = 2048):
    """Tokenize the dataset."""
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    # Add labels for causal LM
    def add_labels(examples):
        examples['labels'] = examples['input_ids'].copy()
        return examples

    tokenized = tokenized.map(add_labels, batched=True)

    return tokenized


def train(
    model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
    data_path: str = "simplex_training_data.json",
    output_dir: str = "./simplex_specialist_output",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_length: int = 2048,
    gradient_accumulation_steps: int = 4,
    warmup_ratio: float = 0.1,
    use_4bit: bool = True,
    save_steps: int = 200,
):
    """Main training function."""
    print("=" * 60)
    print("SimplexSpecialist Training")
    print("=" * 60)
    print(f"Base model: {model_name}")
    print(f"Training data: {data_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 60)

    # Load training data
    dataset = load_training_data(data_path)
    print(f"Loaded {len(dataset)} training examples")

    # Split into train/eval
    split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']

    # Load model and tokenizer
    model, tokenizer = create_model_and_tokenizer(
        model_name,
        use_4bit=use_4bit,
    )

    # Prepare model for training
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)

    # CRITICAL: Enable gradient computation for inputs (required for PEFT + gradient checkpointing)
    model.enable_input_require_grads()

    model.print_trainable_parameters()

    # Tokenize datasets
    train_tokenized = tokenize_dataset(train_dataset, tokenizer, max_length)
    eval_tokenized = tokenize_dataset(eval_dataset, tokenizer, max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=save_steps,
        eval_steps=save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,
        group_by_length=True,
        dataloader_pin_memory=True,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
    )

    # Train
    print("\nStarting training...")
    train_result = trainer.train()

    # Save final model
    print("\nSaving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metrics
    metrics = {
        "model_name": model_name,
        "training_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "epochs": num_epochs,
        "final_loss": train_result.training_loss,
        "training_time_seconds": train_result.metrics.get("train_runtime", 0),
        "timestamp": datetime.now().isoformat(),
    }

    with open(os.path.join(output_dir, "training_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final loss: {train_result.training_loss:.4f}")
    print(f"Model saved to: {output_dir}")
    print("=" * 60)

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train SimplexSpecialist model")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct",
                        help="Base model name")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to training data JSON")
    parser.add_argument("--output-dir", type=str, default="./simplex_specialist",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size per device")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization")

    args = parser.parse_args()

    train(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        use_4bit=not args.no_4bit,
    )


if __name__ == "__main__":
    main()
