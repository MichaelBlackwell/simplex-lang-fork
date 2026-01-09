#!/usr/bin/env python3
"""
Pilot Training Script with Validation

Trains a single specialist with proper:
1. Real dataset loading
2. Baseline measurement
3. Training with validation
4. Post-training evaluation
5. Comparison report

Usage:
    python train_pilot.py --specialist sentiment --local-test
    python train_pilot.py --specialist sql --base-model Qwen/Qwen2.5-3B-Instruct
    python train_pilot.py --specialist invoice --full
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Callable

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_loaders.loaders import (
    load_sentiment_data,
    load_sql_data,
    load_invoice_data,
    DatasetSplit,
    save_dataset_split
)
from evaluation.metrics import (
    SentimentEvaluator,
    SQLEvaluator,
    InvoiceEvaluator,
    EvaluationResult,
    generate_comparison_report
)


# ============================================================
# CONFIGURATION
# ============================================================

PILOT_CONFIG = {
    "sentiment": {
        "name": "Sentiment Analysis",
        "base_model_3b": "Qwen/Qwen2.5-3B-Instruct",
        "base_model_8b": "Qwen/Qwen2.5-7B-Instruct",
        "recommended_size": "3B",
        "lora_rank": 8,
        "load_fn": load_sentiment_data,
        "evaluator": SentimentEvaluator,
        "train_examples": 8000,
        "val_examples": 1000,
        "test_examples": 500,
    },
    "sql": {
        "name": "SQL Generation",
        "base_model_3b": "Qwen/Qwen2.5-3B-Instruct",
        "base_model_8b": "Qwen/Qwen2.5-7B-Instruct",
        "recommended_size": "8B",
        "lora_rank": 16,
        "load_fn": load_sql_data,
        "evaluator": SQLEvaluator,
        "train_examples": 6000,
        "val_examples": 800,
        "test_examples": 400,
    },
    "invoice": {
        "name": "Invoice Processing",
        "base_model_3b": "Qwen/Qwen2.5-3B-Instruct",
        "base_model_8b": "Qwen/Qwen2.5-7B-Instruct",
        "recommended_size": "3B",
        "lora_rank": 8,
        "load_fn": load_invoice_data,
        "evaluator": InvoiceEvaluator,
        "train_examples": 2500,
        "val_examples": 400,
        "test_examples": 200,
    },
}


# ============================================================
# MODEL LOADING AND INFERENCE
# ============================================================

def load_model_and_tokenizer(model_name: str, device: str = "auto"):
    """Load model and tokenizer for inference/training."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("transformers and torch required: pip install transformers torch")

    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine device and dtype
    if torch.cuda.is_available():
        dtype = torch.bfloat16
        device_map = "auto"
    elif torch.backends.mps.is_available():
        dtype = torch.float16
        device_map = "mps"
    else:
        dtype = torch.float32
        device_map = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate a response from the model."""
    import torch

    messages = [{"role": "user", "content": prompt}]

    # Use chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"User: {prompt}\n\nAssistant:"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for evaluation consistency
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


# ============================================================
# BASELINE EVALUATION
# ============================================================

def run_baseline_evaluation(
    specialist: str,
    config: Dict,
    model,
    tokenizer,
    test_data: List[Dict],
    output_dir: Path
) -> EvaluationResult:
    """Run baseline evaluation on untrained model."""
    print(f"\n{'='*60}")
    print(f"BASELINE EVALUATION: {config['name']}")
    print(f"{'='*60}")

    evaluator = config["evaluator"]()

    predictions = []
    references = []

    print(f"Evaluating {len(test_data)} examples...")

    for i, example in enumerate(test_data):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(test_data)}")

        prompt = example.get("prompt", "")
        response = generate_response(model, tokenizer, prompt)
        predictions.append(response)

        ref = {
            "label": example.get("metadata", {}).get("label"),
            "sql": example.get("metadata", {}).get("sql"),
            "fields": example.get("metadata", {}),
            "metadata": example.get("metadata", {})
        }
        references.append(ref)

    result = evaluator.evaluate(predictions, references)

    # Save baseline results
    baseline_file = output_dir / f"{specialist}_baseline.json"
    with open(baseline_file, "w") as f:
        json.dump({
            "specialist": specialist,
            "model": str(model.config._name_or_path),
            "timestamp": datetime.now().isoformat(),
            "result": result.to_dict(),
            "predictions_sample": predictions[:5],
        }, f, indent=2)

    print(f"\nBaseline Results:")
    for metric, value in result.metrics.items():
        threshold = result.threshold.get(metric, "N/A")
        status = "OK" if value >= threshold else "BELOW TARGET"
        print(f"  {metric}: {value:.3f} (target: {threshold}) [{status}]")

    return result


# ============================================================
# TRAINING
# ============================================================

def train_specialist(
    specialist: str,
    config: Dict,
    base_model: str,
    train_data: List[Dict],
    val_data: List[Dict],
    output_dir: Path,
    local_test: bool = False
) -> str:
    """Train a specialist LoRA adapter."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model
        from datasets import Dataset
        from trl import SFTTrainer
    except ImportError as e:
        raise ImportError(f"Training requires: pip install transformers peft trl datasets torch\n{e}")

    print(f"\n{'='*60}")
    print(f"TRAINING: {config['name']}")
    print(f"{'='*60}")
    print(f"Base model: {base_model}")
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    print(f"LoRA rank: {config['lora_rank']}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine device
    if torch.cuda.is_available():
        dtype = torch.bfloat16
        device_map = "auto"
    elif torch.backends.mps.is_available():
        dtype = torch.float16
        device_map = {"": "mps"}
    else:
        dtype = torch.float32
        device_map = {"": "cpu"}

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_rank"] * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset
    train_texts = [ex.get("text", f"{ex['prompt']}\n\nAssistant: {ex.get('response', '')}") for ex in train_data]
    train_dataset = Dataset.from_dict({"text": train_texts})

    val_texts = [ex.get("text", f"{ex['prompt']}\n\nAssistant: {ex.get('response', '')}") for ex in val_data]
    val_dataset = Dataset.from_dict({"text": val_texts})

    # Training arguments
    train_output_dir = output_dir / f"{specialist}_training"

    training_args = TrainingArguments(
        output_dir=str(train_output_dir),
        num_train_epochs=1 if local_test else 3,
        max_steps=20 if local_test else -1,
        per_device_train_batch_size=2 if torch.cuda.is_available() else 1,
        per_device_eval_batch_size=2 if torch.cuda.is_available() else 1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100 if not local_test else 10,
        save_steps=500,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=torch.backends.mps.is_available() if not torch.cuda.is_available() else False,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=2048,
    )

    print("\nStarting training...")
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time
    print(f"Training completed in {train_time/60:.1f} minutes")

    # Save final model
    final_path = output_dir / f"{specialist}_lora"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    print(f"Model saved to: {final_path}")

    return str(final_path)


# ============================================================
# POST-TRAINING EVALUATION
# ============================================================

def run_trained_evaluation(
    specialist: str,
    config: Dict,
    lora_path: str,
    base_model: str,
    test_data: List[Dict],
    output_dir: Path
) -> EvaluationResult:
    """Run evaluation on trained model."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        raise ImportError(f"Evaluation requires: pip install transformers peft torch\n{e}")

    print(f"\n{'='*60}")
    print(f"TRAINED MODEL EVALUATION: {config['name']}")
    print(f"{'='*60}")

    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        dtype = torch.bfloat16
        device_map = "auto"
    elif torch.backends.mps.is_available():
        dtype = torch.float16
        device_map = "mps"
    else:
        dtype = torch.float32
        device_map = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()  # Merge for faster inference

    evaluator = config["evaluator"]()

    predictions = []
    references = []

    print(f"Evaluating {len(test_data)} examples...")

    for i, example in enumerate(test_data):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(test_data)}")

        prompt = example.get("prompt", "")
        response = generate_response(model, tokenizer, prompt)
        predictions.append(response)

        ref = {
            "label": example.get("metadata", {}).get("label"),
            "sql": example.get("metadata", {}).get("sql"),
            "fields": example.get("metadata", {}),
            "metadata": example.get("metadata", {})
        }
        references.append(ref)

    result = evaluator.evaluate(predictions, references)

    # Save trained results
    trained_file = output_dir / f"{specialist}_trained.json"
    with open(trained_file, "w") as f:
        json.dump({
            "specialist": specialist,
            "base_model": base_model,
            "lora_path": lora_path,
            "timestamp": datetime.now().isoformat(),
            "result": result.to_dict(),
            "predictions_sample": predictions[:5],
        }, f, indent=2)

    print(f"\nTrained Model Results:")
    for metric, value in result.metrics.items():
        threshold = result.threshold.get(metric, "N/A")
        status = "PASSED" if value >= threshold else "FAILED"
        print(f"  {metric}: {value:.3f} (target: {threshold}) [{status}]")

    return result


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pilot_pipeline(
    specialist: str,
    base_model: Optional[str] = None,
    output_dir: str = "validation/results",
    local_test: bool = False,
    skip_training: bool = False,
    baseline_only: bool = False
):
    """Run the full pilot validation pipeline."""
    if specialist not in PILOT_CONFIG:
        raise ValueError(f"Unknown specialist: {specialist}. Choose from: {list(PILOT_CONFIG.keys())}")

    config = PILOT_CONFIG[specialist]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine base model
    if base_model is None:
        if config["recommended_size"] == "3B":
            base_model = config["base_model_3b"]
        else:
            base_model = config["base_model_8b"]

    print("\n" + "="*60)
    print(f"PILOT VALIDATION PIPELINE")
    print("="*60)
    print(f"Specialist: {config['name']}")
    print(f"Base Model: {base_model}")
    print(f"Output Dir: {output_path}")
    print(f"Mode: {'LOCAL TEST' if local_test else 'FULL'}")
    print("="*60)

    # Step 1: Load real data
    print("\n[Step 1/5] Loading real datasets...")
    load_fn = config["load_fn"]

    if local_test:
        data = load_fn(max_train=100, max_val=20, max_test=20)
    else:
        data = load_fn(
            max_train=config["train_examples"],
            max_val=config["val_examples"],
            max_test=config["test_examples"]
        )

    print(f"  Train: {len(data.train)} examples")
    print(f"  Val: {len(data.val)} examples")
    print(f"  Test: {len(data.test)} examples")

    # Save datasets
    data_dir = output_path / "data"
    save_dataset_split(data, data_dir, specialist)

    # Convert to dicts for processing
    train_data = [ex.to_dict() for ex in data.train]
    val_data = [ex.to_dict() for ex in data.val]
    test_data = [ex.to_dict() for ex in data.test]

    # Step 2: Baseline evaluation
    print("\n[Step 2/5] Running baseline evaluation...")
    model, tokenizer = load_model_and_tokenizer(base_model)
    baseline_result = run_baseline_evaluation(
        specialist, config, model, tokenizer, test_data, output_path
    )

    if baseline_only:
        print("\n[Baseline only mode - stopping here]")
        return baseline_result, None

    # Free memory
    del model
    del tokenizer
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except:
        pass

    # Step 3: Training
    if not skip_training:
        print("\n[Step 3/5] Training specialist...")
        lora_path = train_specialist(
            specialist, config, base_model,
            train_data, val_data, output_path,
            local_test=local_test
        )
    else:
        lora_path = str(output_path / f"{specialist}_lora")
        print(f"\n[Step 3/5] Skipping training, using: {lora_path}")

    # Step 4: Trained evaluation
    print("\n[Step 4/5] Evaluating trained model...")
    trained_result = run_trained_evaluation(
        specialist, config, lora_path, base_model, test_data, output_path
    )

    # Step 5: Generate comparison report
    print("\n[Step 5/5] Generating comparison report...")
    report = generate_comparison_report(baseline_result, trained_result)
    print(report)

    # Save report
    report_file = output_path / f"{specialist}_report.txt"
    with open(report_file, "w") as f:
        f.write(report)

    # Final summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Baseline passed: {baseline_result.passed}")
    print(f"Trained passed: {trained_result.passed}")

    improvement = {}
    for metric in baseline_result.metrics:
        delta = trained_result.metrics[metric] - baseline_result.metrics[metric]
        improvement[metric] = delta
        print(f"  {metric}: {'+' if delta >= 0 else ''}{delta:.3f}")

    # Verdict
    if trained_result.passed:
        print("\nVERDICT: SUCCESS - Model meets quality thresholds")
    elif sum(improvement.values()) > 0:
        print("\nVERDICT: PARTIAL - Training improved but below threshold")
    else:
        print("\nVERDICT: FAILED - Training did not improve model")

    return baseline_result, trained_result


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Pilot Validation Pipeline")
    parser.add_argument("--specialist", "-s", required=True,
                        choices=["sentiment", "sql", "invoice"],
                        help="Which specialist to validate")
    parser.add_argument("--base-model", "-m", type=str, default=None,
                        help="Base model to use (default: recommended for specialist)")
    parser.add_argument("--output-dir", "-o", type=str, default="validation/results",
                        help="Output directory for results")
    parser.add_argument("--local-test", "-t", action="store_true",
                        help="Run quick local test (small data, few steps)")
    parser.add_argument("--baseline-only", "-b", action="store_true",
                        help="Only run baseline evaluation, skip training")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, use existing LoRA")

    args = parser.parse_args()

    run_pilot_pipeline(
        specialist=args.specialist,
        base_model=args.base_model,
        output_dir=args.output_dir,
        local_test=args.local_test,
        baseline_only=args.baseline_only,
        skip_training=args.skip_training
    )


if __name__ == "__main__":
    main()
