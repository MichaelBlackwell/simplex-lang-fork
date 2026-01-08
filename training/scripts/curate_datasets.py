#!/usr/bin/env python3
"""
Phase 4: Dataset Curation

Downloads and prepares real-world datasets for Simplex cognitive training.
Formats data with confidence annotations for calibration training.
"""

import os
import json
import argparse
from pathlib import Path
import random
from tqdm import tqdm


DATASETS = {
    "squad": ("squad_v2", "Reading comprehension with unanswerable questions"),
    "gsm8k": ("gsm8k", "main", "Grade school math reasoning"),
    "sst2": ("glue", "sst2", "Sentiment analysis"),
    "fever": ("fever", "v1.0", "Fact verification"),
}


def prepare_squad(output_dir: Path, limit: int = 5000):
    """Prepare SQuAD 2.0 for confidence training."""
    from datasets import load_dataset

    print("Loading SQuAD 2.0...")
    ds = load_dataset("squad_v2", split="train")

    examples = []
    for item in tqdm(ds.select(range(min(limit, len(ds)))), desc="SQuAD"):
        q = item["question"]
        ctx = item["context"][:400]
        answers = item["answers"]["text"]

        if answers:
            ans = answers[0]
            conf = random.uniform(0.85, 0.95)
        else:
            ans = "The context does not contain the answer"
            conf = random.uniform(0.10, 0.25)

        examples.append({
            "text": f"Context: {ctx}...\n\nQuestion: {q}\n\nAssistant: {ans} [confidence: {conf:.2f}]",
            "source": "squad",
        })

    out = output_dir / "squad_prepared.jsonl"
    with open(out, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(examples)} to {out}")


def prepare_gsm8k(output_dir: Path, limit: int = 5000):
    """Prepare GSM8K for reasoning with confidence."""
    from datasets import load_dataset

    print("Loading GSM8K...")
    ds = load_dataset("gsm8k", "main", split="train")

    examples = []
    for item in tqdm(ds.select(range(min(limit, len(ds)))), desc="GSM8K"):
        q = item["question"]
        ans = item["answer"]
        conf = random.uniform(0.80, 0.95)

        examples.append({
            "text": f"Math problem: {q}\n\nAssistant: {ans}\n\n[confidence: {conf:.2f}]",
            "source": "gsm8k",
        })

    out = output_dir / "gsm8k_prepared.jsonl"
    with open(out, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(examples)} to {out}")


def prepare_sst2(output_dir: Path, limit: int = 5000):
    """Prepare SST-2 sentiment for confidence training."""
    from datasets import load_dataset

    print("Loading SST-2...")
    ds = load_dataset("glue", "sst2", split="train")

    examples = []
    for item in tqdm(ds.select(range(min(limit, len(ds)))), desc="SST-2"):
        text = item["sentence"]
        label = "positive" if item["label"] == 1 else "negative"
        conf = random.uniform(0.75, 0.95)

        examples.append({
            "text": f"Analyze sentiment: {text}\n\nAssistant: The sentiment is {label}. [confidence: {conf:.2f}]",
            "source": "sst2",
        })

    out = output_dir / "sst2_prepared.jsonl"
    with open(out, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(examples)} to {out}")


def main():
    parser = argparse.ArgumentParser(description="Curate datasets for Simplex training")
    parser.add_argument("--output-dir", type=str, default="data/curated")
    parser.add_argument("--dataset", type=str, choices=["squad", "gsm8k", "sst2", "all"])
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--list", action="store_true", help="List available datasets")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable datasets:")
        for name, info in DATASETS.items():
            print(f"  {name}: {info[-1]}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "all" or args.dataset == "squad":
        prepare_squad(output_dir, args.limit)
    if args.dataset == "all" or args.dataset == "gsm8k":
        prepare_gsm8k(output_dir, args.limit)
    if args.dataset == "all" or args.dataset == "sst2":
        prepare_sst2(output_dir, args.limit)

    print("\nDataset curation complete!")


if __name__ == "__main__":
    main()
