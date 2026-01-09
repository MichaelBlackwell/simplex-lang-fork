#!/usr/bin/env python3
"""
Real Dataset Loaders for Pilot Validation

Loads actual open-source datasets instead of synthetic data.
These are the datasets referenced in our specialist definitions.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    from datasets import load_dataset, Dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: 'datasets' not installed. Run: pip install datasets")


@dataclass
class TrainingExample:
    """A single training example."""
    prompt: str
    response: str
    metadata: Dict = None

    def to_dict(self) -> Dict:
        return {
            "text": f"{self.prompt}\n\nAssistant: {self.response}",
            "prompt": self.prompt,
            "response": self.response,
            "metadata": self.metadata or {}
        }


@dataclass
class DatasetSplit:
    """Train/val/test split."""
    train: List[TrainingExample]
    val: List[TrainingExample]
    test: List[TrainingExample]

    def __repr__(self):
        return f"DatasetSplit(train={len(self.train)}, val={len(self.val)}, test={len(self.test)})"


# ============================================================
# SENTIMENT ANALYSIS - SST-2 + IMDB
# ============================================================

def load_sentiment_data(
    max_train: int = 10000,
    max_val: int = 1000,
    max_test: int = 500
) -> DatasetSplit:
    """
    Load sentiment analysis data from SST-2 and IMDB.

    SST-2: Stanford Sentiment Treebank (movie reviews, binary)
    IMDB: Large Movie Review Dataset (longer reviews, binary)

    License: Both are open for research and commercial use.
    """
    if not HF_AVAILABLE:
        raise ImportError("datasets library required: pip install datasets")

    examples = {"train": [], "val": [], "test": []}

    # Load SST-2 (short sentences)
    print("Loading SST-2...")
    sst2 = load_dataset("glue", "sst2")

    label_map = {0: "negative", 1: "positive"}

    for split, hf_split in [("train", "train"), ("val", "validation")]:
        for item in sst2[hf_split]:
            sentiment = label_map[item["label"]]
            example = TrainingExample(
                prompt=f'Analyze the sentiment of this text:\n\n"{item["sentence"]}"\n\nReturn: sentiment (positive/negative) with confidence.',
                response=f'**Sentiment:** {sentiment.upper()}\n**Confidence:** 0.92\n\n**Analysis:** {"Positive language and tone detected." if sentiment == "positive" else "Negative language and tone detected."}',
                metadata={"source": "sst2", "label": sentiment}
            )
            examples[split].append(example)

    # Load IMDB (longer reviews)
    print("Loading IMDB...")
    imdb = load_dataset("imdb")

    for split, hf_split in [("train", "train"), ("test", "test")]:
        target = "test" if split == "test" else "train"
        for item in imdb[hf_split]:
            # Truncate long reviews
            text = item["text"][:500] + "..." if len(item["text"]) > 500 else item["text"]
            sentiment = "positive" if item["label"] == 1 else "negative"

            example = TrainingExample(
                prompt=f'Analyze the sentiment of this movie review:\n\n"{text}"\n\nReturn: sentiment, confidence, and key phrases.',
                response=f'**Sentiment:** {sentiment.upper()}\n**Confidence:** 0.89\n\n**Key Phrases:**\n- {"Expresses satisfaction" if sentiment == "positive" else "Expresses disappointment"}\n\n**Aspect Analysis:**\n- Overall tone: {sentiment}',
                metadata={"source": "imdb", "label": sentiment}
            )
            examples[target].append(example)

    # Shuffle and limit
    random.shuffle(examples["train"])
    random.shuffle(examples["val"])
    random.shuffle(examples["test"])

    return DatasetSplit(
        train=examples["train"][:max_train],
        val=examples["val"][:max_val],
        test=examples["test"][:max_test]
    )


# ============================================================
# SQL GENERATION - Spider + WikiSQL
# ============================================================

def load_sql_data(
    max_train: int = 8000,
    max_val: int = 1000,
    max_test: int = 500
) -> DatasetSplit:
    """
    Load SQL generation data from Spider and WikiSQL.

    Spider: Complex, cross-domain text-to-SQL (academic license, free for research)
    WikiSQL: Simpler single-table queries (CC BY-SA)

    Note: Spider has academic license - verify for commercial use.
    WikiSQL is CC BY-SA - attribution required.
    """
    if not HF_AVAILABLE:
        raise ImportError("datasets library required: pip install datasets")

    examples = {"train": [], "val": [], "test": []}

    # Load WikiSQL (simpler, larger)
    print("Loading WikiSQL...")
    try:
        wikisql = load_dataset("wikisql")

        for split, hf_split in [("train", "train"), ("val", "validation"), ("test", "test")]:
            for item in wikisql[hf_split]:
                # Build schema description
                table = item["table"]
                columns = table["header"]
                schema = f"Table columns: {', '.join(columns)}"

                # Get the SQL
                sql = item["sql"]
                sql_str = build_wikisql_query(sql, columns, table["name"] if "name" in table else "table")

                example = TrainingExample(
                    prompt=f'Convert this question to SQL:\n\nQuestion: "{item["question"]}"\n\nSchema:\n{schema}\n\nGenerate the SQL query.',
                    response=f'```sql\n{sql_str}\n```\n\n**Explanation:** This query retrieves the requested information by filtering on the appropriate columns.',
                    metadata={"source": "wikisql", "sql": sql_str}
                )
                examples[split].append(example)
    except Exception as e:
        print(f"Warning: Could not load WikiSQL: {e}")

    # Load Spider (more complex)
    print("Loading Spider...")
    try:
        spider = load_dataset("spider")

        for split, hf_split in [("train", "train"), ("val", "validation")]:
            for item in spider[hf_split]:
                # Schema from db_id
                db_id = item.get("db_id", "database")

                example = TrainingExample(
                    prompt=f'Convert this question to SQL:\n\nQuestion: "{item["question"]}"\n\nDatabase: {db_id}\n\nGenerate the SQL query.',
                    response=f'```sql\n{item["query"]}\n```\n\n**Complexity:** {"Simple" if "JOIN" not in item["query"].upper() else "Complex (uses JOIN)"}',
                    metadata={"source": "spider", "sql": item["query"], "db_id": db_id}
                )
                examples[split].append(example)
    except Exception as e:
        print(f"Warning: Could not load Spider: {e}")

    # Shuffle and limit
    for split in examples:
        random.shuffle(examples[split])

    # Use validation as test if no test examples
    if not examples["test"]:
        examples["test"] = examples["val"][max_val:]
        examples["val"] = examples["val"][:max_val]

    return DatasetSplit(
        train=examples["train"][:max_train],
        val=examples["val"][:max_val],
        test=examples["test"][:max_test]
    )


def build_wikisql_query(sql_dict: Dict, columns: List[str], table_name: str) -> str:
    """Convert WikiSQL SQL dict to SQL string."""
    # WikiSQL format: {"sel": col_idx, "conds": [[col_idx, op, value], ...], "agg": agg_idx}
    agg_ops = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
    cond_ops = ["=", ">", "<", ">=", "<=", "!="]

    sel_col = columns[sql_dict["sel"]] if sql_dict["sel"] < len(columns) else "col"
    agg = agg_ops[sql_dict.get("agg", 0)]

    if agg:
        select_clause = f"{agg}({sel_col})"
    else:
        select_clause = sel_col

    query = f"SELECT {select_clause} FROM {table_name}"

    # Add WHERE conditions
    conds = sql_dict.get("conds", [])
    if conds:
        where_parts = []
        for cond in conds:
            if len(cond) >= 3:
                col_idx, op_idx, value = cond[0], cond[1], cond[2]
                col = columns[col_idx] if col_idx < len(columns) else "col"
                op = cond_ops[op_idx] if op_idx < len(cond_ops) else "="
                where_parts.append(f"{col} {op} '{value}'")
        if where_parts:
            query += " WHERE " + " AND ".join(where_parts)

    return query


# ============================================================
# INVOICE PROCESSING - SROIE + CORD
# ============================================================

def load_invoice_data(
    max_train: int = 3000,
    max_val: int = 500,
    max_test: int = 200
) -> DatasetSplit:
    """
    Load invoice/receipt processing data from SROIE and CORD.

    SROIE: Scanned Receipts OCR and Information Extraction (ICDAR 2019)
    CORD: Consolidated Receipt Dataset for Post-OCR Parsing

    Note: These datasets contain OCR'd receipt images with annotations.
    We extract the text and structured fields for training.

    License: Research use - verify for commercial applications.
    """
    if not HF_AVAILABLE:
        raise ImportError("datasets library required: pip install datasets")

    examples = {"train": [], "val": [], "test": []}

    # Load CORD dataset
    print("Loading CORD dataset...")
    try:
        cord = load_dataset("naver-clova-ix/cord-v2")

        for split, hf_split in [("train", "train"), ("val", "validation"), ("test", "test")]:
            for item in cord[hf_split]:
                # CORD has ground_truth as JSON string
                try:
                    gt = json.loads(item["ground_truth"])

                    # Extract text from OCR
                    text_parts = []
                    if "valid_line" in gt:
                        for line in gt["valid_line"]:
                            if "words" in line:
                                line_text = " ".join([w.get("text", "") for w in line["words"]])
                                text_parts.append(line_text)

                    ocr_text = "\n".join(text_parts) if text_parts else "Receipt text unavailable"

                    # Extract structured fields
                    fields = {}
                    if "gt_parse" in gt:
                        fields = gt["gt_parse"]

                    # Build response
                    response_parts = ["**Extracted Information:**\n"]
                    if "menu" in fields:
                        response_parts.append("**Line Items:**")
                        for item_data in fields["menu"][:5]:  # Limit items
                            name = item_data.get("nm", "Item")
                            price = item_data.get("price", "N/A")
                            response_parts.append(f"- {name}: {price}")

                    if "total" in fields:
                        total = fields["total"]
                        if "total_price" in total:
                            response_parts.append(f"\n**Total:** {total['total_price']}")

                    if "sub_total" in fields:
                        response_parts.append(f"**Subtotal:** {fields['sub_total'].get('subtotal_price', 'N/A')}")

                    example = TrainingExample(
                        prompt=f'Extract structured data from this receipt:\n\n{ocr_text[:800]}\n\nReturn: line items, totals, and any other key information.',
                        response="\n".join(response_parts) + "\n\n[confidence: 0.88]",
                        metadata={"source": "cord", "fields": list(fields.keys())}
                    )
                    examples[split].append(example)

                except (json.JSONDecodeError, KeyError) as e:
                    continue

    except Exception as e:
        print(f"Warning: Could not load CORD: {e}")

    # Generate additional synthetic examples if needed (to supplement real data)
    if len(examples["train"]) < max_train // 2:
        print("Supplementing with synthetic invoice examples...")
        examples["train"].extend(generate_synthetic_invoices(max_train - len(examples["train"])))

    # Shuffle and limit
    for split in examples:
        random.shuffle(examples[split])

    return DatasetSplit(
        train=examples["train"][:max_train],
        val=examples["val"][:max_val],
        test=examples["test"][:max_test]
    )


def generate_synthetic_invoices(count: int) -> List[TrainingExample]:
    """Generate synthetic invoice examples to supplement real data."""
    examples = []

    companies = ["Acme Corp", "TechStart Inc", "Global Services", "Metro Supply", "Prime Wholesale"]
    items = ["Widget A", "Service Fee", "Consulting", "Product X", "Subscription", "License", "Support"]

    for i in range(count):
        company = random.choice(companies)
        inv_num = f"INV-{random.randint(10000, 99999)}"
        date = f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}"

        line_items = []
        subtotal = 0
        for _ in range(random.randint(2, 5)):
            item = random.choice(items)
            qty = random.randint(1, 10)
            price = round(random.uniform(50, 500), 2)
            total = round(qty * price, 2)
            subtotal += total
            line_items.append(f"- {item} x{qty} @ ${price:.2f} = ${total:.2f}")

        tax = round(subtotal * 0.1, 2)
        total = round(subtotal + tax, 2)

        prompt = f"""Extract structured data from this invoice:

INVOICE #{inv_num}
From: {company}
Date: {date}

Items:
{chr(10).join(line_items)}

Subtotal: ${subtotal:.2f}
Tax (10%): ${tax:.2f}
TOTAL: ${total:.2f}

Return: invoice number, vendor, line items, and totals as JSON."""

        response = f'''```json
{{
  "invoice_number": "{inv_num}",
  "vendor": "{company}",
  "date": "{date}",
  "line_items": {len(line_items)},
  "subtotal": {subtotal:.2f},
  "tax": {tax:.2f},
  "total": {total:.2f},
  "currency": "USD"
}}
```

[confidence: 0.94]'''

        examples.append(TrainingExample(
            prompt=prompt,
            response=response,
            metadata={"source": "synthetic", "total": total}
        ))

    return examples


# ============================================================
# EXPORT FUNCTIONS
# ============================================================

def save_dataset_split(split: DatasetSplit, output_dir: Path, name: str):
    """Save dataset split to JSONL files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, examples in [("train", split.train), ("val", split.val), ("test", split.test)]:
        output_file = output_dir / f"{name}_{split_name}.jsonl"
        with open(output_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex.to_dict()) + "\n")
        print(f"Saved {len(examples)} examples to {output_file}")


def load_all_pilot_datasets(output_dir: str = "validation/datasets") -> Dict[str, DatasetSplit]:
    """Load all three pilot datasets."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    datasets = {}

    print("\n" + "="*60)
    print("Loading Pilot Datasets")
    print("="*60)

    # Sentiment
    print("\n[1/3] Sentiment Analysis (SST-2 + IMDB)")
    datasets["sentiment"] = load_sentiment_data()
    print(f"  {datasets['sentiment']}")
    save_dataset_split(datasets["sentiment"], output_path, "sentiment")

    # SQL
    print("\n[2/3] SQL Generation (WikiSQL + Spider)")
    datasets["sql"] = load_sql_data()
    print(f"  {datasets['sql']}")
    save_dataset_split(datasets["sql"], output_path, "sql")

    # Invoice
    print("\n[3/3] Invoice Processing (CORD + Synthetic)")
    datasets["invoice"] = load_invoice_data()
    print(f"  {datasets['invoice']}")
    save_dataset_split(datasets["invoice"], output_path, "invoice")

    print("\n" + "="*60)
    print("Dataset Loading Complete")
    print("="*60)

    return datasets


if __name__ == "__main__":
    load_all_pilot_datasets()
