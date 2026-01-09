#!/usr/bin/env python3
"""
Evaluation Metrics for Pilot Specialists

Implements task-specific metrics to measure model quality.
Each specialist has different success criteria.
"""

import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

try:
    import sqlparse
    SQLPARSE_AVAILABLE = True
except ImportError:
    SQLPARSE_AVAILABLE = False


@dataclass
class EvaluationResult:
    """Results from evaluating a model."""
    specialist: str
    num_examples: int
    metrics: Dict[str, float]
    passed: bool
    threshold: Dict[str, float]
    details: List[Dict] = None

    def __repr__(self):
        status = "PASSED" if self.passed else "FAILED"
        metrics_str = ", ".join([f"{k}={v:.3f}" for k, v in self.metrics.items()])
        return f"EvaluationResult({self.specialist}: {status}, {metrics_str})"

    def to_dict(self) -> Dict:
        return {
            "specialist": self.specialist,
            "num_examples": self.num_examples,
            "metrics": self.metrics,
            "passed": self.passed,
            "threshold": self.threshold
        }


# ============================================================
# SENTIMENT ANALYSIS METRICS
# ============================================================

class SentimentEvaluator:
    """
    Evaluates sentiment analysis quality.

    Metrics:
    - Accuracy: % of correct sentiment predictions
    - Macro F1: Average F1 across positive/negative classes

    Success Criteria:
    - Accuracy >= 0.85
    - Macro F1 >= 0.84
    """

    THRESHOLDS = {
        "accuracy": 0.85,
        "macro_f1": 0.84
    }

    def evaluate(self, predictions: List[str], references: List[Dict]) -> EvaluationResult:
        """
        Evaluate sentiment predictions.

        Args:
            predictions: Model outputs (text)
            references: Ground truth with 'label' field (positive/negative)
        """
        correct = 0
        tp = {"positive": 0, "negative": 0}
        fp = {"positive": 0, "negative": 0}
        fn = {"positive": 0, "negative": 0}

        details = []

        for pred, ref in zip(predictions, references):
            true_label = ref.get("label", "").lower()
            pred_label = self._extract_sentiment(pred)

            is_correct = pred_label == true_label
            if is_correct:
                correct += 1
                tp[true_label] += 1
            else:
                if pred_label in fp:
                    fp[pred_label] += 1
                if true_label in fn:
                    fn[true_label] += 1

            details.append({
                "true": true_label,
                "predicted": pred_label,
                "correct": is_correct
            })

        # Calculate metrics
        accuracy = correct / len(predictions) if predictions else 0

        # Macro F1
        f1_scores = []
        for label in ["positive", "negative"]:
            precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0
            recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)

        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

        metrics = {"accuracy": accuracy, "macro_f1": macro_f1}
        passed = all(metrics[k] >= self.THRESHOLDS[k] for k in self.THRESHOLDS)

        return EvaluationResult(
            specialist="sentiment_analysis",
            num_examples=len(predictions),
            metrics=metrics,
            passed=passed,
            threshold=self.THRESHOLDS,
            details=details
        )

    def _extract_sentiment(self, text: str) -> str:
        """Extract sentiment label from model output."""
        text_lower = text.lower()

        # Look for explicit sentiment markers
        if "**sentiment:** positive" in text_lower or "sentiment: positive" in text_lower:
            return "positive"
        if "**sentiment:** negative" in text_lower or "sentiment: negative" in text_lower:
            return "negative"

        # Fallback: look for keywords
        if "positive" in text_lower and "negative" not in text_lower:
            return "positive"
        if "negative" in text_lower and "positive" not in text_lower:
            return "negative"

        return "unknown"


# ============================================================
# SQL GENERATION METRICS
# ============================================================

class SQLEvaluator:
    """
    Evaluates SQL generation quality.

    Metrics:
    - Exact Match: % of queries that match exactly (normalized)
    - Execution Accuracy: % that would return correct results (approximated)
    - Valid SQL: % that parse as valid SQL

    Success Criteria:
    - Exact Match >= 0.60 (SQL is hard!)
    - Valid SQL >= 0.90
    """

    THRESHOLDS = {
        "exact_match": 0.60,
        "valid_sql": 0.90
    }

    def evaluate(self, predictions: List[str], references: List[Dict]) -> EvaluationResult:
        """
        Evaluate SQL predictions.

        Args:
            predictions: Model outputs (text containing SQL)
            references: Ground truth with 'sql' field
        """
        exact_matches = 0
        valid_sql_count = 0
        details = []

        for pred, ref in zip(predictions, references):
            true_sql = ref.get("sql", "")
            pred_sql = self._extract_sql(pred)

            # Normalize for comparison
            norm_true = self._normalize_sql(true_sql)
            norm_pred = self._normalize_sql(pred_sql)

            is_exact = norm_pred == norm_true
            is_valid = self._is_valid_sql(pred_sql)

            if is_exact:
                exact_matches += 1
            if is_valid:
                valid_sql_count += 1

            details.append({
                "true_sql": true_sql,
                "predicted_sql": pred_sql,
                "exact_match": is_exact,
                "valid_sql": is_valid
            })

        metrics = {
            "exact_match": exact_matches / len(predictions) if predictions else 0,
            "valid_sql": valid_sql_count / len(predictions) if predictions else 0
        }

        passed = all(metrics[k] >= self.THRESHOLDS[k] for k in self.THRESHOLDS)

        return EvaluationResult(
            specialist="sql_generation",
            num_examples=len(predictions),
            metrics=metrics,
            passed=passed,
            threshold=self.THRESHOLDS,
            details=details
        )

    def _extract_sql(self, text: str) -> str:
        """Extract SQL from model output (may be in code block)."""
        # Look for SQL in code block
        sql_match = re.search(r'```sql\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()

        # Look for any code block
        code_match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Look for SELECT statement
        select_match = re.search(r'(SELECT\s+.*?)(?:\n\n|$)', text, re.DOTALL | re.IGNORECASE)
        if select_match:
            return select_match.group(1).strip()

        return text.strip()

    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL for comparison."""
        if not sql:
            return ""

        # Lowercase
        sql = sql.lower()

        # Remove extra whitespace
        sql = " ".join(sql.split())

        # Remove trailing semicolon
        sql = sql.rstrip(";")

        # Normalize quotes
        sql = sql.replace('"', "'")

        return sql

    def _is_valid_sql(self, sql: str) -> bool:
        """Check if SQL is syntactically valid."""
        if not sql:
            return False

        # Basic check: starts with valid keyword
        sql_upper = sql.strip().upper()
        valid_starts = ["SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "CREATE"]
        if not any(sql_upper.startswith(kw) for kw in valid_starts):
            return False

        # Use sqlparse if available
        if SQLPARSE_AVAILABLE:
            try:
                parsed = sqlparse.parse(sql)
                return len(parsed) > 0 and parsed[0].tokens
            except:
                return False

        # Basic balance check for parentheses
        return sql.count("(") == sql.count(")")


# ============================================================
# INVOICE PROCESSING METRICS
# ============================================================

class InvoiceEvaluator:
    """
    Evaluates invoice/receipt processing quality.

    Metrics:
    - Field Extraction F1: How many fields correctly extracted
    - Total Accuracy: % of correct total amounts
    - Valid JSON: % of responses with valid JSON structure

    Success Criteria:
    - Field F1 >= 0.75
    - Total Accuracy >= 0.80
    """

    THRESHOLDS = {
        "field_f1": 0.75,
        "total_accuracy": 0.80
    }

    KEY_FIELDS = ["invoice_number", "vendor", "total", "subtotal", "date"]

    def evaluate(self, predictions: List[str], references: List[Dict]) -> EvaluationResult:
        """
        Evaluate invoice processing predictions.

        Args:
            predictions: Model outputs (text with JSON)
            references: Ground truth with field values
        """
        field_tp = 0
        field_fp = 0
        field_fn = 0
        total_correct = 0
        total_count = 0

        details = []

        for pred, ref in zip(predictions, references):
            pred_fields = self._extract_fields(pred)
            true_fields = ref.get("fields", {})

            # If no structured fields, use metadata
            if not true_fields and "metadata" in ref:
                true_fields = ref["metadata"]

            # Field extraction metrics
            for field in self.KEY_FIELDS:
                pred_val = pred_fields.get(field)
                true_val = true_fields.get(field)

                if pred_val and true_val:
                    if self._values_match(pred_val, true_val):
                        field_tp += 1
                    else:
                        field_fp += 1
                        field_fn += 1
                elif pred_val:
                    field_fp += 1
                elif true_val:
                    field_fn += 1

            # Total accuracy
            if "total" in true_fields:
                total_count += 1
                pred_total = pred_fields.get("total")
                true_total = true_fields.get("total")
                if pred_total and self._numbers_match(pred_total, true_total):
                    total_correct += 1

            details.append({
                "predicted_fields": pred_fields,
                "true_fields": true_fields
            })

        # Calculate metrics
        precision = field_tp / (field_tp + field_fp) if (field_tp + field_fp) > 0 else 0
        recall = field_tp / (field_tp + field_fn) if (field_tp + field_fn) > 0 else 0
        field_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        total_accuracy = total_correct / total_count if total_count > 0 else 1.0

        metrics = {
            "field_f1": field_f1,
            "total_accuracy": total_accuracy
        }

        passed = all(metrics[k] >= self.THRESHOLDS[k] for k in self.THRESHOLDS)

        return EvaluationResult(
            specialist="invoice_processing",
            num_examples=len(predictions),
            metrics=metrics,
            passed=passed,
            threshold=self.THRESHOLDS,
            details=details
        )

    def _extract_fields(self, text: str) -> Dict:
        """Extract structured fields from model output."""
        fields = {}

        # Try to find JSON in response
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            try:
                fields = json.loads(json_match.group(1))
                return fields
            except json.JSONDecodeError:
                pass

        # Try to find inline JSON
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                fields = json.loads(json_match.group(0))
                return fields
            except json.JSONDecodeError:
                pass

        # Fallback: extract from text patterns
        patterns = {
            "invoice_number": r'(?:invoice[_\s]?(?:number|#|num)?)[:\s]*([A-Z0-9-]+)',
            "vendor": r'(?:vendor|from|company)[:\s]*([A-Za-z0-9\s]+?)(?:\n|$)',
            "total": r'(?:total)[:\s]*\$?([\d,]+\.?\d*)',
            "subtotal": r'(?:subtotal)[:\s]*\$?([\d,]+\.?\d*)',
        }

        text_lower = text.lower()
        for field, pattern in patterns.items():
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                fields[field] = match.group(1).strip()

        return fields

    def _values_match(self, pred: str, true: str) -> bool:
        """Check if two values match (flexible comparison)."""
        if not pred or not true:
            return False

        # Normalize strings
        pred_norm = str(pred).lower().strip()
        true_norm = str(true).lower().strip()

        # Exact match
        if pred_norm == true_norm:
            return True

        # Numeric match
        try:
            pred_num = float(re.sub(r'[^\d.]', '', str(pred)))
            true_num = float(re.sub(r'[^\d.]', '', str(true)))
            return abs(pred_num - true_num) < 0.01
        except:
            pass

        # Substring match (for invoice numbers, etc.)
        return pred_norm in true_norm or true_norm in pred_norm

    def _numbers_match(self, pred, true, tolerance: float = 0.01) -> bool:
        """Check if two numbers match within tolerance."""
        try:
            pred_num = float(re.sub(r'[^\d.]', '', str(pred)))
            true_num = float(re.sub(r'[^\d.]', '', str(true)))
            return abs(pred_num - true_num) / max(true_num, 1) < tolerance
        except:
            return False


# ============================================================
# BASELINE EVALUATION
# ============================================================

def evaluate_baseline(
    model_name: str,
    specialist: str,
    test_examples: List[Dict],
    generate_fn
) -> EvaluationResult:
    """
    Evaluate a model on test examples.

    Args:
        model_name: Name of model being evaluated
        specialist: Which specialist (sentiment, sql, invoice)
        test_examples: List of test examples with prompts and references
        generate_fn: Function to generate model response given prompt

    Returns:
        EvaluationResult with metrics
    """
    evaluators = {
        "sentiment_analysis": SentimentEvaluator(),
        "sentiment": SentimentEvaluator(),
        "sql_generation": SQLEvaluator(),
        "sql": SQLEvaluator(),
        "invoice_processing": InvoiceEvaluator(),
        "invoice": InvoiceEvaluator(),
    }

    evaluator = evaluators.get(specialist)
    if not evaluator:
        raise ValueError(f"Unknown specialist: {specialist}")

    # Generate predictions
    predictions = []
    references = []

    for example in test_examples:
        prompt = example.get("prompt", example.get("text", ""))
        response = generate_fn(prompt)
        predictions.append(response)

        # Build reference
        ref = {
            "label": example.get("metadata", {}).get("label"),
            "sql": example.get("metadata", {}).get("sql"),
            "fields": example.get("metadata", {}),
            "metadata": example.get("metadata", {})
        }
        references.append(ref)

    return evaluator.evaluate(predictions, references)


# ============================================================
# COMPARISON REPORT
# ============================================================

def generate_comparison_report(
    baseline_result: EvaluationResult,
    trained_result: EvaluationResult
) -> str:
    """Generate a comparison report between baseline and trained model."""
    report = []
    report.append("=" * 60)
    report.append(f"EVALUATION REPORT: {baseline_result.specialist}")
    report.append("=" * 60)
    report.append("")

    report.append("METRICS COMPARISON:")
    report.append("-" * 40)
    report.append(f"{'Metric':<20} {'Baseline':>10} {'Trained':>10} {'Delta':>10} {'Target':>10}")
    report.append("-" * 40)

    for metric in baseline_result.metrics:
        base_val = baseline_result.metrics[metric]
        train_val = trained_result.metrics[metric]
        delta = train_val - base_val
        threshold = baseline_result.threshold.get(metric, "N/A")

        delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
        status = "OK" if train_val >= threshold else "FAIL"

        report.append(f"{metric:<20} {base_val:>10.3f} {train_val:>10.3f} {delta_str:>10} {threshold:>10}")

    report.append("-" * 40)
    report.append("")

    # Verdict
    if trained_result.passed and not baseline_result.passed:
        verdict = "SUCCESS: Training improved model to meet thresholds"
    elif trained_result.passed and baseline_result.passed:
        verdict = "SUCCESS: Model meets thresholds (baseline was already good)"
    elif not trained_result.passed and baseline_result.passed:
        verdict = "FAILURE: Training degraded model performance"
    else:
        improvement = sum(trained_result.metrics.values()) > sum(baseline_result.metrics.values())
        if improvement:
            verdict = "PARTIAL: Training improved but still below threshold"
        else:
            verdict = "FAILURE: Training did not improve model"

    report.append(f"VERDICT: {verdict}")
    report.append("")

    return "\n".join(report)


if __name__ == "__main__":
    # Quick test
    evaluator = SentimentEvaluator()
    predictions = [
        "**Sentiment:** POSITIVE\n**Confidence:** 0.95",
        "**Sentiment:** NEGATIVE\n**Confidence:** 0.88",
        "The sentiment is positive overall.",
    ]
    references = [
        {"label": "positive"},
        {"label": "negative"},
        {"label": "positive"},
    ]
    result = evaluator.evaluate(predictions, references)
    print(result)
