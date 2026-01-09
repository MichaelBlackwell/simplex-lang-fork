"""Tests for evaluation metrics."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.evaluation.metrics import (
    SentimentEvaluator,
    SQLEvaluator,
    InvoiceEvaluator,
)


class TestSentimentEvaluator:
    """Tests for sentiment evaluation."""

    def test_evaluator_initialization(self):
        """Test evaluator can be initialized."""
        evaluator = SentimentEvaluator()
        assert evaluator is not None
        assert hasattr(evaluator, "THRESHOLDS")

    def test_extract_sentiment_positive(self):
        """Test extracting positive sentiment."""
        evaluator = SentimentEvaluator()
        test_response = "**Sentiment:** POSITIVE\n**Confidence:** 0.95"
        result = evaluator.extract_sentiment(test_response)
        assert result == "positive"

    def test_extract_sentiment_negative(self):
        """Test extracting negative sentiment."""
        evaluator = SentimentEvaluator()
        test_response = "The sentiment is NEGATIVE with high confidence."
        result = evaluator.extract_sentiment(test_response)
        assert result == "negative"

    def test_extract_sentiment_handles_variations(self):
        """Test sentiment extraction handles various formats."""
        evaluator = SentimentEvaluator()

        test_cases = [
            ("Sentiment: positive", "positive"),
            ("SENTIMENT: NEGATIVE", "negative"),
            ("This is a positive review", "positive"),
            ("clearly negative sentiment", "negative"),
        ]

        for response, expected in test_cases:
            result = evaluator.extract_sentiment(response)
            assert result == expected, f"Failed for: {response}"

    def test_evaluate_single_correct(self):
        """Test evaluating a single correct prediction."""
        evaluator = SentimentEvaluator()
        prediction = "Sentiment: POSITIVE"
        ground_truth = "positive"

        result = evaluator.evaluate_single(prediction, ground_truth)
        assert result["correct"] == True
        assert result["predicted"] == "positive"

    def test_evaluate_single_incorrect(self):
        """Test evaluating a single incorrect prediction."""
        evaluator = SentimentEvaluator()
        prediction = "Sentiment: NEGATIVE"
        ground_truth = "positive"

        result = evaluator.evaluate_single(prediction, ground_truth)
        assert result["correct"] == False

    def test_evaluate_batch_computes_accuracy(self):
        """Test batch evaluation computes accuracy."""
        evaluator = SentimentEvaluator()

        predictions = [
            "POSITIVE",
            "NEGATIVE",
            "POSITIVE",
            "NEGATIVE",
        ]
        ground_truths = [
            "positive",
            "negative",
            "negative",  # wrong
            "negative",
        ]

        result = evaluator.evaluate_batch(predictions, ground_truths)
        assert "accuracy" in result
        assert result["accuracy"] == 0.75  # 3/4 correct


class TestSQLEvaluator:
    """Tests for SQL evaluation."""

    def test_evaluator_initialization(self):
        """Test evaluator can be initialized."""
        evaluator = SQLEvaluator()
        assert evaluator is not None
        assert hasattr(evaluator, "THRESHOLDS")

    def test_is_valid_sql_simple(self):
        """Test valid SQL detection for simple queries."""
        evaluator = SQLEvaluator()

        valid_queries = [
            "SELECT * FROM users",
            "SELECT name, email FROM users WHERE id = 1",
            "SELECT COUNT(*) FROM orders",
        ]

        for query in valid_queries:
            assert evaluator.is_valid_sql(query), f"Should be valid: {query}"

    def test_is_valid_sql_invalid(self):
        """Test invalid SQL detection."""
        evaluator = SQLEvaluator()

        invalid_queries = [
            "This is not SQL",
            "SELEC * FORM users",  # typos
            "",
        ]

        for query in invalid_queries:
            assert not evaluator.is_valid_sql(query), f"Should be invalid: {query}"

    def test_normalize_sql(self):
        """Test SQL normalization."""
        evaluator = SQLEvaluator()

        sql1 = "SELECT  name,  email FROM  users"
        sql2 = "select name, email from users"

        assert evaluator.normalize_sql(sql1) == evaluator.normalize_sql(sql2)


class TestInvoiceEvaluator:
    """Tests for invoice evaluation."""

    def test_evaluator_initialization(self):
        """Test evaluator can be initialized."""
        evaluator = InvoiceEvaluator()
        assert evaluator is not None
        assert hasattr(evaluator, "THRESHOLDS")

    def test_extract_fields_from_json(self):
        """Test extracting fields from JSON response."""
        evaluator = InvoiceEvaluator()

        response = '''```json
{
    "vendor": "Acme Corp",
    "total": "1234.56",
    "date": "2024-01-15"
}
```'''
        fields = evaluator.extract_fields(response)
        assert fields.get("vendor") == "Acme Corp"
        assert fields.get("total") == "1234.56"

    def test_field_match_exact(self):
        """Test exact field matching."""
        evaluator = InvoiceEvaluator()

        assert evaluator.field_matches("Acme Corp", "Acme Corp")
        assert not evaluator.field_matches("Acme Corp", "Acme Corporation")

    def test_field_match_fuzzy(self):
        """Test fuzzy field matching for totals."""
        evaluator = InvoiceEvaluator()

        # Numbers should match with different formatting
        assert evaluator.field_matches("$1,234.56", "1234.56", field_type="total")
        assert evaluator.field_matches("1234.56", "$1,234.56", field_type="total")


class TestThresholds:
    """Tests for evaluation thresholds."""

    def test_sentiment_thresholds_exist(self):
        """Test sentiment has required thresholds."""
        thresholds = SentimentEvaluator.THRESHOLDS
        assert "accuracy" in thresholds
        assert thresholds["accuracy"] >= 0.8  # Should be high

    def test_sql_thresholds_exist(self):
        """Test SQL has required thresholds."""
        thresholds = SQLEvaluator.THRESHOLDS
        assert "exact_match" in thresholds
        assert "valid_sql" in thresholds

    def test_invoice_thresholds_exist(self):
        """Test invoice has required thresholds."""
        thresholds = InvoiceEvaluator.THRESHOLDS
        assert "field_f1" in thresholds
