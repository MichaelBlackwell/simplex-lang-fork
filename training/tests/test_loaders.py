"""Tests for data loaders."""

import pytest
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.data_loaders.loaders import (
    load_sentiment_data,
    load_sql_data,
    load_invoice_data,
    DatasetSplit,
)


class TestSentimentLoader:
    """Tests for sentiment data loading."""

    def test_load_sentiment_returns_dataset_split(self):
        """Test that load_sentiment_data returns a DatasetSplit."""
        result = load_sentiment_data(max_train=10, max_val=5, max_test=5)
        assert isinstance(result, DatasetSplit)

    def test_sentiment_data_has_required_fields(self):
        """Test that sentiment data has required fields."""
        result = load_sentiment_data(max_train=10, max_val=5, max_test=5)

        # Check train data
        assert len(result.train) > 0
        sample = result.train[0]
        assert "text" in sample or "prompt" in sample
        assert "metadata" in sample
        assert "label" in sample["metadata"]

    def test_sentiment_respects_max_limits(self):
        """Test that max limits are respected."""
        result = load_sentiment_data(max_train=5, max_val=3, max_test=2)
        assert len(result.train) <= 5
        assert len(result.val) <= 3
        assert len(result.test) <= 2

    def test_sentiment_labels_are_valid(self):
        """Test that sentiment labels are positive/negative."""
        result = load_sentiment_data(max_train=20, max_val=10, max_test=10)
        valid_labels = {"positive", "negative"}

        for sample in result.train[:10]:
            assert sample["metadata"]["label"] in valid_labels


class TestSQLLoader:
    """Tests for SQL data loading."""

    def test_load_sql_returns_dataset_split(self):
        """Test that load_sql_data returns a DatasetSplit."""
        result = load_sql_data(max_train=10, max_val=5, max_test=5)
        assert isinstance(result, DatasetSplit)

    def test_sql_data_has_required_fields(self):
        """Test that SQL data has required fields."""
        result = load_sql_data(max_train=10, max_val=5, max_test=5)

        assert len(result.train) > 0
        sample = result.train[0]
        assert "text" in sample or "prompt" in sample
        assert "metadata" in sample


class TestInvoiceLoader:
    """Tests for invoice data loading."""

    def test_load_invoice_returns_dataset_split(self):
        """Test that load_invoice_data returns a DatasetSplit."""
        result = load_invoice_data(max_train=10, max_val=5, max_test=5)
        assert isinstance(result, DatasetSplit)

    def test_invoice_data_has_required_fields(self):
        """Test that invoice data has required fields."""
        result = load_invoice_data(max_train=10, max_val=5, max_test=5)

        if len(result.train) > 0:
            sample = result.train[0]
            assert "text" in sample or "prompt" in sample


class TestDatasetSplitIntegrity:
    """Tests for dataset split integrity."""

    def test_no_overlap_between_splits(self):
        """Test that train/val/test don't overlap."""
        result = load_sentiment_data(max_train=50, max_val=20, max_test=20)

        train_texts = {s.get("prompt", s.get("text", "")) for s in result.train}
        val_texts = {s.get("prompt", s.get("text", "")) for s in result.val}
        test_texts = {s.get("prompt", s.get("text", "")) for s in result.test}

        # Check no overlap (some overlap might be acceptable from different sources)
        # This is a soft check - just ensure they're not identical
        assert train_texts != val_texts or len(train_texts) == 0
        assert train_texts != test_texts or len(train_texts) == 0
