"""Pytest configuration and fixtures for training tests."""

import pytest
import sys


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "requires_torch: marks tests that require PyTorch")
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")


def pytest_collection_modifyitems(config, items):
    """Skip tests based on environment."""
    # Check if torch is available
    try:
        import torch

        torch_available = True
        gpu_available = torch.cuda.is_available()
    except ImportError:
        torch_available = False
        gpu_available = False

    skip_torch = pytest.mark.skip(reason="PyTorch not available")
    skip_gpu = pytest.mark.skip(reason="GPU not available")

    for item in items:
        if "requires_torch" in item.keywords and not torch_available:
            item.add_marker(skip_torch)
        if "requires_gpu" in item.keywords and not gpu_available:
            item.add_marker(skip_gpu)


@pytest.fixture
def sample_sentiment_data():
    """Sample sentiment data for testing."""
    return [
        {
            "prompt": "Analyze the sentiment: This movie was amazing!",
            "response": "POSITIVE",
            "metadata": {"source": "test", "label": "positive"},
        },
        {
            "prompt": "Analyze the sentiment: Terrible experience.",
            "response": "NEGATIVE",
            "metadata": {"source": "test", "label": "negative"},
        },
    ]


@pytest.fixture
def sample_sql_data():
    """Sample SQL data for testing."""
    return [
        {
            "prompt": "Convert to SQL: Show all users",
            "response": "SELECT * FROM users",
            "metadata": {"source": "test", "schema": "users(id, name, email)"},
        },
        {
            "prompt": "Convert to SQL: Count orders by customer",
            "response": "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id",
            "metadata": {"source": "test", "schema": "orders(id, customer_id, amount)"},
        },
    ]


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir
