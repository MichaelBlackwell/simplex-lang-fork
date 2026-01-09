#!/bin/bash
# Run training tests with different configurations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=== Simplex Training Tests ==="
echo ""

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3.11 -m venv .venv
fi

source .venv/bin/activate

# Install test dependencies
pip install pytest pytest-timeout --quiet

# Run tests based on argument
case "${1:-quick}" in
    "quick")
        echo "Running quick tests (no slow tests)..."
        pytest tests/ -v -m "not slow" --tb=short
        ;;
    "all")
        echo "Running all tests..."
        pytest tests/ -v --tb=short
        ;;
    "integration")
        echo "Running integration tests..."
        pytest tests/ -v -m "integration" --tb=short
        ;;
    "loader")
        echo "Running loader tests..."
        pytest tests/test_loaders.py -v --tb=short
        ;;
    "metrics")
        echo "Running metrics tests..."
        pytest tests/test_metrics.py -v --tb=short
        ;;
    "training")
        echo "Running training tests..."
        pytest tests/test_training.py -v --tb=short
        ;;
    *)
        echo "Usage: $0 [quick|all|integration|loader|metrics|training]"
        exit 1
        ;;
esac

echo ""
echo "=== Tests Complete ==="
