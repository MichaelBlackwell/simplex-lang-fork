#!/bin/bash
#
# Pilot Validation Runner
# =======================
#
# Runs the full validation pipeline for pilot specialists.
# This validates our approach before scaling to all 33 specialists.
#
# Estimated costs:
#   - Local test: Free (uses CPU, ~30 min per specialist)
#   - Full run: ~$10-15 on AWS (GPU needed)
#
# Usage:
#   ./run_pilots.sh --local-test          # Quick validation on CPU
#   ./run_pilots.sh --baseline-only       # Just measure baseline
#   ./run_pilots.sh --full                # Full training (needs GPU)
#   ./run_pilots.sh --specialist sentiment # Single specialist
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default options
MODE="local-test"
SPECIALIST="all"
OUTPUT_DIR="results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --local-test|-t)
            MODE="local-test"
            shift
            ;;
        --baseline-only|-b)
            MODE="baseline-only"
            shift
            ;;
        --full|-f)
            MODE="full"
            shift
            ;;
        --specialist|-s)
            SPECIALIST="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --local-test, -t     Quick test on CPU (default)"
            echo "  --baseline-only, -b  Only measure baseline, skip training"
            echo "  --full, -f           Full training run (needs GPU)"
            echo "  --specialist, -s     Run specific specialist (sentiment|sql|invoice|all)"
            echo "  --output, -o         Output directory (default: results)"
            echo "  --help, -h           Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "  PILOT VALIDATION PIPELINE"
echo "=============================================="
echo ""
echo "Mode:       $MODE"
echo "Specialist: $SPECIALIST"
echo "Output:     $OUTPUT_DIR"
echo ""

# Check Python environment
if [ -f "../.venv/bin/activate" ]; then
    source ../.venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Check dependencies
echo "Checking dependencies..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || {
    echo -e "${RED}ERROR: PyTorch not installed${NC}"
    echo "Run: pip install torch"
    exit 1
}

python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null || {
    echo -e "${RED}ERROR: Transformers not installed${NC}"
    echo "Run: pip install transformers"
    exit 1
}

python3 -c "import datasets; print(f'Datasets: {datasets.__version__}')" 2>/dev/null || {
    echo -e "${YELLOW}WARNING: datasets not installed, installing...${NC}"
    pip install datasets
}

# Check GPU
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo -e "${GREEN}GPU: Available (CUDA)${NC}"
    GPU_AVAILABLE=true
elif python3 -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    echo -e "${GREEN}GPU: Available (Apple MPS)${NC}"
    GPU_AVAILABLE=true
else
    echo -e "${YELLOW}GPU: Not available (using CPU)${NC}"
    GPU_AVAILABLE=false
fi

echo ""

# Build command arguments
CMD_ARGS="--output-dir $OUTPUT_DIR"

if [ "$MODE" = "local-test" ]; then
    CMD_ARGS="$CMD_ARGS --local-test"
elif [ "$MODE" = "baseline-only" ]; then
    CMD_ARGS="$CMD_ARGS --baseline-only"
fi

# Run pilots
run_pilot() {
    local specialist=$1
    echo ""
    echo "=============================================="
    echo "  Running: $specialist"
    echo "=============================================="

    python3 pilots/train_pilot.py --specialist "$specialist" $CMD_ARGS

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}$specialist: COMPLETED${NC}"
    else
        echo -e "${RED}$specialist: FAILED${NC}"
        return 1
    fi
}

# Track results
PASSED=0
FAILED=0

if [ "$SPECIALIST" = "all" ]; then
    SPECIALISTS=("sentiment" "sql" "invoice")
else
    SPECIALISTS=("$SPECIALIST")
fi

START_TIME=$(date +%s)

for spec in "${SPECIALISTS[@]}"; do
    if run_pilot "$spec"; then
        ((PASSED++))
    else
        ((FAILED++))
    fi
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "  PILOT VALIDATION COMPLETE"
echo "=============================================="
echo ""
echo "Duration: $((DURATION / 60)) minutes $((DURATION % 60)) seconds"
echo -e "Passed:   ${GREEN}$PASSED${NC}"
echo -e "Failed:   ${RED}$FAILED${NC}"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""

# Show summary
if [ -d "$OUTPUT_DIR" ]; then
    echo "Generated files:"
    ls -la "$OUTPUT_DIR"/*.json "$OUTPUT_DIR"/*.txt 2>/dev/null || true
fi

# Exit with error if any failed
if [ $FAILED -gt 0 ]; then
    exit 1
fi
