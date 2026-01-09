#!/bin/bash
# Simplex Test Runner
# Runs all tests in the test suite
#
# Usage: ./run_tests.sh [category]
#   category: all, language, neural, stdlib, runtime, ai, integration, toolchain
#
# Examples:
#   ./run_tests.sh           # Run all tests
#   ./run_tests.sh neural    # Run only neural IR tests
#   ./run_tests.sh language  # Run only language tests

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RUNTIME="$PROJECT_ROOT/standalone_runtime.c"
COMPILER="$PROJECT_ROOT/stage0.py"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0
SKIPPED=0
WARNINGS=0

# Check if compiler exists
if [ ! -f "$COMPILER" ]; then
    echo -e "${RED}Error: Compiler not found at $COMPILER${NC}"
    exit 1
fi

if [ ! -f "$RUNTIME" ]; then
    echo -e "${RED}Error: Runtime not found at $RUNTIME${NC}"
    exit 1
fi

run_test() {
    local test_file="$1"
    local test_name=$(basename "$test_file" .sx)
    local test_dir=$(dirname "$test_file")
    local display_name="${test_dir#$SCRIPT_DIR/}/$test_name"

    # Skip library/helper files
    if [[ "$test_name" == "mathlib" ]] || [[ "$test_name" == "helpers" ]]; then
        return
    fi

    printf "    %-50s " "$display_name"

    local orig_dir=$(pwd)
    cd "$test_dir"

    # Compile - capture stderr for warnings
    local compile_output
    compile_output=$(python3 "$COMPILER" "$test_name.sx" 2>&1)
    local compile_status=$?

    # Check for static analysis warnings
    if echo "$compile_output" | grep -q "Warning:"; then
        ((WARNINGS++))
    fi

    if [ $compile_status -ne 0 ] || [ ! -f "$test_name.ll" ]; then
        echo -e "${RED}COMPILE FAIL${NC}"
        ((FAILED++))
        cd "$orig_dir"
        return
    fi

    # Link
    if ! clang -O2 "$test_name.ll" "$RUNTIME" -o "$test_name.bin" -lssl -lcrypto -lsqlite3 -lm 2>/dev/null; then
        echo -e "${RED}LINK FAIL${NC}"
        ((FAILED++))
        rm -f "$test_name.ll"
        cd "$orig_dir"
        return
    fi

    # Run
    if ./"$test_name.bin" >/dev/null 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        ((PASSED++))
    else
        echo -e "${RED}FAIL${NC}"
        ((FAILED++))
    fi

    # Cleanup
    rm -f "$test_name.ll" "$test_name.bin"
    cd "$orig_dir"
}

run_category() {
    local category_path="$1"
    local category_name="$2"
    local indent="${3:-  }"

    if [ -d "$category_path" ]; then
        local has_tests=false

        # Check for .sx files (excluding helpers)
        shopt -s nullglob
        for test in "$category_path"/*.sx; do
            if [ -f "$test" ]; then
                local name=$(basename "$test" .sx)
                if [[ "$name" != "mathlib" ]] && [[ "$name" != "helpers" ]]; then
                    has_tests=true
                    break
                fi
            fi
        done

        if $has_tests; then
            echo -e "${indent}${CYAN}$category_name${NC}"
            for test in "$category_path"/*.sx; do
                [ -f "$test" ] && run_test "$test"
            done
        fi

        # Recurse into subdirectories
        for subdir in "$category_path"/*/; do
            if [ -d "$subdir" ]; then
                local subname=$(basename "$subdir")
                run_category "$subdir" "$subname" "${indent}  "
            fi
        done
    fi
}

print_header() {
    echo ""
    echo "=============================================="
    echo "         Simplex Language Test Suite"
    echo "=============================================="
    echo ""
}

print_summary() {
    echo ""
    echo "=============================================="
    echo -e "  ${GREEN}Passed:   $PASSED${NC}"
    echo -e "  ${RED}Failed:   $FAILED${NC}"
    if [ $WARNINGS -gt 0 ]; then
        echo -e "  ${YELLOW}Warnings: $WARNINGS${NC}"
    fi
    TOTAL=$((PASSED + FAILED))
    if [ $TOTAL -gt 0 ]; then
        PERCENT=$((PASSED * 100 / TOTAL))
        echo -e "  Total:    $TOTAL ($PERCENT% pass rate)"
    fi
    echo "=============================================="
}

run_all_tests() {
    # Neural IR Tests (new)
    echo -e "${YELLOW}Neural IR${NC}"
    run_category "$SCRIPT_DIR/neural" "" "  "
    echo ""

    # Language Tests
    echo -e "${YELLOW}Language${NC}"
    run_category "$SCRIPT_DIR/language" "" "  "
    echo ""

    # Standard Library Tests
    echo -e "${YELLOW}Standard Library${NC}"
    run_category "$SCRIPT_DIR/stdlib" "" "  "
    echo ""

    # Runtime Tests
    echo -e "${YELLOW}Runtime${NC}"
    run_category "$SCRIPT_DIR/runtime" "" "  "
    echo ""

    # AI/Cognitive Tests
    echo -e "${YELLOW}AI / Cognitive${NC}"
    run_category "$SCRIPT_DIR/ai" "" "  "
    echo ""

    # Toolchain Tests
    echo -e "${YELLOW}Toolchain${NC}"
    run_category "$SCRIPT_DIR/toolchain" "" "  "
    echo ""

    # Observability Tests
    echo -e "${YELLOW}Observability${NC}"
    run_category "$SCRIPT_DIR/observability" "" "  "
    echo ""

    # Integration Tests
    echo -e "${YELLOW}Integration${NC}"
    run_category "$SCRIPT_DIR/integration" "" "  "
    echo ""
}

# Main execution
CATEGORY="${1:-all}"

print_header

case "$CATEGORY" in
    all)
        run_all_tests
        ;;
    neural)
        echo -e "${YELLOW}Neural IR${NC}"
        run_category "$SCRIPT_DIR/neural" "" "  "
        ;;
    language)
        echo -e "${YELLOW}Language${NC}"
        run_category "$SCRIPT_DIR/language" "" "  "
        ;;
    stdlib)
        echo -e "${YELLOW}Standard Library${NC}"
        run_category "$SCRIPT_DIR/stdlib" "" "  "
        ;;
    runtime)
        echo -e "${YELLOW}Runtime${NC}"
        run_category "$SCRIPT_DIR/runtime" "" "  "
        ;;
    ai)
        echo -e "${YELLOW}AI / Cognitive${NC}"
        run_category "$SCRIPT_DIR/ai" "" "  "
        ;;
    toolchain)
        echo -e "${YELLOW}Toolchain${NC}"
        run_category "$SCRIPT_DIR/toolchain" "" "  "
        ;;
    integration)
        echo -e "${YELLOW}Integration${NC}"
        run_category "$SCRIPT_DIR/integration" "" "  "
        ;;
    observability)
        echo -e "${YELLOW}Observability${NC}"
        run_category "$SCRIPT_DIR/observability" "" "  "
        ;;
    *)
        echo -e "${RED}Unknown category: $CATEGORY${NC}"
        echo "Available: all, neural, language, stdlib, runtime, ai, toolchain, integration, observability"
        exit 1
        ;;
esac

print_summary

if [ $FAILED -gt 0 ]; then
    exit 1
fi
