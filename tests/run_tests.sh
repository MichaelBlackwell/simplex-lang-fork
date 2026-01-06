#!/bin/bash
# Simplex Test Runner
# Runs all tests in the test suite

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BOOTSTRAP_DIR="$PROJECT_ROOT/bootstrap_mini"
RUNTIME="$BOOTSTRAP_DIR/standalone_runtime.c"
COMPILER="$BOOTSTRAP_DIR/stage0.py"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0
SKIPPED=0

run_test() {
    local test_file="$1"
    local test_name=$(basename "$test_file" .sx)

    # Skip library/helper files
    if [[ "$test_name" == "mathlib" ]]; then
        return
    fi

    printf "    %-45s " "$test_name"

    local test_dir=$(dirname "$test_file")
    local orig_dir=$(pwd)

    cd "$test_dir"

    # Compile
    if ! python3 "$COMPILER" "$test_name.sx" >/dev/null 2>&1; then
        echo -e "${RED}COMPILE FAIL${NC}"
        ((FAILED++))
        cd "$orig_dir"
        return
    fi

    # Link
    if ! clang -O2 "$test_name.ll" "$RUNTIME" -o "$test_name.bin" -lssl -lcrypto -lsqlite3 2>/dev/null; then
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

        # Check for .sx files (excluding mathlib and other helpers)
        shopt -s nullglob
        for test in "$category_path"/*.sx; do
            if [ -f "$test" ]; then
                local name=$(basename "$test" .sx)
                if [[ "$name" != "mathlib" ]]; then
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

echo "=============================================="
echo "         Simplex Language Test Suite"
echo "=============================================="
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

# Summary
echo "=============================================="
echo -e "  ${GREEN}Passed:  $PASSED${NC}"
echo -e "  ${RED}Failed:  $FAILED${NC}"
TOTAL=$((PASSED + FAILED))
if [ $TOTAL -gt 0 ]; then
    PERCENT=$((PASSED * 100 / TOTAL))
    echo -e "  Total:   $TOTAL ($PERCENT% pass rate)"
fi
echo "=============================================="

if [ $FAILED -gt 0 ]; then
    exit 1
fi
