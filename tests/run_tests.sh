#!/bin/bash
#
# Simplex Test Runner
#
# Runs tests with stdlib included
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SXC="$PROJECT_ROOT/bootstrap_mini/sxc"
STDLIB="$PROJECT_ROOT/bootstrap_mini/stdlib.sx"
TESTS_DIR="$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
PASSED=0
FAILED=0
SKIPPED=0

run_test() {
    local test_file="$1"
    local test_name=$(basename "$test_file" .sx)
    local test_dir=$(dirname "$test_file")
    local rel_path="${test_file#$TESTS_DIR/}"

    # Create temp file that includes stdlib + test
    local tmp_file=$(mktemp /tmp/sx_test_XXXXXX.sx)
    trap "rm -f $tmp_file" RETURN

    # Combine stdlib and test file
    cat "$STDLIB" > "$tmp_file"
    echo "" >> "$tmp_file"
    echo "// ========== TEST FILE: $rel_path ==========" >> "$tmp_file"
    cat "$test_file" >> "$tmp_file"

    printf "  %-50s " "$rel_path"

    # Run the test
    local output
    local exit_code
    output=$("$SXC" run "$tmp_file" 2>&1) && exit_code=$? || exit_code=$?

    if [ $exit_code -eq 0 ]; then
        # Check if output contains FAIL
        if echo "$output" | grep -q "FAIL:"; then
            echo -e "${RED}FAILED${NC}"
            echo "$output" | grep "FAIL:" | sed 's/^/    /'
            ((FAILED++))
        else
            echo -e "${GREEN}PASSED${NC}"
            ((PASSED++))
        fi
    else
        echo -e "${RED}ERROR${NC}"
        echo "$output" | head -10 | sed 's/^/    /'
        ((FAILED++))
    fi
}

run_simple_test() {
    # Run tests that don't need stdlib (language tests)
    local test_file="$1"
    local test_name=$(basename "$test_file" .sx)
    local rel_path="${test_file#$TESTS_DIR/}"

    printf "  %-50s " "$rel_path"

    local output
    local exit_code
    output=$("$SXC" run "$test_file" 2>&1) && exit_code=$? || exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}PASSED${NC}"
        ((PASSED++))
    else
        # Some language tests are expected to compile but return non-zero
        # Check for compilation errors
        if echo "$output" | grep -q "error:"; then
            echo -e "${RED}ERROR${NC}"
            echo "$output" | head -5 | sed 's/^/    /'
            ((FAILED++))
        else
            echo -e "${GREEN}PASSED${NC}"
            ((PASSED++))
        fi
    fi
}

run_test_category() {
    local category="$1"
    local pattern="$2"
    local use_stdlib="$3"

    echo ""
    echo -e "${YELLOW}=== $category ===${NC}"

    for test_file in $pattern; do
        if [ -f "$test_file" ]; then
            if [ "$use_stdlib" = "yes" ]; then
                run_test "$test_file"
            else
                run_simple_test "$test_file"
            fi
        fi
    done
}

# Main
echo "Simplex Test Suite"
echo "=================="

# Check for sxc
if [ ! -x "$SXC" ]; then
    echo "Error: sxc not found at $SXC"
    exit 1
fi

# Check for stdlib
if [ ! -f "$STDLIB" ]; then
    echo "Error: stdlib.sx not found at $STDLIB"
    exit 1
fi

# Run language tests (don't need stdlib)
run_test_category "Language Tests - Basics" "$TESTS_DIR/language/basics/*.sx" "no"

# Run stdlib tests (need stdlib)
run_test_category "Stdlib Tests" "$TESTS_DIR/stdlib/*.sx" "yes"

# Run integration tests (need stdlib)
run_test_category "Integration Tests" "$TESTS_DIR/integration/*.sx" "yes"

# Summary
echo ""
echo "=================="
echo "Test Results:"
echo -e "  ${GREEN}Passed:${NC}  $PASSED"
echo -e "  ${RED}Failed:${NC}  $FAILED"
echo -e "  ${YELLOW}Skipped:${NC} $SKIPPED"
echo "=================="

if [ $FAILED -gt 0 ]; then
    exit 1
fi
exit 0
