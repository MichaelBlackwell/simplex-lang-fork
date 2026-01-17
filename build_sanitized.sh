#!/bin/bash
# Runtime Safety Build Script
# Builds Simplex runtime with various sanitizers for testing

set -e

echo "=== Runtime Safety Build ==="

# Detect platform
PLATFORM="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux"
fi

echo "Platform: $PLATFORM"

# Set library flags based on platform
if [[ "$PLATFORM" == "macos" ]]; then
    OPENSSL_PREFIX=$(brew --prefix openssl 2>/dev/null || echo "/usr/local/opt/openssl")
    SQLITE_PREFIX=$(brew --prefix sqlite 2>/dev/null || echo "/usr/local/opt/sqlite")
    LIBS="-lm -lssl -lcrypto -lsqlite3 -L$OPENSSL_PREFIX/lib -L$SQLITE_PREFIX/lib"
    INCLUDES="-I$OPENSSL_PREFIX/include -I$SQLITE_PREFIX/include"
elif [[ "$PLATFORM" == "linux" ]]; then
    LIBS="-lm -lssl -lcrypto -lsqlite3 -lpthread"
    INCLUDES=""
else
    echo "Warning: Unsupported platform"
    LIBS="-lm"
    INCLUDES=""
fi

SANITIZER=${1:-"address"}

echo "Building with sanitizer: $SANITIZER"

# Create build directory
mkdir -p build/sanitizers

case $SANITIZER in
  "address")
    echo "Building with AddressSanitizer..."
    gcc -O1 -g -fsanitize=address,undefined \
        -fno-omit-frame-pointer \
        -Wall -Wextra -Wformat-security -Werror \
        $INCLUDES \
        runtime/standalone_runtime.c \
        -o build/sanitizers/test_asan \
        $LIBS
    echo "ASan build complete: build/sanitizers/test_asan"
    ;;
    
  "undefined")
    echo "Building with UBSan..."
    gcc -O1 -g -fsanitize=undefined \
        -fno-omit-frame-pointer \
        -Wall -Wextra -Wformat-security -Werror \
        $INCLUDES \
        runtime/standalone_runtime.c \
        -o build/sanitizers/test_ubsan \
        $LIBS
    echo "UBSan build complete: build/sanitizers/test_ubsan"
    ;;
    
  "thread")
    echo "Building with ThreadSanitizer..."
    gcc -O1 -g -fsanitize=thread \
        -fno-omit-frame-pointer \
        -Wall -Wextra -Wformat-security -Werror \
        $INCLUDES \
        runtime/standalone_runtime.c \
        -o build/sanitizers/test_tsan \
        $LIBS
    echo "TSan build complete: build/sanitizers/test_tsan"
    ;;
    
  "coverage")
    echo "Building with coverage..."
    gcc -O0 -g --coverage \
        -Wall -Wextra -Wformat-security -Werror \
        $INCLUDES \
        runtime/standalone_runtime.c \
        -o build/sanitizers/test_coverage \
        $LIBS
    echo "Coverage build complete: build/sanitizers/test_coverage"
    ;;
    
  "all")
    echo "Building all sanitizer variants..."
    ./build_sanitized.sh address
    ./build_sanitized.sh undefined
    ./build_sanitized.sh thread
    ;;
    
  *)
    echo "Usage: $0 [address|undefined|thread|coverage|all]"
    echo ""
    echo "Available sanitizers:"
    echo "  address    - AddressSanitizer for memory error detection"
    echo "  undefined  - UBSan for undefined behavior detection" 
    echo "  thread     - ThreadSanitizer for data race detection"
    echo "  coverage   - Coverage instrumentation for test analysis"
    echo "  all        - Build all variants"
    exit 1
    ;;
esac

echo ""
echo "=== Runtime Environment Variables ==="
echo "For ASan: export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1"
echo "For UBSan: export UBSAN_OPTIONS=halt_on_error=1"
echo "For TSan: export TSAN_OPTIONS=halt_on_error=1"
echo ""
echo "=== Running Tests ==="
echo "Example: ./build/sanitizers/test_asan"