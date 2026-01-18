# Simplex Playground Docker Image
#
# Builds a secure, sandboxed environment for running user code.
# Deploy at play.simplex-lang.org
#
# Build:
#   docker build -t simplex-playground .
#
# Run:
#   docker run -p 8080:8080 simplex-playground

FROM ubuntu:22.04 AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    llvm \
    libssl-dev \
    libsqlite3-dev \
    python3 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Simplex compiler source
WORKDIR /build
COPY . /build/simplex-lang/

# Build the Simplex compiler
WORKDIR /build/simplex-lang
RUN ./build.sh || echo "Using pre-built binaries"

# Install compiler to /usr/local
RUN mkdir -p /usr/local/bin /usr/local/lib/simplex \
    && cp sxc /usr/local/bin/ || true \
    && cp sxc-compile /usr/local/bin/ || true \
    && cp standalone_runtime.c /usr/local/lib/simplex/ || true \
    && chmod +x /usr/local/bin/sxc /usr/local/bin/sxc-compile || true

# ============================================
# Runtime image - minimal and secure
# ============================================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    clang \
    libssl3 \
    libsqlite3-0 \
    python3 \
    python3-pip \
    coreutils \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install flask flask-cors gunicorn

# Create non-root user for running the server
RUN useradd -m -s /bin/bash playground \
    && mkdir -p /app /tmp/simplex-playground \
    && chown -R playground:playground /app /tmp/simplex-playground

# Copy compiler from builder
COPY --from=builder /usr/local/bin/sxc /usr/local/bin/
COPY --from=builder /usr/local/bin/sxc-compile /usr/local/bin/
COPY --from=builder /usr/local/lib/simplex/ /usr/local/lib/simplex/

# Copy playground application
WORKDIR /app
COPY playground/index.html playground/style.css playground/app.js playground/examples.json ./
COPY playground/server.py ./

# Set permissions
RUN chmod +x /usr/local/bin/sxc /usr/local/bin/sxc-compile \
    && chown -R playground:playground /app

# Environment configuration
ENV SXC_PATH=/usr/local/bin/sxc
ENV RUNTIME_PATH=/usr/local/lib/simplex/standalone_runtime.c
ENV TEMP_BASE=/tmp/simplex-playground
ENV COMPILE_TIMEOUT=30
ENV RUN_TIMEOUT=10
ENV MAX_OUTPUT_SIZE=65536
ENV MAX_CODE_SIZE=102400
ENV MAX_MEMORY=268435456
ENV RATE_LIMIT_REQUESTS=20
ENV RATE_LIMIT_WINDOW=60

# Security: Run as non-root
USER playground

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Start server with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "--timeout", "120", "server:app"]
