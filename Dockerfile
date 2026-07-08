################################################################################
# BUILD STAGE
################################################################################
FROM rust:1.89-trixie AS builder

WORKDIR /usr/src/TVC

# Install build dependencies for Rust + C bindings
RUN apt-get update && apt-get install -y \
    clang \
    libclang-dev \
    build-essential \
    pkg-config \
    libssl-dev \
    procps && \
    rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . .

# Build release binary with ONNX inference enabled
RUN cargo build --release --features onnx-inference

# Strip binary to reduce size
RUN strip target/release/tvc

################################################################################
# RUNTIME STAGE
################################################################################
FROM debian:trixie-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    bash \
    libssl-dev \
    procps && \
    rm -rf /var/lib/apt/lists/*

# Copy binary from builder
COPY --from=builder /usr/src/TVC/target/release/tvc /usr/local/bin/tvc
COPY --from=builder /usr/src/TVC/model.onnx /opt/tvc/model.onnx

# Use the directory containing model.onnx so the default --model-path works.
WORKDIR /opt/tvc

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/tvc"]
