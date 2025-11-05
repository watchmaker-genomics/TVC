################################################################################
# BUILD STAGE
################################################################################
FROM rust:1.89-bookworm AS builder

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

# Build release binary
RUN cargo build --release

# Strip binary to reduce size
RUN strip target/release/tvc

################################################################################
# RUNTIME STAGE
################################################################################
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    bash \
    libssl-dev \
    procps && \
    rm -rf /var/lib/apt/lists/*

# Copy binary from builder
COPY --from=builder /usr/src/TVC/target/release/tvc /usr/local/bin/tvc

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/tvc"]
