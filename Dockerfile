FROM rust:1.94

# System deps
RUN apt-get update && apt-get install -y \
    clang \
    llvm \
    protobuf-compiler \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install wasm-pack
RUN cargo install wasm-pack

WORKDIR /app