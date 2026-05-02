###############################################################################
# llama-cpp-rocm-tq
# ROCm build of llama.cpp with TurboQuant KV cache compression and
# Split Mode Graph (NCCL-based multi-GPU tensor parallelism).
#
# Source: domvox/llama.cpp-turboquant-hip (feature/turboquant-hip-port-clean)
###############################################################################

FROM rocm/dev-ubuntu-22.04 AS builder

# Build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    git \
    build-essential \
    libnccl-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone TurboQuant fork (not vanilla llama.cpp)
RUN git clone --branch feature/turboquant-hip-port-clean --depth 1 \
    https://github.com/domvox/llama.cpp-turboquant-hip.git /opt/llama.cpp
WORKDIR /opt/llama.cpp

# Build with ROCm HIP, NCCL, TurboQuant, and OpenBLAS
RUN cmake -B build \
    -DGGML_HIP=ON \
    -DGGML_NCCL=ON \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=OpenBLAS \
    -DGGML_NATIVE=OFF \
    -DGGML_AVX=OFF \
    -DGGML_AVX2=OFF \
    -DGGML_AVX512=OFF \
    -DGGML_F16C=OFF \
    -DGGML_FMA=OFF \
    -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang
RUN cmake --build build --config Release -j$(nproc)

###############################################################################
# Runtime image
###############################################################################
FROM rocm/dev-ubuntu-22.04

# Runtime dependencies
RUN apt-get update && apt-get install -y \
    libnccl2 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Copy built binaries
COPY --from=builder /opt/llama.cpp/build/bin/llama-server /usr/local/bin/
COPY --from=builder /opt/llama.cpp/build/bin/llama-quantize /usr/local/bin/

# Default: serve with TurboQuant and Split Mode Graph
ENTRYPOINT ["llama-server"]
CMD ["--host", "0.0.0.0", "--port", "8080", "-sm", "graph", "-ctk", "turbo", "-ctv", "turbo"]
