FROM rocm/dev-ubuntu-22.04 AS builder

# Install dependencies including NCCL for Split Mode Graph
RUN apt-get update && apt-get install -y \
    cmake \
    git \
    build-essential \
    libnccl-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone llama.cpp
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp
WORKDIR /opt/llama.cpp

# Build with ROCm, NCCL, and TurboQuant support
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

# Runtime image
FROM rocm/dev-ubuntu-22.04
COPY --from=builder /opt/llama.cpp/build/bin/llama-server /usr/local/bin/
COPY --from=builder /opt/llama.cpp/build/bin/llama-quantize /usr/local/bin/

# Install NCCL for runtime
RUN apt-get update && apt-get install -y libnccl2 && rm -rf /var/lib/apt/lists/*

# Default command with Split Mode Graph and TurboQuant
ENTRYPOINT ["llama-server"]
CMD ["--host", "0.0.0.0", "--port", "8080", "-sm", "graph", "-ctk", "turbo", "-ctv", "turbo"]
