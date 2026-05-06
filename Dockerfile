###############################################################################
# llama-cpp-rocm-tq
# ROCm build of llama.cpp with TurboQuant KV cache compression.
#
# Source: TheTom/llama-cpp-turboquant (feature/turboquant-kv-cache)
###############################################################################

# Use -complete variant which includes hipblas-dev, rocblas-dev, and all CMake configs
# Standard tag is runtime-only; -complete has full dev toolchain (~6.9 GB vs ~1.1 GB)
FROM rocm/dev-ubuntu-24.04:7.2.3-complete AS builder

# Install build dependencies (hipblas/rocblas already in -complete image)
RUN apt-get update && apt-get install -y \
    cmake \
    git \
    build-essential \
    pkg-config \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone TurboQuant fork (TheTom — canonical, actively maintained)
RUN git clone --branch feature/turboquant-kv-cache --depth 1 \
    https://github.com/TheTom/llama-cpp-turboquant.git /opt/llama.cpp
WORKDIR /opt/llama.cpp

# Build with ROCm HIP backend
# GGML_HIP=ON enables the HIP backend (required for GPU compute)
# GGML_HIP_ROCWMMA_FATTN=ON enables rocWMMA flash attention (RDNA3 optimization)
# GPU_TARGETS defaults to "native" which auto-detects the host GPU.
RUN HIPCXX="$(hipconfig -l)/clang" \
    HIP_PATH="$(hipconfig -R)" \
    cmake -B build \
        -DGGML_HIP=ON \
        -DGGML_HIP_ROCWMMA_FATTN=ON \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j$(nproc)

###############################################################################
# Runtime image
###############################################################################
FROM rocm/dev-ubuntu-24.04:7.2.3

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libopenblas0-pthread \
    && rm -rf /var/lib/apt/lists/*

# Copy built binaries + ALL shared libs (llama-server dynamically links to all of them)
COPY --from=builder /opt/llama.cpp/build/bin/llama-server /usr/local/bin/
COPY --from=builder /opt/llama.cpp/build/bin/llama-quantize /usr/local/bin/
COPY --from=builder /opt/llama.cpp/build/bin/llama-bench /usr/local/bin/
COPY --from=builder /opt/llama.cpp/build/bin/llama-perplexity /usr/local/bin/
COPY --from=builder /opt/llama.cpp/build/bin/libggml*.so* /usr/local/lib/
COPY --from=builder /opt/llama.cpp/build/bin/libllama*.so* /usr/local/lib/
COPY --from=builder /opt/llama.cpp/build/bin/libmtmd*.so* /usr/local/lib/

# Copy ROCm runtime libraries (hipblas, rocblas, etc.)
COPY --from=builder /opt/rocm/lib/ /opt/rocm/lib/

ENV LD_LIBRARY_PATH=/usr/local/lib

# Default: serve with TurboQuant 3-bit KV cache (recommended)
ENTRYPOINT ["llama-server"]
CMD ["--host", "0.0.0.0", "--port", "8080", "-ctk", "turbo3", "-ctv", "turbo3"]
