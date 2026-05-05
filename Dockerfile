###############################################################################
# llama-cpp-rocm-tq
# ROCm build of llama.cpp with TurboQuant KV cache compression.
#
# Source: TheTom/llama-cpp-turboquant (feature/turboquant-kv-cache)
###############################################################################

FROM rocm/dev-ubuntu-22.04 AS builder

# Add ROCm apt repo (trusted=yes to avoid keyserver issues in build)
# and install build dependencies.
# NOTE: rocm/dev-ubuntu-22.04 is minimal — hipblas/rocblas not pre-installed.
RUN echo "deb [trusted=yes] https://repo.radeon.com/rocm/apt/7.2.2 jammy main" > /etc/apt/sources.list.d/rocm.list && \
    apt-get update && apt-get install -y \
    cmake \
    git \
    build-essential \
    pkg-config \
    hipblas-dev \
    rocblas-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone TurboQuant fork (TheTom — canonical, actively maintained)
RUN git clone --branch feature/turboquant-kv-cache --depth 1 \
    https://github.com/TheTom/llama-cpp-turboquant.git /opt/llama.cpp
WORKDIR /opt/llama.cpp

# Build with ROCm HIP and OpenBLAS
# Use HIPCXX/HIP_PATH env vars — do NOT override CMAKE_CXX_COMPILER
# (CMake's HIP language support needs to detect the ROCm Clang itself)
RUN HIPCXX="$(hipconfig -l)/clang" \
    HIP_PATH="$(hipconfig -R)" \
    CMAKE_PREFIX_PATH=/opt/rocm \
    cmake -B build \
        -DGGML_HIP=ON \
        -DGPU_TARGETS=gfx1100 \
        -DGGML_BLAS=ON \
        -DGGML_BLAS_VENDOR=OpenBLAS \
        -DGGML_NATIVE=OFF \
        -DGGML_AVX=OFF \
        -DGGML_AVX2=OFF \
        -DGGML_AVX512=OFF \
        -DGGML_F16C=OFF \
        -DGGML_FMA=OFF \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j$(nproc)

###############################################################################
# Runtime image
###############################################################################
FROM rocm/dev-ubuntu-22.04

# Add ROCm apt repo and install runtime dependencies
RUN echo "deb [trusted=yes] https://repo.radeon.com/rocm/apt/7.2.2 jammy main" > /etc/apt/sources.list.d/rocm.list && \
    apt-get update && apt-get install -y \
    hipblas \
    rocblas \
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

ENV LD_LIBRARY_PATH=/usr/local/lib

# Default: serve with TurboQuant 3-bit KV cache (recommended)
ENTRYPOINT ["llama-server"]
CMD ["--host", "0.0.0.0", "--port", "8080", "-ctk", "turbo3", "-ctv", "turbo3"]
