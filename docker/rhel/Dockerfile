###############################################################################
# llama-cpp-rocm-tq (RHEL/Fedora)
# ROCm build of llama.cpp with TurboQuant KV cache compression.
#
# Source: TheTom/llama-cpp-turboquant (feature/turboquant-kv-cache)
# Base: UBI9 + ROCm 6.3 (Red Hat compatible)
###############################################################################

# Builder stage: UBI9 + ROCm dev toolchain
FROM registry.access.redhat.com/ubi9/ubi:9.5 AS builder

# Install EPEL and build dependencies
RUN dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm && \
    dnf install -y \
    cmake \
    git \
    gcc \
    gcc-c++ \
    make \
    openblas-devel \
    python3 \
    && dnf clean all

# Install ROCm 6.3 from AMD repo
RUN dnf install -y https://repo.radeon.com/rocm/rocm-6.3.0-80/ubi9.5/main/x86_64/redhat/6.3.0-80.amdgpu-1.noarch.rpm && \
    dnf install -y rocm-dev hipblas-devel rocblas-devel hip-runtime-amd && \
    dnf clean all

ENV PATH=/opt/rocm/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# Clone TurboQuant fork (TheTom — canonical, actively maintained)
RUN git clone --branch feature/turboquant-kv-cache --depth 1 \
    https://github.com/TheTom/llama-cpp-turboquant.git /opt/llama.cpp
WORKDIR /opt/llama.cpp

# Build with ROCm HIP backend
# GGML_HIP=ON enables the HIP backend (required for GPU compute)
# GGML_HIP_ROCWMMA_FATTN=ON enables rocWMMA flash attention (RDNA3 optimization)
RUN HIPCXX="/opt/rocm/bin/clang++" \
    HIP_PATH="/opt/rocm" \
    cmake -B build \
        -DGGML_HIP=ON \
        -DGGML_HIP_ROCWMMA_FATTN=ON \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j$(nproc)

###############################################################################
# Runtime image
###############################################################################
FROM registry.access.redhat.com/ubi9/ubi:9.5

# Install runtime dependencies
RUN dnf install -y \
    openblas \
    && dnf clean all

# Install ROCm runtime
RUN dnf install -y https://repo.radeon.com/rocm/rocm-6.3.0-80/ubi9.5/main/x86_64/redhat/6.3.0-80.amdgpu-1.noarch.rpm && \
    dnf install -y rocm-libs hip-runtime-amd && \
    dnf clean all

ENV LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib

# Copy built binaries + ALL shared libs (llama-server dynamically links to all of them)
COPY --from=builder /opt/llama.cpp/build/bin/llama-server /usr/local/bin/
COPY --from=builder /opt/llama.cpp/build/bin/llama-quantize /usr/local/bin/
COPY --from=builder /opt/llama.cpp/build/bin/llama-bench /usr/local/bin/
COPY --from=builder /opt/llama.cpp/build/bin/llama-perplexity /usr/local/bin/
COPY --from=builder /opt/llama.cpp/build/bin/libggml*.so* /usr/local/lib/
COPY --from=builder /opt/llama.cpp/build/bin/libllama*.so* /usr/local/lib/
COPY --from=builder /opt/llama.cpp/build/bin/libmtmd*.so* /usr/local/lib/

# Default: serve with TurboQuant 3-bit KV cache (recommended)
ENTRYPOINT ["llama-server"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
