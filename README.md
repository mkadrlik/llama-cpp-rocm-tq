# llama-cpp-rocm-tq

llama.cpp with **TurboQuant** KV cache compression, built for AMD ROCm GPUs.

[![Built with ROCm](https://img.shields.io/badge/Built%20with-ROCm-ED1C24?logo=amd)](https://rocm.docs.amd.com/)

## What is TurboQuant?

TurboQuant compresses the KV cache using Walsh-Hadamard Transform (WHT) rotation + PolarQuant scalar quantization, enabling dramatically larger context windows within the same VRAM budget.

| Type | Bits | Compression | PPL cost |
|------|------|-------------|----------|
| `turbo3` | 3-bit | 5.12x | <1% (recommended) |
| `turbo4` | 4-bit | 3.8x | +0.23% |
| `turbo2` | 2-bit | 7.5x | +3.7% |

**Paper:** [TurboQuant: KV Cache Compression via WHT](https://arxiv.org/abs/2504.19874)

## Quick Start

```bash
# Using pre-built image from ghcr.io
docker run --rm --device /dev/kfd --device /dev/dri \
  --security-opt seccomp=unconfined \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  -p 8080:8080 \
  -v /path/to/model.gguf:/model.gguf:ro \
  ghcr.io/mkadrlik/llama-cpp-rocm-tq:latest \
  --model /model.gguf -ctk turbo3 -ctv turbo3 -ngl 99
```

Or use docker-compose (see [docker-compose.yml](docker-compose.yml)):

```bash
cp .env.example .env
# Edit .env to set MODEL_PATH and other options
docker compose up -d
```

## Building from Source

```bash
docker build -t llama-cpp-rocm-tq .
```

## Attribution & Acknowledgements

This project builds on the excellent work of several open source contributors:

- **[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)** — The canonical TurboQuant fork of llama.cpp. This Docker image uses the `feature/turboquant-kv-cache` branch. All credit for TurboQuant implementation goes to TheTom.
- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** — The upstream llama.cpp project by @ggerganov and contributors.
- **[TurboQuant Paper](https://arxiv.org/abs/2504.19874)** — The research paper describing the WHT + PolarQuant KV cache compression technique.
- **ROCm** — AMD's open software platform for GPU computing, providing the HIP compiler and libraries used in this build.
- **OpenBLAS** — Optimized BLAS implementation used for CPU fallback and ROCm BLAS backend.

## Build Process

This Docker image is built using a multi-stage Dockerfile:

1. **Builder stage** (`rocm/dev-ubuntu-22.04`):
   - Installs ROCm 7.2.2 development packages (hipblas-dev, rocblas-dev, libopenblas-dev)
   - Clones TheTom's TurboQuant fork from `feature/turboquant-kv-cache` branch
   - Configures CMake with `GGML_HIP=ON`, `GPU_TARGETS=gfx1100`, `GGML_BLAS=ON`
   - Builds with ROCm HIP compiler (detected via `hipconfig`)

2. **Runtime stage** (`rocm/dev-ubuntu-22.04`):
   - Installs ROCm runtime packages (hipblas, rocblas, libopenblas0-pthread)
   - Copies built binaries (llama-server, llama-quantize, llama-bench, llama-perplexity)
   - Copies all shared libraries (libggml*, libllama*, libmtmd*)
   - Sets default CMD to serve with turbo3 KV cache compression

### Key Build Decisions

- Uses `HIPCXX`/`HIP_PATH` env vars instead of explicit `CMAKE_CXX_COMPILER` — allows CMake's HIP language support to detect ROCm Clang correctly
- Targets `gfx1100` (RDNA3 / RX 7900 series) — adjust `GPU_TARGETS` for other AMD GPUs
- No NCCL — ROCm doesn't package NCCL; multi-GPU uses HIP native communications
- Uses `libopenblas0-pthread` for runtime (not the meta-package `libopenblas0`)

## Benchmarks

See [benchmarks/README.md](benchmarks/README.md) for detailed benchmark results on the test hardware:
- 3x AMD Radeon RX 7900 XTX (24 GB each, 73.6 GB total VRAM)
- Qwen3.6-27B-UD-Q4_K_XL model (16.39 GiB, 26.90B parameters)

## Repository Structure

- `main` — Turn-key repository, ready to use. No sensitive values.
- `home-lab` — Environment-specific configuration (not for general use).

## License

MIT (same as upstream llama.cpp).
