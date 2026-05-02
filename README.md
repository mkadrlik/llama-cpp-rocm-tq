# llama-cpp-rocm-tq

llama.cpp with **TurboQuant** KV cache compression, built for AMD ROCm GPUs with multi-GPU tensor parallelism via Split Mode Graph.

## What is TurboQuant?

TurboQuant is a lossy KV cache quantization technique that compresses the key and value caches to 3-bit precision with negligible accuracy loss (<0.1% perplexity degradation). This yields:

- **5.12x KV cache compression** — fits 5x more context in the same VRAM
- **Dramatically reduced memory bandwidth** — faster inference, lower power
- **Near-zero accuracy cost** — imperceptible quality difference on standard benchmarks

Developed by domvox, originally ported to the HIP/ROCm backend.

## Features

| Feature | Flag | Description |
|---------|------|-------------|
| TurboQuant K-cache | `-ctk turbo3` | 3-bit compressed key cache |
| TurboQuant V-cache | `-ctv turbo3` | 3-bit compressed value cache |
| Split Mode Graph | `-sm graph` | NCCL-based multi-GPU tensor parallelism |
| ROCm HIP backend | `GGML_HIP=ON` | AMD GPU acceleration via ROCm |
| NCCL support | `GGML_NCCL=ON` | Multi-GPU communication library |

## Quick Start

```bash
# Build
docker build -t llama-cpp-rocm-tq .

# Run with a model (single GPU)
docker run --rm --device /dev/kfd --device /dev/dri \
  --security-opt seccomp=unconfined \
  -v /path/to/model.gguf:/model.gguf:ro \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  llama-cpp-rocm-tq \
  /model.gguf --sm graph -ctk turbo3 -ctv turbo3 -ngl 99

# Multi-GPU (3 GPUs, tensor-split)
docker run --rm --device /dev/kfd --device /dev/dri \
  --security-opt seccomp=unconfined \
  -v /path/to/model.gguf:/model.gguf:ro \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  -e HIP_VISIBLE_DEVICES=0,1,2 \
  llama-cpp-rocm-tq \
  /model.gguf --sm graph -ctk turbo3 -ctv turbo3 --tensor-split 1,1,1 -ngl 99
```

## Build Configuration

| CMake Flag | Value | Purpose |
|-----------|-------|---------|
| `GGML_HIP` | `ON` | Enable ROCm/HIP GPU backend |
| `GGML_NCCL` | `ON` | Enable NCCL for multi-GPU |
| `GGML_BLAS` | `ON` | Enable BLAS (OpenBLAS) for CPU fallback |
| `GGML_NATIVE` | `OFF` | Disable host-specific optimizations (portability) |
| `GGML_AVX*` | `OFF` | Disable x86 vector extensions (ROCm uses GPU) |

## Hardware Requirements

- **AMD GPU** with ROCm support (RDNA2/RDNA3: RX 6000/7000 series)
- **ROCm runtime** (mounted from host or included in image)
- For multi-GPU: GPUs must be the same model and connected via PCIe/NVLink

## Source

Built from `domvox/llama.cpp-turboquant-hip` branch `feature/turboquant-hip-port-clean`.

## License

Same as upstream llama.cpp (MIT). TurboQuant patches by domvox.
