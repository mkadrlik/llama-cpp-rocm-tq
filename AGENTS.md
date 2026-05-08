# llama-cpp-rocm-tq

llama.cpp with **TurboQuant** KV cache compression, built for AMD ROCm GPUs.

## Purpose

The **long-context** backend in the lemonade-tq ecosystem. TurboQuant compresses the KV cache 5.12x (turbo3) with <1% perplexity cost, enabling context windows up to 262K tokens on consumer GPUs.

## Build System

Multi-stage Docker build using ROCm dev toolchain.

```bash
# Build locally
docker build -t llama-cpp-rocm-tq .

# Run standalone
docker run --rm --device /dev/kfd --device /dev/dri \
  --security-opt seccomp=unconfined \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  -p 8080:8080 \
  -v /path/to/model.gguf:/model.gguf:ro \
  llama-cpp-rocm-tq \
  --model /model.gguf -ctk turbo3 -ctv turbo3 -ngl 99
```

## Key Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage build: ROCm dev toolchain → build → runtime |
| `docker-compose.yml` | Deploy with docker-compose |
| `.env.example` | Environment variable template |
| `benchmarks/README.md` | Performance benchmarks on 3x RX 7900 XTX |
| `.upstream-hash` | Tracks upstream llama.cpp commit |

## How to Build

The Dockerfile uses `rocm/dev-ubuntu-24.04:7.2.3-complete` as the builder base (includes hipblas-dev, rocblas-dev, full dev toolchain).

1. **Builder stage**: Clones TheTom/llama-cpp-turboquant (feature/turboquant-kv-cache branch), builds with `GGML_HIP=ON`, `GPU_TARGETS=gfx1100`, `GGML_BLAS=ON`
2. **Runtime stage**: Minimal ROCm runtime, copies binaries + all shared libs

Key build flags:
- `GGML_HIP=ON` — HIP backend for GPU compute
- `GGML_HIP_ROCWMMA_FATTN=ON` — rocWMMA flash attention (RDNA3 optimization)
- `GPU_TARGETS=gfx1100` — RX 7900 series (adjust for other GPUs)
- No NCCL — ROCm doesn't package NCCL; multi-GPU uses HIP native communications

## Performance Benchmarks

Tested on 3x RX 7900 XTX (73.6 GB total VRAM) with Qwen3.6-27B-GGUF (16.4 GiB):

### Speed (compute-bound, no meaningful difference across cache types)

| Cache Type | Prompt (pp512) t/s | Generation (tg256) t/s |
|------------|-------------------|----------------------|
| f16 | 118.92 | 19.41 |
| turbo3 | 118.58 | 19.42 |

### KV Cache Memory at 262K Context

| Cache Type | Total KV Cache | vs f16 savings |
|------------|---------------|---------------|
| f16 | ~18,304 MiB | baseline |
| turbo3 | **5,952 MiB** | **-67.5%** |

### Quality (Perplexity)

| Cache Type | PPL | vs f16 delta |
|------------|-----|-------------|
| f16 | 1.5972 | baseline |
| turbo3 | 1.6044 | +0.45% |

**Recommendation**: Use `--cache-type-k q8_0 --cache-type-v turbo3` — 67.5% VRAM savings with negligible quality loss.

## CI/CD

- Runner: `rocm/linux` (Gitea Actions)
- Pushes to: `ghcr.io/mkadrlik/llama-cpp-rocm-tq:latest`
- Trigger: push to main
- Also mirrors to GitHub

## Dependencies

- ROCm 7.2+ (gfx1100 support)
- TheTom/llama-cpp-turboquant (feature/turboquant-kv-cache branch)
- OpenBLAS
- Docker 24.x+ with AMD GPU plugin

## Known Pitfalls

1. **GPU_TARGETS**: Set to `gfx1100` for RX 7900 series. Change for other AMD GPUs (e.g., `gfx942` for MI250).
2. **ROCWMMA_FATTN**: Enabled in build. If you hit issues, rebuild with `GGML_HIP_ROCWMMA_FATTN=OFF`.
3. **No NCCL**: Multi-GPU uses HIP native communications. Don't expect NCCL performance.
4. **Compute-bound**: At 27B params on 3 GPUs, speed is the same regardless of cache type. TurboQuant's value is VRAM savings, not throughput.
