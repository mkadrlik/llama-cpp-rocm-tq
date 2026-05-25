# llama-cpp-rocm-tq

llama.cpp with **TurboQuant** KV cache compression, built for AMD ROCm GPUs.

## Purpose

The **long-context** backend in the lemonade-tq ecosystem. TurboQuant compresses the KV cache 5.12x (turbo3) with <1% perplexity cost, enabling context windows up to 262K tokens on consumer GPUs.

## Quick Start (Clone → Validate)

### Prerequisites

- Docker 24.x+ with BuildKit
- AMD GPU with ROCm support (gfx1100 = RX 7900 series)
- `HSA_OVERRIDE_GFX_VERSION` set for your GPU architecture

### Clone

```bash
# From Gitea (primary, includes CI workflows)
git clone http://nas.kadrlik.home:3042/mkadrlik/llama-cpp-rocm-tq.git
cd llama-cpp-rocm-tq

# From GitHub (mirror, no CI workflows)
git clone https://github.com/mkadrlik/llama-cpp-rocm-tq.git
cd llama-cpp-rocm-tq
```

### Configure

```bash
cp .env.example .env
# Edit .env — at minimum set MODEL_PATH to a real .gguf file
# Example: MODEL_PATH=/home/you/models/Qwen3.6-27B-Q8_0.gguf
```

### Build

```bash
docker build -t llama-cpp-rocm-tq .
```

The build clones [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (feature/turboquant-kv-cache branch) and compiles with `GGML_HIP=ON`, `GPU_TARGETS=gfx1100`, and `GGML_HIP_ROCWMMA_FATTN=ON`. Build takes 20-40 minutes depending on hardware.

### Run

```bash
docker compose up -d
```

### Validate

```bash
# 1. Container is running
docker ps --filter name=llama-cpp-rocm-tq --format "{{.Status}}"
# Expected: "Up X minutes"

# 2. Server responds
curl -s http://localhost:8080/health | python3 -m json.tool
# Expected: {"status": "ok"}

# 3. Model loads with TurboQuant
# Check server logs for cache type confirmation
docker logs llama-cpp-rocm-tq 2>&1 | grep -i "cache type"
# Expected: cache_type_k=q8_0, cache_type_v=turbo3

# 4. Chat completion works
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Say hello"}],"max_tokens":20}' | python3 -m json.tool
# Expected: JSON response with choices[0].message.content

# 5. Stop
docker compose down
```

### Pull Pre-built Image (skip build)

```bash
docker pull ghcr.io/mkadrlik/llama-cpp-rocm-tq:latest
# Or from Gitea Container Registry
docker pull nas.kadrlik.home:3042/mkadrlik/llama-cpp-rocm-tq:latest
```

## Build System

Multi-stage Docker build using ROCm dev toolchain.

### Key Build Flags

| Flag | Purpose |
|------|---------|
| `GGML_HIP=ON` | HIP backend for GPU compute |
| `GGML_HIP_ROCWMMA_FATTN=ON` | rocWMMA flash attention (RDNA3 optimization) |
| `GPU_TARGETS=gfx1100` | RX 7900 series (change for other GPUs) |
| No NCCL | ROCm doesn't package NCCL; multi-GPU uses HIP native communications |

### Multi-GPU Build

For multi-GPU setups, the build disables NCCL by default. HIP native communications handles tensor splitting across GPUs. Ensure `--tensor-split` is set in your `.env` or EXTRA_ARGS.

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

## Key Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage build: ROCm dev toolchain → build → runtime |
| `docker-compose.yml` | Deploy with docker-compose |
| `.env.example` | Environment variable template |
| `benchmarks/README.md` | Performance benchmarks on 3x RX 7900 XTX |
| `.upstream-hash` | Tracks upstream llama.cpp commit |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_REGISTRY` | `ghcr.io` | Docker image registry |
| `IMAGE_NAME` | `mkadrlik/llama-cpp-rocm-tq` | Image name |
| `IMAGE_TAG` | `latest` | Image tag |
| `MODEL_PATH` | (required) | Path to .gguf model on host |
| `MODEL_MOUNT` | `/model.gguf` | Mount point inside container |
| `SERVER_HOST` | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | `8080` | Host port mapping |
| `HSA_OVERRIDE_GFX_VERSION` | `11.0.0` | ROCm GPU architecture override |
| `CACHE_TYPE_K` | `q8_0` | Key cache type (f16, q8_0, turbo4, turbo3, turbo2) |
| `CACHE_TYPE_V` | `turbo3` | Value cache type (f16, q8_0, turbo4, turbo3, turbo2) |
| `GPU_LAYERS` | `99` | GPU layers to offload (-1 or 99 = all) |
| `THREADS` | (auto) | CPU thread count |
| `CONTEXT_SIZE` | (model default) | Context window size |
| `EXTRA_ARGS` | (empty) | Additional llama-server arguments |

## CI/CD

- Runner: `rocm/linux` (Gitea Actions)
- Pushes to: `ghcr.io/mkadrlik/llama-cpp-rocm-tq:latest` and `nas.kadrlik.home:3042/mkadrlik/llama-cpp-rocm-tq:latest`
- Trigger: push to main
- Also mirrors to GitHub (excludes `.gitea/` and `.upstream-hash`)

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