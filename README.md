# llama-cpp-rocm-tq

llama.cpp with **TurboQuant** KV cache compression, built for AMD ROCm GPUs.

## What is TurboQuant?

TurboQuant compresses the KV cache using Walsh-Hadamard Transform (WHT) rotation + PolarQuant scalar quantization. Results:

| Type | Bits | Compression | PPL cost |
|------|------|-------------|----------|
| `turbo3` | 3-bit | 5.12x | <1% (recommended) |
| `turbo4` | 4-bit | 3.8x | +0.23% |
| `turbo2` | 2-bit | 7.5x | +3.7% |

Paper: [TurboQuant: KV Cache Compression via WHT](https://arxiv.org/abs/2504.19874)

## Quick Start

```bash
docker build -t llama-cpp-rocm-tq .

docker run --rm --device /dev/kfd --device /dev/dri \
  --security-opt seccomp=unconfined \
  -v /path/to/model.gguf:/model.gguf:ro \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  llama-cpp-rocm-tq /model.gguf -ctk turbo3 -ctv turbo3 -ngl 99
```

## Build Notes

- Source: `TheTom/llama-cpp-turboquant` branch `feature/turboquant-kv-cache`
- Uses `HIPCXX`/`HIP_PATH` env vars (NOT explicit CMAKE_CXX_COMPILER)
- No NCCL — ROCm doesn't package it; multi-GPU uses HIP native comms
- `libopenblas0-pthread` for runtime (not meta-package `libopenblas0`)

## License

MIT (same as upstream llama.cpp).
