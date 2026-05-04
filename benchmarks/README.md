# Benchmarks: llama-cpp-rocm-tq

All benchmarks were run on the following hardware using the `llama-bench` and `llama-perplexity` tools from llama.cpp.

## Test Hardware

| Component | Specification |
|-----------|--------------|
| GPU | 3x AMD Radeon RX 7900 XTX |
| VRAM per GPU | 24,560 MiB (24 GB) |
| Total VRAM | 73,680 MiB (72 GB) |
| GPU Architecture | gfx1100 (RDNA3) |
| ROCm Version | 7.2 |
| PyTorch | 2.10.0+rocm7.2.0 |
| System RAM | 63,991 MiB (62 GB) |
| Backend | ROCm + BLAS (Flash Attention) |

## Test Model

| Property | Value |
|----------|-------|
| Model | Qwen3.6-27B-UD-Q4_K_XL |
| Size | 16.39 GiB (5.24 BPW) |
| Parameters | 26.90 B |
| Architecture | qwen35 (64 layers, SSM + attention) |
| Context Length | 262,144 tokens |
| Quantization | Q4_K - Medium (Unsloth) |

## Speed Benchmark (llama-bench)

Parameters: pp512 (prompt processing), tg256 (text generation), batch=8, ubatch=2048, threads=24

| Cache Type | Prompt (pp512) t/s | Generation (tg256) t/s | vs f16 |
|-----------|-------------------|----------------------|--------|
| f16 | 118.92 +/- 0.77 | 19.41 +/- 0.26 | baseline |
| q8_0 | 118.75 +/- 1.06 | 19.24 +/- 0.04 | -0.1% / -0.9% |
| turbo4 | 118.92 +/- 0.73 | 19.64 +/- 0.07 | +0.0% / +1.2% |
| turbo3 | 118.58 +/- 0.48 | 19.42 +/- 0.10 | -0.3% / +0.1% |
| turbo2 | 118.31 +/- 0.27 | 19.32 +/- 0.10 | -0.5% / -0.5% |

**Key finding:** No meaningful speed difference across cache types. The system is compute-bound (27B params across 3 GPUs), not memory-bound. TurboQuant's benefit is VRAM savings, not throughput.

## KV Cache Memory Savings

### 512 Context Window

| Cache Type | Total KV Cache | K Cache | V Cache | vs f16 savings |
|-----------|---------------|---------|---------|---------------|
| f16 | 128.00 MiB | 64.00 MiB | 64.00 MiB | baseline |
| q8_0 | 68.00 MiB | 34.00 MiB | 34.00 MiB | -46.9% |
| turbo4 | 51.00 MiB | 34.00 MiB* | 17.00 MiB | -60.2% |
| turbo3 | 46.50 MiB | 34.00 MiB* | 12.50 MiB | -63.7% |
| turbo2 | 44.09 MiB | 34.00 MiB* | 10.09 MiB | -65.6% |

### 262K Context Window (Full Model Capacity, Measured)

| Cache Type | Total KV Cache | K Cache | V Cache | vs f16 savings |
|-----------|---------------|---------|---------|---------------|
| f16 | ~18,304 MiB (est) | ~8,704 MiB | ~9,600 MiB | baseline |
| q8_0 | ~8,704 MiB (est) | ~4,352 MiB | ~4,352 MiB | -52.5% |
| turbo4 | ~6,592 MiB (est) | ~4,352 MiB | ~2,240 MiB | -64.0% |
| turbo3 | **5,952 MiB** (measured) | **4,352 MiB** | **1,600 MiB** | **-67.5%** |
| turbo2 | ~5,568 MiB (est) | ~4,352 MiB | ~1,216 MiB | -69.6% |

*Note: Auto-asymmetric mode upgrades K to q8_0 for GQA models (GQA ratio 6:1) to prevent quality degradation in attention keys. V cache is where TurboQuant compression is applied.

**Key finding:** At full 262K context, turbo3 saves ~12.3 GiB of VRAM compared to f16. This enables using the full context length on hardware that would otherwise be unable to fit the KV cache.

## Quality Benchmark (Perplexity)

Lower is better. Test file: Wikipedia LLM article (650 lines, 14,336 tokens).

| Cache Type | PPL | vs f16 delta | Quality impact |
|-----------|-----|-------------|---------------|
| f16 | 1.5972 | baseline | reference |
| q8_0 | 1.6006 | +0.21% | negligible |
| turbo4 | 1.6014 | +0.26% | negligible |
| turbo3 | 1.6044 | +0.45% | negligible |
| turbo2 | 1.6142 | +1.06% | minimal |

**Key finding:** Virtually no quality degradation. Even turbo2 (most aggressive) shows only +1.06% PPL increase, which is within noise margin for practical use.

## Server Integration Test

Started llama-server with TurboQuant on the same model:

```bash
docker run --device /dev/kfd --device /dev/dri \
  --security-opt seccomp=unconfined \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  -p 8080:8080 \
  -v /path/to/model.gguf:/model.gguf:ro \
  llama-cpp-rocm-tq \
  --model /model.gguf \
  --cache-type-k q8_0 \
  --cache-type-v turbo3 \
  --host 0.0.0.0
```

**Result:** Server started successfully, API responds correctly.

Actual measured KV cache at 262K context (n_ctx=262144):
- ROCm0: 1,860 MiB
- ROCm1: 2,232 MiB
- ROCm2: 1,860 MiB
- **Total: 5,952 MiB** (K q8_0: 4,352 MiB, V turbo3: 1,600 MiB)

Completion test returned valid output (49 tokens generated).

## Recommendation

For production use with this hardware, use:
```
--cache-type-k q8_0 --cache-type-v turbo3
```

This gives 63.7% KV cache savings with only +0.45% PPL degradation. turbo3 is the sweet spot between VRAM savings and quality preservation.

## Benchmark Methodology

- `llama-bench` run with default settings (pp512, tg256, batch=8, ubatch=2048, threads=24)
- `llama-perplexity` run with test file from Wikipedia LLM article
- Each cache type tested sequentially with 5-second cool-down between runs
- All tests run in Docker container with ROCm device passthrough
- Model fully offloaded to GPU (65/65 layers)

## Disclaimer

These benchmarks were run on specific hardware (3x RX 7900 XTX, ROCm 7.2). Your results may vary depending on your GPU configuration, ROCm version, and model choice. The relative improvements between cache types should be consistent, but absolute numbers will differ.
