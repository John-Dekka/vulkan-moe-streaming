# Benchmarks

## Test machine

| Part | |
|---|---|
| GPU | AMD Radeon RX 7800 XT, 16 GB (Vulkan) |
| CPU | AMD Ryzen 5 7600X, 6 cores (integrated graphics not used) |
| RAM | 32 GB DDR5-5200, of which ~14-22 GB was free during the runs |
| Disk | WD_BLACK SN7100 2 TB, PCIe 4.0 NVMe |
| OS | Windows 11 Pro |
| Model | GLM-5.3-Flash, Unsloth `UD-IQ1_S` (87 GB; experts IQ1_S / IQ2_XXS / IQ3_XXS / IQ4_XS) |
| llama.cpp | release b11518 (official Vulkan build for "stock"; same tag + this patch, built with `NATIVE=1`, otherwise) |

Common settings: `-ngl 999 -t 6 -c 8192 -fa on -b 2048 -ub 2048 --parallel 1 --device Vulkan0`.

## Method

Each configuration starts a fresh server, warms up with a 32-token reply, reads a 1186-token prompt, then generates
128 tokens twice, once for a short story and once for a coding prompt. The decode number is the average of the two
128-token runs. Each run starts from a cold cache, so the numbers include warming up. Runs on the same day vary by
about ±0.2 tokens/s.

## Main results (2026-10-10)

| Configuration | Answer (tok/s) | Prompt (tok/s) | Notes |
|---|---:|---:|---|
| Stock b11518, mmap, `--n-cpu-moe 44` (last expert layer in VRAM) | 2.34 | 11.3 | best stock setup found |
| Streaming, 12 GB RAM cache, one expert layer in VRAM, no VRAM tier | 3.70 | 85.8 | output identical to stock |
| **Streaming, 12 GB RAM cache, 4 GB VRAM tier, all layers streamed** | **4.52** | **81.0** | default settings |
| Streaming, 12 GB cache, 6.2 GB VRAM tier | 4.33 | 28.1 | VRAM overflows into system RAM |
| Streaming, 12 GB cache, 3 GB tier + one full expert layer in VRAM | 4.01 | 29.2 | same overflow |

Chat self-test through the OpenAI endpoint (thinking off):

| Question | Stock | Streaming |
|---|---|---|
| "In one sentence: why is the sky blue?" | 54 s | 32 s |
| "Write a Python function is_prime(n) and nothing else." | 139 s | 34 s |

## Earlier and smaller experiments

| Experiment | Result |
|---|---|
| Stock, all experts on CPU (`--cpu-moe`) | 1.7 tok/s |
| Stock, llama.cpp's VRAM expert cache (`--moe-cache-mib 6486`) | 0.9 tok/s (every miss is paged in through mmap, then copied) |
| Stock, model's own draft layer (MTP) | 1.8-2.1 tok/s |
| Stock, 12 / 18 / 24 threads | no gain over 6 |
| Stock + `PrefetchVirtualMemory` hint before each expert matmul | no gain |
| Streaming, cache 12 GB vs 14 GB (with ~14 GB free) | 14 GB slower (paging) |
| Streaming, prefetch 0 / 4 / 8 experts | 4 and 8 the same, ~5-10% over 0 |
| VRAM tier, first version (3 CPU↔GPU hand-offs per layer) | no gain: the extra syncs ate it |
| VRAM tier with one hand-off per layer, before the shader skip | +5-10% |
| VRAM tier with the Vulkan shader skipping CPU-side pairs | +20% over no tier |

The portable release build (AVX2 only) passed the same crash test. It read the 1372-token test prompt at 67 tok/s
vs 94 tok/s for the native build, and answer speed was similar (3.4-4.7 tok/s).

## Where the time goes (default settings, warm)

From `LLAMA_MOE_STREAM_STATS=1` and `GGML_SCHED_TIMING=64`:

| | per token |
|---|---|
| Waiting for disk (misses) | ~85-100 ms |
| CPU expert math (RAM tier) | ~90 ms |
| GPU: attention, recurrent state, VRAM-tier experts | ~60-70 ms |
| RAM cache hit rate | 65-70% |
| Share of expert uses served by the VRAM tier | 25-40% |
| Disk bytes per token | 0.8-1.2 GB |

## Reproducing

`scripts/run-server.ps1` starts the default configuration. For quick numbers, send a prompt to `/completion` with
`n_predict` 128 and read `timings.predicted_per_second` in the response. Add `LLAMA_MOE_STREAM_STATS=1` for the
cache statistics.
