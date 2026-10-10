# How it works

This page explains what the patch changes inside llama.cpp, for people who want to understand, review or extend it.
The code is in [`patches/llama.cpp-b11518-moe-stream.patch`](../patches/llama.cpp-b11518-moe-stream.patch).
Most of it lives in two new files, `src/llama-moe-stream.{h,cpp}`.

## The problem in numbers

GLM-5.3-Flash (UD-IQ1_S) is 87 GB, of which 80 GB are routed experts: 43 MoE layers × 288 experts × ~7.5 MB.
Every token uses 8 experts per layer, which is **344 experts, ~2.5 GB of weights per token**. With 32 GB of RAM,
at most about a quarter of the experts can be in memory at once, so a share of those 2.5 GB comes from disk on
every token.

Stock llama.cpp memory-maps the file and touches the expert weights from the CPU. Every page that isn't cached
becomes a page fault served by a small read. Measured on Windows with a PCIe 4.0 NVMe: **0.7-1 GB/s**, no matter
how many threads compute, which comes out to about 2.3 tokens/s.

## The pieces

### 1. The streaming cache (from PR #25294)

With `--moe-stream-cache`, the expert tensors (`ffn_{gate,up,down}_exps`) of the CPU-side layers aren't loaded or
mapped. Each one gets a **cache tensor** of N slots instead (N = cache size / bytes per expert / layers), and the
file offsets are kept so that slots can be filled on demand.

In the compute graph, a small CPU **remap op** runs right after the router picks its top-8 experts. It:

- looks up which selected experts are already in a slot,
- picks victims for the missing ones (least "route-hot", with LRU as a tie-break) and queues their loads,
- waits until they're in, then
- rewrites the expert ids the matrix multiplications use into **slot ids**.

The router's choice and the expert weights are unchanged. Only the place they're read from is.

### 2. Disk reads that actually use the SSD (new for Windows)

Loads are split into 1 MB pieces and served by a pool of I/O threads (16 by default). On Windows the file is opened
with `FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING`, and every piece is a positional `ReadFile` with its own
`OVERLAPPED` offset.

- **Overlapped** matters because Windows serializes all I/O on a synchronous file handle, so 16 threads would still
  issue one read at a time. The patch this started from also put every read behind a global mutex on Windows.
- **No buffering** skips the Windows file cache. The patch manages its own cache, and a second copy in the OS cache
  would only take RAM away from it.

Result: 6-9 GB/s while a token waits for misses.

### 3. Next-layer prefetch (from xMytze's port)

While layer L's experts are computing, a tiny op applies **layer L+1's router to layer L's input** and starts
loading the top-4 guesses. The residual stream changes little between layers, so ~80% of these guesses are
used. Wrong guesses cost bandwidth and a cache slot, never correctness.

### 4. The VRAM tier (new)

The 6 CPU cores are slow at decoding IQ2_XXS / IQ3_XXS weights, about as slow as the disk. So the hottest experts
of each layer are moved into a **bank tensor in VRAM**, and those are computed by the GPU instead:

- When a selected expert is far hotter than the coldest expert in the bank, the remap op uploads it there and
  **frees its RAM slot**. Each expert is in exactly one tier, so the bank adds capacity instead of duplicating it.
  A 1.5× hotness margin keeps two similar experts from swapping places every token.
- In the graph, each streamed layer then has two expert branches: CPU experts from the RAM slots and GPU experts from
  the bank. Each side gets ids of `-1` for the pairs the other side computes:
  - the CPU `mul_mat_id` writes zeros for `-1` (small change in `ggml-cpu.c`),
  - the Vulkan mat-vec shaders skip the work for `-1` and write zeros (change in `mul_mat_vec_base.glsl` and the
    IQ shaders),
  - the two results are added before the routing weights are applied.
- **Graph order matters.** Every CPU↔GPU hand-off costs a device sync (~0.6 ms on Windows). The ops are ordered so
  each layer keeps the stock pattern of one GPU→CPU→GPU round trip. A first version had three and lost the whole
  gain to synchronization.

With 4 GB of bank, the GPU serves 25-40% of expert uses once warm.

### 5. Prompt processing: full-layer staging (from xMytze's port, plus a fix)

A batch of 32+ tokens touches nearly every expert of every layer, so caching doesn't help. Instead, the whole
layer's experts are read with large sequential reads into one of two host buffers (~2.6 GB each), the next layer is
read into the other buffer while this one computes, and llama.cpp's op-offload copies the needed experts to the GPU.
That's 81 tokens/s instead of 11.

New here: the buffers are allocated as whole pages and **decommitted as soon as the next answer token is
generated**, so their 5 GB goes back to the expert cache during chat instead of sitting idle.

Batches of 2-31 tokens use a multi-wave version of the cache path. This project fixed two crashes there: a ubatch
routed to fewer distinct experts than the slots each wave reserves, and parking slots reused across ubatches after
being evicted.

## What's left

Per token on the test PC: ~90 ms waiting for the disk, ~90 ms CPU math on the RAM-tier experts, ~60 ms GPU
(attention, recurrent state, VRAM-tier experts). Ideas, roughly in order of value:

- **Faster CPU kernels for IQ2_XXS/IQ3_XXS** (repacking, AVX-512 VNNI paths), which would also shift work toward a
  bigger RAM tier.
- **Faster Vulkan IQ mat-vec kernels.** The IQ3_XXS `mul_mat_vec_id` reaches ~35 GB/s on an RX 7800 XT, a fraction
  of its bandwidth.
- **Overlap the GPU and CPU expert branches** instead of running them back to back.
- **Smarter eviction** using the prefetch predictions.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `LLAMA_MOE_STREAM_GPU_MIB` | 0 | VRAM tier size in MiB |
| `LLAMA_MOE_STREAM_GPU_PROMOTE` | 2 | max experts moved into the VRAM tier per layer per token |
| `LLAMA_MOE_STREAM_PREFETCH` | 0 | next-layer experts to prefetch per token |
| `LLAMA_MOE_STREAM_STATS` | off | log hit rate / disk wait / tier share every 64 tokens |
| `LLAMA_MOE_STREAM_CHUNK_KB` | 1024 | piece size of demand loads |
| `LLAMA_MOE_STREAM_STAGE` | 1 | full-layer staging for prompts (0 is ignored on Vulkan, see README) |
| `GGML_SCHED_TIMING` | off | `N`: log per-backend time per graph every N graphs |

The command-line flags (`--moe-stream-cache`, `--moe-stream-direct`, `--moe-stream-io-threads`) are listed in
`llama-server --help`.
