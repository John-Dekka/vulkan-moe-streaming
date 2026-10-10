# Vulkan MoE Streaming for llama.cpp

**Run Mixture-of-Experts models that are several times bigger than your RAM on an ordinary gaming PC, about twice as fast as stock llama.cpp, on AMD (or any Vulkan) graphics cards.**

On a PC with an RX 7800 XT (16 GB), a 6-core Ryzen and 32 GB of RAM, the 87 GB GLM-5.3-Flash model went from:

|                        | Stock llama.cpp | This project | |
|------------------------|----------------:|-------------:|---|
| Writing the answer     | 2.3 tokens/s    | **4.5 tokens/s** | 1.9× faster |
| Reading a long prompt  | 11 tokens/s     | **81 tokens/s**  | 7× faster |
| "Why is the sky blue?" | 54 s            | 32 s | |

Same model, same quantization, same machine. Nothing is approximated: the model computes exactly the experts it
chooses, they just arrive faster.

---

## Who this is for

You'll get the most out of it if:

- you want to run a **huge MoE model** (GLM-5.x, DeepSeek-class, Kimi, Qwen-MoE...) whose file is **2-4× your RAM**,
- you have **one consumer GPU** with 12-24 GB of VRAM, **especially AMD** (Radeon RX 6000/7000/9000), where the
  CUDA-only tricks for this don't work. Intel Arc and NVIDIA through Vulkan should work too, but are untested,
- the model sits on a **fast NVMe SSD** (PCIe 4.0 or newer; the more GB/s, the better),
- you're on **Windows** (tested) or Linux (builds, not benchmarked yet).

It is **not** for you if:

- the model fits in your RAM. Stock llama.cpp is already as fast as your hardware allows then.
- the model is on a hard disk or a SATA SSD. Streaming needs several GB/s.
- you need many users at once. One conversation at a time.
- you have an NVIDIA card on Linux. The CUDA version this builds on
  ([xMytze/glm53-ssd-streaming](https://github.com/xMytze/glm53-ssd-streaming)) is the better fit.
- you use a Mac. Look at [llama.cpp PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294).

## Why stock llama.cpp is slow here

A MoE model has hundreds of "experts" per layer but uses only a few (8 of 288 for GLM-5.3) for each token. When the
model doesn't fit in RAM, llama.cpp memory-maps the file and lets the operating system fetch experts from disk when
they're touched. On Windows that happens a few kilobytes per page fault, one after another: about **1 GB/s from an
SSD that can do 7 GB/s**. The CPU spends most of every token waiting for it.

## What this project does

It's a patch for llama.cpp that takes expert loading away from the operating system:

1. **Its own expert cache in RAM.** The experts used most often stay in memory (12 GB by default), and the
   rarely-used ones are evicted. Windows' file cache no longer holds a second copy.
2. **Fast, parallel disk reads.** Missing experts are read with 16 simultaneous unbuffered reads, at **6-9 GB/s**
   instead of ~1.
3. **Look-ahead.** While a layer is computing, the next layer's router guesses which experts it will want (it's right
   ~80% of the time), and those start loading early.
4. **A VRAM tier.** The hottest experts move into spare GPU memory (4 GB by default), and the GPU computes them while
   the CPU does the rest. Each expert lives in exactly one place, so the tier adds to the cache.
5. **Fast prompt reading.** Long prompts read each layer in big sequential chunks and the GPU computes them. The
   5 GB of buffers this needs goes back to Windows as soon as the answer starts.

More detail: [docs/how-it-works.md](docs/how-it-works.md). All measurements: [docs/benchmarks.md](docs/benchmarks.md).

---

## Quick start (Windows)

### 1. Get the program

**Option A: download.** Grab `llama-server-moe-stream-win-x64-vulkan.zip` from the
[Releases page](../../releases) and unzip it into this folder (it makes `bin\`). It runs on any 64-bit CPU with
AVX2 (Intel 4th gen / AMD Ryzen and newer) and any GPU with a current Vulkan driver. It's a single exe with no installer. The
benchmarks above used a build tuned for the test CPU, and the download is a little slower (about 40% slower
prompt reading in one test on a Zen 4 CPU, similar answer speed).

**Option B: build it yourself (about 20 minutes).** Install [MSYS2](https://www.msys2.org), open the
**"MSYS2 UCRT64"** terminal, and run:

```bash
cd /c/path/to/this/repo
bash scripts/build-windows.sh
```

It installs the compilers, downloads llama.cpp release b11518, applies the patch and builds
`llama.cpp/build/bin/llama-server.exe`. Use `NATIVE=1 bash scripts/build-windows.sh` for a build tuned to your CPU.

### 2. Get a model

For example GLM-5.3-Flash from [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF).
The `UD-IQ1_S` folder (87 GB, three files) is what was tested. Put it on your fastest NVMe drive.

### 3. Run it

Close browsers, Discord and games first, because free RAM becomes expert cache. Then, in PowerShell:

```powershell
.\scripts\run-server.ps1 -Model D:\models\GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf
```

Loading takes about 10 seconds. Then chat at **http://127.0.0.1:8080/**, or point any OpenAI-compatible app
(Open WebUI, SillyTavern, Continue, ...) at `http://127.0.0.1:8080/v1`.

If your PC also has integrated graphics, add `-Device Vulkan0` (llama.cpp lists the devices when it starts).

## Quick start (Linux)

```bash
sudo apt install git cmake build-essential libvulkan-dev glslc   # or the LunarG Vulkan SDK
bash scripts/build-linux.sh
MODEL=/models/GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf bash scripts/run-server.sh
```

---

## Tuning for your machine

The defaults are for **16 GB VRAM + 32 GB RAM**. Pass these to `run-server.ps1` (or as variables to `run-server.sh`):

| Setting | Default | What it is | How to choose |
|---|---|---|---|
| `-CacheGB` / `CACHE_GB` | 12 | RAM for the expert cache | About **free RAM − 7 GB**. Prompts briefly need ~5 GB more, and if that doesn't fit Windows starts paging and everything crawls. 64 GB PC: try 40. |
| `-GpuTierMB` / `GPU_TIER_MB` | 4000 | VRAM for the hottest experts | Whatever VRAM is left after the model loads, minus ~0.5 GB. If prompt reading suddenly gets 3× slower, it's too big (Windows quietly moves overflow VRAM into system RAM). 24 GB card: try 10000. 12 GB card: try 1000 or 0. |
| `-Prefetch` / `PREFETCH` | 4 | Next-layer experts loaded early | 4 is good; 0 turns look-ahead off. |
| `-IoThreads` / `IO_THREADS` | 16 | Parallel disk reads | 16 suits a PCIe 4.0 NVMe. Slower drives need fewer. |
| `-Threads` / `THREADS` | physical cores | CPU threads | Leave it at the number of physical cores. |

Add `LLAMA_MOE_STREAM_STATS=1` to the environment to get a line every 64 tokens with cache hit rate, disk wait per
token and how much the VRAM tier serves.

**More RAM is the biggest upgrade.** On the test PC each token still spends ~90 ms waiting for the disk, ~90 ms on
CPU math and ~60 ms on the GPU. A bigger cache shrinks the first part.

## Known limitations

- **Tested on one machine and one model** (RX 7800 XT, Ryzen 5 7600X, GLM-5.3-Flash UD-IQ1_S, Windows 11). Other MoE
  models in llama.cpp should work since nothing is GLM-specific, but they haven't been tried. Please open an issue with
  your numbers, good or bad.
- **Linux builds but hasn't been benchmarked** with this Vulkan version.
- **One conversation at a time.** Two contexts on the same streamed model would evict each other's experts.
- **Short prompts** of a few words take ~5 s before the answer starts. Every layer has to fetch nearly every expert
  the words touch, and that dominates.
- **With the VRAM tier on, the exact wording of long answers can drift** from stock llama.cpp, because GPU and CPU
  round slightly differently. The model's quality is the same. Set `-GpuTierMB 0` for byte-identical output.
- `LLAMA_MOE_STREAM_STAGE=0` (prompts without big-chunk reading) gives wrong output with Vulkan, so the build ignores it.

## How this was made, and credits

This builds on other people's work:

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (MIT), the inference engine.
- **[llama.cpp PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294)** by freedomljc, the original idea
  and design for streaming MoE experts from disk (unmerged at the time of writing).
- **[xMytze/glm53-ssd-streaming](https://github.com/xMytze/glm53-ssd-streaming)** (MIT), that PR ported to
  GLM-5.3 with prefill staging and decode prefetch for CUDA on Linux. This project started from their patch.

What this project added: Windows I/O (the earlier code put every Windows read behind a single lock), the Vulkan
port, the VRAM tier and the matching Vulkan shader change, returning prompt buffers to the OS, two crash fixes in
the multi-wave prompt path, and the benchmarking.

**AI disclosure:** most of the code was written with an AI coding assistant (Claude), as was a large part of the
patch it builds on. It was tested as described in [docs/benchmarks.md](docs/benchmarks.md), but treat it as
**experimental**. This is not an official llama.cpp feature and isn't submitted upstream.

## License

MIT, see [LICENSE](LICENSE). The patch modifies llama.cpp, which is MIT-licensed by the ggml authors. Their notice
is in [LICENSE-llama.cpp](LICENSE-llama.cpp).
