#!/usr/bin/env bash
# Start llama-server with MoE expert streaming (Linux / MSYS2).
#
#   MODEL=/models/GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf bash scripts/run-server.sh
#
# Settings via environment (defaults are for a 16 GB GPU and 32 GB of RAM, see the README "Tuning" section):
#   CACHE_GB=12  GPU_TIER_MB=4000  PREFETCH=4  IO_THREADS=16  THREADS=<physical cores>  CTX=16384  PORT=8080
# Extra llama-server arguments can be appended: bash scripts/run-server.sh --device Vulkan0
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
: "${MODEL:?set MODEL to the first .gguf file of the model}"
SERVER=${SERVER:-$here/llama.cpp/build/bin/llama-server}
[ -x "$SERVER" ] || SERVER=$here/llama.cpp/build/bin/llama-server.exe
THREADS=${THREADS:-$(lscpu -p=core 2>/dev/null | grep -v '^#' | sort -u | wc -l || nproc)}

export LLAMA_MOE_STREAM_GPU_MIB=${GPU_TIER_MB:-4000}
export LLAMA_MOE_STREAM_PREFETCH=${PREFETCH:-4}

exec "$SERVER" -m "$MODEL" --host 127.0.0.1 --port "${PORT:-8080}" \
    -ngl 999 -t "$THREADS" -c "${CTX:-16384}" -fa on --jinja -b 2048 -ub 2048 --parallel 1 \
    --cpu-moe --load-mode none \
    --moe-stream-cache "${CACHE_GB:-12}" --moe-stream-direct --moe-stream-io-threads "${IO_THREADS:-16}" \
    "$@"
