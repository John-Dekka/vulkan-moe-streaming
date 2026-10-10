#!/usr/bin/env bash
# Crash/garbage test for the streaming build: sends prompts from 1 to ~1300 tokens (every batch path: per-token
# cache, multi-wave, full-layer staging), each followed by a short answer, and prints speed + the start of each answer.
#
#   MODEL=/path/to/model-00001-of-0000N.gguf bash scripts/stress-test.sh [extra llama-server args]
#
# SERVER, CACHE_GB, GPU_TIER_MB and PORT work like in run-server.sh. Answers should be readable text; "????" or
# repeated symbols mean wrong results.
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
: "${MODEL:?set MODEL to the first .gguf file of the model}"
SERVER=${SERVER:-$here/llama.cpp/build/bin/llama-server}
[ -x "$SERVER" ] || SERVER=$here/llama.cpp/build/bin/llama-server.exe
PORT=${PORT:-8095}
log=${LOG:-stress-test.log}

export LLAMA_MOE_STREAM_GPU_MIB=${GPU_TIER_MB:-4000}
export LLAMA_MOE_STREAM_PREFETCH=${PREFETCH:-4}

"$SERVER" -m "$MODEL" --host 127.0.0.1 --port "$PORT" -ngl 999 -c 8192 -fa on --no-warmup --parallel 1 \
    -b 2048 -ub 2048 --no-webui --cpu-moe --load-mode none \
    --moe-stream-cache "${CACHE_GB:-12}" --moe-stream-direct --moe-stream-io-threads 16 "$@" > "$log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
    curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok && break
    kill -0 $pid 2>/dev/null || { echo "server exited while loading, see $log"; exit 1; }
    sleep 1
done

py=$(command -v python3 || command -v python)
words="the river carried small boats past the old mill while children counted them from the bridge and laughed"
show='import sys, json
j = json.load(sys.stdin); t = j["timings"]
print("prompt %5d tok at %6.1f tok/s, answer %.2f tok/s | %r" % (t["prompt_n"], t["prompt_per_second"], t["predicted_per_second"], j["content"][:60]))'
for n in 1 3 12 25 40 90 300 1200 6 60; do
    prompt=$("$py" -c "w='$words'.split(); print(' '.join(w[i % len(w)] + ('' if i % 7 else ',') for i in range($n)))")
    resp=$(curl -s -m 900 "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
        -d"{\"prompt\":\"$prompt\",\"n_predict\":16,\"cache_prompt\":false}") || true
    if ! kill -0 $pid 2>/dev/null; then
        echo "~$n words: SERVER CRASHED"; grep -E 'ABORT|ASSERT|error' "$log" | head -3; exit 1
    fi
    echo "$resp" | "$py" -c "$show"
done
echo "all prompt sizes passed"
