#!/usr/bin/env bash
# Build llama-server with MoE expert streaming on Linux (Vulkan backend).
#
# Needs: git, cmake, a C++17 compiler, the Vulkan headers + loader and glslc. On Debian/Ubuntu:
#     sudo apt install git cmake build-essential libvulkan-dev glslc
# (or the LunarG Vulkan SDK). Then:
#     bash scripts/build-linux.sh
#
# Status: the Linux I/O path (pread + O_DIRECT) comes from the original CUDA/Linux patch and compiles, but this
# Vulkan build was only benchmarked on Windows. Reports welcome.
set -euo pipefail

LLAMA_TAG=b11518
here=$(cd "$(dirname "$0")/.." && pwd)
src=${SRC_DIR:-$here/llama.cpp}
patch=$here/patches/llama.cpp-$LLAMA_TAG-moe-stream.patch

if [ ! -d "$src/.git" ]; then
    git clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp "$src"
fi

cd "$src"
if git -c core.autocrlf=false apply --check "$patch" 2>/dev/null; then
    git -c core.autocrlf=false apply "$patch"
elif ! git -c core.autocrlf=false apply --reverse --check "$patch" 2>/dev/null; then
    echo "the patch does not apply to $src - it needs a clean llama.cpp $LLAMA_TAG checkout" >&2
    exit 1
fi

# -ffile-prefix-map: no local folder names inside the binaries (source paths appear in assert messages)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_NATIVE=ON \
    -DCMAKE_C_FLAGS="-ffile-prefix-map=$PWD=llama.cpp" -DCMAKE_CXX_FLAGS="-ffile-prefix-map=$PWD=llama.cpp" \
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --config Release -j "$(nproc)" --target llama-server llama-cli llama-bench

echo
echo "done: $src/build/bin/llama-server"
