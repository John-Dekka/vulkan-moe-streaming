#!/usr/bin/env bash
# Build llama-server with MoE expert streaming for Windows + Vulkan (AMD, Intel or NVIDIA GPUs).
#
# Run it inside an MSYS2 "UCRT64" shell (https://www.msys2.org):
#     bash scripts/build-windows.sh            # portable build (any x86-64 CPU with AVX2)
#     NATIVE=1 bash scripts/build-windows.sh   # tuned for this PC's CPU (a bit faster, not portable)
#
# Result: llama.cpp/build/bin/llama-server.exe - one self-contained exe (needs only the GPU driver).
set -euo pipefail

LLAMA_TAG=b11518
here=$(cd "$(dirname "$0")/.." && pwd)
src=${SRC_DIR:-$here/llama.cpp}
patch=$here/patches/llama.cpp-$LLAMA_TAG-moe-stream.patch

if [ "${MSYSTEM:-}" != "UCRT64" ]; then
    echo "please run this from an MSYS2 UCRT64 shell (MSYSTEM is '${MSYSTEM:-unset}')" >&2
    exit 1
fi

echo "== installing the build tools (gcc, cmake, ninja, Vulkan headers, shader compiler)"
pacman -S --noconfirm --needed git \
    mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja \
    mingw-w64-ucrt-x86_64-vulkan-devel mingw-w64-ucrt-x86_64-shaderc mingw-w64-ucrt-x86_64-spirv-headers

if [ ! -d "$src/.git" ]; then
    echo "== downloading llama.cpp $LLAMA_TAG"
    git -c core.autocrlf=false clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp "$src"
fi

cd "$src"
if git -c core.autocrlf=false apply --check "$patch" 2>/dev/null; then
    echo "== applying the streaming patch"
    git -c core.autocrlf=false apply "$patch"
elif git -c core.autocrlf=false apply --reverse --check "$patch" 2>/dev/null; then
    echo "== streaming patch already applied"
else
    echo "the patch does not apply to $src - it needs a clean llama.cpp $LLAMA_TAG checkout" >&2
    exit 1
fi

if [ -n "${NATIVE:-}" ]; then
    cpu_flags=(-DGGML_NATIVE=ON)
else
    cpu_flags=(-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON)
fi

echo "== building (10-20 minutes, most of it compiling Vulkan shaders)"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON "${cpu_flags[@]}" -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DCMAKE_EXE_LINKER_FLAGS="-static-libgcc -static-libstdc++ -static"
cmake --build build --config Release -j "$(nproc)" --target llama-server llama-cli llama-bench

echo
echo "done: $src/build/bin/llama-server.exe"
