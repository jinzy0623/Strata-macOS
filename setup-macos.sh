#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "Requires Apple Silicon macOS in a native ARM64 terminal (no Rosetta)." >&2
  exit 1
fi
xcrun --find clang++ >/dev/null || { echo "Install Apple's tools: xcode-select --install" >&2; exit 1; }
PYTHON="${STRATA_PYTHON:-python3}"
"$PYTHON" -c 'import platform, sys; assert platform.machine() == "arm64", "Use ARM64 Python"; assert sys.version_info >= (3, 10), "Python 3.10+ required"'
if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON" -m venv .venv
fi
.venv/bin/python -c 'import platform; assert platform.machine() == "arm64", "Existing venv must use ARM64 Python"'
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install --upgrade pip
# Force a source build: a cached CPU wheel must not silently replace Metal.
CMAKE_ARGS='-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_OSX_ARCHITECTURES=arm64' \
CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}" \
  .venv/bin/python -m pip install --force-reinstall --no-cache-dir --no-binary llama-cpp-python -r requirements-macos.txt
.venv/bin/cmake -S . -B build-macos -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=OFF
.venv/bin/cmake --build build-macos --parallel 4
.venv/bin/ctest --test-dir build-macos --output-on-failure
.venv/bin/python -c 'from llama_cpp import llama_cpp; assert llama_cpp.llama_supports_gpu_offload(), "Metal build missing"'
echo 'Ready. Copy config/macos.example.json, set model to a small local GGUF, then run:'
echo './run-macos.sh --config config/macos.json --open'
