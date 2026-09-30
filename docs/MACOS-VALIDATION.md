# Apple Silicon validation — 2026-09-30

Tested locally on Apple M4 / ARM64 macOS with native Python 3.12.12,
AppleClang 21, llama-cpp-python 0.3.16 compiled from source with Metal enabled.

| Check | Result |
| --- | --- |
| CMake portable artifact tools + Darwin platform library | Built successfully |
| Native platform test (aligned read, data equality, short EOF, bad alignment, missing file, lock failure) | Passed |
| Original API tests (`serve.test_server`) | 17 passed |
| Backend tests (`tests_macos.test_backend`) | 6 passed |
| Real GGUF: Qwen2.5-0.5B-Instruct Q4_K_M, context 1024 | Loaded on Apple M4 Metal |
| Original Web UI assets, health, metrics and model list over HTTP | 200 OK |
| OpenAI nonstream generation | “Hello! It's a pleasure to meet you!” |
| OpenAI SSE generation | Content chunks and `[DONE]` verified |
| Repeated request after a length-limited generation | Passed, output stayed within 2-token limit |
| Full setup-macos.sh (source compilation, install, native build and ctest) | Passed |
| Installer and launcher shell syntax | Passed |

The llama.cpp load log explicitly reported device **Metal (Apple M4)** and
assigned model layers to Metal. This verifies actual GPU inference, beyond just
checking whether the library was compiled with GPU support.

This is a functional smoke check, not a performance benchmark. No 125B model,
SSD expert streaming, MTP, arbitrary tool schema, image encoder or CUDA
regression run was performed. The original CUDA code remains present; its
Windows/Linux behavior has not been rebuilt on this Mac.

The GitHub workflow performs compilation and the synthetic API/backend tests.
It intentionally does not download model weights and should not be interpreted
as a real-device GPU smoke test.

Reproduce the real-model check:

```bash
.venv/bin/python -m tests_macos.smoke_gguf /path/to/chat-model.gguf
```
