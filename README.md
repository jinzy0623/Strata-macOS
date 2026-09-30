# Strata macOS / Apple Silicon

基于 [Niko1221/Strata v0.1.13](https://github.com/Niko1221/Strata/releases/tag/v0.1.13)，
上游提交 `b89c989a7155e984e544ddd90d1038dda7da9e3d`。
这是首版 macOS 移植：保留 Strata Web UI、Python 服务端、OpenAI 兼容 API，
通过可插拔的 llama.cpp / ggml Metal 后端运行较小 GGUF 模型。

**M4 MacBook Air 16GB 无法运行上游原 125B 配置。** 上游专家权重等内存需求已经超过
16GB，SSD 并不能消除当前方案的常驻内存需求。请先用 0.5B–3B 的量化指令模型、
较短上下文验证；模型文件、KV cache、工作区和 macOS 会共同占用统一内存。

## 已实现的路径

```text
Strata Web UI / OpenAI client
          ↓
Python Service（现有排队、流式响应、设置、监控）
          ↓
Engine protocol + backend registry
          ├─ strata    → 原 CUDA 子进程（Windows / Linux，保留）
          ├─ metal     → llama-cpp-python → llama.cpp / ggml Metal（Apple Silicon）
          └─ ggml-cpu  → llama.cpp / ggml CPU（调试）
```

Metal 实现使用 GGUF 自带分词器和聊天模板，支持文本聊天、SSE 流式输出、采样、
停止 token 和请求间取消。ARM64 的 SIMD 与 Accelerate 由 ggml 提供；不会编译上游 AVX 内核。
这不是 CUDA kernel 的逐项翻译，原 SSD expert streaming、专家缓存和 MTP 推测解码
**尚未接入 Metal**。现阶段不支持 GGUF 工具调用和图像输入；工具请求会明确返回 400。
部分 GGUF 模板使用尚未支持的 Jinja 扩展，因此当前实测支持范围是 Qwen2.5 Instruct GGUF。

## 安装与启动

需要 Apple Silicon Mac、macOS 14+、原生 ARM64 Python 3.10+（建议 3.12）及
Apple Command Line Tools。终端不能运行在 Rosetta 下。

```bash
xcode-select --install   # 已安装时无需再执行
# 如果没有 Python，可从 python.org 安装支持 Apple Silicon 的版本

git clone https://github.com/jinzy0623/Strata-macOS.git
cd Strata-macOS
./setup-macos.sh
```

安装器创建本地 `.venv`，从固定版本 `llama-cpp-python==0.3.16` 源码构建 ggml Metal，
构建 Strata 的 ARM64 工具和文件 I/O 层，并运行平台测试。首次需要网络和编译时间。
可以用 `STRATA_PYTHON=/path/to/python3 ./setup-macos.sh` 指定 Python。
不会自动下载大模型，也不会安装 CUDA。

取得一个允许你使用的较小 GGUF 指令模型。例如本机验证采用
[Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF)
的 `qwen2.5-0.5b-instruct-q4_k_m.gguf`（约 469 MiB）。模型许可证由其发布方提供。

```bash
cp config/macos.example.json config/macos.json
# 编辑 config/macos.json，将 model 改为你的 GGUF 绝对路径
./run-macos.sh --config config/macos.json --open
```

默认网页：`http://127.0.0.1:8080/`，OpenAI base URL：`http://127.0.0.1:8080/v1`。
按 Ctrl+C 停止。配置中的 `context` 默认 4096；上下文越大，KV cache 占用越高。
模型文件大于当前可用内存的 80% 时会拒绝加载；这是基础检查，不能保证其他配置不耗尽内存。
默认使用 mmap，**不强制 mlock**，避免 macOS 锁页配额导致启动失败。

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-0.5b-instruct","messages":[{"role":"user","content":"你好"}],"max_tokens":128}'
```

需要 CPU 调试时：

```bash
.venv/bin/python -m serve.server --engine ggml-cpu --config config/macos.json --port 8080
```

网页保持上游设计。监控中的 NVIDIA GPU / VRAM 指标在 Mac 上显示为空；统一内存使用
CPU/RAM 监控，当前未实现 Apple GPU 温度、功率等读数。仅在需要时开放网络监听，
并通过 `STRATA_API_KEY` 设置 API key。

## 构建和验证

```bash
.venv/bin/cmake -S . -B build-macos -DSTRATA_ENABLE_CUDA=OFF
.venv/bin/cmake --build build-macos --parallel 4
.venv/bin/ctest --test-dir build-macos --output-on-failure
.venv/bin/python -m unittest serve.test_server tests_macos.test_backend -v
# 真实 Metal 推理验证（需要你自己的小型模型）
.venv/bin/python -m tests_macos.smoke_gguf /absolute/path/chat-model.gguf
```

本机 Apple M4 的实测记录见 [docs/MACOS-VALIDATION.md](docs/MACOS-VALIDATION.md)。
CI 构建与测试不下载模型，也不代表实际 GPU 推理已在 CI 验证。
CMake 在 Apple Silicon 下构建 `strata-gguf`、`strata-dequant`、`strata-plan` 和平台层；
Metal 推理库由 Python 安装器独立构建。原生 CUDA `strata` 可执行文件没有移植到 Mac。

## 来源与许可

保留全部上游源码、版权声明、`third_party/ggml/LICENSE`（MIT）、
`serve/web/fonts/OFL.txt`（SIL OFL）。上游 v0.1.13 快照未包含根目录 LICENSE，本移植保留上游随后公开的
MIT LICENSE 原文（来源提交 `a79080535d1b2a71a3419a0d97d8e7dca194b0f1`），
保留 Niko1221 和 Strata contributors 的版权归属。
来源和许可边界见 [ATTRIBUTION.md](ATTRIBUTION.md)。
上游说明完整保存在 [README.upstream.md](README.upstream.md)。

English: This is an initial Apple Silicon port with a working small-GGUF Metal
inference path, the original Python API and Web UI. It does not port Strata's
CUDA expert streaming or MTP features. A 16GB M4 cannot run the original 125B
configuration. Start with Qwen2.5 0.5B Instruct Q4_K_M and a short context.
