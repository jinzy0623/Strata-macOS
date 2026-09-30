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

## 一体化安装（推荐）

下载本仓库 ZIP，解压后双击 **`install-macos.command`**。也可以在终端运行：

```bash
git clone https://github.com/jinzy0623/Strata-macOS.git
cd Strata-macOS
./setup-macos.sh
```

**无需手动选模型、下载权重或编辑配置。** 工具会完成：

1. 检测芯片、原生 ARM64、统一内存、当前可用内存与磁盘空间。
2. 安装 Python/Metal 运行环境，构建 ARM64 工具并运行代码测试。
3. 从固定版本、经过兼容限定的 Qwen2.5 Instruct Q4_K_M 目录中筛选模型，
   下载全部 GGUF 分片并逐个校验 SHA256；下载中断可以继续。
4. 在本机真正运行候选模型，测量短/长提示词下的首字等待、持续生成速度、
   内存余量和交换内存增长。候选不达标时自动降一档。
5. 部署本机后台服务，再验证网页、普通/流式 API、中文和基础算术生成。
   **只有全部通过才显示安装完成**，随后打开聊天网页。

需要 Apple Silicon、macOS 14+ 和 Apple Command Line Tools。如果尚未安装编译工具，
安装器会打开 Apple 的安装窗口；完成后再次打开安装器即可继续。
缺少合适 Python 时使用 Astral uv 安装 Python 3.12。全部操作无需 sudo。

默认采用 **均衡模式**：在能安全放入内存的候选中，选择通过本机响应门槛的最大模型。
目录含 1.5B、3B、7B、14B、32B；**0.5B 不作为日常推荐**。
“最适合”是此兼容目录与所选模式内的本机推荐，不代表对所有模型的质量排名。

| 设备状态示例 | 容量筛选（仍需实测） |
| --- | --- |
| 16GB，当前可用 12GB，4K 上下文 | 先验证 7B |
| 16GB，当前可用 6GB | 先验证 3B |
| 16GB，当前可用 4GB | 先验证 1.5B，并提示释放内存 |
| 24GB，当前可用 16GB，4K 上下文 | 先验证 14B |
| 64GB，当前可用 48GB | 先验证 32B |

大模型或长上下文不能只看芯片型号。工具最多将 60% 总内存作为模型工作预算，
同时要求当前可用内存留至少 1GB；实测内存过低或交换内存增加明显时终止该候选。
如果当前正在运行很多应用，它会说明硬件容量候选和本次实际候选的差异。
关闭不需要的应用后重新运行即可重新评估。

## 安装后使用

默认网页 `http://127.0.0.1:8080/`；端口被占用时自动尝试 8081–8090，
实际地址会在安装完成时显示，并写入验证报告。
OpenAI base URL 为同一地址加 `/v1`。

程序、环境、模型、日志和报告保存在：

```text
~/Library/Application Support/Strata-macOS/
```

该目录生成 **打开 Strata.command** 和 **停止 Strata.command**。
安装完成后关闭终端仍可聊天，登录 macOS 时后台自动启动；停止工具停止本次运行，
下次登录仍按安装设置启动。服务只监听本机，不开放外网访问。

```bash
./setup-macos.sh --status  # 查看部署状态
./setup-macos.sh --stop    # 停止后台服务
./run-macos.sh            # 启动并打开已安装的聊天网页
```

网页保留上游设计。NVIDIA GPU/VRAM 指标在 Mac 上为空；CPU/RAM 监控可用，
当前未提供 Apple GPU 温度与功率读数。GGUF 路径默认 mmap、不强制 mlock。

## 可选偏好与验证

```bash
./setup-macos.sh --profile fast       # 优先流畅，不达标就换小模型
./setup-macos.sh --profile quality    # 接受较慢生成，以容纳较大的候选
./setup-macos.sh --plan               # 只看容量筛选，明确不是已验证推荐
./setup-macos.sh --context 8192       # 更长上下文，重新计算内存并实测
./setup-macos.sh --model qwen2.5-7b    # 指定候选，仍须通过内存和本机验证
```

各模式门槛、模型来源、断点续传、验证范围和故障恢复见
[安装工具说明](docs/MACOS-INSTALLER.md)。
安装报告为 `config/install-report.json`，失败记录另存为
`config/install-report.failed.json`，从不把失败部署标为完成。

开发者 / CI 可以只构建，不下载模型或部署服务：

```bash
./setup-macos.sh --build-only
.venv/bin/python -m unittest serve.test_server tests_macos.test_backend tests_macos.test_installer -v
```

本机实测记录见 [docs/MACOS-VALIDATION.md](docs/MACOS-VALIDATION.md)。
CMake 构建 ARM64 artifact 工具和平台层，Metal 推理库由同一安装工具编译。
原生 CUDA `strata` 可执行文件仍未移植到 Mac。

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
