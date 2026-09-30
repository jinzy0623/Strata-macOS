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

## 一体化安装工具验证（macOS.2）

2026-09-30 在同一台 Apple M4 16GB 上从一个 `setup-macos.sh` 入口执行了
完整安装，随后又完整执行一次重复安装 / 升级。安装目录使用原生 Python
3.14，实际源码编译 llama-cpp-python 0.3.16 + Metal。验证结果：

| 项目 | 结果 |
| --- | --- |
| 推荐规则、源文件完整性、部署回滚等代码测试 | 共 36 项通过 |
| 原生 C++ 构建及 Darwin 平台测试 | 通过 |
| 自动推荐 | 硬件容量候选 7B；当前约 5.9GB 可用内存，实际候选 3B |
| 官方 3B Q4_K_M 下载 | 2,104,932,768 字节，固定 revision 与 SHA256 匹配 |
| 真实 Metal 实测 | 短 / 长提示词均生成 128 tokens |
| 较慢的持续输出 | 37.8 tokens/s |
| 较长的首字等待 | 1.86 秒（约 768-token 提示词） |
| 实测交换内存增长 | 0 字节 |
| 部署位置 | 用户 Application Support 目录 |
| 用户 LaunchAgent 后台部署 | 成功，安装进程退出后继续运行 |
| 部署后网页、普通和流式 API | 通过 |
| 中文生成与基础算术 | “你好！”及“42” |
| 缓存复用、重复安装、暂停旧服务后升级 | 通过 |
| 自动生成启动 / 停止工具 | 通过 |

这是本机当前负载下的选择，不是把所有 16GB Mac 固定到 3B。
释放内存后的容量筛选可先尝试 7B，仍须通过安装时的实际性能检查。
本次没有在该设备上实测 7B、14B 或 32B；这些候选在各自安装者机器上
实际验证通过前不会部署。1.5B 也未在本次流程中下载，因为 3B 已通过。
回答质量没有被这些基础生成检查全面评估；模型家族、量化级别与模式均
明确限定在模型目录内。
