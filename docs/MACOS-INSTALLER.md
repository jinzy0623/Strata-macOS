# macOS installer: recommendation and completion contract

The default `setup-macos.sh` / double-click `install-macos.command` handles the
entire installation. Advanced flags are optional. The installer does not call
successful dependency installation a successful deployment.

## Recommendation scope

The versioned catalog contains official Qwen2.5 Instruct Q4_K_M GGUFs, 1.5B,
3B, 7B, 14B and 32B. Each entry pins a Hugging Face revision and every shard's
size and SHA256 from its LFS metadata. No 0.5B daily-use recommendation exists.
The catalog is deliberately bounded by the tested tokenizer/template/backend
contract. It is not an automatic comparison of all current model families.
Parameter count within this family is a quality preference, not an independently
measured quality score. Greetings and arithmetic are basic functional checks,
not a comprehensive answer-quality evaluation.

Memory estimate = 1.1 × total shard bytes + FP16 KV bytes at the requested
context + 768 MiB workspace. The working budget is the minimum of:

- 60% of physical unified memory;
- physical memory minus 3 GiB;
- current available memory minus 1 GiB.

Both the hardware-capacity candidate and the live-memory candidate are shown.
They can differ when other applications consume memory. A rerun suspends this
installation's existing service first so it does not count its own model twice.
It restores the old deployment if installation fails. The memory estimate is
conservative but is not an absolute out-of-memory guarantee; real trials add a
second check.

## Performance gates

Candidates are tried largest first; the first candidate which passes wins.
Each actual Metal trial includes a shorter prompt and a roughly 800-token
longer prompt, generating up to 128 tokens each. Fewer than 32 generated tokens
is not treated as a reliable speed sample. Warm-up and API checks happen first.
The slower of the two decode rates and the longer time to first token are used.
These are responsiveness preferences, not hardware specifications.

| Mode | Minimum decode | Maximum first-token wait |
| --- | --- | --- |
| balanced (default) | 15 tokens/s | 8 s |
| fast | 30 tokens/s | 4 s |
| quality | 6 tokens/s | 20 s |

A child process isolates every model trial. It is stopped if available system
memory falls below 512 MiB, swap growth exceeds 256 MiB, or the trial exceeds
240 seconds. Global swap can change because of other programs; that can cause
conservative rejection, and the report says so. The report records minimum
available memory, swap growth and observed process peak RSS. These do not
measure every Metal allocation in isolation. Real llama.cpp logs must identify
**device Metal**, not merely a library compiled with GPU support.

A failed candidate is recorded and the next smaller candidate is tried. If no
candidate passes, installation exits unsuccessfully and no new deployment is
claimed. `--model` fixes one candidate and does not silently substitute another.
`--plan` is only a capacity preview, explicitly not a verified recommendation.

## Download integrity and model licenses

All parts are downloaded into `models/<catalog-id>/` under the installation
home. Partial downloads end in `.gguf.part` and are resumed. Each part's exact
size and SHA256 are checked before it becomes a final `.gguf`. Corrupt data
cannot be marked installed. ggml opens the sibling shards from the first shard;
all shards count in the memory budget. `SOURCE.json` records the source,
revision, hashes and license link.

The official 1.5B/7B/14B/32B catalog entries identify Apache 2.0; the official 3B
entry identifies its own Qwen research license. Model terms remain those of the
publishers, and weights are not redistributed in this GitHub repository. Links
are provided with every pinned catalog entry. The installer does not modify or
relicense model files.

## What “complete” means

All of these must pass:

1. Native Python/ARM64 checks and source-built Metal inference runtime.
2. CMake platform build and native tests (Darwin I/O, EOF, alignment, optional locks).
3. Original Python service and backend/installer regression tests.
4. Pinned model integrity, actual Metal generation and selected performance gates.
5. Background deployment and identity-checked HTTP readiness.
6. Deployed web assets, model list, metrics, ordinary/streaming chat, Chinese
   generation and a basic arithmetic answer.

The resulting JSON report includes hardware snapshots, candidate attempts,
performance results, synthetic answers, source hashes and deployment address.
A failure is saved separately, with status **incomplete**. A failed deployment
restores the previous config, LaunchAgent and runtime source. Previous runtime
snapshots are retained in `runtime.backup-*`; failed runtime copies are kept
for diagnosis. A failed install does not overwrite the last successful report.

The backend still supports text only, with no generic tool calling or vision.
The original 125B CUDA expert cache, SSD expert streaming and MTP are not part
of this automatic GGUF deployment.

## Runtime and controls

The default home is `~/Library/Application Support/Strata-macOS`. The Python
venv and copied server source live there, so a background service does not need
access to the user's Documents folder. No administrator or sudo is required.
A per-user LaunchAgent `com.jinzy0623.strata-macos` starts the local server and
restarts it after an unexpected exit. It starts at login. Ports 8080–8090 are
tried; readiness checks require the unique installation id, so an unrelated
server cannot pass the installer's checks. Listening is restricted to loopback.

Generated “打开 Strata.command” / “停止 Strata.command” controls live in the
installation home. `setup-macos.sh --start`, `--stop`, and `--status` also work.
Stopping removes the running job for this login; the plist remains and starts
again on the next login. Close the terminal after a completed installation:
the service keeps running. `--no-open` completes deployment without opening a
browser. `STRATA_HOME` can select a different data home; the fixed service label
prevents silently overwriting an installation owned by another directory.

Dependency and native/test logs, per-model trial logs, and server stdout/stderr
live in the installation home `logs/`. The installer console is captured as
`logs/installer.log`. `--build-only` is intended for CI and explicitly does not
claim a verified model or completed deployment.
