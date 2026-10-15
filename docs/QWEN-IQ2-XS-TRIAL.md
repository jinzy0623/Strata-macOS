# Qwen3.8-Flash-Next IQ2_XS on the macOS port

## Swift files on this Mac

Use `./start-swift-qwen.command --open` for the downloaded Swift model. This
launcher uses the separately repacked files in
`~/models/qwen3.8-flash-next/swift-IQ2_XS-metal`, with a 65,536-token context.
The original downloads in `swift-IQ2_XS` are preserved.

The downloaded layout places the 28.8 GB lazy embedding table between Metal
weights in the first shard. That layout failed with Metal out-of-memory errors
on this 64 GiB Mac. `tools/resplit_gguf_for_metal.py` copies all ordinary weights
into the first shard and isolates `per_layer_token_embd.weight` in the second,
preserving tensor bytes and metadata except split fields and tensor offsets.
All 1,224 copied tensors passed SHA256 comparison with the source tensors;
the verification manifest is saved next to the repacked files.

The launcher now tests real generation before starting the UI and stops the UI
if its model process exits. At 8K, a Strata chat request returned `42` for
`17 + 25`. The 64K setting is intended to accommodate the existing conversation;
it is not a full-context performance benchmark.

The existing files in `~/models/qwen3.8-flash-next/IQ2_XS` work with a new
`llama-server` adapter. It keeps this repository's Strata chat UI, request queue,
OpenAI and Anthropic APIs, reasoning parser, and monitor. The native engine is
the installed `/opt/homebrew/bin/llama-server`, build b11146-7fe450e19.

The pinned llama-cpp-python 0.3.16 engine cannot load these GGUFs: it rejects
GGML type 42. Its memory admission check also rejects the combined 63.4 GiB
file size. The native engine supports `qwen4exp` and reads the large embedding
table on demand with `--lazy-mode on`. The adapter leaves the existing `metal`
backend and its checks intact.

## Start

Double-click `try-qwen-iq2-xs.command`, or run it in a terminal from this folder:

```bash
./try-qwen-iq2-xs.command --open
```

Chat: http://127.0.0.1:8095/

OpenAI base URL: http://127.0.0.1:8095/v1

Model name: `qwen3.8-flash-next`

The native engine listens on loopback port 8096. The launcher waits until it is
ready, starts Strata, and stops both processes when closed with Ctrl+C. It does
not download or modify the GGUFs or install a login service. An occupied port
causes an explicit failure, without stopping another service.

Double-click `stop-qwen-iq2-xs.command` to stop the trial from another window.

The launcher defaults to 120000 tokens (120K), with one slot, Metal
acceleration, flash attention, and Q4 KV cache. The original measurements
below used 8192 tokens. To override the context for a launch:

```bash
STRATA_CONTEXT=16384 ./try-qwen-iq2-xs.command --open
```

Longer contexts require separate memory and response-time validation.
Logs: `logs/qwen-iq2-xs-metal.log` (native engine) and the Strata launcher
terminal. No vision projector is loaded. Strata's CUDA expert cache and MTP
are not implemented by this adapter.

## Validation — October 5, 2026

Machine: Apple M1 Max, 64 GiB unified memory.

- Both GGUF shard headers parsed: 1224 tensors total, architecture `qwen4exp`.
- Model ready after approximately 19 seconds in the initial trial.
- Native process resident memory approximately 36 GiB at 8K context.
- Arithmetic reply: `17 + 25` returned `42`.
- Portuguese explanation: 74 completion tokens, approximately 24.8 tokens/s
  from native generation timings, total response approximately 3.5 seconds.
- Streaming reply preserved `ação`, `coração`, and `São Paulo`; first visible
  text arrived after approximately 0.52 seconds for the short test prompt.
- Low reasoning effort returned reasoning separately and answered `9 * 7` as `63`.
- Anthropic Messages returned `8 + 6` as `14` with `end_turn`.
- Cancellation during an 8112-token prompt returned control in approximately
  0.13 seconds; the next request completed successfully.
- The browser displayed `42` for `17 + 25`, with approximately 25 tokens/s.
- Start/stop launchers verified: missing shards and occupied ports fail without
  disturbing the trial; stopping releases both ports and removes its PID record.
- 30 automated adapter/backend/server tests passed, covering cancellation during
  prefill, progress events, EOS/timing handling, truncated streams, and request
  formatting. The loading-progress event's placeholder token is excluded from
  model output.

Measurements are short local trials, not a long-context benchmark.

The adapter uses the native `/completion`, `/tokenize`, `/detokenize`, and
`/apply-template` endpoints described in the
[official llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

## 32K context validation — October 5, 2026

At this validation step, the launcher default was raised to 32,768 tokens. Both native `/props` and
Strata `/health` reported the new limit after restart. A 10,047-token prompt
completed successfully and recalled a validation code from its beginning,
returning 10 output tokens in approximately 47.7 seconds. This tests a prompt
beyond the previous 8K limit; it is not a full 32K performance benchmark.
Monitor displayed 10K / 32K and all 14 file/browser tools remained ready.
System memory stayed around 54 / 64 GiB during the check, with swap usage
unchanged at approximately 841 MiB.

## 120K context configuration — October 5, 2026

The saved launcher default is now 120,000 tokens. The native engine rounds
its allocation to 120,064 tokens, which Strata reports through `/health` and
displays as 120K in Monitor. Both services started, all 14 tools remained
ready, and a short inference check returned the correct arithmetic answer.
This verifies startup and generation with the enlarged allocation; a full
120K prompt has not been benchmarked.

## Faster decoding and two fixes — October 7, 2026

The launcher now runs the upstream llama.cpp build in `~/bin/llama-b11476/` (0.6.0-dev, with Qwen4Exp MTP
support; the Homebrew 0.5.0 build is too old) with the MTP draft head
`~/models/qwen3.8-flash-next/mtp/Qwen3.8-Flash-Next-MTP-Q8_0.gguf`, prompt lookup (`--spec-type
draft-mtp,ngram-mod`) and larger batches (`-b 2048 -ub 1024`). Measured on this M1 Max at temperature 0, same
output in every configuration: chat 24.5 → 34 tok/s, a 6.5K-token prompt read at 183 → 262 tok/s, a code edit
written at 24 → 100 tok/s (prompt lookup accepted every draft). The web app and the shared Chat settings now
default to **low** thinking effort; `medium` wrote 6,600 tokens of reasoning for one answer (4 min 21 s).

**Lost tokens (`�` in replies).** The native `/completion` stream holds back a chunk whose text ends inside a
multi-byte character (an emoji's first byte, or a whole speculative step) and then reports only the newest id
in `tokens`, so the adapter lost 10–20% of the ids on emoji-heavy text (63 of 310 on one request). The adapter
now watches the stream's cumulative `tokens_predicted`: when it moves further than `tokens` says, that chunk's `content` (which is complete) is tokenized instead; every other chunk keeps the server's own ids, so a continuation prompt matches the engine's cache exactly and costs no extra round trip. Measured: a
169-token continuation built this way from a stream with 12 lossy chunks was served with 1 token evaluated
and 168 cached, and a 600-token code edit ran at 103.6 tok/s through Strata against 103.9 native.
Test: `tests_macos/test_llama_server.py::test_streamed_text_wins_over_lossy_token_field`.

**Replies that "stop thinking" with no answer.** Two causes, both seen in the server log: the model ends its
turn inside `<think>` (a 2,895-character reasoning then `<|im_end|>`, no `</think>`), or it goes round in
circles there (143,140 characters of `- Tavaires? no.` until the token limit, 9 minutes). At temperature 0.6 a
haiku request looped 4 times out of 6, and no sampling setting helps (6 requests each, two haiku prompts,
low effort, 2,500-token cap): temperature 0.6 → 4 loops, Qwen's official 1.0 → 5, 0.6 with presence
penalty 1.5 → 4, 1.0 with presence penalty 1.5 → 6. Sampling stays at 0.6. Three guards from upstream
Strata are now in `serve/server.py`:

- `reasoning_close_retry` (default on): a reply that ends inside its thinking with no answer and no tool call is
  closed once with `</think>` and the model answers from the prefix the engine already holds (upstream #1053).
- `reasoning_loop_guard` (default on): from 512 thinking tokens on, every 256 tokens, the last 3,000
  characters of reasoning are checked for lines or sentences of 12+ characters repeated three or more times;
  at 40% coverage the thinking is wrapped up (`I have thought about this long enough; time to give my
  answer.` + `</think>`) and the model answers. A simplified form of upstream #728.
- `reasoning_budget_tokens`: a hard cap on thinking tokens, in the config or per request in the body
  (upstream #123). A number caps every request; a map by effort caps only the efforts it names. This config
  sets `{"low": 2000, "medium": 8000}` (high stays uncapped): on this model, normal low-effort thinking
  measured under 700 tokens, while rumination that never repeats itself exactly ("Let me try again" with
  ever-different syllable counts, 5 of 12 looped samples) ran past 1,500 and is invisible to the loop guard.

**Tool loops.** With the web app's tools on, the first live request after the guards still came back empty
after 259 s: every time a thinking cut let the model answer, it answered with a pointless
`web__browser_snapshot` call instead of text, the tool ran, the next round looped again, and the 8-round
limit returned nothing. A tool call repeated with the same arguments in one turn is now treated as a loop:
it is not run, the model gets `This exact tool call was already made in this turn ... answer the user now,
in plain text.` as its result, and a second repeat ends the turn. Test:
`serve/test_mcp.py::ToolLoop::test_a_repeated_identical_call_is_not_run_again`. For casual chat, the 14
tool definitions in every prompt now cost more time than any decoding setting: that trivial request would
have taken a few seconds with tools off. Measured in the web app: the same question took 8 s with tools off
(906-token prompt read at 291 tok/s) against 12 s with them on. **Tools are now off by default** in the web
app (`serve/web/app.js`, `DEFAULTS.tools`); the ⚙ panel's "Use tools from MCP servers" switch turns them on
per browser.

A thinking cut aborts the native stream, and llama-server then drops the generated tokens from its cache
(measured: 307 of 338 tokens re-evaluated after an abort), so a cut re-reads the reasoning at prompt speed,
about 5 s for 1,300 tokens. A reply that ends on its own keeps the cache (1 of 169 evaluated).

Finished assistant turns with no text are also left out of the next prompt, so an earlier empty reply is not
an example for the next one (upstream #843; `STRATA_KEEP_EMPTY_TURNS=1` keeps them). All three guards are set
in `config/qwen-iq2-xs.json` (`"reasoning_close_retry": false` turns one off). Tests:
`serve/test_server.py::ReasoningGuards`.

## Thinking off and an uncompressed KV cache — October 7, 2026

The web app and the shared Chat settings now default to thinking **off** (a simple question: 1.4 s against
9.1 s at low). Pick Low/Medium/High in the chat menu for hard questions. The launcher no longer compresses the
KV cache (`-ctk/-ctv q4_0` removed): Swift 1.5 IQ2_XS on this M1 Max, temperature 0, mean of 2 runs:

| Setting | Chat tok/s | 6.5K prompt read tok/s | Code edit tok/s |
|---|---:|---:|---:|
| KV q4_0, draft 3 (previous) | 32.5 | 261 | 107.3 |
| **KV f16, draft 3 (now)** | **34.1** | **279** | **109.9** |
| KV q8_0 | 34.7 | 265 | 107.5 |
| draft 4 | 30.1 | 250 | 106.3 |
| draft 2 | 27.1 | 254 | 101.1 |
| batch 4096/2048 | 31.9 | 258 | 105.5 |

Memory stayed at 20% free with the f16 cache at 64K context (only 12 of 48 layers keep a KV cache).

`STRATA_KV_TYPE` selects the cache type (default `f16`). A later controlled check (a fresh server per type and
three prompts neither server had seen, 131K context, the same weights) measured q8_0 at 34.6 and f16 at 34.5 tok/s,
so `STRATA_KV_TYPE=q8_0` costs no speed and halves the cache. When benchmarking speculative decoding, repeat runs
of one prompt overstate speed: the prompt cache changes the sampled text and ngram-mod drafts from earlier answers,
which produced 52 and 127 tok/s readings on the same setup.
