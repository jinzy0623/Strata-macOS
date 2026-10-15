#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
STRATA_MODEL_DIR="${STRATA_MODEL_DIR:-$HOME/models/qwen3.8-flash-next/IQ2_XS}"
STRATA_LLAMA_SERVER="${STRATA_LLAMA_SERVER:-$HOME/bin/llama-b11476/llama-server}"
STRATA_CONTEXT="${STRATA_CONTEXT:-120000}"
STRATA_MODEL="${STRATA_MODEL:-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS}"
STRATA_MTP="${STRATA_MTP:-$HOME/models/qwen3.8-flash-next/mtp/Qwen3.8-Flash-Next-MTP-Q8_0.gguf}"
# Speculative decoding: the MTP head drafts 3 tokens; prompt lookup (ngram-mod) drafts repeated context.
# Benchmarked 2026-10-07 on this M1 Max: chat 24.5 -> 34 tok/s, code edits 24 -> 100 tok/s, same output.
if [ -f "$STRATA_MTP" ]; then
  STRATA_SPEC=(-md "$STRATA_MTP" -ngld 99 --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3)
else
  echo "MTP head not found ($STRATA_MTP); using prompt lookup only."
  STRATA_SPEC=(--spec-type ngram-mod)
fi
for part in 00001 00002; do
  test -f "$STRATA_MODEL_DIR/$STRATA_MODEL-$part-of-00002.gguf" || {
    echo "Missing model shard $part in $STRATA_MODEL_DIR"; exit 1;
  }
done
test -x "$STRATA_LLAMA_SERVER" || { echo "llama-server not found: $STRATA_LLAMA_SERVER"; exit 1; }
test -x .venv/bin/python || { echo "Strata's Python environment is missing."; exit 1; }
for port in 8095 8096; do
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN -t >/dev/null; then
    echo "Port $port is already in use. Stop that service before starting this trial."; exit 1
  fi
done
mkdir -p logs work
STRATA_NATIVE_PID=""
STRATA_UI_PID=""
cleanup() {
  trap - EXIT INT TERM
  for pid in "$STRATA_UI_PID" "$STRATA_NATIVE_PID"; do
    if [ -n "$pid" ]; then kill "$pid" 2>/dev/null || true; fi
  done
  for pid in "$STRATA_UI_PID" "$STRATA_NATIVE_PID"; do
    if [ -n "$pid" ]; then wait "$pid" 2>/dev/null || true; fi
  done
  if [ "$(cat work/qwen-iq2-xs.pid 2>/dev/null || true)" = "$$" ]; then
    rm -f work/qwen-iq2-xs.pid
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo "$$" > work/qwen-iq2-xs.pid
# KV cache type: f16 (default) or q8_0. Both measured at the same decode speed on an M1 Max (see
# docs/QWEN-IQ2-XS-TRIAL.md); q8_0 halves the cache, which matters at long contexts on a 64 GB Mac.
STRATA_KV_TYPE="${STRATA_KV_TYPE:-f16}"
"$STRATA_LLAMA_SERVER" -m "$STRATA_MODEL_DIR/$STRATA_MODEL-00001-of-00002.gguf" \
  -ngl 99 -np 1 -c "$STRATA_CONTEXT" -b 2048 -ub 1024 "${STRATA_SPEC[@]}" \
  -ctk "$STRATA_KV_TYPE" -ctv "$STRATA_KV_TYPE" \
  --flash-attn on --lazy-mode on --jinja --host 127.0.0.1 --port 8096 \
  --alias qwen3.8-flash-next >logs/qwen-iq2-xs-metal.log 2>&1 &
STRATA_NATIVE_PID=$!
echo "Loading Qwen IQ2_XS with Metal (context: $STRATA_CONTEXT). Log: logs/qwen-iq2-xs-metal.log"
STRATA_READY=0
for ((attempt=0; attempt<300; attempt++)); do
  kill -0 "$STRATA_NATIVE_PID" 2>/dev/null || { tail -40 logs/qwen-iq2-xs-metal.log; exit 1; }
  if curl -fsS --max-time 2 http://127.0.0.1:8096/health >/dev/null 2>&1; then STRATA_READY=1; break; fi
  sleep 1
done
[ "$STRATA_READY" = 1 ] || { echo "Model load timed out. See logs/qwen-iq2-xs-metal.log"; exit 1; }
echo "Checking that the model can generate a reply."
if ! .venv/bin/python - <<'PY'
import json
import urllib.request
request = urllib.request.Request(
    'http://127.0.0.1:8096/completion',
    data=json.dumps({'prompt': 'Hello', 'n_predict': 1, 'temperature': 0,
                     'stream': False}).encode(),
    headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(request, timeout=120) as response:
    result = json.load(response)
if result.get('error') or result.get('tokens_predicted', 0) < 1:
    raise RuntimeError(f'Model generation check failed: {result}')
print('Model generation check passed.')
PY
then
  tail -40 logs/qwen-iq2-xs-metal.log
  exit 1
fi
.venv/bin/python -m serve.server --engine llama-server --config config/qwen-iq2-xs.json --port 8095 "$@" &
STRATA_UI_PID=$!
echo "Strata chat: http://127.0.0.1:8095/ — stop with Ctrl+C."
while kill -0 "$STRATA_UI_PID" 2>/dev/null; do
  if ! kill -0 "$STRATA_NATIVE_PID" 2>/dev/null; then
    echo "The model service stopped. See logs/qwen-iq2-xs-metal.log"
    tail -40 logs/qwen-iq2-xs-metal.log
    exit 1
  fi
  sleep 1
done
wait "$STRATA_UI_PID"
