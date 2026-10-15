#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export STRATA_MODEL_DIR="${STRATA_MODEL_DIR:-$HOME/models/qwen3.8-flash-next/swift-IQ2_XS-metal}"
export STRATA_MODEL="Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS"
export STRATA_CONTEXT="${STRATA_CONTEXT:-65536}"
exec ./try-qwen-iq2-xs.command "$@"
