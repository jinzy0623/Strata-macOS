#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python -m serve.server --engine metal --port 8080 "$@"
