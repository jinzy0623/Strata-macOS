#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
STRATA_INSTALL_HOME="${STRATA_HOME:-$HOME/Library/Application Support/Strata-macOS}"
if [[ $# -eq 0 && -f "$STRATA_INSTALL_HOME/config/install-report.json" ]]; then
  export STRATA_HOME="$STRATA_INSTALL_HOME"
  cd "$STRATA_INSTALL_HOME/runtime"
  exec "$STRATA_INSTALL_HOME/.venv/bin/python" -m tools.macos_installer --start
fi
# Advanced/manual mode retained for existing v0.1.13-macos.1 configurations.
exec .venv/bin/python -m serve.server --engine metal --port 8080 "$@"
