#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
STRATA_TRIAL_PID="$(cat work/qwen-iq2-xs.pid 2>/dev/null || true)"
case "$STRATA_TRIAL_PID" in
  ''|*[!0-9]*) echo "No managed Qwen trial is running."; exit 0 ;;
esac
STRATA_TRIAL_COMMAND="$(ps -p "$STRATA_TRIAL_PID" -o command= 2>/dev/null || true)"
case "$STRATA_TRIAL_COMMAND" in
  *try-qwen-iq2-xs.command*)
    kill -TERM "$STRATA_TRIAL_PID"
    echo "Stopping the Strata Qwen trial."
    for ((attempt=0; attempt<300; attempt++)); do
      if ! kill -0 "$STRATA_TRIAL_PID" 2>/dev/null; then
        echo "Strata Qwen trial stopped."
        exit 0
      fi
      sleep 0.1
    done
    echo "Strata is still shutting down. Wait before starting it again."
    exit 1
    ;;
  *) echo "The saved trial process is no longer running." ;;
esac
