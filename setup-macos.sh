#!/bin/bash
# One entry point: runtime, hardware recommendation, verified model, tests and deployment.
set -euo pipefail
cd "$(dirname "$0")"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "需要 Apple Silicon macOS 和原生 ARM64 终端，请关闭 Rosetta。" >&2
  exit 1
fi
mkdir -p work logs
if ! xcrun --find clang++ >/dev/null 2>&1; then
  xcode-select --install || true
  echo "已打开 Apple 编译工具安装窗口。完成后重新打开本安装工具即可继续。" >&2
  exit 1
fi
PYTHON="${STRATA_PYTHON:-python3}"
if ! "$PYTHON" -c 'import platform,sys; assert platform.machine()=="arm64" and sys.version_info>=(3,10)' >/dev/null 2>&1; then
  if command -v uv >/dev/null 2>&1; then
    STRATA_UV="$(command -v uv)"
  else
    echo '正在安装本目录专用的 Python 引导工具（Astral uv）…'
    curl -fLsS --retry 3 https://astral.sh/uv/install.sh -o work/install-uv.sh
    UV_INSTALL_DIR="$PWD/.tools" UV_NO_MODIFY_PATH=1 sh work/install-uv.sh
    STRATA_UV="$PWD/.tools/uv"
  fi
  "$STRATA_UV" python install 3.12
  PYTHON="$("$STRATA_UV" python find 3.12)"
fi
STRATA_INSTALL_HOME="${STRATA_HOME:-$HOME/Library/Application Support/Strata-macOS}"
STRATA_VENV="$STRATA_INSTALL_HOME/.venv"
for STRATA_ARG in "$@"; do
  if [[ "$STRATA_ARG" == --build-only ]]; then STRATA_VENV="$PWD/.venv"; fi
done
mkdir -p "$(dirname "$STRATA_VENV")"
if [[ ! -x "$STRATA_VENV/bin/python" ]]; then
  "$PYTHON" -m venv "$STRATA_VENV"
fi
"$STRATA_VENV/bin/python" -c 'import platform,sys; assert platform.machine()=="arm64" and sys.version_info>=(3,10), "请使用 ARM64 Python 3.10+"'
"$STRATA_VENV/bin/python" -m ensurepip --upgrade > logs/bootstrap.log 2>&1
"$STRATA_VENV/bin/python" -m pip install 'jinja2>=3.1.6,<4' 'psutil>=5.9,<8' 'cmake>=3.24,<5' >> logs/bootstrap.log 2>&1
export STRATA_HOME="$STRATA_INSTALL_HOME"
mkdir -p "$STRATA_INSTALL_HOME/logs"
"$STRATA_VENV/bin/python" -m tools.macos_installer "$@" 2>&1 | tee -a "$STRATA_INSTALL_HOME/logs/installer.log"
