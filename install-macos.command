#!/bin/bash
cd "$(dirname "$0")"
./setup-macos.sh "$@"
STRATA_INSTALL_EXIT=$?
if [[ $STRATA_INSTALL_EXIT -ne 0 ]]; then
  echo "安装尚未完成，请查看上方提示或 logs/installer.log。重新打开可以继续。"
fi
if [[ -t 0 ]]; then
  read -r -p '按回车关闭安装窗口…' _
fi
exit "$STRATA_INSTALL_EXIT"
