#!/bin/bash
# 推理视频2D bbox叠加工具 - 启动脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/run_interactive.py" "$@"
