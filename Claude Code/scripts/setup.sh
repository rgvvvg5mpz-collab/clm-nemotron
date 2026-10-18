#!/bin/bash
# Create .venv and install CLM with what this machine needs. Safe to run again.
#   Mac (Apple silicon) / CPU:  clm + sentence-transformers (local encoder server)
#   Linux + NVIDIA GPU:         the same; add CLM_WITH_VLLM=1 for the 8B behind vLLM
source "$(dirname "$0")/common.sh"

kind="$(os_kind)"
[ "$kind" = unsupported ] && die "unsupported OS $(uname -s): use macOS or Linux (on Windows, use WSL2)"
if [ ! -x "$PY" ]; then
  base="$(find_python)" || die "Python 3.10 or newer is needed. On a Mac: brew install python@3.12. On Ubuntu: sudo apt install python3.12-venv"
  say "creating $VENV with $("$base" --version)"
  "$base" -m venv "$VENV"
fi
"$PY" -m pip install -q --upgrade pip
extras="mac,test"
if [ "$kind" = linux-cpu ]; then
  say "installing CPU-only torch (no GPU found)"
  "$PY" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
fi
[ "${CLM_WITH_VLLM:-0}" = 1 ] && extras="$extras,gpu"
say "installing clm-nemotron[$extras] (first time: a few minutes)"
"$PY" -m pip install -q -e "$REPO[$extras]"
"$PY" -c "import clm, torch; print('[clm] ready: clm', clm.__version__, '· torch', torch.__version__, '· device', 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))"
