#!/bin/bash
# Shared settings for the scripts in this folder. Sourced, never run directly.
# Every path is derived from the repo root, so the scripts work from any directory.
set -euo pipefail
shopt -s nullglob                         # an empty glob is empty, not a literal pattern

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV="${CLM_VENV:-$REPO/.venv}"
PY="$VENV/bin/python"
MODELS_DIR="${CLM_MODELS_DIR:-$REPO/models}"
LOGS="$REPO/logs"
RUN="$REPO/.run"                          # pid files
RUNS_DIR="${CLM_RUNS_DIR:-$REPO/runs/playground}"   # models trained in the playground
ENCODER="${CLM_ENCODER:-nemotron-3-embed-1b}"
ENC_PORT="${CLM_ENC_PORT:-8090}"
APP_PORT="${CLM_PORT:-8700}"
MODELS_TAG="${CLM_MODELS_TAG:-models-v1}"
MODELS_REPO="${CLM_MODELS_REPO:-}"        # owner/name; default: this clone's GitHub origin

export CLM_MODELS_DIR="$MODELS_DIR" CLM_ENCODER="$ENCODER" CLM_RUNS_DIR="$RUNS_DIR"
mkdir -p "$LOGS" "$RUN"

say()  { printf '\033[1m[clm]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[clm] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m[clm] %s\033[0m\n' "$*" >&2; exit 1; }

os_kind() {                               # mac | linux-gpu | linux-cpu
  case "$(uname -s)" in
    Darwin) echo mac ;;
    Linux)  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then echo linux-gpu; else echo linux-cpu; fi ;;
    *)      echo unsupported ;;
  esac
}

find_python() {                           # a Python 3.10+ to build the venv with
  for c in python3.13 python3.12 python3.11 python3.10 python3.14 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      command -v "$c"; return 0
    fi
  done
  return 1
}

github_slug() {                           # owner/name of the GitHub repo holding the model release
  if [ -n "$MODELS_REPO" ]; then echo "$MODELS_REPO"; return; fi
  local url; url="$(git -C "$REPO" remote get-url origin 2>/dev/null || true)"
  url="${url%.git}"
  case "$url" in
    *github.com[:/]*) echo "${url##*github.com[:/]}" ;;
    *) echo "" ;;
  esac
}

pid_alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
port_busy() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }
wait_http() {                             # url seconds
  local i
  for ((i = 0; i < $2; i++)); do curl -sf -m 2 "$1" >/dev/null 2>&1 && return 0; sleep 1; done
  return 1
}
