#!/bin/bash
# Run the test suite (no GPU, no network, no running services needed).
source "$(dirname "$0")/common.sh"
[ -x "$PY" ] || bash "$HERE/setup.sh"
"$PY" -m pyflakes "$REPO/src/clm" "$REPO/research" "$REPO/Claude Code/tools"
"$PY" -m pytest "$REPO/tests" -q "$@"
