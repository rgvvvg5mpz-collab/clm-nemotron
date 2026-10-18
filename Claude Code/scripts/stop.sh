#!/bin/bash
# Stop clm-serve and the encoder started by start.sh.
source "$(dirname "$0")/common.sh"
for name in clm-serve encoder; do
  f="$RUN/$name.pid"
  if pid_alive "$f"; then
    kill "$(cat "$f")" && say "stopped $name (pid $(cat "$f"))"
    for _ in 1 2 3 4 5 6 7 8 9 10; do pid_alive "$f" || break; sleep 0.5; done
    pid_alive "$f" && kill -9 "$(cat "$f")" 2>/dev/null || true
  else
    say "$name was not running"
  fi
  rm -f "$f"
done
