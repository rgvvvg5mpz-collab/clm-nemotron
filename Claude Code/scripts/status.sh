#!/bin/bash
# Is CLM running? Shows both processes, the loaded models and the last log lines on trouble.
source "$(dirname "$0")/common.sh"
ok=1
for name in encoder clm-serve; do
  if pid_alive "$RUN/$name.pid"; then say "$name: running (pid $(cat "$RUN/$name.pid"))"; else warn "$name: not running"; ok=0; fi
done
if curl -sf -m 3 "http://127.0.0.1:$APP_PORT/health" > "$RUN/health.json" 2>/dev/null; then
  "$PY" - "$RUN/health.json" "$APP_PORT" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))
print(f"[clm] playground: http://localhost:{sys.argv[2]}/   encoder reachable: {h.get('embedder')}   training: {h.get('training')}")
print(f"[clm] models: {', '.join(h.get('models', []))}")
PY
else
  ok=0
fi
[ -f "$MODELS_DIR/.complete-$MODELS_TAG" ] && say "models: $MODELS_TAG in $MODELS_DIR" || warn "models: not downloaded yet (fetch-models.sh)"
if [ "$ok" = 0 ]; then
  for f in encoder clm-serve; do [ -f "$LOGS/$f.log" ] && { warn "last lines of logs/$f.log:"; tail -8 "$LOGS/$f.log" >&2; }; done
  exit 1
fi
