#!/bin/bash
# Start CLM: the Nemotron encoder (port 8090) and clm-serve with the playground (port 8700).
# Installs and downloads on first run. Logs: logs/encoder.log, logs/clm-serve.log.
source "$(dirname "$0")/common.sh"

[ -x "$VENV/bin/clm-encoder" ] || bash "$HERE/setup.sh"
[ -f "$MODELS_DIR/.complete-$MODELS_TAG" ] || [ -f "$MODELS_DIR/encoders/$ENCODER/config.json" ] || bash "$HERE/fetch-models.sh"

# 1. encoder
if pid_alive "$RUN/encoder.pid"; then
  say "encoder already running (pid $(cat "$RUN/encoder.pid"))"
else
  port_busy "$ENC_PORT" && die "port $ENC_PORT is in use by another program; stop it or set CLM_ENC_PORT"
  if [ -f "$MODELS_DIR/encoders/$ENCODER/config.json" ]; then export HF_HUB_OFFLINE=1; fi
  say "starting the $ENCODER encoder on :$ENC_PORT ($(os_kind))"
  nohup "$VENV/bin/clm-encoder" --encoder "$ENCODER" --port "$ENC_PORT" \
    > "$LOGS/encoder.log" 2>&1 &
  echo $! > "$RUN/encoder.pid"
fi
say "waiting for the encoder to load (about 30 s the first time)"
wait_http "http://127.0.0.1:$ENC_PORT/v1/models" 300 || { tail -20 "$LOGS/encoder.log" >&2; die "the encoder did not start; see logs/encoder.log"; }

# 2. clm-serve
if pid_alive "$RUN/clm-serve.pid"; then
  say "clm-serve already running (pid $(cat "$RUN/clm-serve.pid"))"
else
  port_busy "$APP_PORT" && die "port $APP_PORT is in use by another program; stop it or set CLM_PORT"
  heads=()
  [ -n "$(ls "$MODELS_DIR"/heads/*.pt 2>/dev/null)" ] && heads=(--ckpt-dir "$MODELS_DIR/heads")
  say "starting clm-serve on :$APP_PORT"
  nohup "$VENV/bin/clm-serve" --encoder "$ENCODER" --port "$APP_PORT" \
    --emb-url "http://127.0.0.1:$ENC_PORT/v1/embeddings" --runs-dir "$RUNS_DIR" ${heads[@]+"${heads[@]}"} \
    > "$LOGS/clm-serve.log" 2>&1 &
  echo $! > "$RUN/clm-serve.pid"
fi
wait_http "http://127.0.0.1:$APP_PORT/health" 120 || { tail -20 "$LOGS/clm-serve.log" >&2; die "clm-serve did not start; see logs/clm-serve.log"; }

models="$(curl -s "http://127.0.0.1:$APP_PORT/v1/models" | "$PY" -c "import json,sys; print(', '.join(m['name'] for m in json.load(sys.stdin)['models']))")"
say "running. Open http://localhost:$APP_PORT/  (models: $models)"
say "stop with: bash \"Claude Code/scripts/stop.sh\""
