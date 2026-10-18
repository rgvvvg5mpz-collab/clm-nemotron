#!/bin/bash
# Export a trained model (its .pt, a .json card and .sha256 checksums) to a folder or an S3 bucket.
#   export-model.sh NAME DEST [--endpoint-url URL] [--name NEWNAME] [--archive]
# NAME is a served model (models/heads/ or a playground run); DEST is a folder or s3://bucket/prefix/.
source "$(dirname "$0")/common.sh"
[ $# -ge 2 ] || die "usage: export-model.sh NAME DEST [--endpoint-url URL] [--name NEWNAME] [--archive]"
[ -x "$VENV/bin/clm-export" ] || die "no .venv yet: run start.sh (or setup.sh) first"
case "$2" in
  s3://*) "$PY" -c "import boto3" 2>/dev/null || { say "installing boto3 for S3 exports"; "$VENV/bin/pip" install -q boto3; } ;;
esac
exec "$VENV/bin/clm-export" "$1" "$2" --ckpt-dir "$MODELS_DIR/heads" --runs-dir "$RUNS_DIR" "${@:3}"
