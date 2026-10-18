#!/bin/bash
# Put the encoder and the trained heads in models/, from this repo's GitHub release.
#   models/encoders/nemotron-3-embed-1b/   NVIDIA Nemotron-3-Embed-1B (OpenMDW-1.1, notices included)
#   models/heads/*.pt                       trained heads, served by start.sh
# Checksums are verified before anything is unpacked. Safe to run again.
#   --from-hub   get the encoder from Hugging Face instead (no heads)
#   --from-dir D install from release files already in folder D (offline machines)
source "$(dirname "$0")/common.sh"

force=0; from_dir=""; from_hub=0
while [ $# -gt 0 ]; do
  case "$1" in
    --force) force=1 ;;
    --from-hub) from_hub=1 ;;
    --from-dir) from_dir="${2:-}"; shift ;;
    *) die "unknown option $1 (use --force, --from-dir DIR, --from-hub)" ;;
  esac
  shift
done
marker="$MODELS_DIR/.complete-$MODELS_TAG"
if [ -f "$marker" ] && [ "$force" = 0 ]; then
  say "models already in place ($MODELS_TAG): $MODELS_DIR"; exit 0
fi
mkdir -p "$MODELS_DIR/encoders" "$MODELS_DIR/heads"

if [ "$from_hub" = 1 ]; then
  [ -x "$PY" ] || die "run setup.sh first"
  say "downloading $ENCODER from Hugging Face (no trained heads this way)"
  "$PY" - <<PY
from huggingface_hub import snapshot_download
from clm.encoders import get_encoder
spec = get_encoder("$ENCODER")
snapshot_download(spec.hf_id, local_dir="$MODELS_DIR/encoders/" + spec.name)
PY
  exit 0
fi

dl="$MODELS_DIR/.download"
mkdir -p "$dl"
if [ -n "$from_dir" ]; then
  [ -f "$from_dir/SHA256SUMS" ] || die "--from-dir needs a folder with the release files (SHA256SUMS missing)"
  say "copying release files from $from_dir"
  cp "$from_dir"/* "$dl/"
else
slug="$(github_slug)"
[ -n "$slug" ] || die "can't tell which GitHub repo holds the models: set CLM_MODELS_REPO=owner/name"
say "downloading release $MODELS_TAG from github.com/$slug (about 2.4 GB the first time)"
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  gh release download "$MODELS_TAG" -R "$slug" -D "$dl" --clobber
elif [ -n "${GITHUB_TOKEN:-}" ]; then
  api="https://api.github.com/repos/$slug/releases/tags/$MODELS_TAG"
  curl -sfL -H "Authorization: Bearer $GITHUB_TOKEN" "$api" > "$dl/release.json" || die "release $MODELS_TAG not found in $slug"
  "$PY" -c "import json,sys; [print(a['id'], a['name']) for a in json.load(open(sys.argv[1]))['assets']]" "$dl/release.json" |
  while read -r id name; do
    say "  $name"
    curl -sfL -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/octet-stream" \
      "https://api.github.com/repos/$slug/releases/assets/$id" -o "$dl/$name"
  done
else
  die "need GitHub access to download the models: run 'gh auth login' (or set GITHUB_TOKEN), then try again"
fi
fi

say "verifying checksums"
( cd "$dl" && shasum -a 256 -c SHA256SUMS >/dev/null ) || die "a downloaded file is corrupt; run again with --force"

say "unpacking the encoder"
rm -rf "$MODELS_DIR/encoders/$ENCODER" "$MODELS_DIR"/heads/*
cat "$dl"/encoder-*.tar.part-* | tar -x -C "$MODELS_DIR/encoders"
cp "$dl"/heads-*.pt "$MODELS_DIR/heads/" 2>/dev/null || true
for f in "$dl"/heads-*.json; do [ -f "$f" ] && cp "$f" "$MODELS_DIR/heads/"; done
for f in "$MODELS_DIR"/heads/heads-*; do mv "$f" "$MODELS_DIR/heads/${f##*/heads-}"; done
cp "$dl/MANIFEST.json" "$MODELS_DIR/"
rm -rf "$dl"
touch "$marker"
say "models ready in $MODELS_DIR:"
ls -1 "$MODELS_DIR/encoders" "$MODELS_DIR/heads" | sed 's/^/  /'
