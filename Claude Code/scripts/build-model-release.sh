#!/bin/bash
# Maintainers only: package the models for a GitHub release (what fetch-models.sh downloads).
#   bash "Claude Code/scripts/build-model-release.sh" HEAD_NAME=RUN_DIR [HEAD_NAME=RUN_DIR ...]
# RUN_DIR is a training output folder (best_head.pt, summary.json, optional questions.json).
# Writes dist/$MODELS_TAG/: the encoder as a tar split into parts under GitHub's 2 GB
# asset limit, one .pt and one .json card per head, MANIFEST.json and SHA256SUMS.
# Publish with:  gh release create $MODELS_TAG dist/$MODELS_TAG/* --title ... --notes-file ...
source "$(dirname "$0")/common.sh"
[ $# -ge 1 ] || die "usage: build-model-release.sh NAME=RUN_DIR [NAME=RUN_DIR ...]"
out="$REPO/dist/$MODELS_TAG"
rm -rf "$out"; mkdir -p "$out"

src="$("$PY" - <<PY
from huggingface_hub import snapshot_download
from clm.encoders import get_encoder
print(snapshot_download(get_encoder("$ENCODER").hf_id))
PY
)"
say "packing $ENCODER from $src"
# A reproducible archive: sorted entries, fixed time and owner, symlinks resolved, so the same
# encoder always gives the same checksums.
"$PY" - "$src" "$ENCODER" <<'PY' | split -b 1900m - "$out/encoder-$ENCODER.tar.part-"
import os, sys, tarfile
src, name = sys.argv[1:]
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|", format=tarfile.PAX_FORMAT) as tar:
    def add(path, arc):
        info = tar.gettarinfo(os.path.realpath(path), arcname=arc)
        info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
        info.mode = 0o755 if info.isdir() else 0o644
        if info.isdir():
            tar.addfile(info)
        else:
            with open(os.path.realpath(path), "rb") as f:
                tar.addfile(info, f)
    add(src, name)
    for root, dirs, files in os.walk(src):
        dirs.sort()
        rel = os.path.relpath(root, src)
        for d in dirs:
            add(os.path.join(root, d), os.path.normpath(os.path.join(name, rel, d)))
        for f in sorted(files):
            add(os.path.join(root, f), os.path.normpath(os.path.join(name, rel, f)))
PY

for spec in "$@"; do
  name="${spec%%=*}"; dir="${spec#*=}"
  [ -f "$dir/best_head.pt" ] || die "$dir has no best_head.pt"
  cp "$dir/best_head.pt" "$out/heads-$name.pt"
  "$PY" - "$dir" "$name" "$out/heads-$name.json" <<'PY'
import json, os, sys
d, name, dest = sys.argv[1:]
summary = json.load(open(os.path.join(d, "summary.json"))) if os.path.exists(os.path.join(d, "summary.json")) else \
          json.load(open(os.path.join(d, "finetune_summary.json")))
card = {"name": name, "test_accuracy": summary["test"]["acc"], "clm_raw_accuracy": summary.get("raw_test"),
        "majority_accuracy": summary.get("majority_test"), "per_question": summary["test"]["per_question"],
        "best_epoch": summary.get("best_epoch")}
q = os.path.join(d, "questions.json")
if os.path.exists(q):
    card["questions"] = json.load(open(q))
json.dump(card, open(dest, "w"), indent=1)
PY
done

"$PY" - "$out" "$ENCODER" "$MODELS_TAG" <<'PY'
import hashlib, json, os, sys, time
out, enc, tag = sys.argv[1:]
from clm.encoders import get_encoder
spec = get_encoder(enc)
files = sorted(f for f in os.listdir(out) if f not in ("MANIFEST.json", "SHA256SUMS"))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()
sums = {f: sha(os.path.join(out, f)) for f in files}
manifest = {"tag": tag, "built": time.strftime("%Y-%m-%d"), "encoder": {**spec.identity(), "origin": spec.origin,
            "license": "OpenMDW-1.1 (LICENSE, NOTICE and THIRD_PARTY_NOTICES.md inside the archive)"},
            "files": {f: {"bytes": os.path.getsize(os.path.join(out, f)), "sha256": sums[f]} for f in files}}
json.dump(manifest, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
sums["MANIFEST.json"] = sha(os.path.join(out, "MANIFEST.json"))
open(os.path.join(out, "SHA256SUMS"), "w").write("".join(f"{v}  {k}\n" for k, v in sums.items()))
PY
say "built $out:"; ls -la "$out" | sed 's/^/  /'
