"""Export a trained model: its head, a card saying what it answers, and a checksum.

    clm-export ticket-topic ./exports/                 # a folder on this machine
    clm-export ticket-topic s3://my-bucket/clm/models/  # an S3 bucket (needs boto3)

An export is three files:

    <name>.pt       the trained heads (the checkpoint clm-serve loads)
    <name>.json     the card: encoder identity, questions and labels, test accuracy, hashes
    <name>.sha256   checksums of the two files above, in `shasum -a 256 -c` format

Another CLM server loads it with ``clm-serve --model <name>=<name>.pt`` (or by dropping the
files into its ``--ckpt-dir``); the card beside the head is what the playground's model list
and ``/v1/models`` describe it from. The head only works with the encoder it was trained
on, which the card records and ``clm-serve`` checks.

S3 uses boto3's normal credential chain (environment, ``~/.aws``, instance roles);
``--endpoint-url`` points it at an S3-compatible store.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from typing import Any

from .encoders import checkpoint_encoder

FILES = (".pt", ".json", ".sha256")


class ExportError(Exception):
    pass


# ----------------------------------------------------------------------------- card
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_card(head_path: str, name: str) -> dict[str, Any]:
    """Everything a reader (or another server) needs to know about the head, gathered from the
    files saved beside it: a ``<stem>.json`` card (release heads, clm-train) or a training
    run's ``summary.json`` + ``questions.json`` (playground)."""
    import torch
    ck = torch.load(head_path, map_location="cpu", weights_only=True)
    cfg = dict(ck.get("cfg") or {}) if isinstance(ck, dict) else {}
    stem = os.path.splitext(head_path)[0]
    folder = os.path.dirname(head_path)
    card: dict[str, Any] = {}
    if os.path.exists(stem + ".json"):
        card.update(_load_json(stem + ".json"))
    else:
        summary = os.path.join(folder, "summary.json")
        if os.path.exists(summary):
            s = _load_json(summary)
            card.update({"test_accuracy": (s.get("test") or {}).get("acc"), "clm_raw_accuracy": s.get("raw_test"),
                         "majority_accuracy": s.get("majority_test"),
                         "per_question": (s.get("test") or {}).get("per_question"), "best_epoch": s.get("best_epoch")})
        questions = os.path.join(folder, "questions.json")
        if os.path.exists(questions):
            card["questions"] = _load_json(questions)
    card = {k: v for k, v in card.items() if v is not None}
    card["name"] = name
    card["encoder"] = checkpoint_encoder(cfg)
    card["head"] = {k: cfg.get(k) for k in ("width", "depth", "projection_dim", "hidden_size") if k in cfg}
    card["source"] = os.path.basename(head_path)
    card["exported"] = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return card


def bundle(head_path: str, name: str, out_dir: str) -> list[str]:
    """Write ``<name>.pt``, ``<name>.json`` and ``<name>.sha256`` into ``out_dir``; returns their paths."""
    if not os.path.isfile(head_path):
        raise ExportError(f"no checkpoint at {head_path}")
    os.makedirs(out_dir, exist_ok=True)
    pt = os.path.join(out_dir, name + ".pt")
    if os.path.abspath(pt) != os.path.abspath(head_path):
        shutil.copyfile(head_path, pt)
    card = build_card(head_path, name)
    card["sha256"] = {name + ".pt": _sha256(pt)}
    js = os.path.join(out_dir, name + ".json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=1)
    sums = os.path.join(out_dir, name + ".sha256")
    with open(sums, "w", encoding="utf-8") as f:
        f.write(f"{card['sha256'][name + '.pt']}  {name}.pt\n{_sha256(js)}  {name}.json\n")
    return [pt, js, sums]


def archive(head_path: str, name: str, out_tgz: str) -> str:
    """The three export files in one ``.tar.gz`` (what the playground's Download button gives)."""
    tmp = tempfile.mkdtemp(prefix="clm-export-")
    try:
        paths = bundle(head_path, name, tmp)
        with tarfile.open(out_tgz, "w:gz") as tar:
            for p in paths:
                tar.add(p, arcname=os.path.basename(p))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out_tgz


def display_path(path: str) -> str:
    """A path as shown to people: relative to the working directory when under it, else with ~."""
    cwd = os.getcwd()
    if path == cwd or path.startswith(cwd + os.sep):
        return os.path.relpath(path, cwd)
    home = os.path.expanduser("~")
    return "~" + path[len(home):] if path.startswith(home + os.sep) else path


# ----------------------------------------------------------------------------- destinations
def _s3_client(endpoint_url: str | None):
    try:
        import boto3
    except ImportError as e:        # optional dependency: pip install 'clm-nemotron[s3]'
        raise ExportError("exporting to S3 needs boto3: run  pip install boto3  (or  pip install 'clm-nemotron[s3]')") from e
    return boto3.client("s3", endpoint_url=endpoint_url or None)


def _split_s3(uri: str) -> tuple[str, str]:
    rest = uri[len("s3://"):]
    bucket, _, prefix = rest.partition("/")
    if not bucket:
        raise ExportError("an S3 destination looks like s3://bucket/prefix/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix


def export_model(head_path: str, dest: str, name: str, endpoint_url: str | None = None) -> dict[str, Any]:
    """Export to a folder (any path) or a bucket (``s3://bucket/prefix/``).

    Returns ``{"name", "dest", "files": [{"name", "size", "location"}], "card"}``."""
    dest = (dest or "").strip()
    if not dest:
        raise ExportError("say where to export to: a folder, or s3://bucket/prefix/")
    if dest.startswith("s3://"):
        bucket, prefix = _split_s3(dest)
        tmp = tempfile.mkdtemp(prefix="clm-export-")
        try:
            paths = bundle(head_path, name, tmp)
            s3 = _s3_client(endpoint_url)
            files = []
            for p in paths:
                key = prefix + os.path.basename(p)
                try:
                    s3.upload_file(p, bucket, key)
                except Exception as e:      # botocore raises many classes; the message is what helps
                    raise ExportError(f"upload to s3://{bucket}/{key} failed: {e}") from e
                files.append({"name": os.path.basename(p), "size": os.path.getsize(p), "location": f"s3://{bucket}/{key}"})
            card = _load_json(paths[1])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        for f in files:
            f["display"] = f["location"]
        return {"name": name, "dest": f"s3://{bucket}/{prefix}", "dest_display": f"s3://{bucket}/{prefix}",
                "files": files, "card": card}
    folder = os.path.abspath(os.path.expanduser(dest))
    if os.path.exists(folder) and not os.path.isdir(folder):
        raise ExportError(f"{dest} is a file; the destination must be a folder")
    try:
        paths = bundle(head_path, name, folder)
    except OSError as e:
        raise ExportError(f"cannot write to {folder}: {e.strerror or e}") from e
    return {"name": name, "dest": folder, "dest_display": display_path(folder),
            "files": [{"name": os.path.basename(p), "size": os.path.getsize(p), "location": p,
                       "display": display_path(p)} for p in paths],
            "card": _load_json(paths[1])}


# ----------------------------------------------------------------------------- command line
def resolve(model: str, ckpt_dir: str | None, runs_dir: str | None) -> tuple[str, str]:
    """A model name or a checkpoint path -> (checkpoint path, export name).

    Names are looked up as ``<ckpt_dir>/<name>.pt`` and then as a playground / clm-train run
    under ``runs_dir`` whose ``job.json`` names the model."""
    if os.path.isfile(model):
        name = os.path.splitext(os.path.basename(model))[0]
        if name in ("best_head", "final_head"):        # a run folder: name it after the folder
            name = os.path.basename(os.path.dirname(os.path.abspath(model)))
        return model, name
    if ckpt_dir:
        p = os.path.join(ckpt_dir, model + ".pt")
        if os.path.isfile(p):
            return p, model
    if runs_dir and os.path.isdir(runs_dir):
        for run in sorted(os.listdir(runs_dir), reverse=True):      # newest first
            job = os.path.join(runs_dir, run, "job.json")
            if os.path.isfile(job):
                try:
                    j = _load_json(job)
                except (OSError, ValueError):
                    continue
                if j.get("model") == model and j.get("checkpoint") and os.path.isfile(j["checkpoint"]):
                    return j["checkpoint"], model
    looked = [d for d in (ckpt_dir, runs_dir) if d]
    raise ExportError(f"no model {model!r}" + (f" in {', '.join(looked)}" if looked else "") + "; give a checkpoint path instead")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="clm-export", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("model", help="a served model's name, or the path to its .pt checkpoint")
    ap.add_argument("dest", help="a folder, or s3://bucket/prefix/")
    ap.add_argument("--name", help="export under this name instead of the model's")
    ap.add_argument("--ckpt-dir", default=os.environ.get("CLM_CKPT_DIR", "models/heads"),
                    help="where named models live (default models/heads)")
    ap.add_argument("--runs-dir", default=os.environ.get("CLM_RUNS_DIR", os.path.expanduser("~/.cache/clm/runs")),
                    help="where playground training runs live")
    ap.add_argument("--endpoint-url", default=os.environ.get("CLM_S3_ENDPOINT"),
                    help="S3-compatible endpoint (MinIO, Ceph, ...)")
    ap.add_argument("--archive", action="store_true", help="write one <name>.tar.gz instead of three files (folders only)")
    a = ap.parse_args(argv)
    try:
        path, name = resolve(a.model, a.ckpt_dir, a.runs_dir)
        name = a.name or name
        if a.archive:
            if a.dest.startswith("s3://"):
                ap.error("--archive writes to a folder; upload the three files to S3 instead")
            os.makedirs(a.dest, exist_ok=True)
            out = archive(path, name, os.path.join(a.dest, name + ".tar.gz"))
            print(f"[clm-export] {name}: {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
            return
        res = export_model(path, a.dest, name, a.endpoint_url)
    except ExportError as e:
        sys.exit(f"[clm-export] {e}")
    for f in res["files"]:
        print(f"[clm-export] {f['display']}  ({f['size'] / 1e6:.1f} MB)")
    enc = (res["card"].get("encoder") or {}).get("hf_id") or "unrecorded encoder"
    acc = res["card"].get("test_accuracy")
    print(f"[clm-export] {name}: {enc}" + (f", {acc:.0%} on its test set" if isinstance(acc, (int, float)) else ""))
    print(f"[clm-export] serve it elsewhere:  clm-serve --model {name}={res['files'][0]['display']}")


if __name__ == "__main__":
    main()
