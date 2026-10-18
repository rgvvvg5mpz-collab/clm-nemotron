"""Encoder profiles, prompts, truncation and the embedder."""
from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest

from helpers import NEMO, OTHER, REPO, CharTokenizer, fake_embedder
from clm.embedder import EmbedderError
from clm.encoders import ENCODERS, IncompatibleEncoder, check_compatible, checkpoint_encoder, get_encoder


# ----------------------------------------------------------------------------- profiles
def test_nemotron_is_the_only_encoder(monkeypatch):
    monkeypatch.delenv("CLM_ENCODER", raising=False)
    assert get_encoder().hf_id == "nvidia/Nemotron-3-Embed-8B-BF16"
    assert sorted(ENCODERS) == ["nemotron-3-embed-1b", "nemotron-3-embed-8b"]
    assert all(e.hf_id.startswith("nvidia/") for e in ENCODERS.values())


def test_repo_has_no_banned_model_references():
    banned = "q" + "wen"                                       # spelled out, this file would match itself
    # files git tracks or would track (skips .venv, caches and everything else .gitignore'd)
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=REPO,
                            capture_output=True, text=True, check=True).stdout.split()
    hits = []
    for rel in listed:
        if rel.startswith("Model Governance/"):      # data lineage must name what the encoders' makers used
            continue
        try:
            text = open(os.path.join(REPO, rel), encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue                                           # binary assets, deleted files
        hits += [f"{rel}:{n}" for n, line in enumerate(text.splitlines(), 1) if banned in line.lower()]
    assert hits == []


def test_prepare_adds_role_prefix():
    assert NEMO.prepare("Is it raining?", "state") == "query: Is it raining?"
    assert NEMO.prepare("Yes.", "action") == "passage: Yes."
    assert NEMO.prepare("raw", None) == "raw"
    assert OTHER.prepare("Is it raining?", "state") == "Is it raining?"


def test_truncation_keeps_prefix_and_the_right_end():
    tok = CharTokenizer()
    body = "".join(chr(ord("a") + i % 26) for i in range(100))
    s = NEMO.prepare(body, "state", tok, max_tokens=30)
    a = NEMO.prepare(body, "action", tok, max_tokens=30)
    assert s.startswith("query: ") and body.endswith(s[len("query: "):])        # states keep the tail
    assert a.startswith("passage: ") and body.startswith(a[len("passage: "):])  # candidates keep the head
    assert len(tok(s)["input_ids"]) <= 30 and len(tok(a)["input_ids"]) <= 30


def test_unrecorded_or_other_encoder_is_refused():
    assert checkpoint_encoder({"width": 1536}) is None         # upstream release: nothing recorded
    with pytest.raises(IncompatibleEncoder, match="records no encoder"):
        check_compatible(checkpoint_encoder({}), NEMO)
    with pytest.raises(IncompatibleEncoder, match="example/other-embed"):
        check_compatible(OTHER.identity(), NEMO)
    check_compatible(checkpoint_encoder({"encoder": NEMO.identity()}), NEMO)


# ----------------------------------------------------------------------------- embedder
def test_embedder_sends_prompts_and_caches_per_role():
    e = fake_embedder()
    s, _ = e.embed(["same text"], "state")
    a, _ = e.embed(["same text"], "action")
    assert e.session.sent == ["query: same text", "passage: same text"]
    assert not np.allclose(s, a)                       # different prompts, different vectors
    e.embed(["same text"], "state")
    assert len(e.session.sent) == 2                    # cache hit
    assert e.model == "nemotron-3-embed-8b" and e.max_tokens == NEMO.max_tokens


def test_embedder_rejects_wrong_width():
    e = fake_embedder(dim=1024)
    with pytest.raises(EmbedderError, match="1024-d"):
        e.embed(["x"], "state")


def test_local_model_copy_wins_over_the_hub(tmp_path, monkeypatch):
    spec = ENCODERS["nemotron-3-embed-1b"]
    monkeypatch.delenv("CLM_MODELS_DIR", raising=False)
    assert spec.source() == "nvidia/Nemotron-3-Embed-1B-BF16"
    monkeypatch.setenv("CLM_MODELS_DIR", str(tmp_path))
    assert spec.source() == "nvidia/Nemotron-3-Embed-1B-BF16"          # nothing there yet
    local = tmp_path / "encoders" / "nemotron-3-embed-1b"
    local.mkdir(parents=True)
    (local / "config.json").write_text("{}")
    assert spec.source() == str(local)
