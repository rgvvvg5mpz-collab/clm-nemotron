"""Encoder profiles: prompts, truncation, and refusing heads from another encoder.

No GPU, network or encoder server: a character-level tokenizer and an in-process
fake ``/v1/embeddings`` stand in for the real ones.

    pytest tests/
"""
from __future__ import annotations

import base64
import dataclasses
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "train"))

from clm.embedder import Embedder, EmbedderError  # noqa: E402
from clm.encoders import (ENCODERS, IncompatibleEncoder, check_compatible, checkpoint_encoder,  # noqa: E402
                          get_encoder)
from clm.engine import Engine  # noqa: E402
from clm.heads import HeadPair, make_head  # noqa: E402

NEMO = ENCODERS["nemotron-3-embed-8b"]
OTHER = dataclasses.replace(NEMO, name="other-embed", hf_id="example/other-embed", pooling="last",
                            state_prefix="", action_prefix="")   # stands in for any other encoder


class CharTokenizer:
    """One token per character, plus one special token when asked for."""

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": ([0] if add_special_tokens else []) + [ord(c) for c in text]}

    def num_special_tokens_to_add(self, pair=False):
        return 1

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if i)


class FakeResponse:
    status_code = 200

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


class FakeSession:
    """Records the texts posted; returns a deterministic vector per text."""

    def __init__(self, dim):
        self.dim, self.sent, self.headers = dim, [], {}

    def post(self, url, json, timeout):
        self.sent.extend(json["input"])
        data = []
        for i, t in enumerate(json["input"]):
            v = np.random.default_rng(abs(hash(t)) % 2**32).standard_normal(self.dim).astype(np.float32)
            data.append({"index": i, "embedding": base64.b64encode(v.tobytes()).decode()})
        return FakeResponse({"data": data, "usage": {"prompt_tokens": sum(len(t) for t in json["input"])}})


def fake_embedder(spec=NEMO, dim=None, **kw):
    e = Embedder(encoder=spec, tokenizer=CharTokenizer(), **kw)
    e.session = FakeSession(dim or spec.hidden_size)
    return e


def save_head(path, encoder=None, hidden=4096, proj=8):
    torch.manual_seed(0)
    sh, ah = make_head(16, 2, proj, hidden=hidden), make_head(16, 2, proj, hidden=hidden)
    cfg = {"width": 16, "depth": 2, "projection_dim": proj, "hidden_size": hidden}
    if encoder is not None:
        cfg["encoder"] = encoder.identity()
    torch.save({"state_head": sh.state_dict(), "action_head": ah.state_dict(),
                "logit_scale": torch.tensor(2.0), "cfg": cfg}, path)
    return str(path)


# ----------------------------------------------------------------------------- profiles
def test_nemotron_is_the_only_encoder(monkeypatch):
    monkeypatch.delenv("CLM_ENCODER", raising=False)
    assert get_encoder().hf_id == "nvidia/Nemotron-3-Embed-8B-BF16"
    assert list(ENCODERS) == ["nemotron-3-embed-8b"]


def test_repo_has_no_banned_model_references():
    banned = "q" + "wen"                                       # spelled out, this file would match itself
    hits = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache")]
        for f in files:
            path = os.path.join(root, f)
            try:
                text = open(path, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue                                       # binary assets
            hits += [f"{os.path.relpath(path, REPO)}:{n}" for n, line in enumerate(text.splitlines(), 1)
                     if banned in line.lower()]
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


# ----------------------------------------------------------------------------- heads / engine
def test_head_from_another_encoder_is_refused(tmp_path):
    upstream = save_head(tmp_path / "upstream.pt")             # no cfg["encoder"]: an upstream release
    with pytest.raises(IncompatibleEncoder, match="records no encoder"):
        HeadPair("upstream", upstream, "cpu", NEMO).ensure()
    other = save_head(tmp_path / "other.pt", OTHER)
    with pytest.raises(IncompatibleEncoder, match="example/other-embed"):
        HeadPair("other", other, "cpu", NEMO).ensure()
    HeadPair("other", other, "cpu", OTHER).ensure()            # loads for the encoder it belongs to


def test_engine_answers_with_nemotron_head(tmp_path):
    ck = save_head(tmp_path / "nemo.pt", NEMO)
    eng = Engine(fake_embedder(), checkpoint=ck, device="cpu", action_cache="1MiB")
    out = eng.answer("Customer: charged twice!", {
        "dept": {"type": "choice", "instructions": "Which team?",
                 "criteria": {"billing": "Charges and refunds", "tech": "Bugs"}}})
    p = out["answers"]["dept"]["probabilities"]
    assert set(p) == {"billing", "tech"} and abs(sum(p.values()) - 1) < 1e-6
    sent = eng.embedder.session.sent
    assert "query: Customer: charged twice!\n\nWhich team?" in sent
    assert {"passage: Charges and refunds", "passage: Bugs"} <= set(sent)
    raw = eng.answer("s", {"q": {"type": "noul", "instructions": "ok?"}}, model="clm-raw")
    assert 0 <= raw["answers"]["q"]["noul"] <= 1
    assert "Nemotron" in eng.models()[0]["description"]


def test_engine_has_no_default_head_and_refuses_upstream_ones(tmp_path, monkeypatch):
    monkeypatch.delenv("CLM_CKPT", raising=False)
    assert Engine(fake_embedder(), device="cpu", action_cache="0").heads == {}
    upstream = save_head(tmp_path / "upstream.pt")
    with pytest.raises(IncompatibleEncoder):
        Engine(fake_embedder(), checkpoint=upstream, device="cpu", action_cache="0")
    monkeypatch.setenv("CLM_CKPT", upstream)
    with pytest.raises(IncompatibleEncoder):
        Engine(fake_embedder(), device="cpu", action_cache="0")


def test_engine_rejects_mismatched_encoder_argument():
    with pytest.raises(ValueError, match="differs"):
        Engine(fake_embedder(NEMO), encoder=OTHER, device="cpu", action_cache="0")
    with pytest.raises(ValueError, match="unknown encoder"):
        get_encoder("not-an-encoder")


# ----------------------------------------------------------------------------- training
def test_recipe_matches_serving_text():
    import embed_utils
    r = embed_utils.Recipe.__new__(embed_utils.Recipe)
    r.spec, r.tok, r.max_len = NEMO, CharTokenizer(), 40
    msgs = [{"role": "user", "content": "fix the bug"}, {"role": "assistant", "content": "ok"}]
    ids = r.ids(msgs, "state")
    assert CharTokenizer().decode(ids) == "query: user: fix the bug\n\nassistant: ok"
    assert ids[0] == 0                                         # special tokens, as the server adds them
    long_ids = r.ids("x" * 500, "action")
    assert len(long_ids) <= 40 and CharTokenizer().decode(long_ids).startswith("passage: ")


def test_finetune_needs_a_data_source(tmp_path):
    env = {**os.environ, "CLM_ENCODER": ""}
    p = subprocess.run([sys.executable, os.path.join(REPO, "train", "finetune.py"), "--task", "clm",
                        "--out-dir", str(tmp_path)], capture_output=True, text=True, env=env)
    assert p.returncode != 0 and "clm needs exactly one of --data" in p.stderr


def test_finetune_refuses_upstream_embedding_dir(tmp_path):
    import json
    emb = tmp_path / "emb"
    emb.mkdir()
    torch.save(torch.zeros(4, 4096, dtype=torch.float16), emb / "state_embeddings.pt")
    torch.save(torch.zeros(4, 4096, dtype=torch.float16), emb / "action_embeddings.pt")
    samples = [{"task_id": f"t{i % 2}", "step_idx": i, "trajectory_id": str(i), "reward": 1} for i in range(4)]
    json.dump({"num_samples": 4, "hidden_size": 4096, "samples": samples}, open(emb / "metadata.json", "w"))
    p = subprocess.run([sys.executable, os.path.join(REPO, "train", "finetune.py"), "--task", "clm",
                        "--emb-dir", str(emb), "--out-dir", str(tmp_path / "out"), "--batch", "2",
                        "--epochs", "1"], capture_output=True, text=True)
    assert p.returncode != 0 and "records no encoder" in p.stderr
