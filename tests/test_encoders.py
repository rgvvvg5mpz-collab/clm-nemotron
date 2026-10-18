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
    assert sorted(ENCODERS) == ["nemotron-3-embed-1b", "nemotron-3-embed-8b"]
    assert all(e.hf_id.startswith("nvidia/") for e in ENCODERS.values())


def test_repo_has_no_banned_model_references():
    banned = "q" + "wen"                                       # spelled out, this file would match itself
    # files git tracks or would track (skips .venv, caches and everything else .gitignore'd)
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=REPO,
                            capture_output=True, text=True, check=True).stdout.split()
    hits = []
    for rel in listed:
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


# ----------------------------------------------------------------------------- no head yet: clm-raw
def test_engine_without_head_defaults_to_raw(monkeypatch):
    monkeypatch.delenv("CLM_CKPT", raising=False)
    eng = Engine(fake_embedder(), device="cpu", action_cache="0")
    assert eng.default_model == "clm-raw" and [m["name"] for m in eng.models()] == ["clm-raw"]
    out = eng.answer("Customer: charged twice!", {"q": {"type": "noul", "instructions": "Is it about billing?"}})
    assert out["model"] == "clm-raw" and 0 <= out["answers"]["q"]["noul"] <= 1
    assert eng.rank("Tides?", ["The Moon.", "Plants."])[0]["rank"] == 1


def test_server_serves_raw_without_a_head(monkeypatch):
    from fastapi.testclient import TestClient
    from clm.server import create_app
    monkeypatch.delenv("CLM_CKPT", raising=False)
    eng = Engine(fake_embedder(), device="cpu", action_cache="0")
    eng.embedder.healthy = lambda: True
    c = TestClient(create_app(eng))
    h = c.get("/health").json()
    assert h["default_model"] == "clm-raw" and h["encoder"]["name"] == "nemotron-3-embed-8b"
    r = c.post("/v1/systemone", json={"state": "s", "questions": {"q": {"type": "noul", "instructions": "ok?"}}})
    assert r.status_code == 200 and r.json()["model"] == "clm-raw"
    r = c.post("/v1/rank", json={"context": "c", "answers": ["a", "b"]})
    assert r.status_code == 200 and r.json()["model"] == "clm-raw"
    r = c.post("/v1/systemone", json={"state": "s", "model": "clm-latest",
                                      "questions": {"q": {"type": "noul", "instructions": "ok?"}}})
    assert r.status_code == 422                                # asking for a head that is not there


def test_client_lets_the_server_pick_the_model():
    from clm.client import CLMClient

    class Capture:
        headers = {}

        def post(self, url, json, timeout):
            self.body = json
            return type("R", (), {"status_code": 200, "headers": {}, "json": lambda s: {
                "model": "clm-raw", "answers": {}, "usage": {}}})()
    client = CLMClient()
    client._s = Capture()
    assert client.system_one("s", {}).model == "clm-raw"
    assert "model" not in client._s.body
    client.system_one("s", {}, model="clm-latest")
    assert client._s.body["model"] == "clm-latest"


# ----------------------------------------------------------------------------- typed-decisions training
def test_choice_training_runs_and_reports_raw_baseline(tmp_path):
    """End to end on CPU: embeddings come from a pre-filled cache, so no encoder is needed."""
    import hashlib
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq
    import adapters

    rng = np.random.default_rng(0)
    rows = []
    for i in range(24):
        team = "billing" if i % 2 else "technical"
        rows.append({"id": f"r{i}", "state": f"Ticket {i}: {'refund please' if i % 2 else 'app crashes'}",
                     "questions": json.dumps({"dept": {"type": "choice", "instructions": "Which team?",
                                                       "criteria": {"billing": "Charges and refunds",
                                                                    "technical": "Bugs and outages"}}}),
                     "gold": json.dumps({"dept": {"label": team}})})
    data = tmp_path / "typed.parquet"
    pq.write_table(pa.Table.from_pylist(rows), data)

    keys = set()
    for e in adapters.typed_decision_examples(rows):
        keys.add("state\x00" + e.state_text)
        keys.update("action\x00" + t for t in e.candidates)
    keys = sorted(keys)
    sha = [hashlib.sha1(k.encode()).hexdigest() for k in keys]
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    np.savez(cache_dir / "choice_nemotron-3-embed-8b_2048.npz", keys=np.array(sha),
             vecs=rng.standard_normal((len(keys), 4096)).astype(np.float16))

    out = tmp_path / "run"
    p = subprocess.run([sys.executable, os.path.join(REPO, "train", "finetune.py"), "--task", "choice",
                        "--data", str(data), "--embed-cache", str(cache_dir), "--out-dir", str(out),
                        "--epochs", "2", "--batch", "8", "--width", "32", "--depth", "2", "--gpu", "0"],
                       capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert p.returncode == 0, p.stderr[-2000:]
    assert "clm-raw (nemotron-3-embed-8b, no head)" in p.stdout
    summary = json.load(open(out / "finetune_summary.json"))
    assert 0 <= summary["raw_test"] <= 1 and "test" in summary
    ck = torch.load(out / "best_head.pt", map_location="cpu", weights_only=False)
    assert ck["cfg"]["encoder"] == NEMO.identity()
    HeadPair("trained", str(out / "best_head.pt"), "cpu", NEMO).ensure()   # servable as-is


# ----------------------------------------------------------------------------- tools/local_embed_server.py
def test_local_embed_server_speaks_the_vllm_subset():
    from fastapi.testclient import TestClient
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import local_embed_server as les

    class FakeST(torch.nn.Module):    # mean of per-token vectors, like the real pooling
        def forward(self, f):
            table = torch.randn(256, 8, generator=torch.Generator().manual_seed(0))
            tok = table[f["input_ids"] % 256] * f["attention_mask"].unsqueeze(-1)
            return {"sentence_embedding": tok.sum(1) / f["attention_mask"].sum(1, keepdim=True)}

    class Tok:
        pad_token_id = 0

        def __call__(self, text, add_special_tokens=True):
            return {"input_ids": ([1] if add_special_tokens else []) + [ord(c) % 250 + 2 for c in text]}

    enc = les.LocalEncoder(FakeST(), Tok(), "cpu", batch_tokens=16)
    c = TestClient(les.create_app(enc, {"nemotron-3-embed-1b", "nvidia/Nemotron-3-Embed-1B-BF16"}, 64, 8))
    assert {m["id"] for m in c.get("/v1/models").json()["data"]} >= {"nemotron-3-embed-1b"}
    r = c.post("/v1/embeddings", json={"model": "nemotron-3-embed-1b", "encoding_format": "base64",
                                       "input": ["query: a", "passage: much longer text here"]}).json()
    v = [np.frombuffer(base64.b64decode(d["embedding"]), dtype=np.float32) for d in r["data"]]
    assert [d["index"] for d in r["data"]] == [0, 1] and all(abs(np.linalg.norm(x) - 1) < 1e-5 for x in v)
    assert r["usage"]["prompt_tokens"] == len("query: a") + len("passage: much longer text here") + 2
    # batching and padding must not change a vector
    one = c.post("/v1/embeddings", json={"input": "query: a"}).json()["data"][0]["embedding"]
    assert np.allclose(one, v[0], atol=1e-5)
    # token ids (what the trainer sends) give the same vector as the text
    ids = Tok()("query: a")["input_ids"]
    assert np.allclose(c.post("/v1/embeddings", json={"input": [ids]}).json()["data"][0]["embedding"], one, atol=1e-5)
    assert c.post("/v1/embeddings", json={"input": "x" * 100}).status_code == 400       # over max-model-len
    assert c.post("/v1/embeddings", json={"input": "x" * 100, "truncate_prompt_tokens": 10}).status_code == 200
    assert c.post("/v1/embeddings", json={"model": "other", "input": "x"}).status_code == 404
    # and clm's own Embedder accepts its responses
    one_b = dataclasses.replace(ENCODERS["nemotron-3-embed-1b"], hidden_size=8)
    e = Embedder(url="http://t/v1/embeddings", encoder=one_b, tokenizer=None)
    e.session = c
    vec, tokens = e.embed(["hello"], "state")
    assert vec.shape == (1, 8) and tokens > 0
