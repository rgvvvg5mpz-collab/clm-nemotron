"""Heads, the engine, compatibility checks and clm-raw serving."""
from __future__ import annotations

import pytest

from helpers import NEMO, OTHER, fake_embedder, save_head
from clm.encoders import IncompatibleEncoder, get_encoder
from clm.engine import Engine
from clm.heads import HeadPair


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
