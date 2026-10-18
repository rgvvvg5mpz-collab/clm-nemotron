"""The playground's training API (/v1/train/*)."""
from __future__ import annotations

import base64
import os


from helpers import REPO, fake_embedder
from clm.engine import Engine


# ----------------------------------------------------------------------------- playground training API
def _train_client(tmp_path):
    from fastapi.testclient import TestClient
    from clm.server import create_app
    from clm.train_jobs import TrainingJobs
    eng = Engine(fake_embedder(), device="cpu", action_cache="0")
    eng.embedder.healthy = lambda: True
    jobs = TrainingJobs(eng, str(tmp_path / "runs"))
    return TestClient(create_app(eng, jobs=jobs)), eng, jobs


def _wait(client, job_id, timeout=120):
    import time as _t
    end = _t.time() + timeout
    while _t.time() < end:
        j = client.get(f"/v1/train/jobs/{job_id}").json()
        if j["status"] in ("done", "failed", "stopped"):
            return j
        _t.sleep(0.2)
    raise AssertionError("training did not finish")


def test_playground_trains_and_serves_a_model(tmp_path, monkeypatch):
    monkeypatch.delenv("CLM_CKPT", raising=False)
    c, eng, jobs = _train_client(tmp_path)
    assert c.get("/health").json()["training"] is True
    sample = os.path.join(REPO, "src", "clm", "static", "sample", "support-tickets.csv")
    info = c.post("/v1/train/inspect", json={"filename": "support-tickets.csv",
                                             "content_b64": base64.b64encode(open(sample, "rb").read()).decode()}).json()
    assert info["format"] == "simple" and info["suggested"] == {"text": "text", "label": "label"}
    assert info["has_split"] and info["rows"] == 400 and set(info["stats"]["label"]["values"]) == {
        "account", "billing", "delivery", "refund", "technical"}

    spec = {"upload_id": info["upload_id"], "text_column": "text", "label_column": "label",
            "question": "What is this conversation about?", "type": "choice", "epochs": 2, "name": "Ticket topic",
            "options": {k: f"{k} problems" for k in info["stats"]["label"]["values"]}}
    # a label without a description entry is refused before anything runs
    bad = dict(spec, options={"billing": "x"})
    assert c.post("/v1/train/jobs", json=bad).status_code == 422
    job = c.post("/v1/train/jobs", json=spec).json()
    assert job["model"] == "ticket-topic" and job["rows"] == {"train": 300, "test": 100}
    done = _wait(c, job["id"])
    assert done["status"] == "done", done.get("error")
    assert done["served"] and 0 <= done["result"]["test_acc"] <= 1 and done["history"]
    assert "ticket-topic" in [m["name"] for m in c.get("/v1/models").json()["models"]]
    r = c.post("/v1/systemone", json={"state": "I was charged twice", "model": "ticket-topic",
                                      "questions": {"label": done["question"]}})
    assert r.status_code == 200 and r.json()["answers"]["label"]["choice"] in spec["options"]

    # a restarted server serves it again
    from clm.train_jobs import TrainingJobs
    eng2 = Engine(fake_embedder(), device="cpu", action_cache="0")
    TrainingJobs(eng2, str(tmp_path / "runs"))
    assert "ticket-topic" in eng2.heads


def test_playground_training_errors(tmp_path):
    c, _, _ = _train_client(tmp_path)
    up = lambda name, data: c.post("/v1/train/inspect", json={"filename": name,  # noqa: E731
                                                              "content_b64": base64.b64encode(data).decode()})
    assert up("notes.txt", b"hello").status_code == 422
    assert up("bad.jsonl", b"{not json").status_code == 422
    info = up("t.csv", b"text,label\n" + b"".join(b"row %d,%s\n" % (i, b"a" if i % 2 else b"b") for i in range(30))).json()
    base = {"upload_id": info["upload_id"], "question": "Which?", "options": {"a": "A things", "b": "B things"}}
    assert c.post("/v1/train/jobs", json=dict(base, text_column="text", label_column="text")).status_code == 422
    assert c.post("/v1/train/jobs", json=dict(base, text_column="text", label_column="label",
                                              question=" ")).status_code == 422
    assert c.post("/v1/train/jobs", json=dict(base, text_column="text", label_column="label",
                                              upload_id="../../etc")).status_code == 422
    assert c.get("/v1/train/jobs/nope").status_code == 404


def test_server_without_training_hides_it():
    from fastapi.testclient import TestClient
    from clm.server import create_app
    eng = Engine(fake_embedder(), device="cpu", action_cache="0")
    eng.embedder.healthy = lambda: True
    c = TestClient(create_app(eng))
    assert c.get("/health").json()["training"] is False
    assert c.get("/v1/train/jobs").status_code in (404, 405)
