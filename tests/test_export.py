"""clm-export: a trained model to a folder, an S3 bucket, or a .tar.gz download."""
from __future__ import annotations

import hashlib
import json
import os
import tarfile

import pytest

from helpers import NEMO, fake_embedder, save_head
from clm import export
from clm.engine import Engine


def _head_with_card(tmp_path, name="tickets"):
    pt = save_head(tmp_path / f"{name}.pt", encoder=NEMO)
    card = {"name": name, "test_accuracy": 0.88,
            "questions": {"label": {"type": "choice", "instructions": "Topic?", "criteria": {"billing": "Money.", "tech": "Bugs."}}}}
    (tmp_path / f"{name}.json").write_text(json.dumps(card))
    return pt


def test_export_to_folder_writes_head_card_and_checksums(tmp_path):
    pt = _head_with_card(tmp_path)
    res = export.export_model(pt, str(tmp_path / "out"), "tickets")
    names = [f["name"] for f in res["files"]]
    assert names == ["tickets.pt", "tickets.json", "tickets.sha256"]
    card = json.load(open(tmp_path / "out" / "tickets.json"))
    assert card["encoder"]["hf_id"] == NEMO.hf_id and card["test_accuracy"] == 0.88
    assert card["questions"]["label"]["criteria"]["billing"] == "Money." and card["head"]["width"] == 16
    # the checksum file verifies the copy
    sums = dict(reversed(line.split("  ")) for line in (tmp_path / "out" / "tickets.sha256").read_text().splitlines())
    for fname, digest in sums.items():
        assert hashlib.sha256((tmp_path / "out" / fname).read_bytes()).hexdigest() == digest
    assert card["sha256"]["tickets.pt"] == sums["tickets.pt"]


def test_export_card_from_a_training_run_folder(tmp_path):
    run = tmp_path / "20261018-120000-abcd"
    run.mkdir()
    pt = save_head(run / "best_head.pt", encoder=NEMO)
    (run / "summary.json").write_text(json.dumps({"test": {"acc": 0.5, "per_question": {"label": 0.5}}, "raw_test": 0.4,
                                                  "majority_test": 0.3, "best_epoch": 3}))
    (run / "questions.json").write_text(json.dumps({"label": {"type": "choice", "instructions": "Q?", "criteria": {"a": "A"}}}))
    res = export.export_model(pt, str(tmp_path / "out"), "my-model")
    card = res["card"]
    assert card["test_accuracy"] == 0.5 and card["clm_raw_accuracy"] == 0.4 and card["best_epoch"] == 3
    assert card["questions"]["label"]["criteria"] == {"a": "A"} and card["name"] == "my-model"
    # resolve(): a run checkpoint is named after its folder unless the job names it
    assert export.resolve(str(run / "best_head.pt"), None, None) == (str(run / "best_head.pt"), "20261018-120000-abcd")
    (run / "job.json").write_text(json.dumps({"model": "my-model", "checkpoint": str(run / "best_head.pt")}))
    assert export.resolve("my-model", str(tmp_path / "nowhere"), str(tmp_path)) == (str(run / "best_head.pt"), "my-model")
    with pytest.raises(export.ExportError, match="no model"):
        export.resolve("other", str(tmp_path / "nowhere"), str(tmp_path))


def test_export_to_s3_uploads_three_files(tmp_path, monkeypatch):
    pt = _head_with_card(tmp_path)
    uploads = []

    class FakeS3:
        def upload_file(self, path, bucket, key):
            uploads.append((os.path.basename(path), bucket, key))

    seen = {}
    monkeypatch.setattr(export, "_s3_client", lambda endpoint: seen.setdefault("endpoint", endpoint) and FakeS3() or FakeS3())
    res = export.export_model(pt, "s3://my-bucket/clm/models", "tickets", endpoint_url="http://minio:9000")
    assert seen["endpoint"] == "http://minio:9000"
    assert [u[2] for u in uploads] == ["clm/models/tickets.pt", "clm/models/tickets.json", "clm/models/tickets.sha256"]
    assert res["dest"] == "s3://my-bucket/clm/models/" and res["files"][0]["location"] == "s3://my-bucket/clm/models/tickets.pt"
    with pytest.raises(export.ExportError, match="s3://bucket/prefix"):
        export.export_model(pt, "s3://", "tickets")


def test_export_without_boto3_explains(tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_boto(name, *a, **k):
        if name == "boto3":
            raise ImportError("no boto3")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_boto)
    pt = _head_with_card(tmp_path)
    with pytest.raises(export.ExportError, match="boto3"):
        export.export_model(pt, "s3://bucket/x/", "tickets")


def test_export_errors(tmp_path):
    pt = _head_with_card(tmp_path)
    with pytest.raises(export.ExportError, match="where to export"):
        export.export_model(pt, "   ", "tickets")
    (tmp_path / "afile").write_text("x")
    with pytest.raises(export.ExportError, match="is a file"):
        export.export_model(pt, str(tmp_path / "afile"), "tickets")
    with pytest.raises(export.ExportError, match="no checkpoint"):
        export.export_model(str(tmp_path / "missing.pt"), str(tmp_path / "out"), "x")


def test_export_cli_and_archive(tmp_path, capsys):
    pt = _head_with_card(tmp_path)
    export.main([str(pt), str(tmp_path / "cli-out")])
    out = capsys.readouterr().out
    assert "tickets.pt" in out and "clm-serve --model tickets=" in out and "88% on its test set" in out
    export.main([str(pt), str(tmp_path / "tgz"), "--archive"])
    with tarfile.open(tmp_path / "tgz" / "tickets.tar.gz") as tar:
        assert sorted(tar.getnames()) == ["tickets.json", "tickets.pt", "tickets.sha256"]


# ----------------------------------------------------------------------------- the API
def _client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from clm.server import create_app
    monkeypatch.delenv("CLM_CKPT", raising=False)
    pt = _head_with_card(tmp_path)
    eng = Engine(fake_embedder(NEMO), models={"tickets": pt}, device="cpu", action_cache="0")
    return TestClient(create_app(eng, api_key="k"))


def test_api_download_and_export(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    h = {"Authorization": "Bearer k"}
    assert c.get("/v1/models/tickets/download").status_code == 401
    r = c.get("/v1/models/tickets/download", headers=h)
    assert r.status_code == 200 and r.headers["content-type"] == "application/gzip"
    assert 'filename="tickets.tar.gz"' in r.headers["content-disposition"]
    (tmp_path / "dl.tar.gz").write_bytes(r.content)
    with tarfile.open(tmp_path / "dl.tar.gz") as tar:
        assert sorted(tar.getnames()) == ["tickets.json", "tickets.pt", "tickets.sha256"]
    assert c.get("/v1/models/nope/download", headers=h).status_code == 404
    assert c.get("/v1/models/clm-raw/download", headers=h).status_code == 422

    r = c.post("/v1/models/tickets/export", headers=h, json={"dest": str(tmp_path / "api-out")})
    assert r.status_code == 200, r.text
    assert [f["name"] for f in r.json()["files"]] == ["tickets.pt", "tickets.json", "tickets.sha256"]
    assert os.path.exists(tmp_path / "api-out" / "tickets.pt")
    assert c.post("/v1/models/tickets/export", headers=h, json={"dest": ""}).status_code == 422
    assert c.post("/v1/models/tickets/export", headers=h, json={"nope": 1}).status_code == 422
