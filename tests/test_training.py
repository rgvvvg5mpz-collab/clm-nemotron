"""Training heads: research/finetune.py, the shared trainer and clm-train."""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from helpers import NEMO, REPO, CharTokenizer
from clm.heads import HeadPair


# ----------------------------------------------------------------------------- training
def test_recipe_matches_serving_text():
    from clm.embed_backends import Recipe
    r = Recipe.__new__(Recipe)
    r.spec, r.tok, r.max_len = NEMO, CharTokenizer(), 40
    msgs = [{"role": "user", "content": "fix the bug"}, {"role": "assistant", "content": "ok"}]
    ids = r.ids(msgs, "state")
    assert CharTokenizer().decode(ids) == "query: user: fix the bug\n\nassistant: ok"
    assert ids[0] == 0                                         # special tokens, as the server adds them
    long_ids = r.ids("x" * 500, "action")
    assert len(long_ids) <= 40 and CharTokenizer().decode(long_ids).startswith("passage: ")


def test_finetune_needs_a_data_source(tmp_path):
    env = {**os.environ, "CLM_ENCODER": ""}
    p = subprocess.run([sys.executable, os.path.join(REPO, "research", "finetune.py"), "--task", "clm",
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
    p = subprocess.run([sys.executable, os.path.join(REPO, "research", "finetune.py"), "--task", "clm",
                        "--emb-dir", str(emb), "--out-dir", str(tmp_path / "out"), "--batch", "2",
                        "--epochs", "1"], capture_output=True, text=True)
    assert p.returncode != 0 and "records no encoder" in p.stderr


# ----------------------------------------------------------------------------- typed-decisions training
def test_choice_training_runs_and_reports_raw_baseline(tmp_path):
    """End to end on CPU: embeddings come from a pre-filled cache, so no encoder is needed."""
    import hashlib
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq
    from clm import typed_data as adapters

    rng = np.random.default_rng(0)
    rows = []
    for i in range(24):
        team = "billing" if i % 2 else "technical"
        rows.append({"id": f"r{i}", "split": "test" if i >= 16 else "train",
                     "state": f"Ticket {i}: {'refund please' if i % 2 else 'app crashes'}",
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
    p = subprocess.run([sys.executable, os.path.join(REPO, "research", "finetune.py"), "--task", "choice",
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

    # a file without a split column must not double as its own test set
    nosplit = tmp_path / "nosplit.parquet"
    pq.write_table(pa.Table.from_pylist([{k: v for k, v in r.items() if k != "split"} for r in rows]), nosplit)
    p = subprocess.run([sys.executable, os.path.join(REPO, "research", "finetune.py"), "--task", "choice",
                        "--data", str(nosplit), "--embed-cache", str(cache_dir), "--out-dir", str(tmp_path / "x")],
                       capture_output=True, text=True)
    assert p.returncode != 0 and "no 'split' column" in p.stderr


# ----------------------------------------------------------------------------- clm-train
def _prefill_cache(rows, cache_dir, name="nemotron-3-embed-8b", max_len=2048):
    """Write the embedding cache clm-train would build, so no encoder is needed."""
    import hashlib
    from clm.typed_data import typed_decision_examples
    keys = set()
    for e in typed_decision_examples(rows):
        keys.add("state\x00" + e.state_text)
        keys.update("action\x00" + t for t in e.candidates)
    keys = sorted(keys)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(os.path.join(cache_dir, f"choice_{name}_{max_len}.npz"),
             keys=np.array([hashlib.sha1(k.encode()).hexdigest() for k in keys]),
             vecs=np.random.default_rng(0).standard_normal((len(keys), 4096)).astype(np.float16))


def test_clm_train_on_a_text_label_csv(tmp_path):
    import csv
    import json
    from clm.train_cli import main as clm_train
    from clm.typed_data import read_rows, simple_rows

    data = tmp_path / "tickets.csv"
    with open(data, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "customer"])
        for i in range(40):
            w.writerow([f"Ticket {i}: {'I was charged twice' if i % 2 else 'the app crashes'}",
                        "billing" if i % 2 else "technical", f"c{i // 2}"])
    options = {"billing": "A charge, invoice, refund or payment problem.",
               "technical": "The product is broken, erroring or down."}
    question = {"type": "choice", "instructions": "Which team should handle this?", "criteria": options}
    _prefill_cache(simple_rows(read_rows(str(data)), question), tmp_path / "cache")

    out = tmp_path / "run"
    clm_train(["--data", str(data), "--question", "Which team should handle this?",
               "--options", json.dumps(options), "--group-column", "customer", "--test-frac", "0.25",
               "--embed-cache", str(tmp_path / "cache"), "--encoder", "nemotron-3-embed-8b",
               "--out", str(out), "--epochs", "2", "--width", "32", "--depth", "2", "--batch", "8"])
    summary = json.load(open(out / "summary.json"))
    assert summary["encoder"] == "nemotron-3-embed-8b" and 0 <= summary["test"]["acc"] <= 1
    assert json.load(open(out / "questions.json")) == {"label": question}
    HeadPair("t", str(out / "best_head.pt"), "cpu", NEMO).ensure()

    # held out by customer: no customer appears in both splits
    from clm.typed_data import holdout
    rows = simple_rows(read_rows(str(data)), question, group_column="customer")
    tr, te = holdout(rows, 0.25, 1234, lambda r: r["group"])
    assert not {r["group"] for r in tr} & {r["group"] for r in te} and len(te) == 10

    # labels outside the options are refused before anything is embedded
    with pytest.raises(ValueError, match="not one of the options"):
        simple_rows([{"text": "x", "label": "sales"}], question)
