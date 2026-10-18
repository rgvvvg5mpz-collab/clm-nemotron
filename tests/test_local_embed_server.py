"""tools/local_embed_server.py against a fake model."""
from __future__ import annotations

import base64
import dataclasses
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from helpers import (NEMO, OTHER, REPO, CharTokenizer, FakeSession, fake_embedder, save_head)  # noqa: F401
from clm.embedder import Embedder, EmbedderError  # noqa: F401
from clm.encoders import (ENCODERS, IncompatibleEncoder, check_compatible, checkpoint_encoder,  # noqa: F401
                          get_encoder)
from clm.engine import Engine  # noqa: F401
from clm.heads import HeadPair, make_head  # noqa: F401


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
