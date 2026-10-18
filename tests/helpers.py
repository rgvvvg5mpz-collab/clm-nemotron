"""Shared fakes for the tests: no GPU, network or encoder server needed."""
from __future__ import annotations

import base64
import dataclasses
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "train"))

from clm.embedder import Embedder  # noqa: E402
from clm.encoders import ENCODERS  # noqa: E402
from clm.heads import make_head  # noqa: E402

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
