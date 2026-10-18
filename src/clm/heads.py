"""Projection heads: architecture, checkpoint loading and download.

A CLM checkpoint (as trained by ``train/finetune.py``) is a ``torch.save``
dict with ``state_head`` / ``action_head`` state dicts, ``logit_scale`` (log of
the InfoNCE temperature inverse) and ``cfg`` (``width``, ``depth``, optional
``projection_dim``, ``activation``, ``layernorm``, ``residual``, ``hidden_size``
and ``encoder``, the ``clm.encoders`` identity it was trained on).  Each head maps
an encoder embedding to a ``projection_dim``-d vector; the score of a
(state, candidate) pair is ``exp(logit_scale) * cos(state_head(s), action_head(c))``.

There is no default head: serve one trained on the configured encoder, or fetch
your own from the Hub with ``clm-download --repo ... --file ...``.
"""
from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np

from .encoders import EncoderSpec, check_compatible, checkpoint_encoder

HIDDEN = 4096          # default encoder embedding width (Nemotron-3-Embed-8B)
PROJ_DIM = 512
DEFAULT_CKPT_DIR = os.environ.get("CLM_CKPT_DIR", os.path.join(os.path.expanduser("~"), ".cache", "clm"))


def default_device() -> str:
    """``CLM_DEVICE`` if set, else the GPU when torch sees one (the heads are tiny but the
    projection then runs next to the encoder instead of copying embeddings back to host)."""
    d = os.environ.get("CLM_DEVICE")
    if d:
        return d
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def make_head(width: int, depth: int = 2, proj: int = PROJ_DIM, activation: str = "gelu",
              layernorm: bool = False, residual: bool = False, hidden: int = HIDDEN):
    """``hidden -> width -> ... -> proj`` MLP with optional LayerNorm / residual hidden blocks."""
    import torch.nn as nn
    act = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(hidden, width)
            self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 2))
            self.norms = nn.ModuleList((nn.LayerNorm(width) if layernorm else nn.Identity())
                                       for _ in range(depth - 2))
            self.out = nn.Linear(width, proj)
            self.act = act()
            self.residual = residual

        def forward(self, x):
            x = self.act(self.inp(x))
            for lin, nrm in zip(self.hidden, self.norms):
                h = self.act(nrm(lin(x)))
                x = x + h if self.residual else h
            return self.out(x)

    return Head()


class HeadPair:
    """State head + action head from one checkpoint, hot-reloaded when the file changes.

    With ``encoder`` set, a checkpoint trained on any other encoder is refused at load.
    """

    def __init__(self, name: str, path: str, device: str | None = None, encoder: EncoderSpec | None = None):
        self.name, self.path, self.device, self.encoder = name, path, device or default_device(), encoder
        self.mtime: float | None = None
        self.state_head = self.action_head = None
        self.generation = 0          # bumped on every (re)load; stamps cached projections
        self.proj_dim = PROJ_DIM
        self.scale = 1.0
        self.cfg: dict[str, Any] = {}
        self._lock = threading.Lock()

    def _load(self) -> None:
        import torch
        ck = torch.load(self.path, map_location="cpu", weights_only=True)   # tensors and plain data only: no code
        cfg = dict(ck["cfg"])
        if self.encoder is not None:
            check_compatible(checkpoint_encoder(cfg), self.encoder, what=f"head {self.name!r} ({self.path})")
        kw = dict(width=cfg["width"], depth=cfg["depth"],
                  proj=ck.get("projection_dim", cfg.get("projection_dim", PROJ_DIM)),
                  activation=cfg.get("activation", "gelu"), layernorm=cfg.get("layernorm", False),
                  residual=cfg.get("residual", False), hidden=cfg.get("hidden_size", HIDDEN))
        sh, ah = make_head(**kw), make_head(**kw)
        sh.load_state_dict(ck["state_head"]); ah.load_state_dict(ck["action_head"])
        sh.eval().to(self.device); ah.eval().to(self.device)
        self.state_head, self.action_head, self.cfg = sh, ah, cfg
        self.generation += 1
        self.proj_dim = kw["proj"]
        self.scale = float(torch.as_tensor(ck["logit_scale"]).float().exp().clamp(max=100.0))

    def ensure(self) -> "HeadPair":
        with self._lock:
            m = os.path.getmtime(self.path)
            if m != self.mtime:
                self._load(); self.mtime = m
        return self

    def _project(self, head, x: np.ndarray):
        """-> [n, proj] L2-normalised projections, left on ``self.device``."""
        import torch
        self.ensure()
        with torch.no_grad():
            return torch.nn.functional.normalize(head(torch.from_numpy(x).to(self.device)), dim=-1)

    def project_states(self, states: np.ndarray):
        return self._project(self.state_head, states)

    def project_actions(self, candidates: np.ndarray):
        return self._project(self.action_head, candidates)

    def project(self, states: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """L2-normalised projections of [n, hidden] state and candidate embeddings."""
        zs, zc = self.project_states(states), self.project_actions(candidates)
        return zs.cpu().numpy(), zc.cpu().numpy()

    @property
    def namespace(self) -> str:
        """Identity of these exact weights, for cache keys."""
        return f"{self.name}@{self.generation}"

    @property
    def n_params(self) -> int:
        self.ensure()
        return sum(p.numel() for h in (self.state_head, self.action_head) for p in h.parameters())


def download(repo: str, filename: str, dest_dir: str = DEFAULT_CKPT_DIR, force: bool = False) -> str:
    """Fetch a checkpoint from the Hugging Face Hub; returns the local path."""
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest) and not force:
        return dest
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo, filename, local_dir=dest_dir, force_download=force)
    except ImportError:
        import requests
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        os.replace(tmp, dest)
        return dest


def download_main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Download a CLM projection-head checkpoint from the Hugging Face Hub.")
    ap.add_argument("--repo", required=True, help="Hugging Face model repo holding the head")
    ap.add_argument("--file", required=True, help="checkpoint file in that repo, e.g. best_head.pt")
    ap.add_argument("--dest", default=DEFAULT_CKPT_DIR)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    print(download(a.repo, a.file, a.dest, a.force))
