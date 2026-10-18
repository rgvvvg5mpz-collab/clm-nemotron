"""Encoder side: an OpenAI-compatible ``/v1/embeddings`` endpoint (vLLM pooling
server) plus an LRU cache of L2-normalised embeddings.

What is sent for a text is set by the encoder profile (``clm.encoders``): states
and candidates get their role's prefix, and when the encoder's tokenizer is
available an over-long body is truncated here, so the prefix always survives.
The default encoder is NVIDIA Nemotron-3-Embed-8B, e.g.

    vllm serve nvidia/Nemotron-3-Embed-8B-BF16 --served-model-name nemotron-3-embed-8b \\
         --enable-prefix-caching --max-model-len 8192 --port 8090
"""
from __future__ import annotations

import base64
import threading
from collections import OrderedDict
from typing import Any

import numpy as np
import requests

from .encoders import EncoderSpec, get_encoder, load_tokenizer


class EmbedderError(RuntimeError):
    pass


def l2(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


class Embedder:
    def __init__(self, url: str = "http://127.0.0.1:8090/v1/embeddings", model: str | None = None,
                 max_tokens: int | None = None, cache_size: int = 200_000, batch: int = 32,
                 timeout: float = 300.0, api_key: str | None = None,
                 encoder: "str | EncoderSpec | None" = None, tokenizer: Any = "auto"):
        self.spec = get_encoder(encoder)
        self.url, self.batch, self.timeout = url, batch, timeout
        self.model = model or self.spec.name
        self.max_tokens = self.spec.max_tokens if max_tokens is None else max_tokens
        # "auto": the encoder's own tokenizer when transformers is installed, for truncation
        self.tokenizer = load_tokenizer(self.spec) if tokenizer == "auto" else tokenizer
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.cache_size = cache_size
        self._lock = threading.Lock()
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def _fetch(self, texts: list[str]) -> tuple[list[np.ndarray], int]:
        body: dict[str, Any] = {"model": self.model, "input": texts, "encoding_format": "base64"}
        if self.max_tokens:
            body["truncate_prompt_tokens"] = self.max_tokens
        try:
            r = self.session.post(self.url, json=body, timeout=self.timeout)
        except requests.RequestException as e:
            raise EmbedderError(f"embedder unreachable at {self.url}: {e}") from e
        if r.status_code != 200:
            raise EmbedderError(f"embedder error {r.status_code}: {r.text[:300]}")
        j = r.json()
        out: list[np.ndarray] = [None] * len(texts)  # type: ignore[list-item]
        for d in j["data"]:
            e = d["embedding"]
            v = (np.frombuffer(base64.b64decode(e), dtype=np.float32) if isinstance(e, str)
                 else np.asarray(e, dtype=np.float32))
            if v.shape[-1] != self.spec.hidden_size:
                raise EmbedderError(f"embedder returned {v.shape[-1]}-d vectors but encoder "
                                    f"{self.spec.name!r} is {self.spec.hidden_size}-d; is {self.url} "
                                    f"serving {self.spec.hf_id}?")
            out[d["index"]] = l2(v.astype(np.float32))
        return out, int((j.get("usage") or {}).get("prompt_tokens", 0) or 0)

    @property
    def dim(self) -> int:
        return self.spec.hidden_size

    def embed(self, texts: list[str], role: str | None = None) -> tuple[np.ndarray, int]:
        """-> ([n, hidden] L2-normalised embeddings, encoder tokens spent on cache misses).

        ``role`` (``"state"`` / ``"action"``) picks the encoder's prefix; ``None``
        embeds the texts verbatim.  The cache is keyed by the text actually sent.
        """
        texts = [self.spec.prepare(t, role, self.tokenizer, self.max_tokens) for t in texts]
        vecs: dict[str, np.ndarray] = {}
        todo: list[str] = []
        with self._lock:
            for t in dict.fromkeys(texts):
                v = self.cache.get(t)
                if v is None:
                    todo.append(t)
                else:
                    self.cache.move_to_end(t); vecs[t] = v
        tokens = 0
        for i in range(0, len(todo), self.batch):
            chunk = todo[i:i + self.batch]
            got, tk = self._fetch(chunk)
            tokens += tk
            with self._lock:
                for t, v in zip(chunk, got):
                    vecs[t] = v; self.cache[t] = v
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
        return np.stack([vecs[t] for t in texts]), tokens

    def healthy(self) -> bool:
        try:
            return self.session.get(self.url.rsplit("/v1/", 1)[0] + "/v1/models", timeout=5).status_code == 200
        except requests.RequestException:
            return False
