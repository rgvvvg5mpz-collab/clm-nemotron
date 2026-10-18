"""Encoder embeddings for training, with the same input format ``clm-serve`` uses.

``Recipe(encoder, max_len).ids(x, role)`` turns a state (``role="state"``: text or
chat messages) or a candidate (``role="action"``) into token ids:

chat messages are rendered as ``role: content`` text, then ``EncoderSpec.prepare``
adds the role prefix (Nemotron's ``query: `` / ``passage: ``) and truncates the
body exactly as the serving embedder does, and the result is tokenized with the
encoder's special tokens.

Backends take token ids: ``OfflineBackend`` (in-process vLLM) and ``ServerBackend``
(``vllm serve <model> --runner pooling``).
"""
from __future__ import annotations

import base64

import numpy as np

from .embedder import EmbedderError, l2
from .encoders import EncoderSpec, get_encoder, render_messages


class Recipe:
    def __init__(self, encoder: "str | EncoderSpec | None" = None, max_len: int = 8192, model: str | None = None):
        """``model`` overrides where the tokenizer is loaded from (default: the encoder's HF id)."""
        from transformers import AutoTokenizer
        self.spec = get_encoder(encoder)
        self.tok = AutoTokenizer.from_pretrained(model or self.spec.source())
        self.max_len = max_len

    def ids(self, x, role: str) -> list[int]:
        """Token ids for a state (text or chat messages) or a candidate, as ``clm-serve`` sends them."""
        if not isinstance(x, str):
            x = render_messages(x)
        text = self.spec.prepare(x, role, self.tok, self.max_len)
        ids = self._flatten(self.tok(text, add_special_tokens=self.spec.add_special_tokens)["input_ids"])
        return ids[:self.max_len]       # decode/re-encode can overshoot by a token or two

    @staticmethod
    def _flatten(out) -> list[int]:
        # transformers 4.x returns List[int]; 5.x may return a BatchEncoding / nested list
        if hasattr(out, "keys"):
            out = out["input_ids"]
        if out and isinstance(out[0], list):
            out = out[0]
        return list(out)


class OfflineBackend:
    """In-process vLLM pooling model, loaded on first use."""

    def __init__(self, model: str, max_len: int, gpu_mem: float = 0.85):
        self.model, self.max_len, self.gpu_mem = model, max_len, gpu_mem
        self._llm = None

    def embed(self, id_lists: list[list[int]]) -> np.ndarray:
        from vllm.inputs import TokensPrompt
        if self._llm is None:
            from vllm import LLM
            self._llm = LLM(model=self.model, runner="pooling", enforce_eager=False,
                            max_model_len=self.max_len, trust_remote_code=True,
                            gpu_memory_utilization=self.gpu_mem)
        outs = self._llm.embed([TokensPrompt(prompt_token_ids=ids) for ids in id_lists], use_tqdm=True)
        return l2(np.stack([np.asarray(o.outputs.embedding, dtype=np.float32) for o in outs]))


class ServerBackend:
    """OpenAI-compatible ``/v1/embeddings`` endpoint."""

    def __init__(self, url: str, model: str, batch: int = 64, timeout: float = 600.0,
                 api_key: str | None = None):
        import requests
        self.url, self.model, self.batch, self.timeout = url, model, batch, timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def embed(self, id_lists: list[list[int]]) -> np.ndarray:
        import requests
        out: list[np.ndarray] = []
        for i in range(0, len(id_lists), self.batch):
            chunk = id_lists[i:i + self.batch]
            body = {"model": self.model, "input": chunk, "encoding_format": "base64"}
            try:
                r = self.session.post(self.url, json=body, timeout=self.timeout)
            except requests.RequestException as e:
                raise EmbedderError(f"embedder unreachable at {self.url}: {e}") from e
            if r.status_code != 200:
                raise EmbedderError(f"embedder error {r.status_code}: {r.text[:300]}")
            got: list = [None] * len(chunk)
            for d in r.json()["data"]:
                e = d["embedding"]
                got[d["index"]] = (np.frombuffer(base64.b64decode(e), dtype=np.float32) if isinstance(e, str)
                                   else np.asarray(e, dtype=np.float32))
            out.extend(got)
            if (i // self.batch) % 50 == 0:
                print(f"[embed] server {min(i + self.batch, len(id_lists))}/{len(id_lists)}", flush=True)
        return l2(np.stack(out).astype(np.float32))


def make_backend(url: str | None, model: str, max_len: int, gpu_mem: float = 0.85, served_name: str | None = None,
                 hidden_size: int | None = None):
    """Server backend when ``url`` is given, else offline vLLM; checks the embedding width."""
    backend = ServerBackend(url, served_name or model) if url else OfflineBackend(model, max_len, gpu_mem)
    if hidden_size is None:
        return backend
    inner = backend.embed

    def embed(id_lists):
        out = inner(id_lists)
        if out.shape[-1] != hidden_size:
            raise EmbedderError(f"encoder returned {out.shape[-1]}-d vectors, expected {hidden_size}-d for {model}")
        return out
    backend.embed = embed
    return backend
