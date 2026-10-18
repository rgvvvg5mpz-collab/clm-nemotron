#!/usr/bin/env python3
"""A small OpenAI-compatible ``/v1/embeddings`` server for machines without vLLM.

vLLM needs Linux and an NVIDIA GPU.  This serves a ``clm.encoders`` encoder with
sentence-transformers on Apple silicon (MPS) or the CPU, behind the same two
routes ``clm-serve`` and ``train/finetune.py --embed-url`` use, so either can
point at it unchanged:

    pip install -e ".[mac]"
    python tools/local_embed_server.py --encoder nemotron-3-embed-1b      # :8090
    clm-serve --encoder nemotron-3-embed-1b

It behaves like vLLM where the callers depend on it:

* ``input`` is a string, a list of strings, or a list of token-id lists (the
  trainer sends ids); strings are tokenized with the model's special tokens;
* ``truncate_prompt_tokens: k`` keeps the *last* k tokens; an input longer than
  ``--max-model-len`` without it is a 400;
* no prompts are added here: callers send ``query: `` / ``passage: `` themselves
  (``clm.encoders``), as vLLM's OpenAI route requires;
* ``encoding_format`` ``float`` or ``base64`` (float32), and ``usage.prompt_tokens``.

It is one process running one batch at a time: fine for the playground and small
training sets, far slower than vLLM on a GPU.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import os
import sys
import threading
import time
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from clm.encoders import ENCODERS, get_encoder  # noqa: E402


class LocalEncoder:
    """Token ids -> L2-normalised sentence embeddings, batched by token budget."""

    def __init__(self, model, tokenizer, device: str, batch_tokens: int = 16384):
        self.model, self.tokenizer, self.device, self.batch_tokens = model, tokenizer, device, batch_tokens
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self._lock = threading.Lock()

    def tokenize(self, text: str) -> list[int]:
        ids = self.tokenizer(text, add_special_tokens=True)["input_ids"]
        return list(ids[0] if ids and isinstance(ids[0], list) else ids)

    def embed(self, id_lists: list[list[int]]) -> np.ndarray:
        import torch
        order = sorted(range(len(id_lists)), key=lambda i: len(id_lists[i]))
        out: list[np.ndarray | None] = [None] * len(id_lists)
        batch: list[int] = []

        def run(idx: list[int]) -> None:
            width = max(len(id_lists[i]) for i in idx)
            ids = torch.full((len(idx), width), self.pad_id, dtype=torch.long)
            mask = torch.zeros((len(idx), width), dtype=torch.long)
            for row, i in enumerate(idx):
                ids[row, :len(id_lists[i])] = torch.tensor(id_lists[i])
                mask[row, :len(id_lists[i])] = 1
            with self._lock, torch.inference_mode():
                emb = self.model({"input_ids": ids.to(self.device),
                                  "attention_mask": mask.to(self.device)})["sentence_embedding"]
            emb = emb.float().cpu().numpy()
            for row, i in enumerate(idx):
                out[i] = emb[row]

        for i in order:      # sorted by length, so a batch's padding stays small
            longest = max([len(id_lists[j]) for j in batch] + [len(id_lists[i])])
            if batch and longest * (len(batch) + 1) > self.batch_tokens:
                run(batch); batch = []
            batch.append(i)
        if batch:
            run(batch)
        v = np.stack(out).astype(np.float32)
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def create_app(encoder: LocalEncoder, names: set[str], max_model_len: int, dim: int):
    app = FastAPI(title="local /v1/embeddings")
    served = sorted(names)[0]

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": n, "object": "model", "owned_by": "local"} for n in sorted(names)]}

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        body: dict[str, Any] = await request.json()
        model = body.get("model") or served
        if model not in names:
            raise HTTPException(404, f"model {model!r} not served; available: {sorted(names)}")
        inp = body.get("input")
        if isinstance(inp, str) or (isinstance(inp, list) and inp and isinstance(inp[0], int)):
            inp = [inp]
        if not isinstance(inp, list) or not inp:
            raise HTTPException(400, "input must be a string, a list of strings or a list of token-id lists")
        id_lists = [encoder.tokenize(x) if isinstance(x, str) else [int(t) for t in x] for x in inp]
        k = body.get("truncate_prompt_tokens")
        if k:
            id_lists = [ids[-int(k):] for ids in id_lists]          # vLLM: keep the last k tokens
        too_long = [i for i, ids in enumerate(id_lists) if len(ids) > max_model_len]
        if too_long:
            raise HTTPException(400, f"input {too_long[0]} has {len(id_lists[too_long[0]])} tokens, "
                                     f"over --max-model-len {max_model_len}; send truncate_prompt_tokens")
        t0 = time.perf_counter()
        vecs = await asyncio.get_running_loop().run_in_executor(None, encoder.embed, id_lists)
        n_tok = sum(len(ids) for ids in id_lists)
        dt = time.perf_counter() - t0
        print(f"[embed] {len(id_lists)} inputs, {n_tok} tokens in {dt:.2f}s ({n_tok / max(dt, 1e-9):.0f} tok/s)",
              flush=True)
        b64 = body.get("encoding_format") == "base64"
        data = [{"object": "embedding", "index": i,
                 "embedding": base64.b64encode(v.tobytes()).decode() if b64 else v.tolist()}
                for i, v in enumerate(vecs)]
        return {"object": "list", "model": model, "data": data,
                "usage": {"prompt_tokens": n_tok, "total_tokens": n_tok}}

    app.state.dim = dim
    return app


def pick_device(want: str) -> str:
    import torch
    if want != "auto":
        return want
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", choices=sorted(ENCODERS), default="nemotron-3-embed-1b",
                    help="clm.encoders profile to serve (default: the 1B, which fits a laptop)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default="auto", help="auto (cuda > mps > cpu), mps, cpu or cuda")
    ap.add_argument("--max-model-len", type=int, default=None, help="default: the encoder's token budget")
    ap.add_argument("--batch-tokens", type=int, default=16384,
                    help="padded tokens per forward pass; lower it if memory runs short")
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer
    spec = get_encoder(args.encoder)
    device = pick_device(args.device)
    src = spec.source()                     # the release copy in $CLM_MODELS_DIR, else the Hub
    print(f"[local-embed] loading {spec.hf_id} from {src} on {device} (bf16)...", flush=True)
    t0 = time.time()
    model = SentenceTransformer(src, device=device, model_kwargs={"dtype": torch.bfloat16})
    model.eval()
    dim = model.get_sentence_embedding_dimension()
    if dim != spec.hidden_size:
        raise SystemExit(f"{spec.hf_id} gives {dim}-d embeddings, profile {spec.name} expects {spec.hidden_size}")
    encoder = LocalEncoder(model, model.tokenizer, device, args.batch_tokens)
    encoder.embed([encoder.tokenize("query: warm-up")])
    max_len = args.max_model_len or spec.max_tokens
    print(f"[local-embed] ready in {time.time() - t0:.0f}s: {spec.name} ({dim}-d) at "
          f"http://{args.host}:{args.port}/v1/embeddings, max-model-len {max_len}", flush=True)
    import uvicorn
    app = create_app(encoder, {spec.name, spec.hf_id}, max_len, dim)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
