"""The inference engine: state + typed questions -> distributions, no HTTP needed.

    from clm import Engine
    engine = Engine(checkpoint="heads/nemotron.pt")        # embedder at :8090, default encoder
    out = engine.answer(state, {"ok": {"type": "noul", "instructions": "Is this fine?"}})
    out["answers"]["ok"]["noul"]

For each question the state (with the question's instructions appended) goes
through the state head and every option's description through the action head;
the softmax over ``scale * cosine`` is the answer.  ``clm-raw`` skips the heads
(cosine in the encoder's own space) as an ablation.

The encoder (``clm.encoders``; default Nemotron-3-Embed-8B) fixes the embedding
width, the state / candidate prefixes, and which checkpoints may be served.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any

from .cache import CacheDisabled, VectorArena
from .client import question_to_dict
from .embedder import Embedder
from .encoders import EncoderSpec, get_encoder
from .heads import HeadPair, default_device
from .schema import answer_from_logits, build_pairs

DEFAULT_MODEL = "clm-latest"
RAW_MODEL = "clm-raw"
RAW_SCALE = 100.0
RAW_SHARE = 0.125      # of the arena, for the raw ablation's wider vectors
RELEASE = "2026-09-19"


class ModelNotFound(KeyError):
    pass


class Engine:
    def __init__(self, embedder: Embedder | None = None, checkpoint: str | None = None,
                 models: dict[str, str] | None = None, checkpoint_dir: str | None = None, device: str | None = None,
                 emb_url: str | None = None, emb_model: str | None = None, action_cache: Any = None,
                 encoder: "str | EncoderSpec | None" = None):
        if embedder is not None and encoder is not None and get_encoder(encoder) != embedder.spec:
            raise ValueError(f"encoder {get_encoder(encoder).name!r} differs from the embedder's "
                             f"{embedder.spec.name!r}")
        self.embedder = embedder or Embedder(url=emb_url or os.environ.get("CLM_EMB_URL", "http://127.0.0.1:8090/v1/embeddings"),
                                             model=emb_model or os.environ.get("CLM_EMB_MODEL"), encoder=encoder)
        self.encoder: EncoderSpec = self.embedder.spec
        device = device or default_device()
        self.heads: dict[str, HeadPair] = {}
        ck = checkpoint or os.environ.get("CLM_CKPT")      # no built-in default head

        def pair(name: str, path: str) -> HeadPair:
            return HeadPair(name, path, device, self.encoder)
        if ck:
            self.heads[DEFAULT_MODEL] = pair(DEFAULT_MODEL, ck)
        for p in sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt"))) if checkpoint_dir else []:
            if not ck or os.path.abspath(p) != os.path.abspath(ck):
                stem = os.path.splitext(os.path.basename(p))[0]
                self.heads.setdefault(stem, pair(stem, p))
        for name, path in (models or {}).items():
            self.heads[name] = pair(name, path)
        for h in self.heads.values():
            h.ensure()
        self.device = device
        self.arena = self._reserve(action_cache)

    def _reserve(self, budget: Any) -> VectorArena | None:
        """Claim the arena up front, so its cost is paid at start-up or not at all.

        Projections are 512-d and encoder embeddings (typically) 4096-d, so the raw
        ablation's pool gets an eighth of the arena and still holds a useful number of rows.
        """
        hidden = self.encoder.hidden_size
        if budget is None:
            budget = os.environ.get("CLM_ACTION_CACHE")
        try:
            arena = VectorArena(self.device, budget)
        except CacheDisabled:
            return None
        for dim in sorted({h.proj_dim for h in self.heads.values()}):
            arena.reserve(dim, RAW_SHARE if dim == hidden else (1.0 - RAW_SHARE))
        arena.reserve(hidden, RAW_SHARE)       # clm-raw works in the encoder's own space
        return arena

    # ------------------------------------------------------------------ cached vectors
    def _cached(self, namespace: str, dim: int, texts: list[str], tokens: list[int], role: str, project=None):
        """Vectors for ``texts`` in ``role`` (state / action), from the arena where possible;
        ``tokens`` collects misses."""
        def compute(missing: list[str]):
            emb, spent = self.embedder.embed(missing, role)
            tokens.append(spent)
            return project(emb) if project else self._to_device(emb)

        if self.arena is None:
            return compute(texts)
        return self.arena.get(namespace, dim, texts, compute)

    def _to_device(self, x):
        import torch
        return torch.from_numpy(x).to(self.device)

    # ------------------------------------------------------------------ models
    def models(self) -> list[dict[str, Any]]:
        """Every served model, with what it answers when the head has a card beside it."""
        desc = {DEFAULT_MODEL: f"Contrastive language model: {self.encoder.hf_id} encoder + trained projection heads",
                RAW_MODEL: (f"Cosine in the raw {self.encoder.hf_id} embedding space, no projection head; "
                            "scores are uncalibrated")}
        names = ([DEFAULT_MODEL] if DEFAULT_MODEL in self.heads else []) + \
            sorted(n for n in self.heads if n != DEFAULT_MODEL) + [RAW_MODEL]
        out = []
        for n in names:
            card = head_card(self.heads[n].path) if n in self.heads else {}
            out.append({"name": n, "release_date": RELEASE, "kind": "raw" if n == RAW_MODEL else "trained",
                        "description": desc.get(n) or card.get("description")
                        or f"Projection-head checkpoint {os.path.basename(self.heads[n].path)}",
                        **{k: v for k, v in card.items() if k != "description"}})
        return out

    @property
    def default_model(self) -> str:
        """What a request without ``model`` gets: ``clm-latest``, else another served head,
        else ``clm-raw`` (a server with no trained head yet)."""
        if DEFAULT_MODEL in self.heads:
            return DEFAULT_MODEL
        return min(self.heads) if self.heads else RAW_MODEL

    def has(self, model: str) -> bool:
        return model == RAW_MODEL or model in self.heads

    # ------------------------------------------------------------------ inference
    def answer(self, state: Any, questions: dict[str, dict], model: str | None = None,
               temperature: float = 1.0) -> dict:
        """-> {"model", "answers": {id: Answer}, "usage"}; raises ValueError on bad questions."""
        model = model or self.default_model
        if not questions:
            raise ValueError("questions must not be empty")
        if not (0 < temperature <= 100):
            raise ValueError("temperature must be in (0, 100]")
        if model == RAW_MODEL:
            head = None
        elif model in self.heads:
            head = self.heads[model].ensure()
        else:
            raise ModelNotFound(f"unknown model {model!r}; available: {[m['name'] for m in self.models()]}")
        questions = {k: question_to_dict(q) for k, q in questions.items()}   # Noul/Choice/Score or dicts
        pairs = build_pairs(state, questions)          # ValueError on malformed questions
        states = [p[0] for p in pairs.values()]
        cands = [t for p in pairs.values() for t in p[2]]
        tokens: list[int] = []
        if head is None:
            # The raw ablation reuses the encoder's own vectors, in the encoder's own space.
            hidden = self.encoder.hidden_size
            zq = self._cached("raw/state", hidden, states, tokens, "state")
            za = self._cached("raw/action", hidden, cands, tokens, "action")
            scale = RAW_SCALE
        else:
            ns, dim = head.namespace, head.proj_dim
            zq = self._cached(f"{ns}/state", dim, states, tokens, "state", head.project_states)
            za = self._cached(f"{ns}/action", dim, cands, tokens, "action", head.project_actions)
            scale = head.scale
        answers, k = {}, 0
        for i, (qid, (_, keys, texts)) in enumerate(pairs.items()):
            cos = za[k:k + len(texts)] @ zq[i]
            k += len(texts)
            answers[qid] = answer_from_logits(questions[qid], keys, (scale * cos / temperature).tolist())
        return {"model": model, "answers": answers,
                "usage": {"billing_units": len(questions), "input_tokens": sum(tokens), "output_tokens": 0}}

    def rank(self, state: Any, candidates: list[str], instructions: str | None = None,
             model: str | None = None, temperature: float = 1.0) -> list[dict]:
        """Rank free-form candidate strings against a state (best first).

        ``state`` is the context and ``instructions`` the question; the state head sees
        ``context + question`` and the action head sees each candidate verbatim.
        """
        q = {"type": "choice", "instructions": instructions,
             "criteria": {str(i): c for i, c in enumerate(candidates)}}
        a = self.answer(state, {"rank": q}, model, temperature)["answers"]["rank"]
        order = sorted(a["probabilities"].items(), key=lambda kv: -kv[1])
        return [{"rank": r + 1, "candidate": candidates[int(i)], "prob": p} for r, (i, p) in enumerate(order)]


def head_card(path: str) -> dict[str, Any]:
    """What a head answers, read from the files saved beside it: ``<stem>.json`` (release
    heads, ``clm-train``) or ``questions.json`` + ``summary.json`` (playground runs).
    Returns ``{}`` when there is nothing to read; never raises."""
    stem = os.path.splitext(path)[0]
    folder = os.path.dirname(path)
    questions, acc = None, None
    try:
        if os.path.exists(stem + ".json"):
            with open(stem + ".json", encoding="utf-8") as f:
                card = json.load(f)
            questions, acc = card.get("questions"), card.get("test_accuracy")
            if not questions and isinstance(card.get("per_question"), dict):
                questions = {q: {} for q in card["per_question"]}      # names only
        elif os.path.exists(os.path.join(folder, "questions.json")):
            with open(os.path.join(folder, "questions.json"), encoding="utf-8") as f:
                questions = json.load(f)
            summary = os.path.join(folder, "summary.json")
            if os.path.exists(summary):
                with open(summary, encoding="utf-8") as f:
                    acc = (json.load(f).get("test") or {}).get("acc")
    except (OSError, ValueError, AttributeError):
        return {}
    if not isinstance(questions, dict) or not questions:
        return {}
    qs = [{"name": q, "type": spec.get("type"), "instructions": spec.get("instructions", ""),
           "labels": list(spec.get("criteria") or {})}
          for q, spec in questions.items() if isinstance(spec, dict)]
    if len(qs) == 1:
        q = qs[0]
        text = q["instructions"] or q["name"]
        if q["labels"]:
            text += " (" + ", ".join(q["labels"]) + ")"
    else:
        text = f"Answers {len(qs)} questions: " + ", ".join(q["name"] for q in qs)
    if isinstance(acc, (int, float)):
        text += f". {acc:.0%} on its test set"
    out: dict[str, Any] = {"description": text, "questions": qs}
    if isinstance(acc, (int, float)):
        out["test_accuracy"] = round(float(acc), 4)
    return out
