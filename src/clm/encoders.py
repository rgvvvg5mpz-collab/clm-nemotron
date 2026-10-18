"""Encoder profiles: everything the heads assume about the frozen encoder.

A projection head only makes sense with the exact encoder, pooling and input
format it was trained against.  An ``EncoderSpec`` pins those down, and the
same spec drives both sides:

* serving (``clm.embedder.Embedder``) turns a state or a candidate into the text
  sent to ``/v1/embeddings`` with ``spec.prepare(text, role, tokenizer)``, and
* training (``clm.embed_backends.Recipe``) turns the same text into token ids
  with the same call, so a head never sees inputs at serving time that differ
  from the ones it was trained on.

Checkpoints record ``spec.identity()`` as ``cfg["encoder"]``; ``check_compatible``
refuses to serve a head against an encoder it was not trained on, and refuses a
checkpoint or embedding set that records no encoder at all (upstream CLM releases,
which were trained on a different encoder).

Asymmetric retrieval encoders embed the state as the *query* and each
candidate as the *passage*; ``role`` is ``"state"`` or ``"action"``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

ROLES = ("state", "action")


class IncompatibleEncoder(ValueError):
    pass


@dataclass(frozen=True)
class EncoderSpec:
    name: str                      # registry key; also the default --served-model-name
    hf_id: str                     # Hugging Face repo: weights for `vllm serve`, and the tokenizer
    hidden_size: int               # embedding width the heads take as input
    pooling: str                   # "last" | "mean" (set by the model / vLLM, recorded for checks)
    state_prefix: str = ""         # prepended to every state text
    action_prefix: str = ""        # prepended to every candidate text
    max_tokens: int = 2048         # default token budget, prefix and special tokens included
    add_special_tokens: bool = True  # what the server's own tokenisation of a text does
    state_keep: str = "tail"       # which end of an over-long state survives truncation
    action_keep: str = "head"      # ... and of an over-long candidate
    origin: str = ""               # provenance note, shown in `clm-serve` start-up output

    def prefix(self, role: str | None) -> str:
        if role is None:
            return ""
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")
        return self.state_prefix if role == "state" else self.action_prefix

    def keep(self, role: str | None) -> str:
        return self.action_keep if role == "action" else self.state_keep

    def source(self) -> str:
        """Where to load the encoder's weights and tokenizer from: a local copy when one exists
        (``$CLM_MODELS_DIR/encoders/<name>``, filled by the model release), else the Hub id."""
        root = os.environ.get("CLM_MODELS_DIR")
        if root:
            local = os.path.join(root, "encoders", self.name)
            if os.path.isfile(os.path.join(local, "config.json")):
                return local
        return self.hf_id

    def vllm_command(self, port: int = 8090, max_model_len: int | None = None) -> str:
        """The ``vllm serve`` line that starts this encoder behind clm-serve."""
        return (f"vllm serve {self.hf_id} --served-model-name {self.name} --runner pooling \\\n"
                f"     --enable-prefix-caching --max-model-len {max_model_len or self.max_tokens} "
                f"--gpu-memory-utilization 0.35 --port {port}")

    def identity(self) -> dict[str, Any]:
        """The fields a head depends on, written to checkpoints as ``cfg["encoder"]``."""
        return {"name": self.name, "hf_id": self.hf_id, "hidden_size": self.hidden_size,
                "pooling": self.pooling, "state_prefix": self.state_prefix,
                "action_prefix": self.action_prefix}

    def prepare(self, text: str, role: str | None, tokenizer=None, max_tokens: int | None = None) -> str:
        """Prefix ``text`` for its role, truncating the body (never the prefix) to fit.

        Without a tokenizer the text is only prefixed; the server's own truncation
        then applies, which keeps the *last* tokens and so can cut a prefix off an
        over-long input.
        """
        prefix = self.prefix(role)
        budget = max_tokens or self.max_tokens
        if tokenizer is None or not budget:
            return prefix + text
        if len(text.encode()) + len(prefix.encode()) + 8 <= budget:
            return prefix + text            # byte-level BPE: never more tokens than UTF-8 bytes
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        room = budget - len(tokenizer(prefix, add_special_tokens=False)["input_ids"]) if prefix else budget
        if self.add_special_tokens:
            room -= tokenizer.num_special_tokens_to_add()
        room = max(1, room)
        if len(ids) <= room:
            return prefix + text
        ids = ids[:room] if self.keep(role) == "head" else ids[-room:]
        return prefix + tokenizer.decode(ids, skip_special_tokens=True)


ENCODERS: dict[str, EncoderSpec] = {
    # NVIDIA retrieval embedder on Ministral-3-8B (Mistral AI); OpenMDW-1.1, commercial use allowed.
    # Bidirectional attention + mean pooling; asymmetric "query: " / "passage: " prompts.
    "nemotron-3-embed-8b": EncoderSpec(
        name="nemotron-3-embed-8b", hf_id="nvidia/Nemotron-3-Embed-8B-BF16", hidden_size=4096,
        pooling="mean", state_prefix="query: ", action_prefix="passage: ", max_tokens=8192,
        origin="NVIDIA, built on Mistral AI's Ministral-3-8B; OpenMDW-1.1"),
    # Its 1B sibling (same prompts and pooling, 2048-d): small enough for a laptop
    # (clm-encoder).  Heads trained on one size do not transfer to the other.
    "nemotron-3-embed-1b": EncoderSpec(
        name="nemotron-3-embed-1b", hf_id="nvidia/Nemotron-3-Embed-1B-BF16", hidden_size=2048,
        pooling="mean", state_prefix="query: ", action_prefix="passage: ", max_tokens=8192,
        origin="NVIDIA, Ministral-3 architecture, 1B; OpenMDW-1.1"),
}

DEFAULT_ENCODER = "nemotron-3-embed-8b"


def get_encoder(spec: "str | EncoderSpec | None" = None) -> EncoderSpec:
    """A spec by name; ``None`` -> ``CLM_ENCODER`` or the default."""
    if isinstance(spec, EncoderSpec):
        return spec
    name = spec or os.environ.get("CLM_ENCODER") or DEFAULT_ENCODER
    if name not in ENCODERS:
        raise ValueError(f"unknown encoder {name!r}; known: {sorted(ENCODERS)}")
    return ENCODERS[name]


def checkpoint_encoder(cfg: dict) -> dict[str, Any] | None:
    """The encoder identity recorded in a checkpoint ``cfg`` or an embedding dir's
    ``metadata.json``; None when nothing is recorded (every upstream CLM release)."""
    enc = cfg.get("encoder")
    return dict(enc) if enc else None


_CHECKED = ("hf_id", "hidden_size", "pooling", "state_prefix", "action_prefix")


def check_compatible(trained_on: dict[str, Any] | None, expected: "EncoderSpec | dict[str, Any] | None",
                     what: str = "checkpoint") -> None:
    """Raise ``IncompatibleEncoder`` unless ``trained_on`` matches ``expected`` on every checked field.

    An unrecorded encoder (None) on either side never matches: it is an upstream
    release trained on an encoder this fork does not use.
    """
    want = expected.identity() if isinstance(expected, EncoderSpec) else expected
    if trained_on is None or want is None:
        raise IncompatibleEncoder(
            f"{what} records no encoder, so it predates this fork (an upstream CLM release trained on "
            "a different encoder); heads and embeddings must be built with a clm.encoders encoder")
    diff = {k: (trained_on.get(k), want.get(k)) for k in _CHECKED if trained_on.get(k) != want.get(k)}
    if diff:
        detail = ", ".join(f"{k}: {a!r} vs expected {b!r}" for k, (a, b) in diff.items())
        raise IncompatibleEncoder(
            f"{what} belongs to encoder {trained_on.get('hf_id')!r}, not {want.get('hf_id')!r} ({detail}); "
            "heads and embeddings must come from the same encoder")


def load_tokenizer(spec: EncoderSpec):
    """The encoder's tokenizer, or None when transformers is unavailable."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(spec.source())
    except Exception as e:  # noqa: BLE001  (not installed, offline, gated repo)
        print(f"[clm] no tokenizer for {spec.hf_id} ({type(e).__name__}: {e}); over-long texts are "
              "truncated by the embedder, which keeps the last tokens and can drop the prefix", flush=True)
        return None


def render_messages(messages: list[dict]) -> str:
    """Chat messages as plain ``role: content`` blocks, for encoders without a chat template."""
    parts = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):   # OpenAI-style content parts
            content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
        parts.append(f"{m.get('role', 'user')}: {content}")
    return "\n\n".join(parts)


__all__ = ["EncoderSpec", "ENCODERS", "DEFAULT_ENCODER", "IncompatibleEncoder",
           "get_encoder", "checkpoint_encoder", "check_compatible", "load_tokenizer", "render_messages"]
