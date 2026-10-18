"""Typed-question training data: rows of ``state`` + ``questions`` + ``gold``.

A row is one state with any number of System One questions and their gold answers::

    {"id": "t-17", "state": {...} or "text",
     "questions": {"dept": {"type": "choice", "instructions": "...", "criteria": {...}}},
     "gold": {"dept": {"label": "billing", "probabilities": {"billing": 0.9, ...}}}}

``state``, ``questions`` and ``gold`` may also be JSON strings (as in parquet).  Each
(row, question) becomes one ``ChoiceExample``: the state text and candidate texts that
``clm.schema.build_pairs`` builds for serving, plus the gold distribution.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import random
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from .schema import build_pairs


def _maybe_json(x):
    """Decode JSON-encoded objects; other strings are returned as is."""
    if isinstance(x, str):
        try:
            v = json.loads(x)
        except ValueError:
            return x
        return v if isinstance(v, (dict, list)) else x
    return x


def _key_of(label) -> str:
    if isinstance(label, bool):
        return "true" if label else "false"
    return str(label)


@dataclass
class ChoiceExample:
    qid: str
    state_text: str
    keys: list[str]
    candidates: list[str]
    target: list[float]      # gold distribution over ``keys``
    label: int               # index of the gold label in ``keys``
    group: str               # source row id
    workflow: str | None = None
    meta: dict = field(default_factory=dict)


def typed_decision_examples(rows: Iterable[dict]) -> Iterator[ChoiceExample]:
    """Rows with ``state``, ``questions`` (System One wire format) and ``gold``
    (``{qid: {"label", "probabilities"}}``) -> one ChoiceExample per answered question."""
    for r in rows:
        state, questions, gold = _maybe_json(r["state"]), _maybe_json(r["questions"]), _maybe_json(r["gold"])
        for qid, (stext, keys, cands) in build_pairs(state, questions).items():
            g = gold.get(qid) if isinstance(gold, dict) else None
            if not g or "label" not in g:
                continue
            label = _key_of(g["label"])
            if label not in keys:
                raise ValueError(f"row {r.get('id')}: gold label {label!r} not among options {keys} for {qid!r}")
            probs = {_key_of(k): float(v) for k, v in (g.get("probabilities") or {}).items()}
            target = [probs.get(k, 0.0) for k in keys]
            z = sum(target)
            target = [t / z for t in target] if z > 0 else [float(k == label) for k in keys]
            yield ChoiceExample(qid=qid, state_text=stext, keys=keys, candidates=cands, target=target,
                                label=keys.index(label), group=str(r.get("id", "")),
                                workflow=r.get("workflow"))


# ----------------------------------------------------------------------------- reading rows
def read_rows(path: str) -> list[dict]:
    """Rows from a ``.jsonl``, ``.json`` (array), ``.csv`` or ``.parquet`` file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if ext == ".json":
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f"{path}: expected a JSON array of rows")
        return rows
    if ext == ".csv":
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    if ext == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    raise ValueError(f"{path}: unsupported file type {ext!r}; use .jsonl, .json, .csv or .parquet")


def load_typed_rows(data: str, split: str, workflow: str = "all", cache_dir: str | None = None) -> list[dict]:
    """``split`` rows ("train" / "test") of a typed dataset.

    ``data`` is a directory or Hugging Face dataset id laid out as
    ``<workflow>/<split>-*.parquet`` (the ``LocalLLaMA/typed-decisions`` layout), or a
    single file with a ``split`` column.  A single file without that column is refused:
    using it for both splits would score the head on its own training data.
    """
    if os.path.isfile(data):
        rows = read_rows(data)
        if rows and "split" not in rows[0]:
            raise SystemExit(f"{data} has no 'split' column, so it cannot provide a {split!r} split; add one, "
                             "or use clm-train --test FILE / --test-frac")
        return [r for r in rows if r.get("split") == split]
    root = data
    if not os.path.isdir(root):
        from huggingface_hub import snapshot_download
        root = snapshot_download(data, repo_type="dataset", cache_dir=cache_dir,
                                 allow_patterns=[f"{workflow}/*.parquet"])
    files = sorted(f for f in glob.glob(os.path.join(root, workflow, "*.parquet"))
                   if os.path.basename(f).startswith(split))
    if not files:
        raise SystemExit(f"no {split!r} parquet for workflow {workflow!r} in {data}")
    rows: list[dict] = []
    for f in files:
        rows += read_rows(f)
    return rows


def holdout(rows: list[dict], frac: float, seed: int, group_of=None) -> tuple[list[dict], list[dict]]:
    """Split rows into (train, test), whole groups at a time when ``group_of`` is given,
    so near-duplicates of one customer / account / incident never straddle the split."""
    group_of = group_of or (lambda r: r["id"])
    groups = sorted({str(group_of(r)) for r in rows})
    random.Random(seed).shuffle(groups)
    n = max(1, round(frac * len(groups)))
    if n >= len(groups):
        raise ValueError(f"cannot hold out {frac:.0%} of {len(groups)} groups and keep any for training")
    test = set(groups[:n])
    return [r for r in rows if str(group_of(r)) not in test], [r for r in rows if str(group_of(r)) in test]


# ----------------------------------------------------------------------------- one question per row
_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f"}


def simple_rows(rows: list[dict], question: dict, qid: str = "label", text_column: str = "text",
                label_column: str = "label", id_column: str | None = None,
                group_column: str | None = None) -> list[dict]:
    """``text`` + ``label`` rows (one question for every row) -> typed rows.

    ``question`` is one System One question (``type``, ``instructions``, ``criteria``).
    Labels are option keys for ``choice``, level indices or level texts for ``score``,
    and true/false (yes/no, 1/0) for ``noul``.  Rows with an empty label are skipped.
    """
    t = question.get("type")
    levels = question.get("criteria") if t == "score" else None
    out = []
    for i, r in enumerate(rows):
        if text_column not in r or label_column not in r:
            raise ValueError(f"row {i} needs '{text_column}' and '{label_column}' columns; has {sorted(r)}")
        raw = r[label_column]
        if raw is None or str(raw).strip() == "":
            continue
        label = str(raw).strip()
        if t == "noul":
            low = label.lower()
            if low not in _TRUE | _FALSE:
                raise ValueError(f"row {i}: noul label {label!r} is not true/false")
            label = "true" if low in _TRUE else "false"
        elif t == "score":
            if label.isdigit() and int(label) < len(levels):
                pass
            elif label in levels:
                label = str(levels.index(label))
            else:
                raise ValueError(f"row {i}: score label {label!r} is neither a level index nor a level text")
        elif t == "choice" and label not in question.get("criteria", {}):
            raise ValueError(f"row {i}: label {label!r} is not one of the options {sorted(question['criteria'])}")
        row = {"id": str(r[id_column]) if id_column else f"row{i}", "state": r[text_column],
               "questions": {qid: question}, "gold": {qid: {"label": label}}}
        if group_column:
            row["group"] = str(r[group_column])
        out.append(row)
    return out
