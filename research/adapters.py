"""Data adapters for ``research/finetune.py``.

* ``read_transitions``: step transitions (``state`` chat messages, ``action``,
  ``task_id``, ``step_idx``, ``trajectory_id``, ``reward``) as written by
  the DeepSWE trace builder in the research repo's ``main`` branch, returned unchanged.
* ``typed_decision_examples`` / ``ChoiceExample``: moved to ``clm.typed_data``,
  re-exported here.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from clm.typed_data import ChoiceExample, typed_decision_examples  # noqa: E402,F401

__all__ = ["read_transitions", "ChoiceExample", "typed_decision_examples"]
TRANSITION_KEYS = ("state", "action", "task_id", "step_idx")


def read_transitions(path: str) -> list[dict]:
    """Load step transitions from a ``.jsonl`` (one object per line) or ``.json`` (array) file."""
    if path.endswith(".jsonl"):
        with open(path) as f:
            recs = [json.loads(line) for line in f if line.strip()]
    else:
        with open(path) as f:
            recs = json.load(f)
    for i, r in enumerate(recs):
        missing = [k for k in TRANSITION_KEYS if k not in r]
        if missing:
            raise ValueError(f"{path}: record {i} is missing {missing}")
    return recs
