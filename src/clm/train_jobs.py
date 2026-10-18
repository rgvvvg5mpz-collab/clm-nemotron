"""Training heads from the playground: uploads, background jobs, and serving the result.

``clm-serve`` keeps one ``TrainingJobs`` per process.  A browser uploads a file
(``inspect``), describes the question, and starts a job; the job runs the same
``clm.trainer.train_choice`` as ``clm-train`` in a background thread, embedding through
the server's own ``Embedder`` (so training and serving see identical vectors), and on
success adds the head to the engine as a new model.

Everything lives under ``runs_dir``: ``uploads/<id>.<ext>`` and ``<job id>/`` with
``best_head.pt``, ``questions.json``, ``summary.json`` and ``job.json``.  Finished jobs
are served again after a restart.  One job runs at a time: the encoder is the
bottleneck, and two jobs would only slow each other down.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import threading
import time
import traceback
import uuid
from collections import Counter
from typing import Any

from .heads import HeadPair
from .typed_data import holdout, read_rows, simple_rows

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
EXTENSIONS = (".csv", ".jsonl", ".json", ".parquet")
TYPED_COLUMNS = {"state", "questions", "gold"}
RESERVED = {"clm-raw", "clm-latest"}
TEXT_NAMES = ("text", "message", "body", "content", "input", "query", "ticket", "description", "state")
LABEL_NAMES = ("label", "category", "class", "target", "y", "intent", "topic", "answer")


class JobError(ValueError):
    """A request the user can fix; its message is shown in the page."""


class Cancelled(Exception):
    pass


def _preview(v: Any, n: int = 240) -> Any:
    if isinstance(v, (dict, list)):
        v = json.dumps(v, ensure_ascii=False)
    if isinstance(v, str) and len(v) > n:
        return v[:n] + "…"
    return v


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s[:48] or "model"


def summarize(rows: list[dict]) -> dict:
    """Columns, a preview, and which columns look like the text and the label."""
    columns = list(rows[0]) if rows else []
    stats = {}
    for c in columns:
        vals = [r.get(c) for r in rows]
        strs = [v for v in vals if isinstance(v, str)]
        counts = Counter(str(v) for v in vals if v is not None and str(v).strip() != "")
        stats[c] = {"distinct": len(counts),
                    "avg_len": round(sum(map(len, strs)) / len(strs), 1) if strs else 0,
                    "values": dict(counts.most_common()) if len(counts) <= 100 else None}
    typed = TYPED_COLUMNS <= set(columns)

    def by_name(names, pool):
        for n in names:
            for c in pool:
                if c.lower() == n:
                    return c
        return None

    text_col = label_col = None
    if not typed and columns:
        text_pool = [c for c in columns if stats[c]["avg_len"] > 0]
        text_col = by_name(TEXT_NAMES, text_pool) or (max(text_pool, key=lambda c: stats[c]["avg_len"])
                                                       if text_pool else columns[0])
        label_pool = [c for c in columns if c != text_col and c != "split" and 2 <= stats[c]["distinct"] <= 100]
        label_col = by_name(LABEL_NAMES, label_pool) or (min(label_pool, key=lambda c: stats[c]["distinct"])
                                                         if label_pool else None)
    questions = {}
    if typed:
        for r in rows[:200]:
            q = r["questions"]
            q = json.loads(q) if isinstance(q, str) else q
            for k, v in (q or {}).items():
                questions.setdefault(k, {"type": v.get("type"), "instructions": v.get("instructions")})
    return {"rows": len(rows), "columns": columns, "stats": stats, "format": "typed" if typed else "simple",
            "suggested": {"text": text_col, "label": label_col}, "has_split": "split" in columns,
            "questions": questions, "preview": [{c: _preview(r.get(c)) for c in columns} for r in rows[:5]]}


class TrainingJobs:
    def __init__(self, engine, runs_dir: str):
        self.engine, self.runs_dir = engine, runs_dir
        self.uploads_dir = os.path.join(runs_dir, "uploads")
        os.makedirs(self.uploads_dir, exist_ok=True)
        self.jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._restore()

    # ------------------------------------------------------------------ persistence
    def _job_path(self, job_id: str) -> str:
        return os.path.join(self.runs_dir, job_id, "job.json")

    def _write(self, job: dict) -> None:
        path = self._job_path(job["id"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in job.items() if not k.startswith("_")}, f, indent=1)
        os.replace(tmp, path)

    def _restore(self) -> None:
        """Reload past jobs; serve the finished ones whose encoder matches this server."""
        for name in sorted(os.listdir(self.runs_dir)):
            path = self._job_path(name)
            if not os.path.isfile(path):
                continue
            try:
                job = json.load(open(path, encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if job.get("status") in ("queued", "running"):
                job.update(status="failed", error="clm-serve stopped while this was training")
            self.jobs[job["id"]] = job
            if job.get("status") == "done":
                try:
                    self._serve(job)
                except Exception as e:  # noqa: BLE001  another encoder, deleted file, ...
                    job["served"] = False
                    job["serve_error"] = str(e)
                    print(f"[clm] not serving trained model {job.get('model')!r}: {e}", flush=True)

    # ------------------------------------------------------------------ uploads
    def inspect(self, filename: str, content_b64: str) -> dict:
        ext = os.path.splitext(filename or "")[1].lower()
        if ext not in EXTENSIONS:
            raise JobError(f"use a .csv, .jsonl, .json or .parquet file, not {ext or 'a file without an extension'}")
        try:
            data = base64.b64decode(content_b64, validate=True)
        except (ValueError, TypeError) as e:
            raise JobError(f"the upload is not valid base64: {e}") from e
        if len(data) > MAX_UPLOAD_BYTES:
            raise JobError(f"the file is {len(data) / 1e6:.0f} MB; the limit is {MAX_UPLOAD_BYTES // 10**6} MB")
        upload_id = uuid.uuid4().hex[:12]
        path = os.path.join(self.uploads_dir, upload_id + ext)
        with open(path, "wb") as f:
            f.write(data)
        try:
            rows = read_rows(path)
        except Exception as e:  # noqa: BLE001  parse errors of every format
            os.remove(path)
            raise JobError(f"could not read {filename}: {e}") from e
        if not rows or not isinstance(rows[0], dict):
            os.remove(path)
            raise JobError(f"{filename} has no rows of named columns")
        return {"upload_id": upload_id, "filename": filename, **summarize(rows)}

    def _upload_path(self, upload_id: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{12}", upload_id or ""):
            raise JobError("unknown upload; add the file again")
        for ext in EXTENSIONS:
            p = os.path.join(self.uploads_dir, upload_id + ext)
            if os.path.exists(p):
                return p
        raise JobError("unknown upload; add the file again")

    # ------------------------------------------------------------------ jobs
    def list(self) -> list[dict]:
        with self._lock:
            jobs = sorted(self.jobs.values(), key=lambda j: j.get("created", 0), reverse=True)
            return [self._public(j) for j in jobs]

    def get(self, job_id: str) -> dict:
        with self._lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return self._public(self.jobs[job_id])

    @staticmethod
    def _public(job: dict) -> dict:
        return {k: v for k, v in job.items() if not k.startswith("_")}

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["status"] in ("queued", "running"):
                job["_cancel"] = True
            return self._public(job)

    def _rows(self, spec: dict) -> tuple[list[dict], list[dict], dict | None]:
        """-> (train rows, test rows, the question for text + label data) in the typed schema."""
        raw = read_rows(self._upload_path(spec.get("upload_id")))
        typed = TYPED_COLUMNS <= set(raw[0])
        group = spec.get("group_column") or None
        question = None
        if typed:
            def convert(rows):
                for i, r in enumerate(rows):
                    r.setdefault("id", f"row{i}")
                    if group:
                        r["group"] = str(r.get(group))
                return rows
        else:
            text_col, label_col = spec.get("text_column"), spec.get("label_column")
            if not text_col or not label_col or text_col == label_col:
                raise JobError("choose one column for the text and a different one for the label")
            qtext = (spec.get("question") or "").strip()
            if not qtext:
                raise JobError("write the question the label answers")
            qtype = spec.get("type") or "choice"
            options = {str(k): (v or "").strip() for k, v in (spec.get("options") or {}).items()}
            labels = {str(r.get(label_col)).strip() for r in raw if str(r.get(label_col) or "").strip()}
            question = {"type": qtype, "instructions": qtext}
            if qtype == "choice":
                missing = sorted(labels - set(options))
                if missing:
                    raise JobError(f"describe every label; missing: {', '.join(missing[:8])}")
                if len(labels) < 2:
                    raise JobError("the label column needs at least two different labels")
                question["criteria"] = {k: v for k, v in options.items() if k in labels}
            elif qtype == "noul":
                crit = {k: options[k] for k in ("true", "false") if options.get(k)}
                if crit:
                    question["criteria"] = crit
            else:
                raise JobError("the page trains choice and yes/no questions; use clm-train for scores")

            def convert(rows):
                try:
                    return simple_rows(rows, question, "label", text_col, label_col, None, group)
                except ValueError as e:
                    raise JobError(str(e)) from e

        splits = {str(r.get("split")) for r in raw} if "split" in raw[0] else set()
        if {"train", "test"} <= splits:
            train = convert([r for r in raw if r.get("split") == "train"])
            test = convert([r for r in raw if r.get("split") == "test"])
        else:
            frac = float(spec.get("test_frac") or 0.2)
            if not 0.05 <= frac <= 0.5:
                raise JobError("hold out between 5% and 50% for testing")
            rows = convert(raw)
            if len(rows) < 12:
                raise JobError(f"need at least a dozen labelled examples; have {len(rows)}")
            train, test = holdout(rows, frac, int(spec.get("seed") or 1234), lambda r: r.get("group") or r["id"])
        if len(train) < 10 or len(test) < 2:
            raise JobError(f"need at least 10 training and 2 test examples; have {len(train)} and {len(test)}")
        return train, test, question

    def start(self, spec: dict) -> dict:
        with self._lock:
            busy = [j for j in self.jobs.values() if j["status"] in ("queued", "running")]
            if busy:
                raise JobError(f"model {busy[0]['model']!r} is still training; wait for it or stop it")
            name = _slug(spec.get("name") or spec.get("question") or "model")
            taken = set(self.engine.heads) | RESERVED | {j.get("model") for j in self.jobs.values()}
            base, n = name, 2
            while name in taken:
                name, n = f"{base}-{n}", n + 1
        train, test, question = self._rows(spec)
        epochs = int(spec.get("epochs") or 20)
        if not 1 <= epochs <= 200:
            raise JobError("training rounds must be between 1 and 200")
        job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        job = {"id": job_id, "model": name, "status": "queued", "created": time.time(),
               "encoder": self.engine.encoder.name, "question": question,
               "rows": {"train": len(train), "test": len(test)},
               "progress": {"stage": "queued"}, "history": [], "result": None, "error": None}
        with self._lock:
            self.jobs[job_id] = job
        self._write(job)
        threading.Thread(target=self._run, args=(job, train, test, epochs, spec), daemon=True,
                         name=f"train-{job_id}").start()
        return self._public(job)

    def _run(self, job: dict, train: list[dict], test: list[dict], epochs: int, spec: dict) -> None:
        from .trainer import train_choice
        out_dir = os.path.join(self.runs_dir, job["id"])
        embedder = self.engine.embedder
        cfg = argparse.Namespace(
            data=spec.get("filename") or "upload", workflow=None, out_dir=out_dir, encoder=self.engine.encoder.name,
            embed_url=None, embed_model=None, served_model_name=None, max_len=embedder.max_tokens, gpu_mem=0.85,
            embed_cache=os.path.join(self.runs_dir, "embeddings"), seed=int(spec.get("seed") or 1234),
            val_frac=0.1, init_ckpt=None, width=None, depth=None, proj=512, targets="soft", loss="infonce",
            lr=None, weight_decay=0.0, epochs=epochs, batch=256, patience=5, gpu=0)

        def progress(event: dict) -> None:
            with self._lock:
                if job.get("_cancel"):
                    raise Cancelled()
                job["progress"] = event
                if event["stage"] == "training":
                    job["history"].append({"epoch": event["epoch"], "val_acc": event["val_acc"]})
                if event["stage"] == "baselines":
                    job["baselines"] = {k: v for k, v in event.items() if k != "stage"}
                job["status"] = "running"

        def embed_fn(texts: list[str], role: str):
            return embedder.embed(texts, role)[0]

        t0 = time.time()
        try:
            res = train_choice(cfg, train, test, source=job["model"], progress=progress, embed_fn=embed_fn)
            res["minutes"] = round((time.time() - t0) / 60, 2)
            with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(res, f, indent=1, default=str)
            if job["question"]:
                with open(os.path.join(out_dir, "questions.json"), "w", encoding="utf-8") as f:
                    json.dump({"label": job["question"]}, f, indent=1)
            with self._lock:
                job["result"] = {"test_acc": res["test"]["acc"], "raw_test": res["raw_test"],
                                 "majority_test": res["majority_test"], "best_epoch": res["best_epoch"],
                                 "per_question": res["test"]["per_question"],
                                 "raw_per_question": res.get("raw_test_per_question"), "minutes": res["minutes"]}
                job["checkpoint"] = os.path.join(out_dir, "best_head.pt")
                job["status"] = "done"
                job["progress"] = {"stage": "done"}
            self._serve(job)
        except Cancelled:
            with self._lock:
                job.update(status="stopped", progress={"stage": "stopped"})
        except Exception as e:  # noqa: BLE001  surfaced in the page
            traceback.print_exc()
            with self._lock:
                job.update(status="failed", error=str(e) or type(e).__name__, progress={"stage": "failed"})
        finally:
            job["finished"] = time.time()
            self._write(job)

    def _serve(self, job: dict) -> None:
        pair = HeadPair(job["model"], job["checkpoint"], device=self.engine.device, encoder=self.engine.encoder)
        pair.ensure()
        self.engine.heads[job["model"]] = pair
        job["served"] = True
