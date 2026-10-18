"""``clm-train``: train CLM heads for your own classification problem.

The frozen encoder must be running (``vllm serve`` on a GPU box, or
``tools/local_embed_server.py`` on a Mac).  Two ways to give it data:

One question for every row: a CSV / JSONL / JSON / parquet file of ``text`` and
``label`` columns, plus the question and its options:

    clm-train --data tickets.csv --encoder nemotron-3-embed-1b \\
        --question "Which team should handle this?" \\
        --options '{"billing": "A charge, invoice, refund or payment problem.",
                    "technical": "The product is broken, erroring or down."}' \\
        --out runs/tickets

Typed rows: any number of questions per state, each row
``{"id", "state", "questions", "gold"}`` (the ``LocalLLaMA/typed-decisions`` schema),
from a file, a directory laid out as ``<workflow>/{train,test}-*.parquet``, or a
Hugging Face dataset id:

    clm-train --data LocalLLaMA/typed-decisions --encoder nemotron-3-embed-1b --out runs/typed

The test split comes from, in order: ``--test FILE``; a ``split`` column (``train`` /
``test``); the directory layout; otherwise ``--test-frac`` of the rows, held out by
``--group-column`` when given.  The run writes ``best_head.pt`` (serve it with
``clm-serve --ckpt``), ``questions.json`` (the question to send at serving time, for
one-question data), ``history.json`` and ``summary.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .encoders import ENCODERS, get_encoder
from .typed_data import holdout, load_typed_rows, read_rows, simple_rows

DEFAULT_EMB_URL = "http://127.0.0.1:8090/v1/embeddings"


def _json_arg(value: str, what: str):
    """A JSON literal, or the path of a JSON file."""
    if value is None:
        return None
    if os.path.isfile(value):
        with open(value, encoding="utf-8") as f:
            return json.load(f)
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise SystemExit(f"{what} must be JSON or a path to a JSON file: {e}") from e


def build_question(args) -> dict:
    """The one System One question that every row of one-question data answers."""
    t = args.type
    q: dict = {"type": t, "instructions": args.question}
    options = _json_arg(args.options, "--options")
    if t == "choice":
        if not isinstance(options, dict) or len(options) < 2:
            raise SystemExit('--options must be an object of >= 2 options: {"key": "description", ...}')
        q["criteria"] = options
    elif t == "score":
        if not isinstance(options, list) or len(options) < 2:
            raise SystemExit('--type score needs --options as an ordered list of >= 2 levels: ["Low: ...", ...]')
        q["criteria"] = options
    elif options is not None:
        if not isinstance(options, dict) or set(options) - {"true", "false"}:
            raise SystemExit('--type noul takes optional --options {"true": "...", "false": "..."}')
        q["criteria"] = options
    return q


def load_data(args) -> tuple[list[dict], list[dict], str, dict | None]:
    """-> (train rows, test rows, a label for the source, the question for one-question data)."""
    question = build_question(args) if args.question else None

    def convert(rows):
        if question is None:
            missing = {"state", "questions", "gold"} - set(rows[0]) if rows else set()
            if missing:
                raise SystemExit(f"rows are missing {sorted(missing)}; for text + label data pass --question "
                                 "and --options")
            for i, r in enumerate(rows):
                r.setdefault("id", f"row{i}")
                if args.group_column:
                    r["group"] = str(r[args.group_column])
            return rows
        return simple_rows(rows, question, args.qid, args.text_column, args.label_column,
                           args.id_column, args.group_column)

    if os.path.isfile(args.data):
        rows = read_rows(args.data)
        if not rows:
            raise SystemExit(f"{args.data} has no rows")
        if args.test:
            train, test = convert(rows), convert(read_rows(args.test))
            how = f"test from {args.test}"
        elif "split" in rows[0]:
            train = convert([r for r in rows if r.get("split") == "train"])
            test = convert([r for r in rows if r.get("split") == "test"])
            how = "split column"
        else:
            all_rows = convert(rows)
            train, test = holdout(all_rows, args.test_frac, args.seed,
                                  (lambda r: r.get("group") or r["id"]))
            how = f"{args.test_frac:.0%} held out" + (f" by {args.group_column}" if args.group_column else "")
    else:
        if question is not None:
            raise SystemExit("--question applies to a single text + label file, not a directory or dataset id")
        train = convert(load_typed_rows(args.data, "train", args.workflow, args.hf_cache))
        test = convert(load_typed_rows(args.data, "test", args.workflow, args.hf_cache))
        how = f"{args.workflow}/train, {args.workflow}/test"
    return train, test, f"{args.data} ({how})", question


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="clm-train", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    d = ap.add_argument_group("data")
    d.add_argument("--data", required=True, help="file (.csv/.jsonl/.json/.parquet), typed-dataset directory, "
                                                 "or Hugging Face dataset id")
    d.add_argument("--test", help="separate test file (same format as --data)")
    d.add_argument("--test-frac", type=float, default=0.2,
                   help="held-out test fraction when the data has no split (default 0.2)")
    d.add_argument("--group-column", help="column whose values must not straddle splits (customer, account, ...)")
    d.add_argument("--workflow", default="all", help="typed-dataset subfolder (default: all)")
    d.add_argument("--hf-cache", default=None, help="Hugging Face download cache")
    q = ap.add_argument_group("one question per row (text + label data)")
    q.add_argument("--question", help="the question, e.g. 'Which team should handle this?'")
    q.add_argument("--type", choices=["choice", "noul", "score"], default="choice")
    q.add_argument("--options", help="choice: {key: description}; score: [level, ...]; noul: optional "
                                     "{true: ..., false: ...}. JSON, or a path to a JSON file")
    q.add_argument("--text-column", default="text")
    q.add_argument("--label-column", default="label")
    q.add_argument("--id-column", default=None)
    q.add_argument("--qid", default="label", help="name of the question in requests (default: label)")
    e = ap.add_argument_group("encoder")
    e.add_argument("--encoder", choices=sorted(ENCODERS), default=get_encoder().name,
                   help="frozen encoder; serve the head with the same one (default: CLM_ENCODER or "
                        "nemotron-3-embed-8b)")
    e.add_argument("--embed-url", default=os.environ.get("CLM_EMB_URL", DEFAULT_EMB_URL),
                   help=f"the running encoder's /v1/embeddings (default: {DEFAULT_EMB_URL})")
    e.add_argument("--served-model-name", default=None, help="model name the encoder serves (default: the encoder's)")
    e.add_argument("--offline", action="store_true", help="embed with in-process vLLM instead (Linux + NVIDIA GPU)")
    e.add_argument("--max-len", type=int, default=2048, help="token budget per text, prompt included (default 2048)")
    e.add_argument("--embed-cache", default=None, help="embedding cache dir (default: OUT/embeddings)")
    h = ap.add_argument_group("training")
    h.add_argument("--out", required=True, help="output directory")
    h.add_argument("--epochs", type=int, default=20)
    h.add_argument("--patience", type=int, default=5, help="stop after this many epochs without a better validation score")
    h.add_argument("--batch", type=int, default=256)
    h.add_argument("--lr", type=float, default=None, help="default 5e-4")
    h.add_argument("--width", type=int, default=None, help="head hidden width (default 1536)")
    h.add_argument("--depth", type=int, default=None, help="head layers (default 3)")
    h.add_argument("--targets", choices=["soft", "hard"], default="soft",
                   help="soft: train on gold probabilities when given; hard: on the label only")
    h.add_argument("--loss", choices=["infonce", "softce"], default="infonce")
    h.add_argument("--val-frac", type=float, default=0.1, help="share of training rows (or groups) for validation")
    h.add_argument("--init-ckpt", default=None, help="continue from a head trained on the same encoder")
    h.add_argument("--seed", type=int, default=1234)
    h.add_argument("--gpu", type=int, default=0, help="CUDA device for the heads when one exists")
    args = ap.parse_args(argv)

    if args.question is None and (args.options or args.type != "choice"):
        ap.error("--options / --type need --question")
    if args.question is not None and args.type in ("choice", "score") and not args.options:
        ap.error(f"--type {args.type} needs --options")
    if not 0 < args.test_frac < 1 or not 0 < args.val_frac < 1:
        ap.error("--test-frac and --val-frac must be between 0 and 1")
    if args.batch < 2 or args.epochs < 1 or args.patience < 1:
        ap.error("--batch must be >= 2, --epochs and --patience >= 1")

    spec = get_encoder(args.encoder)
    train, test, source, question = load_data(args)
    os.makedirs(args.out, exist_ok=True)
    if question is not None:
        with open(os.path.join(args.out, "questions.json"), "w", encoding="utf-8") as f:
            json.dump({args.qid: question}, f, indent=1)

    # the options train_choice shares with train/finetune.py --task choice
    cfg = argparse.Namespace(
        data=args.data, workflow=args.workflow if question is None else None, out_dir=args.out,
        encoder=spec.name, embed_url=None if args.offline else args.embed_url, embed_model=None,
        served_model_name=args.served_model_name or (None if args.offline else spec.name),
        max_len=args.max_len, gpu_mem=0.85, embed_cache=args.embed_cache, seed=args.seed,
        val_frac=args.val_frac, init_ckpt=args.init_ckpt, width=args.width, depth=args.depth, proj=512,
        targets=args.targets, loss=args.loss, lr=args.lr, weight_decay=0.0, epochs=args.epochs,
        batch=args.batch, patience=args.patience, gpu=args.gpu)

    from .trainer import train_choice
    t0 = time.time()
    res = train_choice(cfg, train, test, source=source)
    res["minutes"] = round((time.time() - t0) / 60, 2)
    res["encoder"] = spec.name
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)

    ckpt = os.path.join(args.out, "best_head.pt")
    print(f"\n  test accuracy   head {res['test']['acc']:.1%}   clm-raw {res['raw_test']:.1%}   "
          f"majority {res['majority_test']:.1%}   ({res['minutes']} min)", flush=True)
    if res["test"]["acc"] <= res["majority_test"]:
        print("  the head does not beat always predicting the most common label; see the guide's "
              "'What to change when' before serving it", flush=True)
    print(f"  serve it:  clm-serve --encoder {spec.name} --ckpt {ckpt}", flush=True)
    if question is not None:
        print(f"  ask it:    POST /v1/systemone with \"questions\": the contents of "
              f"{os.path.join(args.out, 'questions.json')}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
