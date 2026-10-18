#!/usr/bin/env python3
"""Run a frontier model on the same test rows as CLM, and measure accuracy, latency and cost.

Needs Anthropic API credentials (ANTHROPIC_API_KEY, or a profile from `ant auth login`).
It sends the 100 synthetic test conversations of src/clm/static/sample/support-tickets.csv
to the API, so it costs money: a few cents with Haiku, about a dollar with Opus.

    pip install anthropic
    python "Claude Code/tools/compare_frontier.py" --models claude-haiku-5-5 claude-sonnet-5-5 claude-opus-5-5

Each request asks the same question with the same label descriptions the CLM model was
trained with, and constrains the answer to one label with structured output. Effort is
"low", the usual setting for classification. Opus 5.5 and Sonnet 5.5 requests opt into
the API's server-side refusal fallback; a refusal that still ends the chain counts as
wrong. Results go to Marketing/benchmark/frontier-results.json, which build_marketing.py reads.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time


# USD per million tokens, Anthropic first-party list prices (cached 2026-10-06).
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-fable-5-1": (10.00, 50.00),
}
SERVER_FALLBACK = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1"}

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def prompt(question: dict, text: str) -> str:
    opts = "\n".join(f"- {k}: {v}" for k, v in question["criteria"].items())
    return (f"{question['instructions']}\n\nOptions:\n{opts}\n\n"
            f"Conversation:\n<conversation>\n{text}\n</conversation>\n\nAnswer with the one option key that fits best.")


def classify(client, model, question, text):
    schema = {"type": "object", "properties": {"label": {"type": "string", "enum": list(question["criteria"])}},
              "required": ["label"], "additionalProperties": False}
    kw = dict(model=model, max_tokens=2048, messages=[{"role": "user", "content": prompt(question, text)}],
              output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}})
    t0 = time.perf_counter()
    if model in SERVER_FALLBACK:
        resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
    else:
        resp = client.messages.create(**kw)
    ms = (time.perf_counter() - t0) * 1000
    label = None
    if resp.stop_reason != "refusal":
        txt = next((b.text for b in resp.content if b.type == "text"), "")
        label = json.loads(txt).get("label") if txt else None
    return label, ms, resp.usage.input_tokens, resp.usage.output_tokens


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["claude-opus-5-5"], choices=sorted(PRICES))
    ap.add_argument("--data", default=os.path.join(REPO, "src/clm/static/sample/support-tickets.csv"))
    ap.add_argument("--card", default=os.path.join(REPO, "models/heads/support-topic-1b.json"))
    ap.add_argument("--out", default=os.path.join(REPO, "Marketing", "benchmark", "frontier-results.json"))
    a = ap.parse_args()

    question = json.load(open(a.card))["questions"]["label"]
    rows = [r for r in csv.DictReader(open(a.data, encoding="utf-8")) if r.get("split") == "test"]
    import anthropic  # optional dependency: pip install anthropic

    client = anthropic.Anthropic()
    results = {}
    for model in a.models:
        hits, lat, tin, tout, errors = 0, [], [], [], 0
        for r in rows:
            try:
                label, ms, i, o = classify(client, model, question, r["text"])
            except anthropic.RateLimitError:
                time.sleep(20); errors += 1; continue
            except anthropic.APIStatusError as e:
                print(f"  {model}: API error {e.status_code}: {e.message}"); errors += 1; continue
            except anthropic.APIConnectionError:
                print(f"  {model}: network error"); errors += 1; continue
            hits += label == r["label"]
            lat.append(ms); tin.append(i); tout.append(o)
        pin, pout = PRICES[model]
        per_req = (statistics.mean(tin) * pin + statistics.mean(tout) * pout) / 1e6 if tin else None
        results[model] = {"accuracy": hits / len(rows), "errors": errors,
                          "latency_ms": {"p50": round(sorted(lat)[len(lat) // 2], 1),
                                         "p95": round(sorted(lat)[int(len(lat) * .95) - 1], 1)} if lat else None,
                          "tokens_per_request": {"input": round(statistics.mean(tin), 1),
                                                 "output": round(statistics.mean(tout), 1)} if tin else None,
                          "usd_per_request": per_req, "usd_per_million_requests": per_req * 1e6 if per_req else None,
                          "settings": {"effort": "low", "structured_output": True}}
        print(f"{model}: accuracy {hits / len(rows):.1%}, p50 {results[model]['latency_ms']['p50'] if lat else '-'} ms, "
              f"${per_req * 1e6:,.0f} per million requests" if per_req else f"{model}: no successful requests")
    json.dump({"date": time.strftime("%Y-%m-%d"), "rows": len(rows), "results": results}, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
