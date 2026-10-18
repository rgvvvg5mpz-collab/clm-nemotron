#!/usr/bin/env python3
"""Measure a running clm-serve: accuracy, latency and throughput on labelled test rows.

    python evaluation/benchmark_service.py --model support-topic-1b \\
        --data src/clm/static/sample/support-tickets.csv --questions models/heads/support-topic-1b.json \\
        --out "Marketing/benchmark/results.json"

* accuracy: the model and clm-raw on the rows with split == test (or all rows);
* latency, first time: each test text once, sequentially, so the encoder runs every call;
* latency, repeated: the same texts again (served from the vector cache);
* throughput: unique texts sent by --concurrency parallel clients.

Numbers depend on the machine; the report records it.  Nothing here calls an external API.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import os
import platform
import statistics
import subprocess
import time

import requests


def pct(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def ask(base, model, text, questions):
    t0 = time.perf_counter()
    r = requests.post(f"{base}/v1/systemone", json={"state": text, "model": model, "questions": questions}, timeout=120)
    r.raise_for_status()
    ms = (time.perf_counter() - t0) * 1000
    j = r.json()
    return j["answers"]["label"], ms, j["usage"].get("input_tokens", 0)


def machine():
    info = {"os": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
            "cpu": platform.processor() or platform.machine()}
    try:
        if platform.system() == "Darwin":
            info["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            info["memory_gb"] = round(int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)) / 2**30)
    except Exception:  # noqa: BLE001
        pass
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8700")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="CSV with text, label and optional split columns")
    ap.add_argument("--questions", required=True, help="JSON with a 'questions' object (a head card) or the questions themselves")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--throughput-requests", type=int, default=400)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    q = json.load(open(a.questions))
    questions = q.get("questions", q)
    rows = list(csv.DictReader(open(a.data, encoding="utf-8")))
    test = [r for r in rows if r.get("split") == "test"] or rows
    health = requests.get(f"{a.base}/health", timeout=10).json()

    report = {"model": a.model, "rows": len(test), "machine": machine(), "encoder": health.get("encoder"),
              "date": time.strftime("%Y-%m-%d")}
    for model in (a.model, "clm-raw"):
        hits, lat, toks = 0, [], []
        # a marker no earlier run used, so the first pass is a true cache miss for the encoder
        salt = f" [{model}:{time.time_ns()}]"
        for r in test:
            ans, ms, tk = ask(a.base, model, r["text"] + salt, questions)
            hits += ans["choice"] == r["label"]
            lat.append(ms); toks.append(tk)
        warm = [ask(a.base, model, r["text"] + salt, questions)[1] for r in test]
        report[model] = {"accuracy": hits / len(test),
                         "latency_ms_first_time": {"p50": round(pct(lat, .5), 1), "p95": round(pct(lat, .95), 1)},
                         "latency_ms_repeated": {"p50": round(pct(warm, .5), 1), "p95": round(pct(warm, .95), 1)},
                         "encoder_tokens_per_request": round(statistics.mean(toks), 1)}
        print(f"{model:>20}: accuracy {hits / len(test):.1%}  first-time p50 {pct(lat, .5):.0f} ms  "
              f"p95 {pct(lat, .95):.0f} ms  repeated p50 {pct(warm, .5):.1f} ms", flush=True)

    texts = [test[i % len(test)]["text"] + f" [t{i}:{time.time_ns()}]" for i in range(a.throughput_requests)]
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(lambda t: ask(a.base, a.model, t, questions), texts))
    secs = time.perf_counter() - t0
    report["throughput"] = {"requests": len(texts), "concurrency": a.concurrency, "seconds": round(secs, 1),
                            "requests_per_second": round(len(texts) / secs, 1), "all_first_time": True}
    print(f"throughput: {len(texts) / secs:.1f} requests/s ({len(texts)} unique texts, {a.concurrency} clients)")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(report, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
