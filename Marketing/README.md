# Marketing: executive briefing

Why a small trained classifier (CLM) beats calling a frontier language model for
high-volume classification (cost, accuracy, latency, control), and where it doesn't.

| File | What it is |
| --- | --- |
| [`CLM-Executive-Briefing.pptx`](CLM-Executive-Briefing.pptx) | 10-slide deck, editable: native text, charts and table, with the narration as speaker notes |
| [`CLM-Executive-Briefing.pdf`](CLM-Executive-Briefing.pdf) | The same deck as PDF |
| [`CLM-Executive-Briefing-Video.mp4`](CLM-Executive-Briefing-Video.mp4) | 2:51 narrated video of the deck, 1080p, English subtitles built in |
| [`CLM-Executive-Briefing-Video.srt`](CLM-Executive-Briefing-Video.srt) | Subtitles / transcript |
| [`slides/`](slides/) | Each slide as a 1920×1080 PNG, for email or chat |
| [`benchmark/`](benchmark/) | The measurements behind the numbers, and the script to measure a frontier model |
| [`production/build_marketing.py`](production/build_marketing.py) | Rebuilds everything from the benchmark files |

## The story in one minute

1. **The job:** millions of short, repetitive decisions with fixed labels: routing, tagging, triage.
2. **The approach:** NVIDIA Nemotron reads the text (frozen); a small model learns our labels
   in minutes; answers come back in milliseconds, on our hardware.
3. **Accuracy (measured):** trained on 300 examples, 88–96% on unseen tickets, against 57% untrained.
4. **Speed (measured):** 33 ms median on one laptop; about 36 answers a second, 3 million a day.
5. **Cost:** about $8 per million decisions on own hardware, against $58 (Haiku 5.5),
   $1,160 (Sonnet 5.5) and $2,320 (Opus 5.5) at list prices.
6. **Control:** data stays inside; reproducible; governance pack included.
7. **Use each where strongest:** CLM for fixed labels at volume; frontier models for open
   questions, writing and reasoning; or both, with CLM escalating its uncertain cases.
8. **Ask:** a four-week pilot on one workflow, with 500 real labelled examples.

## Where every number comes from

| Claim | Source | Status |
| --- | --- | --- |
| 88% shipped / 88–96% over 9 runs; 57% untrained; 18% majority | `benchmark/results.json`, `Model Governance/04-evaluation.md` | **Measured** on synthetic demo data labelled by a model |
| 33 ms p50, 43 ms p95; 36.5 requests/s | `benchmark/results.json` (Apple M5 Pro laptop, 2026-10-18) | **Measured** |
| ≈ $8 per million (CLM) | 7.6 compute-hours per million at the measured throughput × **assumed** $1/hour | Estimate; state the assumption |
| $58 / $1,160 / $2,320 per million (frontier) | Anthropic list prices (cached 2026-10-06) × **estimated** 330 input + 50 output tokens per ticket | Estimate; a floor, since thinking tokens add output |
| Frontier accuracy and latency on our data | Not measured (no API access when this was built) | **Run `benchmark/compare_frontier.py`** before quoting any figure |

Do not present the demo accuracy as accuracy on real customer data; the deck's footnotes
say so. The pilot in the last slide is how to get that number.

## Measure the frontier side

```bash
.venv/bin/pip install anthropic
```

```bash
.venv/bin/python Marketing/benchmark/compare_frontier.py --models claude-haiku-5-5 claude-sonnet-5-5 claude-opus-5-5
```

It needs Anthropic API credentials (`ANTHROPIC_API_KEY`) and costs well under a dollar for
the 100 test tickets. It writes `benchmark/frontier-results.json`; rebuilding then adds the
measured frontier results to the console summary, and they can be added to the slides.

## Rebuild

With CLM running (`bash "Claude Code/scripts/start.sh"`):

```bash
.venv/bin/python evaluation/benchmark_service.py --model support-topic-1b --data src/clm/static/sample/support-tickets.csv --questions models/heads/support-topic-1b.json --out Marketing/benchmark/results.json
```

```bash
.venv/bin/python Marketing/production/build_marketing.py
```

Needs Playwright, python-pptx, macOS `say` and ffmpeg. Narration speed: `STUDIO_RATE`
(words per minute, the video uses 174); voice: `STUDIO_VOICE`.
