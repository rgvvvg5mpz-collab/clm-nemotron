# 04 · Evaluation

## Method

- **Splits.** Test rows are separated before training and used once, at the end. A check
  set (10% of training rows, held out by group when a group column is given) picks the best
  training round. Test data never influences training or model selection.
- **Baselines on the same test rows.** (1) Always predicting the most common training label
  ("majority"). (2) The encoder with no trained head (`clm-raw`). A head is only useful if
  it clearly beats both.
- **Metric.** Top-1 accuracy against the gold label: exact match for pick-one and
  yes/no questions, exact level for ordered scales.
- **Uncertainty.** The 95% binomial interval is about ±4–5 points at 100 test items near 95%
  accuracy, and about ±9 points at 100 items near 50–70%.
- **Reproduce.** `clm-benchmark` (accuracy, latency, throughput against
  a running service); `clm-train` and `finetune.py` print the three-way comparison at the end
  of every run.

## Results

### `support-topic-1b`: 5 topics, 100 test conversations

| Model | Accuracy |
| --- | ---: |
| `support-topic-1b` (shipped) | **88.0%** (training run) · **89.0%** (service benchmark) |
| Same recipe, 9 seeds | 88–96%, mean 92.4% |
| `clm-raw` | 56.0% · 57.0% |
| Majority label | 18.0% |

The service benchmark appends a short run marker to each text so the cache can't answer,
which explains the one-point difference. Run-to-run variation, and the head-size and
seeding changes that cut it, are described in the model card.

### `typed-decisions-1b`: 15 questions, 2,000 test questions

| Question | Type | `clm-raw` | Head |
| --- | --- | ---: | ---: |
| category | choice | 56% | 92% |
| duplicate | yes/no | 11% | 89% |
| needs_review | yes/no | 45% | 80% |
| true_positive | yes/no | 73% | 78% |
| risk | scale | 45% | 75% |
| credential_compromise | yes/no | 54% | 74% |
| needs_human | yes/no | 34% | 73% |
| severity | scale | 47% | 72% |
| churn_risk | scale | 28% | 67% |
| matches_order | yes/no | 46% | 67% |
| action | choice | 22% | 66% |
| disposition | choice | 28% | 64% |
| urgency | scale | 32% | 63% |
| outcome | choice | 50% | 59% |
| discrepancy_severity | scale | 41% | 48% |
| **All** | | **37.8%** | **69.2%** |

Majority label overall: 45.0%. The labeller's agreement with itself: 73.5%.

### Latency and throughput (service benchmark)

Apple M5 Pro laptop (24 GB), encoder on the Mac GPU (MPS), heads on CPU, 2026-10-18,
`Marketing/benchmark/results.json`:

| | p50 | p95 |
| --- | ---: | ---: |
| Unseen text (encoder runs) | 33 ms | 43 ms |
| Repeated text (vector cache) | 1.0 ms | 1.3 ms |
| Throughput, 8 parallel clients, all unseen texts | 36.5 requests/s | |

Two earlier runs on the same machine measured 30 / 38 ms and 41 requests/s; expect a few
milliseconds of variation between runs.

Latency is measured end to end from the client on the same machine. A server GPU would be
faster; network hops add latency.

## What these results do not show

- **Correctness on real data.** All test data is synthetic and labelled by a model.
- **Calibration.** Accuracy says nothing about whether a reported 0.95 means "right 95% of the
  time". Trained heads put more than 0.99 on their top choice for typical inputs.
  Calibration is not yet measured (no ECE or Brier score); see risk R2.
- **Robustness.** No tests yet for typos, other languages, very long texts, adversarial
  wording or topic drift.
- **Fairness.** No group attributes in the data; no fairness measurement was possible.
- **Frontier comparison.** Frontier model accuracy and latency on this data were not
  measured (no API access during evaluation). `Claude Code/tools/compare_frontier.py`
  does it when credentials are available.
