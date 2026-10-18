# Model card · Head `typed-decisions-1b`

| | |
| --- | --- |
| Purpose | Demo: one head answering 15 different typed questions across four business workflows |
| Workflows | customer service, invoice processing, security incidents, agent-trace observability |
| Questions | 5 per workflow: pick-one (`action`, `category`, `disposition`, `outcome`), yes/no (`credential_compromise`, `duplicate`, `matches_order`, `needs_human`, `needs_review`, `true_positive`), ordered scales (`churn_risk`, `discrepancy_severity`, `risk`, `severity`, `urgency`) |
| Encoder | `nemotron-3-embed-1b` |
| Architecture | Two MLPs (state, action): 2048 → 1536 → 1536 → 512, GELU, LayerNorm; 12,596,224 parameters; learned logit scale 14.8. Trained before head weights were seeded, so a retrain won't reproduce it exactly (two later seeded runs: 69.6% and 70.0%). |
| Training | `finetune.py --task choice` (same trainer as `clm-train`): InfoNCE, soft targets from three teacher samples, batch 256, best of 20 rounds (round 15) |
| Data | 1,200 synthetic rows (1,080 train, 120 check) with 6,000 labelled questions; test: 400 rows, 2,000 questions. `LocalLLaMA/typed-decisions`, Apache-2.0 |
| Status | **Demo only.** |

## Results (test split, 2,000 questions)

| | Accuracy |
| --- | ---: |
| This head | **69.2%** |
| Encoder alone (`clm-raw`) | 37.8% |
| Always the most common label | 45.0% |
| Labeller agreeing with itself (dataset card) | 73.5% |

The head sits close to the ceiling set by the noisy labels. Per question, from 48%
(`discrepancy_severity`) to 92% (`category`); see [04-evaluation.md](../04-evaluation.md).
Each question has 100 test items (400 for `urgency`), so per-question figures move by about
±9 points by chance.

## Limits

As for `support-topic-1b`, plus: ordered-scale answers are graded as exact-level matches,
which understates near misses.
