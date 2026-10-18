# Model card · Head `support-topic-1b`

| | |
| --- | --- |
| Purpose | Demo: route a customer-support conversation to one of five topics |
| Labels | `account` · `billing` · `delivery` · `refund` · `technical`, each with the one-line description it was trained with (in `models/heads/support-topic-1b.json`) |
| Question | "What is this customer conversation primarily about?" |
| Encoder | `nemotron-3-embed-1b` (recorded in the checkpoint; enforced at load) |
| Architecture | Two MLPs (state, action): 2048 → 512 → 512, GELU; 2,623,488 parameters; learned logit scale 14.3 |
| Training | `clm-train` with default settings: InfoNCE loss on soft targets, AdamW, one-cycle LR (max 5e-4), batch 256, best of 20 rounds by check-set accuracy (round 7), seed 1234. The smaller head is the default under 3,000 training examples (see "Run-to-run variation"). |
| Data | 300 synthetic conversations for training (30 held out as the check set), 100 for test. From `LocalLLaMA/typed-decisions`, workflow `customer_service`, gold `category` label. See [data lineage](../03-data-lineage.md). |
| Status | **Demo only.** Not for decisions about real customers. |

## Results (test split, 100 conversations)

| | Accuracy |
| --- | ---: |
| This head | **88.0%** at training time; **89.0%** in the service benchmark (texts carry a run marker, so they're slightly altered) |
| Same recipe, 9 training seeds | 88–96%, mean 92.4% |
| Encoder alone (`clm-raw`) | 56–57% |
| Always the most common label | 18.0% |

At this test size, ±6 points is ordinary chance variation (95% binomial interval at 88%).
The shipped head is the run with the project's standard seed, not the best of several:
choosing among runs by their test score would be tuning on the test set. The service
answered in 33 ms at the median (43 ms p95) for unseen texts on an Apple M5 Pro, and in
about 1 ms for repeated texts. Source: `Marketing/benchmark/results.json`.

## Run-to-run variation

With the earlier default head (1536 wide, 3 layers, 12.6M parameters) and unseeded weights,
the same 300 examples gave anywhere from 75% to 96% across training runs. Two fixes:

- Weights are now seeded, so the same data and seed reproduce the same head.
- Heads are smaller by default on small data. Over 8 seeds: 1536×3 gave 82–95% (mean
  90.9%), 512×2 gave 91–96% (mean 93.0%). On the 5,400-question typed-decisions data the
  two sizes scored alike (68–70%), so the larger head stays the default there.

## Limits

- The labels come from a model, not people. 95% means agreement with that labeller.
- Probabilities are over-confident (top class often above 0.99) and uncalibrated.
- Five topics only. Messages about two topics get the dominant one.
- English, synthetic, short conversations. No evidence for other languages, channels or lengths.
