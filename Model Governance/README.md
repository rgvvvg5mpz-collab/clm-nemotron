# Model Governance

The system of record for what models this application runs, how they were built and
evaluated, what they must not be used for, and how changes are controlled. Written for
model risk, compliance and architecture reviewers. Every figure here comes from files in
this repo or the `models-v1` release, and each section names the file that produced it.

| | |
| --- | --- |
| **System** | CLM-Nemotron: contrastive classifiers and rankers served over HTTP (`clm-serve`) with a browser playground and in-browser training |
| **Version** | Application `clm-nemotron` 0.2.0 · model release `models-v1` (2026-10-18) |
| **Repository** | `github.com/rgvvvg5mpz-collab/clm-nemotron` (private); fork of `Contrastive-LM/CLM` (Apache-2.0) |
| **Models in scope** | 1 encoder (frozen) + 2 trained heads, plus the head-less `clm-raw` mode. See [02-model-inventory.md](02-model-inventory.md) |
| **Decision type** | Text classification and ranking; outputs are probabilities over a fixed set of labels |
| **Status of shipped heads** | **Demonstration only.** Trained on synthetic data labelled by an unknown model. Not approved for production decisions. See [05-risks-limitations-controls.md](05-risks-limitations-controls.md) |
| **Model sourcing rule** | No Chinese-origin model is run anywhere in the stack. Training-data lineage is disclosed, not prohibited. See [06-model-sourcing-policy.md](06-model-sourcing-policy.md) |

## Documents

| # | Document | Answers |
| --- | --- | --- |
| 01 | [System architecture](01-system-architecture.md) | Components, data flow, trust boundaries, where data is stored, security controls |
| 02 | [Model inventory](02-model-inventory.md) | Every model, its version, hash, licence, owner role and status |
| — | [Model cards](model-cards/) | One card per model: purpose, design, training, evaluation, limits |
| 03 | [Data lineage](03-data-lineage.md) | Where training and evaluation data came from, how it was labelled, known gaps |
| 04 | [Evaluation](04-evaluation.md) | Method, results with uncertainty, latency and throughput, what the numbers do not show |
| 05 | [Risks, limitations and controls](05-risks-limitations-controls.md) | Risk register with likelihood, impact, controls and residual risk |
| 06 | [Model sourcing policy](06-model-sourcing-policy.md) | Origin rule, licences, how compliance is checked |
| 07 | [Operations and monitoring](07-operations-monitoring.md) | Deployment, monitoring, retraining triggers, incident steps, approvals |
| 08 | [Change log](08-change-log.md) | Model and application changes with dates |
| — | [New model checklist](templates/new-model-checklist.md) | What must be true before a new head is served for real decisions |

## Summary for reviewers

- **What the models do.** A frozen NVIDIA embedding model turns each text into a vector.
  Two small trained networks (2.6M–12.6M parameters, depending on data size) compare a text against the
  description of each label, and a softmax turns the scores into probabilities. Nothing
  generates text.
- **What has been measured.** On a held-out support-ticket test set (100 conversations),
  the shipped topic head is right 88–89% of the time (88–96% across 9 training runs),
  against 56–57% for the untrained encoder and 18% for always guessing the most common
  label. Latency is about 33 ms per request on a laptop. On a harder 15-question
  benchmark the shipped head reaches 69.2%. The labeller of that benchmark agreed with
  itself only 73.5% of the time.
- **Main risks.** The demo data is synthetic and labelled by an unknown model. Probabilities
  are over-confident until calibrated. Accuracy depends heavily on how labels are described.
  The encoder's training data includes synthetic data from Chinese-origin models, which is
  disclosed and allowed under the sourcing rule. Each risk has controls in document 05.
- **Before any production use:** train on owned, human-labelled data, calibrate, validate on
  a holdout split by customer, and complete the [new model checklist](templates/new-model-checklist.md).
