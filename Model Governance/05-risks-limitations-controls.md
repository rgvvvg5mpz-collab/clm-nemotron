# 05 · Risks, limitations and controls

Rating: likelihood × impact, each Low / Medium / High, for a deployment that follows the
operating guidance. "Residual" is the rating after the listed controls.

| # | Risk | L | I | Controls in place | Further action before production | Residual |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | **Shipped heads are demos.** Their labels come from an unknown model, so their accuracy does not transfer to real use. | High | High | Marked "demo only" in the UI, docs, model cards and release notes | Train on owned, human-reviewed labels; validate per the checklist | Low once retrained |
| R2 | **Over-confident probabilities.** Trained heads put more than 0.99 on the top class, so thresholds on raw probabilities mislead. | High | Medium | UI and docs warn; `temperature` parameter on every request; `confidence` shown as a ranking signal | Fit temperature on validation data; measure ECE; set thresholds from validation | Medium |
| R3 | **Label-description sensitivity.** Accuracy depends on how labels are described; edited descriptions silently change behaviour. | Medium | Medium | Live hints in the Train view; `questions.json` saved with each head; guide on writing descriptions | Version-control questions with the head; re-evaluate after any wording change | Low |
| R4 | **`clm-raw` misuse.** Without a head, scores fall below the majority baseline on some tasks. | Medium | Medium | Status shows "no head (clm-raw)"; model card; docs | Don't route decisions to `clm-raw` | Low |
| R5 | **Truncation.** Texts beyond 8,192 tokens lose their beginning (prompt kept). | Low | Medium | Truncation keeps the state's end, which holds the question; documented | Put decision-relevant text near the end; monitor input lengths | Low |
| R6 | **Distribution shift.** Real traffic differs from the training data. | High | Medium | Held-out test, majority/raw baselines | Monitoring sample with human labels (doc 07); retraining triggers | Medium |
| R7 | **Encoder data lineage.** NVIDIA's training data for the encoders includes synthetic data from Chinese-origin models. | Certain | Policy | Disclosed here; no such model runs in the stack; guard test against run-time use | Policy owner sign-off recorded in doc 06 | Accepted |
| R8 | **Network exposure.** An exposed service accepts uploads and starts training jobs. | Low | High | Binds to 127.0.0.1 by default; optional API key; 50 MB upload cap; one job at a time; parsing as data only | Put behind an authenticating proxy for shared use; set `CLM_API_KEY` | Low |
| R9 | **Retention of uploaded data.** Uploads and their embeddings stay in `runs/playground/`. | Medium | Medium | Documented in docs 01 and 03 | Operator retention schedule; delete runs when a model retires | Medium |
| R10 | **Malicious model files.** A crafted `.pt` file could run code. | Low | High | `torch.load(weights_only=True)`; SHA-256-verified release | Load heads only from the release or your own training | Low |
| R11 | **Encoder–head mismatch.** Serving a head with another encoder produces meaningless scores. | Low | High | Every head records its encoder; serving refuses mismatches and unrecorded heads; tested | none | Low |
| R12 | **No fairness evaluation.** Neither NVIDIA nor this project measured bias. | Unknown | High for decisions about people | Documented | Fairness testing on representative data before any decision that affects people | Open |
| R14 | **Run-to-run variation.** Retraining on the same small dataset gave 75–96% before the fixes. | Medium | Medium | Seeded weights (reproducible); smaller default head under 3,000 examples (88–96% across 9 seeds) | Train several seeds and pick by check-set score, never test score; report the spread | Low |
| R13 | **Single-machine operation.** No high availability; training competes with serving for CPU. | Medium | Low | One job at a time; status script; restart restores models | Separate training from serving for production load | Low |

## Limitations to state wherever results are presented

1. The demo results measure agreement with a model labeller on synthetic data.
2. One test split per model; differences under about 5 points (at 100 items) are within noise.
3. A frontier-model comparison of accuracy and latency has not been measured; only list-price
   cost is compared.
4. English only has been tested, although the encoder is multilingual.

## Not suitable for

- Decisions with legal or similarly significant effects on people (credit, employment,
  insurance, healthcare, law enforcement) without independent validation, human review
  and fairness testing.
- Open-ended questions, generation, multi-step reasoning, or label sets that change per
  request without retraining. Use a frontier language model for those.
