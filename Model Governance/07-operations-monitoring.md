# 07 · Operations and monitoring

## Deploy and run

`Claude Code/scripts/start.sh` installs, downloads and verifies the models, and starts both
services. `status.sh` and `stop.sh` check and stop them. Full runbook:
[`Claude Code/RUNBOOK.md`](../Claude%20Code/RUNBOOK.md).

## What to monitor

| Signal | How | Act when |
| --- | --- | --- |
| Service health | `GET /health` (encoder reachable, models loaded) or `status.sh` | Any check fails |
| Latency | `X-CLM-Latency-Ms` response header; `evaluation/benchmark_service.py` | p95 exceeds 2× the baseline in doc 04 |
| Accuracy on live traffic | Each week, a random sample of at least 100 decisions labelled by people, scored with `benchmark_service.py` | Drops more than 5 points below validation accuracy |
| Confidence distribution | Log the top-class probability; compare week-over-week histograms | A large shift, or growth in low-confidence answers |
| Label mix | Share of each predicted label per week | A label's share changes by more than half without a business reason |
| Input length | Characters per request | Many inputs near the 8,192-token budget |

## Retraining triggers

Retrain (with `clm-train` or the Train view) and re-validate when: accuracy on the
monitoring sample drops by more than 5 points; labels or their descriptions change; a new
data source is added; or the encoder changes (heads must always be retrained then).

## Change control

1. Train on approved data and save the run folder (`best_head.pt`, `questions.json`, `summary.json`).
2. The validator completes the [new model checklist](templates/new-model-checklist.md).
3. Add the head to the inventory (doc 02), write its model card and log it in doc 08.
4. Ship it in a new model release (`build-model-release.sh`, tag `models-vN`) and update
   `CLM_MODELS_TAG`. Never overwrite the files of an existing release.
5. Keep the previous release available for rollback: `CLM_MODELS_TAG=models-v(N-1) fetch-models.sh --force`.

## Incidents

| Situation | First steps |
| --- | --- |
| Wrong answers reported | Re-run the case in the playground and in the API (JSON tab) to reproduce. Check that the request used the head's exact `questions.json`. Check whether it is near the decision boundary (low confidence). |
| Service down | `status.sh`, read the logs it prints, then `stop.sh` and `start.sh` |
| Suspected data exposure | Stop the service. Check whether `--host` was changed from 127.0.0.1 and whether `CLM_API_KEY` was set. Review `runs/playground/uploads/`. |
| Corrupt or unexpected model file | `fetch-models.sh --force` re-verifies against `SHA256SUMS`. Never bypass the encoder compatibility check. |
