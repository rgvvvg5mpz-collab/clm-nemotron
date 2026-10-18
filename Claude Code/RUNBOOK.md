# CLM runbook (for Claude Code)

Operating procedures for this repo. Run every command from the repo root and quote
`"Claude Code/..."` paths. Prefer the scripts over hand-typed commands: they encode the
ports, paths and checks below.

## Start

```bash
bash "Claude Code/scripts/start.sh"
```

What it does, in order. Each step is skipped when already done:

1. **No `.venv`?** It runs `setup.sh`. That finds Python 3.10+, creates `.venv` and
   installs `clm-nemotron[mac,test]`. On Linux without a GPU it installs CPU-only torch
   first. Takes 2–5 minutes.
2. **No models?** It runs `fetch-models.sh`. That downloads release `models-v1` from this
   repo's GitHub origin with `gh` (or `GITHUB_TOKEN`), verifies `SHA256SUMS` and unpacks
   to `models/`. About 2.4 GB.
3. It starts `tools/local_embed_server.py --encoder nemotron-3-embed-1b` on :8090,
   loading from `models/encoders/nemotron-3-embed-1b`. `HF_HUB_OFFLINE=1` is set, so
   there is no network access. It picks the device itself: CUDA, then Apple MPS, then CPU.
4. It starts `clm-serve` on :8700 with `--ckpt-dir models/heads` (the shipped models) and
   `--runs-dir runs/playground` (models trained in the browser).
5. It waits for both to answer, then prints the URL and the model names.

Success looks like:

```text
[clm] running. Open http://localhost:8700/  (models: support-topic-1b, typed-decisions-1b, clm-raw)
```

Check it any time with `bash "Claude Code/scripts/status.sh"`, which exits 1 if
something is down.

## Verify it answers

```bash
curl -s localhost:8700/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "customer: my parcel never arrived",
  "model": "support-topic-1b",
  "questions": {"label": {"type": "choice", "instructions": "What is this customer conversation primarily about?",
    "criteria": {"billing": "A charge, invoice, subscription or payment problem.",
                 "delivery": "Shipping, fulfilment or delivery of a physical item."}}}}'
```

Expect `"choice":"delivery"`. The exact question each shipped model was trained with is in
`models/heads/<name>.json` under `questions`; send that for the accuracy it reports.

## Stop, restart, logs

- Stop: `bash "Claude Code/scripts/stop.sh"`. Restart: stop, then start.
- Logs: `logs/encoder.log` and `logs/clm-serve.log`. PIDs are in `.run/`.

## Troubleshooting

| Symptom (script output or log) | Cause | Fix |
| --- | --- | --- |
| `Python 3.10 or newer is needed` | no suitable Python | Mac: `brew install python@3.12`. Ubuntu: `sudo apt install python3.12-venv`. Ask the user before installing. Then start again. |
| `need GitHub access to download the models` | `gh` not signed in | Ask the user to run `gh auth login` (it opens a browser), then start again. |
| `release models-v1 not found` / 404 | wrong origin, or no access | Check `git remote get-url origin`. Set `CLM_MODELS_REPO=rgvvvg5mpz-collab/clm-nemotron`. |
| `a downloaded file is corrupt` | interrupted download | `bash "Claude Code/scripts/fetch-models.sh" --force` |
| `port 8090 is in use` / `port 8700 is in use` | another program, or an old run without its pid file | `lsof -nP -iTCP:8700 -sTCP:LISTEN` to see it. Stop it only if it is this project's process; otherwise set `CLM_PORT` / `CLM_ENC_PORT`. |
| `the encoder did not start` | see the log tail it prints | Out of memory: close apps (the encoder needs ~3 GB). `No module named sentence_transformers`: run `setup.sh`. |
| clm-serve log: `belongs to encoder …` / `records no encoder` | a head from another encoder in `models/heads/` or `runs/playground/` | Remove that head, or start with the matching `CLM_ENCODER`. Never bypass the check. |
| playground shows "encoder down" | encoder crashed | `status.sh`, read `logs/encoder.log`, restart. |
| no internet at all | | Copy a `dist/models-v1/` folder over, then run `fetch-models.sh --from-dir PATH` |

## Training

**In the browser** (simplest; point users here first): http://localhost:8700/ → **Train a
model**. Upload a CSV, describe the labels, train. The result is served at once as a
new model and kept in `runs/playground/`. Full walk-through: `docs/USER_GUIDE.md`.

**From the command line**, with the service running:

```bash
.venv/bin/clm-train --data path/to/data.csv --encoder nemotron-3-embed-1b \
  --question "Which team should handle this?" --options path/to/options.json \
  --group-column customer --out runs/my-model
```

- `data.csv` needs a `text` and a `label` column. `options.json` maps every label to a
  one-sentence description: `{"billing": "A charge, invoice or payment problem.", ...}`.
- It prints test accuracy next to `clm-raw` and the majority label. If it doesn't beat
  the majority label, say so plainly and suggest better label descriptions
  (`docs/USER_GUIDE.md` → "Improving accuracy").
- To serve the result with the shipped models, copy it into `models/heads/` and restart:
  `cp runs/my-model/best_head.pt models/heads/my-model.pt`. `runs/my-model/questions.json`
  holds the question to send.
- All options: `docs/TRAINING.md`.

## Bigger encoder (optional, NVIDIA GPU)

The 8B encoder is more accurate but needs about 18 GB of GPU memory and downloads from
Hugging Face:

```bash
CLM_WITH_VLLM=1 bash "Claude Code/scripts/setup.sh"
bash "Claude Code/scripts/serve-encoder-gpu.sh" &
```

Then run `clm-serve --encoder nemotron-3-embed-8b`. Heads trained on the 1B don't work
with the 8B; retrain them.

## Releasing new models (maintainers)

```bash
bash "Claude Code/scripts/build-model-release.sh" NAME=RUN_DIR [NAME=RUN_DIR ...]
gh release create models-v2 dist/models-v2/* --title "Models v2" --notes "..."
```

Set `CLM_MODELS_TAG=models-v2` for both commands, then bump the default in
`Claude Code/scripts/common.sh`. Record the new models in `Model Governance/`
(model inventory and evaluation results) before announcing them.

## Tests

`bash "Claude Code/scripts/test.sh"` runs pyflakes and pytest: no GPU, network or running
service needed, about 5 seconds. CI (`.github/workflows/tests.yml`) runs the same.
