# CLM-Nemotron: instructions for Claude Code

This repo is CLM (Contrastive Language Models) running on NVIDIA's Nemotron embedding
models. It serves fast classifiers and rankers behind an HTTP API and a browser
playground, where users can also train their own classifier. Developers working here
may only have Claude Code, so you run everything for them.

## What the developer says → what you run

All scripts live in `Claude Code/scripts/`. Quote the path, since it contains a space.
Run them from the repo root. They are idempotent.

| They say | You run | Then tell them |
| --- | --- | --- |
| "start the service", "run it", "start CLM" | `bash "Claude Code/scripts/start.sh"` | the URL it prints (http://localhost:8700/) and the models it lists |
| "stop", "shut it down" | `bash "Claude Code/scripts/stop.sh"` | that both processes stopped |
| "status", "is it running?" | `bash "Claude Code/scripts/status.sh"` | running or not; on trouble, the log lines it prints |
| "run the tests" | `bash "Claude Code/scripts/test.sh"` | the pass/fail count |
| "update the models", "re-download models" | `bash "Claude Code/scripts/fetch-models.sh" --force` | what it installed |
| "train a classifier on FILE" | see "Training" in `Claude Code/RUNBOOK.md` | test accuracy against the baselines |
| "export MODEL to DEST" (a folder or `s3://bucket/prefix/`) | `bash "Claude Code/scripts/export-model.sh" MODEL DEST` | the three files it wrote and the `clm-serve --model` line to load them elsewhere |

`start.sh` does everything on first run: it creates `.venv` (`setup.sh`), downloads and
verifies the models from this repo's GitHub release `models-v1` (`fetch-models.sh`, about
2.4 GB, needs `gh auth login`), starts the encoder on port 8090 and `clm-serve` on port
8700, and waits until both answer. The first run takes a few minutes; later runs take
seconds. Run it in the foreground and let it finish. It returns once both services are
up and leaves them running in the background.

If a step fails, read `Claude Code/RUNBOOK.md` → "Troubleshooting" and the log it names
(`logs/encoder.log`, `logs/clm-serve.log`) before changing anything.

## Rules for this repo

- Never push to the upstream `Contrastive-LM/CLM` repository. This fork lives at
  `github.com/rgvvvg5mpz-collab/clm-nemotron`.
- Never commit `models/`, `dist/`, `runs/`, `logs/` or `.venv/`. Models ship as GitHub
  release assets (`Claude Code/scripts/build-model-release.sh`), not in git.
- No Chinese-origin models anywhere: not as encoders, tokenizers, defaults or examples.
  Check where a model comes from before suggesting it. The encoders here are NVIDIA's,
  built on Mistral AI's Ministral 3. `tests/test_encoders.py` guards the one model family
  this fork replaced.
- A head only works with the encoder it was trained on. `clm-serve` refuses mismatches; don't
  work around that check.
- Run `bash "Claude Code/scripts/test.sh"` after code changes. CI runs the same on every push.

## Where things are

- `src/clm/`: the product. `server.py` is `clm-serve`, `train_cli.py` is `clm-train`,
  `train_jobs.py` handles training from the playground, `encoders.py` holds the encoder
  profiles, and `static/` is the playground.
- `clm-encoder` (`src/clm/local_encoder.py`): the encoder server used on Macs and CPUs (and by default everywhere).
- `Claude Code/scripts/` (operations) and `Claude Code/tools/` (builders for the videos, deck,
  guide screenshots and the frontier comparison).
- `docs/`: `USER_GUIDE.md` (step by step), `TRAINING.md`, `API.md`, `site/` (HTML pages).
- `research/`: upstream's method and results, agent-trace training (`finetune.py`),
  best-of-N evaluation, the T-Rex demo. Not needed to run the product.
- `Model Governance/`: architecture, model cards, data lineage, evaluation, risks.
- `Crew Training/`, `Marketing/`: training video and materials, executive deck and video.
