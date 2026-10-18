# Run CLM with Claude Code

You don't need to know Python, GPUs or the command line. Clone the repo, open Claude Code
in it, and ask in plain words.

```text
git clone https://github.com/rgvvvg5mpz-collab/clm-nemotron.git
cd clm-nemotron
claude
> start the service
```

Claude reads `CLAUDE.md` at the repo root, which maps your request to the scripts in this
folder. The first start takes a few minutes:

1. It creates a Python environment in `.venv/` and installs CLM.
2. It downloads the models from the repo's GitHub release (`models-v1`, about 2.4 GB) and
   checks their checksums.
3. It starts the Nemotron encoder on port 8090 and the CLM service on port 8700.

Then open **http://localhost:8700/**. Later starts take seconds.

## Things you can ask

| Ask | What happens |
| --- | --- |
| start the service | installs on first run, then starts both services |
| is it running? | status of both services, the loaded models, recent errors |
| stop the service | stops both |
| run the tests | the automated test suite (about 5 seconds) |
| train a classifier on data/tickets.csv | trains from the command line and reports accuracy |
| re-download the models | fetches and verifies the release again |
| export ticket-topic to ./exports | writes the model, its card and checksums to that folder |
| export ticket-topic to s3://my-bucket/clm/ | the same three files, uploaded with the machine's AWS credentials |

You can also train in the browser: open the playground and choose **Train a model**.
The [user guide](../docs/USER_GUIDE.md) walks through it with screenshots.

## What your machine needs

- macOS on Apple silicon, or Linux (Ubuntu 22.04+). On Windows, use WSL2.
- Python 3.10 or newer. Claude will tell you how to install it if it's missing.
- About 8 GB of free memory and 6 GB of disk.
- GitHub access to this repo. If `gh` isn't signed in yet, Claude will ask you to run
  `gh auth login` once, to download the models from the private release.

No NVIDIA GPU is needed: the 1B encoder runs on a Mac's GPU or on a CPU. With an NVIDIA
GPU it runs there automatically.

## What's in this folder

| File | Purpose |
| --- | --- |
| `RUNBOOK.md` | Detailed operating steps and troubleshooting, written for Claude |
| `scripts/start.sh` | set up if needed, fetch models if needed, start both services |
| `scripts/stop.sh`, `scripts/status.sh` | stop, check |
| `scripts/setup.sh` | create `.venv` and install (`CLM_WITH_VLLM=1` also installs vLLM) |
| `scripts/fetch-models.sh` | download, verify and unpack the model release (`--from-dir`, `--from-hub`) |
| `scripts/test.sh` | lint and tests |
| `scripts/serve-encoder-gpu.sh` | optional: the 8B encoder behind vLLM on an NVIDIA GPU |
| `scripts/build-model-release.sh` | maintainers: package models for a new release |
| `scripts/common.sh` | shared settings (ports, paths) |

Settings you can change with environment variables: `CLM_PORT` (8700), `CLM_ENC_PORT`
(8090), `CLM_ENCODER` (`nemotron-3-embed-1b`), `CLM_MODELS_DIR` (`models/`), `CLM_RUNS_DIR`
(`runs/playground/`), `CLM_MODELS_REPO` (where the release lives).
