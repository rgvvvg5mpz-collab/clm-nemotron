# HTTP API and `clm-serve` options

## `POST /v1/systemone`

| field | |
| --- | --- |
| `state` | string, object or array (objects are rendered as `key: value` text, arrays as `- item` lines; never JSON, the heads are trained on prose) |
| `model` | `clm-latest`, `clm-raw`, or any model from `GET /v1/models`; default `clm-latest`, or `clm-raw` when no head is served |
| `questions` | `{id: Question}`, at least one |
| `temperature` | optional, `(0, 100]`, default 1; divides the logits before the softmax |

| question | required | answer |
| --- | --- | --- |
| `noul` | `instructions`; optional `criteria: {"true": …, "false": …}` | `{"noul": p_true}` |
| `choice` | `instructions` (the question), `criteria: {option: description}` (each option is embedded as its description, or its key when the description is empty) | `{"choice", "confidence", "probabilities"}` |
| `score` | `instructions`, `criteria: [level0, level1, …]` (ordered, ≥2) | `{"score", "confidence", "legend", "probabilities"}` |

- `confidence` = top probability minus the mean of the others.
- `score` = expected level index; `legend` maps indices back to the rubric.
- `usage.input_tokens` counts encoder tokens spent on cache misses;
  `billing_units` is the number of questions.
- Errors: `401` bad key · `422` malformed request or unknown model · `502`
  embedder unreachable. `X-CLM-Latency-Ms` carries the server-side time.

## `POST /v1/rank`

The same primitive in its plain form: `{"context": ..., "question": ..., "answers": [...]}`
returns `{"model", "ranked": [{"rank", "candidate", "prob"}, ...]}`, best first. The
state head sees `context + question`, the action head sees each answer verbatim.
`CLMClient.rank(context, question, answers)` and `Engine.rank(context, answers, question)`
are the client and in-process forms.

## `GET /`

The playground (see [above](#playground)), unless `clm-serve --no-ui`. Static
files only; every API route above shadows it.

## `GET /v1/models`

```json
{"models": [{"name": "support-topic-1b", "kind": "trained", "release_date": "2026-09-19",
             "description": "What is this customer conversation primarily about? (account, billing, ...). 88% on its test set",
             "questions": [{"name": "label", "type": "choice", "instructions": "...", "labels": ["account", "billing", "..."]}],
             "test_accuracy": 0.88},
            {"name": "clm-raw", "kind": "raw", "description": "Cosine in the raw encoder embedding space, no projection head; ...", ...}]}
```

`questions` and `test_accuracy` come from the card saved beside the head (`<name>.json`, or
`questions.json` + `summary.json` for a playground run); they are left out when there is none.
The playground's status light lists these when clicked.

## Export routes

| Route | Body / result |
| --- | --- |
| `GET /v1/models/{name}/download` | `{name}.tar.gz` (`application/gzip`) holding `{name}.pt`, `{name}.json` and `{name}.sha256` |
| `POST /v1/models/{name}/export` | `{"dest": "folder on the server" \| "s3://bucket/prefix/", "endpoint_url"?: "..."}` → `{name, dest, files: [{name, size, location}], card}`. S3 needs `boto3` and credentials on the server. 404 for an unknown model, 422 for `clm-raw` or an unusable destination. |

The card (`{name}.json`) records the encoder identity, the questions and labels, test
accuracy, the head's shape and the `.pt` file's SHA-256. Command line: `clm-export`.

## Training routes (`/v1/train/*`)

Used by the playground's **Train a model** view; off with `clm-serve --no-train`. All take
the same `Authorization: Bearer` key as the rest of the API.

| Route | Body / result |
| --- | --- |
| `POST /v1/train/inspect` | `{filename, content_b64}` (CSV, JSONL, JSON or Parquet, up to 50 MB) → `upload_id`, row count, columns, a preview, per-column value counts, suggested text and label columns, and whether the file is in the typed format |
| `POST /v1/train/jobs` | `{upload_id, text_column, label_column, question, type: "choice"|"noul", options: {label: description}, group_column?, test_frac?, epochs?, name?}` → the job. For typed files only `upload_id` (and optional `group_column`, `test_frac`, `epochs`, `name`) is needed. One job runs at a time (422 otherwise). |
| `GET /v1/train/jobs` | `{"jobs": [...]}`, newest first |
| `GET /v1/train/jobs/{id}` | `status` (`queued`, `running`, `done`, `failed`, `stopped`), `progress` (stage, counts), `history` (check-set accuracy per round), `baselines`, `result` (test accuracy of the model, of `clm-raw` and of the majority label), `model`, `served`, `checkpoint` |
| `POST /v1/train/jobs/{id}/stop` | Stop a queued or running job |

A finished job's model is served at once under its `model` name, and again after a
restart (`--runs-dir`). `GET /health` reports `"training": true` when these routes are on.

## `clm-serve` options

```
clm-serve [--port 8700] [--encoder nemotron-3-embed-8b|nemotron-3-embed-1b]
          [--emb-url http://127.0.0.1:8090/v1/embeddings] [--emb-model NAME]
          [--max-tokens 8192] [--ckpt PATH] [--ckpt-dir DIR] [--model NAME=PATH ...] [--device cpu|cuda]
          [--action-cache 0.02|512MiB|0] [--host 127.0.0.1] [--runs-dir DIR] [--no-train] [--no-ui] [--cors]
```

`--encoder` names the frozen encoder behind `--emb-url`. It sets the
prompts, the token budget and the expected width, and heads trained on
any other encoder are refused. `--emb-model` defaults to the encoder's name.
`--ckpt PATH` (or `CLM_CKPT`) serves a head as `clm-latest`; there is no default.
`--ckpt-dir DIR` serves every
`*.pt` there under its file stem; `--model NAME=PATH` adds one more.
The heads run on the GPU when torch sees one, else on the CPU; `--device` (or
`CLM_DEVICE`) forces one. Checkpoints hot-reload when the file changes. Set `CLM_API_KEY` to require
`Authorization: Bearer <key>` (the playground has a field for it). Environment
equivalents: `CLM_ENCODER`, `CLM_PORT`, `CLM_EMB_URL`, `CLM_EMB_MODEL`, `CLM_EMB_MAX_TOKENS`, `CLM_CKPT`,
`CLM_DEVICE`, `CLM_ACTION_CACHE`.

`--runs-dir` (default `~/.cache/clm/runs`, or `CLM_RUNS_DIR`) is where models trained in the
playground are kept. Each one is served again when `clm-serve` restarts, provided it was
trained on the same encoder. `--no-train` turns the Train view and the `/v1/train/*` routes off.
`--host` defaults to `127.0.0.1`. Pass `0.0.0.0` to expose the API, file uploads and training
to the network, and then set `CLM_API_KEY`.

`--no-ui` drops the playground and serves the API alone. `--cors` allows browser
requests from any origin and is off by default, because an API key otherwise
travels in a header any page would then be free to send.

### The vector cache

An agent asks about a changing state but a mostly fixed set of actions, and it
revisits states it has already seen. Neither their embeddings nor their
projections change while the head does not, so `clm-serve` reserves a slab of
device memory at start-up — the way vLLM claims its KV cache — and keeps them in
it:

```
[clm] vector cache 505.0 MB reserved on cuda (215,764x512d + 3,852x4096d)
```

`--action-cache` takes a fraction of the device (`0.02`, the default), an
absolute size (`512MiB`), or `0` to switch it off; `CLM_ACTION_CACHE` does the
same. It covers states and actions on every served head, and `clm-raw` in the
encoder's own space — the two widths are pools carved from the one allocation,
which never grows, so a long-running server cannot drift into an out-of-memory
kill. Entries are keyed by head and generation, so several heads share the arena
and a hot-reloaded head stops matching rows its previous weights produced;
eviction is least-recently-used. `GET /health` reports occupancy and hit rate.

A hit skips the encoder call, the host-to-device copy and the head's forward
pass. Measured on one RTX 4090, server-side p50, against a fixed action set:

| | 3 actions | 50 actions |
|---|---|---|
| new state every call | 28.6 → 28.0 ms | 28.8 → 28.1 ms |
| revisited states (20 rooms) | 1.7 → 0.6 ms | 2.0 → 0.7 ms |
| one repeated state | 1.7 → 0.6 ms | 2.0 → 0.7 ms |

So a loop that revisits states answers about 2.8x faster, and a loop that never
repeats itself pays the encoder either way. A cached vector costs no encoder
tokens, so `usage.input_tokens` counts only what the encoder actually did.
