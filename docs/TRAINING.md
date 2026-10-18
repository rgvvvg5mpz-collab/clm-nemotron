# Training CLM heads with `clm-train`

**In the browser:** start `clm-serve`, open `http://localhost:8700/`, and choose **Train a
model** in the top bar. It walks through the same steps with your file: pick columns,
describe labels, train, read the results, try it. The model is served as soon as it
finishes. The rest of this page is the command-line version.

`clm-train` turns labelled examples into a head that `clm-serve` can serve. The
Nemotron encoder stays frozen. Every text is embedded once, then only the two small
projection heads (state and action) and the logit scale are trained. The illustrated
version of this page is `docs/site/training.html`.

## 1. Install and start the encoder

```bash
pip install -e .            # Linux
pip install -e ".[mac]"     # macOS (Apple silicon)
```

| `--encoder` | Use it for | Start it with |
| --- | --- | --- |
| `nemotron-3-embed-1b` | laptops, quick iteration (2.3 GB, 2048-d) | `python tools/local_embed_server.py --encoder nemotron-3-embed-1b` |
| `nemotron-3-embed-8b` (default) | production on an NVIDIA GPU (15.9 GB, 4096-d) | `./serve_nemotron3_embed_8b.sh` |

Training reaches the encoder at `http://127.0.0.1:8090/v1/embeddings` (`--embed-url`
to change it, `--offline` for in-process vLLM on Linux). A head only works with the
encoder it was trained on, so train with the one you will serve with.

## 2. Prepare the data

**Simple format**: every row answers the same question. Use a CSV, JSONL, JSON or
parquet file with `text` and `label` columns (`--text-column` and `--label-column`
rename them), plus the options:

```csv
text,label,customer
"customer: I was charged twice…",billing,c-1042
"customer: The app crashes on login",technical,c-0007
```

```json
{"billing":   "A charge, invoice, subscription or payment problem.",
 "technical": "The product or service is not working as expected."}
```

| Type | Flags | Labels |
| --- | --- | --- |
| `choice` | `--options '{"key": "description", ...}'` | an option key |
| `score` | `--type score --options '["level 0", "level 1", ...]'` | a level index or the exact level text |
| `noul` | `--type noul` (optional `--options '{"true": "...", "false": "..."}'`) | true/false, yes/no, 1/0 |

**Typed rows**: one state with several questions, the `LocalLLaMA/typed-decisions`
schema. Rows can come from a file, a directory laid out as
`<workflow>/{train,test}-*.parquet`, or a Hub dataset id:

```json
{"id": "t-17", "split": "train", "state": {"thread": [...]},
 "questions": {"category": {"type": "choice", "instructions": "...", "criteria": {...}}},
 "gold": {"category": {"label": "delivery", "probabilities": {"delivery": 0.95, "account": 0.05}}}}
```

The test split comes from, in order:

1. `--test FILE`;
2. a `split` column (`train` / `test`);
3. the directory layout;
4. `--test-frac` (default 0.2).

Pass `--group-column` (customer, account, incident) so a group never sits on both
sides of the test or validation split.

## 3. Run

```bash
clm-train --data tickets_train.csv --test tickets_test.csv \
  --question "What is this customer conversation primarily about?" --options options.json \
  --encoder nemotron-3-embed-1b --out runs/tickets

clm-train --data LocalLLaMA/typed-decisions --workflow all --encoder nemotron-3-embed-1b --out runs/typed
```

The run goes through four stages:

1. **Split.** Train, validation (`--val-frac` of train, by group) and test.
2. **Embed.** Every text goes through the same `query: ` / `passage: ` prompts and
   truncation as `clm-serve`, and is cached in `OUT/embeddings/`. Re-runs embed only
   new texts.
3. **Train.** Up to `--epochs`; the epoch with the best validation accuracy is kept,
   with early stopping after `--patience` epochs without improvement.
4. **Score.** One pass on the test split, next to `clm-raw` and the majority label.

Real runs on an Apple M5 Pro with Nemotron-3-Embed-1B:

| Data | Train rows | Majority | clm-raw | Head | Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| customer-service category (1 question, 5 options) | 300 | 18.0% | 56.0% | 95.0% | 16 s |
| typed-decisions, all workflows (15 questions) | 1,200 | 45.0% | 37.8% | 69.2% | 4.2 min |

## 4. Outputs (`--out`)

| File | What it is |
| --- | --- |
| `best_head.pt` | the head to serve (weights, logit scale, encoder stamp) |
| `final_head.pt` | last epoch, for comparison |
| `questions.json` | simple format: the exact question to send when serving |
| `summary.json` | test accuracy per question: head, clm-raw, majority |
| `history.json` | validation metrics per epoch |
| `embeddings/` | cached vectors |

## 5. Serve

```bash
clm-serve --encoder nemotron-3-embed-1b --ckpt runs/tickets/best_head.pt
```

```python
import json
from clm import CLMClient
questions = json.load(open("runs/tickets/questions.json"))
r = CLMClient().system_one(state="customer: I was billed twice this month.", questions=questions)
r.answers["label"].choice
```

Trained heads are very confident. Calibrate a temperature and any thresholds on
held-out data before a probability gates anything. A running `clm-serve` reloads
`best_head.pt` when the file changes.

## Errors

| Message contains | Fix |
| --- | --- |
| `embedder unreachable` | start the encoder or fix `--embed-url` |
| `not served` / `returned 2048-d vectors` | the encoder server runs another size; match `--encoder` |
| `is not one of the options` | a label is not a key in `--options` |
| `has no 'split' column` | add a `split` column, or use `--test` / `--test-frac` |
| `rows are missing` | text + label data needs `--question` and `--options` |
| `need labelled questions in train, validation and test` | too few rows for the splits |
| `belongs to encoder` / `records no encoder` | the head is from another encoder; retrain with the matching `--encoder` |

Agent trajectories (state → action, best-of-N verification) use
`train/finetune.py --task clm` and `evaluation/bon_eval.py` instead. `docs/FINETUNING.md`
is upstream's script for an AI agent running fine-tuning experiments in a loop, not
a user guide.
