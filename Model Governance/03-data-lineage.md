# 03 · Data lineage

## Encoder pre-training (not done here)

| Item | Detail | Source |
| --- | --- | --- |
| Maker | NVIDIA | Model card |
| Base weights | Mistral AI Ministral-3-3B-Instruct-2512 (1B) / Ministral-3-8B-Instruct-2512 (8B), Apache-2.0 | Model card, NOTICE file in the release |
| Data | 8.5M+ examples, 161 files: public commercially permissible retrieval datasets (MIRACL, MLDR and others) plus synthetic query–passage pairs | Model card |
| Synthetic data generators (1B) | Alibaba Qwen: Qwen3-Next-80B-A3B-Instruct, Qwen3-235B-A22B, Qwen3.5-397B-A17B, Qwen3.6-27B, Qwen3.6-35B-A3B | Model card, "Synthetic Datasets" |
| Synthetic data generators (8B) | The Qwen models above, plus Google gemma-4-31B-it, OpenAI gpt-oss-120b / -20b, NVIDIA Nemotron-3 Super and Ultra | Model card |
| Personal data | "None Known"; user-interaction data from NVIDIA's models was used | Model card, Privacy table |

**Sourcing assessment.** None of these generator models is downloaded, run or called by this
application; they shaped NVIDIA's training data only. Under the sourcing rule (run-time models only, lineage disclosed) this
is permitted and recorded as risk R7. See [06-model-sourcing-policy.md](06-model-sourcing-policy.md).

## Head training data (demo heads)

| Item | `support-topic-1b` | `typed-decisions-1b` |
| --- | --- | --- |
| Dataset | `LocalLLaMA/typed-decisions` (Hugging Face), Apache-2.0, tagged synthetic | same |
| Subset | `customer_service` workflow, `category` question; conversation text only | all four workflows, all 15 questions |
| Rows | 300 train (30 of them used as the check set), 100 test, from the dataset's own split | 1,080 train, 120 check, 400 test |
| Labels | "Gold" is the mean of three samples from an unnamed teacher model of roughly 4B-class capability. Teacher self-agreement: 0.735. | same |
| Labeller identity | **Unknown** (not disclosed by the dataset authors) | same |
| Personal data | Synthetic; names and order numbers are invented | same |
| Copy in this repo | `src/clm/static/sample/support-tickets.csv` (text, label, split), with `NOTICE.txt` | not copied; downloaded by `finetune.py` |

**Assessment.** The labels measure agreement with an unknown model, so accuracy against them
says nothing about correctness for real customers. This is why both heads are demo-only
(risk R1).

## Data the application handles at run time

| Data | Enters through | Kept |
| --- | --- | --- |
| Texts to classify | `POST /v1/systemone`, `/v1/rank`, playground | In memory (vector caches) until evicted; never written to disk or logs |
| Training files | Playground Train view (`/v1/train/inspect`), `clm-train --data` | `runs/playground/uploads/`; embeddings of them in `runs/playground/embeddings/`. Kept until an operator deletes them. |
| Label descriptions, questions | Requests, training jobs | Inside each trained head's `questions.json` / `summary.json` |

Embeddings derived from texts can leak information about them. Treat
`runs/playground/embeddings/` with the same sensitivity as the uploads.
