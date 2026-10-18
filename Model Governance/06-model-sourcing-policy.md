# 06 · Model sourcing policy

## Rule

1. **No Chinese-origin model is run anywhere in this application.** That covers encoders,
   tokenizers, heads, default downloads, examples, and any model called at run time or
   during training.
2. **Training-data lineage is disclosed, not prohibited.** If a third-party model we run was
   trained on data that other models (of any origin) helped generate, record it in
   [03-data-lineage.md](03-data-lineage.md) and in that model's card.
3. **Licences must allow commercial use and redistribution** of anything shipped in a
   release, and the licence and notice files must ship with it.

Rules 1 and 2 were confirmed with the system owner on 2026-10-18.

## Current status

| Model | Maker / base | Origin of what runs | Lineage notes | Licence | Compliant |
| --- | --- | --- | --- | --- | --- |
| Nemotron-3-Embed-1B | NVIDIA / Mistral AI Ministral-3-3B | USA / France | Synthetic training data partly generated with Alibaba Qwen models (doc 03) | OpenMDW-1.1 (+ Apache-2.0 for the base) | Yes (rule 2: disclosed) |
| Nemotron-3-Embed-8B | NVIDIA / Mistral AI Ministral-3-8B | USA / France | As above, plus Google, OpenAI and NVIDIA generators | OpenMDW-1.1 | Yes (rule 2: disclosed) |
| `support-topic-1b`, `typed-decisions-1b` | Trained here | n/a | Labels from an unnamed teacher model of unknown origin | Apache-2.0 | Yes; demo-only status covers the unknown labeller |

## How compliance is checked

- **Guard test.** `tests/test_encoders.py::test_repo_has_no_banned_model_references` fails
  CI if the name of the model family this fork replaced appears in any tracked file outside
  `Model Governance/`, which is the only place lineage may name it.
- **Encoder registry.** Only models listed in `src/clm/encoders.py` can be served, and
  `test_nemotron_is_the_only_encoder` pins that list to NVIDIA models.
- **New models.** The [new model checklist](templates/new-model-checklist.md) requires an
  origin and lineage check against the maker's model card before anything is added.

## History

The upstream project (`Contrastive-LM/CLM`) used a Chinese-origin 8B model as its encoder,
and its released heads and embedding datasets were built with it. This fork removed that
encoder, its serve script, the reference head and the dataset defaults (2026-10-18). Heads
or embeddings that don't record their encoder, which includes every upstream artifact, are
refused at load. Upstream's git history still contains the original code.
