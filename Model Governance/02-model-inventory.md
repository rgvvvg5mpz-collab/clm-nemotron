# 02 · Model inventory

Every model this application can run. Hashes are SHA-256 from `models/MANIFEST.json` (release `models-v1`).

| Id (as served) | Type | Version / source | Size | Licence | Status | Card |
| --- | --- | --- | --- | --- | --- | --- |
| `nemotron-3-embed-1b` | Encoder, frozen | `nvidia/Nemotron-3-Embed-1B-BF16`, Hub revision `c0c9fea9`, released by NVIDIA 2026-07-16 | 1.14B parameters, 2.28 GB | OpenMDW-1.1 | **In use** (default encoder) | [card](model-cards/encoder-nemotron-3-embed-1b.md) |
| `nemotron-3-embed-8b` | Encoder, frozen | `nvidia/Nemotron-3-Embed-8B-BF16` (downloaded from Hugging Face on demand; not in the release) | ~8B parameters, 15.9 GB | OpenMDW-1.1 | Available, opt-in, **no trained heads** | [card](model-cards/encoder-nemotron-3-embed-8b.md) |
| `support-topic-1b` | Trained head pair | `models-v1`, trained 2026-10-18 | 2.6M parameters, 10.5 MB | Apache-2.0 (this repo) | **Demo only** | [card](model-cards/head-support-topic-1b.md) |
| `typed-decisions-1b` | Trained head pair | `models-v1`, trained 2026-10-18 | 12.6M parameters, 50.4 MB | Apache-2.0 (this repo) | **Demo only** | [card](model-cards/head-typed-decisions-1b.md) |
| `clm-raw` | Scoring mode, no trained weights | Cosine similarity in the encoder's space, fixed scale 100 | none | n/a | Available; for smoke tests and ranking only | [card](model-cards/clm-raw.md) |
| Playground-trained heads | Trained head pairs | Created by users in the Train view, stored in `runs/playground/` | 10.5 MB (under 3,000 examples) or 50.4 MB | n/a | **Unreviewed** until they pass the [checklist](templates/new-model-checklist.md) | per-run `summary.json` |

## Release file hashes (`models-v1`)

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `encoder-nemotron-3-embed-1b.tar.part-aa` | 1,992,294,400 | `6d0456e94fb4f47f9610ef87a0ce218fc2270d046071761f85dfc7d164ec9de7` |
| `encoder-nemotron-3-embed-1b.tar.part-ab` | 306,759,680 | `d5f2a2e62bff2a34d84e0761b151bdce55878b5d9ece8e9028ae5f97c41beb0c` |
| `heads-support-topic-1b.pt` | 10,498,663 | `22c0135bace33446a75049ea6be2fce7ae97e0906d980f0aef97d2fc1d51d83c` |
| `heads-typed-decisions-1b.pt` | 50,392,158 | `9eb391b60e5607bcfcac8709f5629c3fd05c12edd8287b4b505a6184d9160cba` |

`Claude Code/scripts/fetch-models.sh` verifies all of these before unpacking. The unpacked
encoder was compared byte for byte with NVIDIA's published snapshot and is identical. The
archive is reproducible: rebuilding it from the same snapshot gives the same checksums
(sorted entries, fixed timestamps and owners).

## Roles

This fork has no named owners yet. Assign people to these roles before production use.

| Role | Responsible for |
| --- | --- |
| Model owner | Each head's purpose, training data, approval and retirement |
| Model validator (independent of the owner) | Reviewing evaluation and limits against the checklist |
| Service operator | Running `clm-serve`, access, monitoring, incident response |
| Data steward | Rights to training data, retention and deletion in `runs/` |
