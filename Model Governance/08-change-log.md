# 08 · Change log

| Date | Version | Change | Models affected |
| --- | --- | --- | --- |
| 2026-10-18 | app 0.2.0, `models-v1` | Head weights seeded (reproducible training); smaller default head under 3,000 examples, after measuring run-to-run variation; reproducible release archive. Playground Train view with tooltips; `clm-train`; Mac encoder server; model release with verified download; Claude Code runbook; Model Governance created; safe checkpoint loading (`weights_only=True`); services bind to 127.0.0.1 | New: `support-topic-1b`, `typed-decisions-1b` (demo). Encoder: `nemotron-3-embed-1b` shipped in the release. |
| 2026-10-18 | app 0.2.0-dev | `clm-serve` can run with no head (`clm-raw`); a typed-decisions training bug fixed | none |
| 2026-10-18 | fork created | Encoder replaced with NVIDIA Nemotron-3-Embed (8B default, 1B added); upstream encoder, heads and datasets removed; encoder compatibility enforced | All upstream artifacts retired |
| 2026-10-05 | upstream `d5f9ef0` | Starting point: `Contrastive-LM/CLM` | n/a |
