# Model card · Encoder `nemotron-3-embed-1b`

| | |
| --- | --- |
| Source | `nvidia/Nemotron-3-Embed-1B-BF16`, Hugging Face revision `c0c9fea93ea424587517f2c59e20db9f1d6bf615`, released 2026-07-16 |
| Maker | NVIDIA. Derived from Mistral AI's `Ministral-3-3B-Instruct-2512` (Apache-2.0) by structured pruning and distillation from Nemotron-3-Embed-8B |
| Licence | OpenMDW-1.1 (redistribution allowed with the licence and notices; both ship inside the release archive) |
| Role here | Frozen feature extractor. It is never trained or modified in this application. |
| Architecture | Transformer, bidirectional attention, mean pooling; 1.14B parameters; 2048-dimension output; up to 32,768 tokens (this app caps texts at 8,192) |
| Prompts | `query: ` before states, `passage: ` before label descriptions (NVIDIA's retrieval convention) |
| Runs on | Apple MPS, NVIDIA CUDA or CPU, via `tools/local_embed_server.py` (sentence-transformers, bf16) |

## Intended use (by its maker)

Multilingual retrieval and semantic similarity. NVIDIA reports state-of-the-art results
among models of its size on retrieval benchmarks (RTEB 72.38, MMTEB retrieval 71.04
NDCG@10). Classification through CLM heads is a use this application adds; NVIDIA did not
evaluate it.

## Training data (from NVIDIA's model card)

8.5M+ examples from 161 dataset files: public, commercially permissible retrieval datasets
(MIRACL, MLDR and others) and synthetic query–passage pairs. The synthetic pairs were
generated with Qwen3-Next-80B-A3B-Instruct, Qwen3-235B-A22B, Qwen3.5-397B-A17B,
Qwen3.6-27B and Qwen3.6-35B-A3B (Alibaba). NVIDIA states that data from user interactions
with its models was used, and that personal data in it is "None Known". See
[03-data-lineage.md](../03-data-lineage.md) and the [sourcing policy](../06-model-sourcing-policy.md).

## Known limits (from NVIDIA, and observed here)

- NVIDIA: "does not always guarantee to retrieve the correct passage(s)". No bias
  measurement or mitigation was done (its Bias table answers "None" to all three questions).
- Observed: with no trained head, it ranked "Because the Earth is round." above "The Moon's
  gravitational pull." for "What causes tides on Earth?" (cosine 0.365 against 0.319). Word
  overlap pulls similarity, which is why classification needs trained heads.
- Observed: alone (`clm-raw`), it scored **below** the majority-label baseline on the
  15-question typed-decisions test (37.8% against 45.0%).
