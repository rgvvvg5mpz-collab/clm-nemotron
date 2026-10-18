# Model card · Encoder `nemotron-3-embed-8b`

| | |
| --- | --- |
| Source | `nvidia/Nemotron-3-Embed-8B-BF16`, released 2026-07-16; downloaded from Hugging Face when selected (not in the `models-v1` release) |
| Maker | NVIDIA, built on Mistral AI's `Ministral-3-8B-Instruct-2512` |
| Licence | OpenMDW-1.1 |
| Architecture | Transformer, bidirectional attention, mean pooling; 4096-dimension output; up to 32,768 tokens |
| Role here | Optional encoder for NVIDIA GPU servers (vLLM, `serve-encoder-gpu.sh`) |
| Status | **No trained heads exist for it.** Heads trained on the 1B are refused with it by design. |

NVIDIA reports RTEB 78.46 and MMTEB retrieval 75.45 NDCG@10 (against 72.38 and 71.04 for the
1B). It has not been evaluated for classification in this application.

Training data per NVIDIA's card: public retrieval datasets plus synthetic pairs generated
with Alibaba Qwen models (Qwen3-Next-80B-A3B-Instruct, Qwen3-235B-A22B, Qwen3.5-397B-A17B,
Qwen3.6-27B, Qwen3.6-35B-A3B), Google gemma-4-31B-it, OpenAI gpt-oss-120b and gpt-oss-20b,
and NVIDIA Nemotron-3 Super and Ultra.

Before using it: train and evaluate heads on it, and add a card for each head.
