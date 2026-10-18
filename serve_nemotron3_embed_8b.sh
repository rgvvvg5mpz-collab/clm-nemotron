#!/bin/bash
# Serve NVIDIA Nemotron-3-Embed-8B (Ministral-3-8B base, OpenMDW-1.1) as the encoder behind
# `clm-serve --encoder nemotron-3-embed-8b` (the default encoder).
# The model does its own bidirectional mean pooling. The "query: " / "passage: " prompts are
# added by clm-serve (clm/encoders.py), not here, because the OpenAI /v1/embeddings route
# does not add them. NVIDIA validated vLLM 0.21-0.25 and nvcr.io/nvidia/vllm:26.06-py3.
#
# Usage: GPU=0 PORT=8090 UTIL=0.35 MAXLEN=8192 ./serve_nemotron3_embed_8b.sh
#        FP8=1 on Hopper / Ada adds --quantization fp8_per_tensor
set -u
GPU="${GPU:-0}"
PORT="${PORT:-8090}"
UTIL="${UTIL:-0.35}"
MAXLEN="${MAXLEN:-8192}"     # model supports 32768; clm-serve --max-tokens must not exceed this
LOGDIR="${LOGDIR:-$(cd "$(dirname "$0")/.." && pwd)/logs}"
EXTRA=()
if [ "${FP8:-0}" = "1" ]; then EXTRA+=(--quantization fp8_per_tensor); fi
mkdir -p "$LOGDIR"
echo "serving Nemotron-3-Embed-8B on GPU $GPU port $PORT (util $UTIL, max-model-len $MAXLEN)"
CUDA_VISIBLE_DEVICES=$GPU exec vllm serve nvidia/Nemotron-3-Embed-8B-BF16 \
    --served-model-name nemotron-3-embed-8b \
    --runner pooling \
    --enable-prefix-caching \
    --max-model-len "$MAXLEN" \
    --gpu-memory-utilization "$UTIL" \
    --max-num-seqs 32 \
    --port "$PORT" \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    >> "$LOGDIR/vllm_nemotron3_embed_8b.log" 2>&1
