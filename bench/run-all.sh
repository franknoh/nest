#!/usr/bin/env bash
# Measures every model in the registry and records what each produces:
# `models/<name>/bench.json`, `samples.json`, and `samples/`. Small models go
# first so a broken setup shows in minutes, not hours. One model failing is
# one model's rows failing; the run carries on.
#
#     bash bench/setup-pod.sh && bash bench/run-all.sh
set -uo pipefail

# shellcheck disable=SC1091
source /workspace/venv/bin/activate
export LINNET_BIN=/workspace/Linnet/build/release/linnet
export NEST_VLLM_PYTHON=/workspace/vllm/bin/python
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p logs

run() {
    echo "=== $* ($(date -u +%H:%M:%S))"
    python -m bench.run "$@" 2>&1 | tee -a "logs/$1.log" | grep -E "^\[" || true
}

# Encoders, vision, audio, segmentation, diffusion: minutes each.
for model in all-MiniLM-L6-v2 bert-base-uncased roberta-base modernbert-base \
    resnet-18 resnet-50 vit-base-patch16-224 dinov2-base siglip-base-patch16-224 \
    whisper-tiny whisper-large-v3 sam-vit-base sd-vae-ft-mse sdxl-base-unet; do
    run "$model"
done

# Decoders, smallest first.
for model in gpt2 qwen2.5-0.5b-instruct tinyllama-1.1b-chat smollm2-1.7b-instruct \
    phi-3-mini-4k-instruct qwen3-4b mistral-7b-instruct-v0.3 gpt-oss-20b; do
    run "$model"
done

# Two 8B decoders also get the placement rows: every GPU, and one GPU capped
# so that half the model streams in from the host.
PLACED=transformers-eager,transformers-compile,vllm,linnet-torch,linnet-cudagraphs,linnet-jax,linnet-gpus,linnet-offload
for model in llama-3.1-8b-instruct qwen3-8b; do
    CUDA_VISIBLE_DEVICES=0,1 run "$model" --methods "$PLACED" --offload-gib 8
done

echo "all done ($(date -u +%H:%M:%S))"
