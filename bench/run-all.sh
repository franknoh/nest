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
NEST_VLLM_SITE=$(/workspace/vllm/bin/python -c "import site; print(site.getsitepackages()[0])")
export NEST_VLLM_SITE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p logs

# Samples were recorded with the last full run; this one measures.
run() {
    echo "=== $* ($(date -u +%H:%M:%S))"
    python -m bench.run "$@" --samples skip 2>&1 | tee -a "logs/$1.log" | grep -E "^\[" || true
}

# Encoders, vision, audio, segmentation, diffusion: minutes each.
for model in all-MiniLM-L6-v2 bert-base-uncased roberta-base modernbert-base \
    resnet-18 resnet-50 vit-base-patch16-224 dinov2-base siglip-base-patch16-224 \
    whisper-tiny whisper-large-v3 sam-vit-base sd-vae-ft-mse sdxl-base-unet; do
    run "$model"
done

# Decoders, smallest first.
for model in gpt2 qwen2.5-0.5b-instruct tinyllama-1.1b-chat smollm2-1.7b-instruct \
    phi-3-mini-4k-instruct qwen3-4b mistral-7b-instruct-v0.3; do
    run "$model"
done

# The two 8B decoders: every method, then the offloading row (one GPU capped
# so that half the model streams in from the host) merged beside them.
for model in llama-3.1-8b-instruct qwen3-8b; do
    run "$model"
    run "$model" --methods linnet-offload --offload-gib 8 --merge
done

# The largest last: gpt-oss now has KV-cache entries, so it gets every row.
run gpt-oss-20b

echo "all done ($(date -u +%H:%M:%S))"
