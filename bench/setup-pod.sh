#!/usr/bin/env bash
# Prepares a GPU pod to run the Nest benchmarks. The pod runs NVIDIA's Triton
# Inference Server image (`nvcr.io/nvidia/tritonserver:26.08-py3`, Ubuntu
# 24.04, CUDA 13), so the server and its ONNX Runtime and Python backends are
# already there; this adds the vLLM backend, builds the Linnet compiler, and
# installs Linnet and every reference stack in one environment and vLLM in a
# second, since it pins its own PyTorch. The host driver must support CUDA 13
# (580 or newer): on RunPod, create the pod with `--min-cuda-version 13.0`.
# Run it from the registry checkout:
#
#     git clone https://github.com/franknoh/nest.git && cd nest
#     bash bench/setup-pod.sh
#     bash bench/run-all.sh
set -euo pipefail

if ! nvidia-smi | grep -qE "CUDA Version: (1[3-9]|[2-9][0-9])"; then
    echo "the driver does not support CUDA 13; create the pod with --min-cuda-version 13.0" >&2
    exit 1
fi
if [ ! -x /opt/tritonserver/bin/tritonserver ]; then
    echo "no Triton Inference Server here; use the nvcr.io/nvidia/tritonserver image" >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends cmake ninja-build g++-13 gcc-13 git \
    python3-venv python3-dev >/dev/null

# The compiler, release build, from Linnet's main branch.
if [ ! -d /workspace/Linnet ]; then
    git clone --depth 1 https://github.com/franknoh/Linnet.git /workspace/Linnet
fi
(
    cd /workspace/Linnet
    CC=gcc-13 CXX=g++-13 cmake --preset release >/dev/null
    cmake --build --preset release
    build/release/linnet --version
)

# Linnet and the reference stacks in an environment of their own, built from
# the image's Python so Triton's Python backend can import from it too.
python3 -m venv /workspace/venv
# shellcheck disable=SC1091
source /workspace/venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet torch torchvision
python -m pip install --quiet -e "/workspace/Linnet/python/linnet[torch,jax,onnx,nest]"
python -m pip install --quiet "jax[cuda13]" onnxruntime-gpu onnxscript nvidia-ml-py \
    transformers accelerate diffusers sentence-transformers datasets soundfile \
    pillow safetensors huggingface_hub sentencepiece protobuf keras keras-hub \
    "tritonclient[http]" requests
# onnxruntime-gpu is built for CUDA 12; its libraries sit beside PyTorch's
# CUDA 13 ones under different names, and the worker preloads them.
python -m pip install --quiet nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 \
    "nvidia-cudnn-cu12>=9,<10" nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cuda-nvrtc-cu12
python - <<'EOF'
import jax, torch, onnxruntime
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count(), "GPUs")
print("jax", jax.__version__, jax.default_backend(), len(jax.devices()), "devices")
print("onnxruntime", onnxruntime.__version__, onnxruntime.get_available_providers())
import keras_hub
print("keras_hub", keras_hub.__version__, "jax still on", jax.default_backend())
EOF
deactivate

# vLLM pins its own PyTorch, so it gets its own environment -- also the one
# Triton's vLLM backend imports from (`NEST_VLLM_SITE`).
python3 -m venv /workspace/vllm
/workspace/vllm/bin/python -m pip install --quiet --upgrade pip
/workspace/vllm/bin/python -m pip install --quiet vllm nvidia-ml-py numpy
/workspace/vllm/bin/python -c "import vllm; print('vllm', vllm.__version__)"

# Triton's vLLM backend is Python; its source goes where the server looks
# for backends.
if [ ! -f /opt/tritonserver/backends/vllm/model.py ]; then
    rm -rf /tmp/vllm_backend
    git clone --depth 1 https://github.com/triton-inference-server/vllm_backend.git /tmp/vllm_backend
    mkdir -p /opt/tritonserver/backends/vllm
    cp -r /tmp/vllm_backend/src/* /opt/tritonserver/backends/vllm/
fi
ls /opt/tritonserver/backends
echo "setup done"
