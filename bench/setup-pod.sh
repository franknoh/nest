#!/usr/bin/env bash
# Prepares a fresh GPU pod (RunPod's `runpod/pytorch:*-cu128*-ubuntu2404`
# image, or any Ubuntu 24.04 with a CUDA PyTorch) to run the Nest benchmarks:
# builds the Linnet compiler, installs Linnet and every reference stack next
# to the image's PyTorch, and gives vLLM an environment of its own, since it
# pins its own PyTorch. Run it from the registry checkout:
#
#     git clone https://github.com/franknoh/nest.git && cd nest
#     bash bench/setup-pod.sh
#     bash bench/run-all.sh
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends cmake ninja-build g++-13 gcc-13 git >/dev/null

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

# Linnet and the reference stacks in an environment of their own. Not beside
# the image's PyTorch: the stacks pull a newer PyTorch, and the image's
# torchaudio and NCCL, still visible through system site packages, then fail
# to load against it. JAX uses the same CUDA major as that PyTorch for the
# same reason.
python -m venv /workspace/venv
# shellcheck disable=SC1091
source /workspace/venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet torch torchvision
python -m pip install --quiet -e "/workspace/Linnet/python/linnet[torch,jax,onnx,nest]"
python -m pip install --quiet "jax[cuda13]" onnxruntime-gpu nvidia-ml-py \
    transformers accelerate diffusers sentence-transformers datasets soundfile \
    pillow safetensors huggingface_hub
python - <<'EOF'
import jax, torch, onnxruntime
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count(), "GPUs")
print("jax", jax.__version__, jax.default_backend(), len(jax.devices()), "devices")
print("onnxruntime", onnxruntime.__version__, onnxruntime.get_available_providers())
EOF
deactivate

# vLLM pins its own PyTorch, so it gets its own environment.
python -m venv /workspace/vllm
/workspace/vllm/bin/python -m pip install --quiet --upgrade pip
/workspace/vllm/bin/python -m pip install --quiet vllm nvidia-ml-py numpy
/workspace/vllm/bin/python -c "import vllm; print('vllm', vllm.__version__)"
echo "setup done"
