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
#     bash bench/setup-pod.sh              # add --engines for SGLang and TGI
#     bash bench/run-all.sh
set -euo pipefail

# Read the whole output first: under pipefail, `grep -q` quitting early kills
# nvidia-smi with SIGPIPE and fails the check on a driver that passes it.
driver=$(nvidia-smi)
if ! grep -qE "CUDA Version: (1[3-9]|[2-9][0-9])" <<<"$driver"; then
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
# Its TensorRT provider links TensorRT 10; the image's own is 11, under
# another soname. `run-all.sh` puts these on the library path.
python -m pip install --quiet "tensorrt-cu13-libs>=10,<11"
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
# The backend's main branch trails vLLM: `build_1_2_5_buckets` moved from
# `vllm.v1.metrics.loggers` to `vllm.v1.metrics.buckets`.
/workspace/vllm/bin/python - <<'EOF'
import importlib.util
from pathlib import Path

if importlib.util.find_spec("vllm.v1.metrics.buckets") is not None:
    path = Path("/opt/tritonserver/backends/vllm/utils/metrics.py")
    text = path.read_text()
    old = "from vllm.v1.metrics.loggers import StatLoggerBase, build_1_2_5_buckets"
    new = "from vllm.v1.metrics.buckets import build_1_2_5_buckets\nfrom vllm.v1.metrics.loggers import StatLoggerBase"
    path.write_text(text.replace(old, new))
EOF
ls /opt/tritonserver/backends

# llama.cpp with CUDA, for the GGUF rows: its converter needs `gguf` in our
# environment, its build CUDA's compiler. Best effort -- without it, those
# rows fail and the rest run.
(
    set +e
    if [ ! -x /workspace/llama.cpp/build/bin/llama-bench ]; then
        if ! command -v nvcc >/dev/null && [ ! -x /usr/local/cuda/bin/nvcc ]; then
            version=$(nvidia-smi | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+")
            major=${version%%.*}
            wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/keyring.deb \
                && dpkg -i /tmp/keyring.deb >/dev/null && apt-get update -qq
            apt-get install -y -qq --no-install-recommends "cuda-nvcc-${major}-0" \
                "cuda-cudart-dev-${major}-0" "libcublas-dev-${major}-0" >/dev/null
        fi
        export PATH=/usr/local/cuda/bin:$PATH
        rm -rf /workspace/llama.cpp
        git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /workspace/llama.cpp
        cmake -S /workspace/llama.cpp -B /workspace/llama.cpp/build -G Ninja \
            -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DLLAMA_CURL=OFF >/dev/null \
            && cmake --build /workspace/llama.cpp/build --target llama-bench >/dev/null
    fi
    /workspace/venv/bin/python -m pip install --quiet gguf
    ls -la /workspace/llama.cpp/build/bin/llama-bench || echo "llama.cpp build failed; its rows will fail"
)

# SGLang and Text Generation Inference, for the engine rows, only with
# `--engines`. SGLang pins its own PyTorch, so it gets an environment of its
# own, as vLLM does. TGI ships only as an image; its layers are unpacked
# into /workspace/tgi-root (with resumable downloads: the registry drops
# long transfers) and linked at the paths it was built for.
if [ "${1:-}" = "--engines" ]; then
    python3 -m venv /workspace/sglang
    /workspace/sglang/bin/python -m pip install --quiet --upgrade pip
    /workspace/sglang/bin/python -m pip install --quiet "sglang[all]" nvidia-ml-py numpy
    /workspace/sglang/bin/python -c "import sglang; print('sglang', sglang.__version__)"

    TGI_REPO=huggingface/text-generation-inference
    TGI_TAG=3.3.7
    cd /workspace
    curl -sL https://github.com/google/go-containerregistry/releases/latest/download/go-containerregistry_Linux_x86_64.tar.gz \
        | tar -xz crane
    GODEBUG=http2client=0 ./crane manifest --platform linux/amd64 "ghcr.io/$TGI_REPO:$TGI_TAG" \
        | python3 -c "import json, sys; [print(l['digest']) for l in json.load(sys.stdin)['layers']]" \
        > tgi-layers.txt
    mkdir -p tgi-layers tgi-root
    n=0
    while read -r digest; do
        n=$((n + 1))
        for attempt in $(seq 1 30); do
            token=$(curl -s "https://ghcr.io/token?scope=repository:$TGI_REPO:pull" \
                | python3 -c "import json, sys; print(json.load(sys.stdin)['token'])")
            curl -sL -C - -H "Authorization: Bearer $token" -o "tgi-layers/$n.tar.gz" \
                "https://ghcr.io/v2/$TGI_REPO/blobs/$digest" || true
            if gzip -t "tgi-layers/$n.tar.gz" 2>/dev/null; then break; fi
            echo "TGI layer $n: resuming ($attempt)"
            sleep 3
        done
        tar -xzf "tgi-layers/$n.tar.gz" -C tgi-root --exclude=".wh.*" 2>/dev/null || true
    done < tgi-layers.txt
    mkdir -p /usr/src /root/.local/share
    [ -e /usr/src/.venv ] || ln -s /workspace/tgi-root/usr/src/.venv /usr/src/.venv
    [ -e /usr/src/server ] || ln -s /workspace/tgi-root/usr/src/server /usr/src/server
    [ -e /root/.local/share/uv ] || ln -s /workspace/tgi-root/root/.local/share/uv /root/.local/share/uv
    [ -e /kernels ] || ln -s /workspace/tgi-root/kernels /kernels
    # The image's environment, which the launcher and its server expect.
    cat > /workspace/tgi-launcher.sh <<'LAUNCHER'
#!/bin/bash
unset PYTHONPATH
export PATH=/usr/src/.venv/bin:/workspace/tgi-root/usr/local/bin:$PATH
export LD_LIBRARY_PATH=/root/.local/share/uv/python/cpython-3.11.11-linux-x86_64-gnu/lib:${LD_LIBRARY_PATH:-}
export VIRTUAL_ENV=/usr/src/.venv HF_KERNELS_CACHE=/kernels EXLLAMA_NO_FLASH_ATTN=1
exec /workspace/tgi-root/usr/local/bin/text-generation-launcher "$@"
LAUNCHER
    chmod +x /workspace/tgi-launcher.sh
    /workspace/tgi-launcher.sh --version
fi
echo "setup done"
