"""Linnet models in other engines, through Linnet's own exports.

`linnet.hf.export` writes a Transformers checkpoint directory for a model
whose structure one of the serving engines implements (the Llama family,
GPT-2); vLLM then serves it like any checkpoint. `linnet.gguf.export` goes one
step further to llama.cpp's GGUF, which `llama-bench` times. These rows show
that the exported model runs in those engines and at what speed; the
engines' own kernels do the work, so the numbers belong to the engines.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from bench.harness import Result, python_for, run_isolated


def _export_hf(data: dict[str, Any], workload: dict[str, Any]) -> Path:
    from linnet import hf

    out = Path(workload["workdir"]) / "linnet-hf"
    if not (out / "config.json").exists():
        hf.export(data["directory"], out)
    return out


def linnet_engine(method: str, label: str) -> Any:
    """The card exported by `linnet.hf.export`, run by another row (`vllm`,
    `serve-sglang`, ...) in that engine's own environment."""

    def run(data: dict[str, Any], workload: dict[str, Any]) -> Result:
        start = time.perf_counter()
        directory = _export_hf(data, workload)
        export_s = time.perf_counter() - start
        child = {**workload, "model_path": str(directory)}
        result = run_isolated(
            python_for(method),
            data["model"]["name"],
            method,
            child,
            float(workload.get("timeout", 3600)),
        )
        result.kind = "linnet"
        result.method = f"Linnet -> {label}"
        result.notes = (
            f"linnet.hf.export ({export_s:.0f} s), then {label} on the exported checkpoint; "
            + result.notes
        )
        return result

    return run


def linnet_llamacpp(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    """The card exported by `linnet.gguf.export` (f16), timed by llama.cpp's
    `llama-bench` with every layer on the GPU: prompt processing at the
    workload's prompt length, generation at its new tokens."""
    from linnet import gguf

    converter = os.environ.get("NEST_LLAMA_CONVERTER")
    bench = os.environ.get("NEST_LLAMA_BENCH")
    if not converter or not bench:
        raise RuntimeError("llama.cpp is not set up here (NEST_LLAMA_CONVERTER, NEST_LLAMA_BENCH)")
    start = time.perf_counter()
    exported = gguf.export(
        data["directory"], Path(workload["workdir"]) / "linnet-gguf", converter=converter
    )
    export_s = time.perf_counter() - start
    prompt, new = int(workload["prompt_tokens"]), int(workload["new_tokens"])
    command = [bench, "-m", str(exported.gguf), "-p", str(prompt), "-n", str(new)]
    command += ["-ngl", "99", "-r", str(max(3, int(workload["iters"]) // 2)), "-o", "json"]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout).strip().splitlines()[-3:]
        raise RuntimeError("llama-bench failed: " + " | ".join(tail))
    rows = json.loads(completed.stdout)
    prompt_rate = next(
        r["avg_ts"] for r in rows if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0
    )
    decode_rate = next(
        r["avg_ts"] for r in rows if r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0
    )
    return Result(
        "Linnet -> llama.cpp (GGUF, f16)",
        "linnet",
        {
            "ttft_ms": prompt / prompt_rate * 1e3,
            "decode_tok_s": decode_rate,
            "load_s": export_s,
        },
        notes="linnet.gguf.export, then llama-bench with every layer on the GPU; the first "
        "token is the prompt at llama-bench's prompt rate",
    )


METHODS = {
    "linnet-vllm": linnet_engine("vllm", "vLLM"),
    "serve-linnet-vllm": linnet_engine("serve-vllm", "vLLM (offline, continuous batching)"),
    "linnet-llamacpp": linnet_llamacpp,
    "linnet-sglang": linnet_engine("sglang", "SGLang"),
    "serve-linnet-sglang": linnet_engine("serve-sglang", "SGLang (offline, continuous batching)"),
    "linnet-tgi": linnet_engine("tgi", "Text Generation Inference"),
    "serve-linnet-tgi": linnet_engine(
        "serve-tgi", "Text Generation Inference (continuous batching)"
    ),
}

# The families `linnet.hf.export` recognizes, by card family.
EXPORTABLE = {"llama", "gpt2"}
