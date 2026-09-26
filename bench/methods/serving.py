"""Serving: many requests at once, the way a server sees them.

A batch-1 decode speed says how fast one conversation goes; a server's cost
is how many tokens it produces per second while many are in flight. Every
method here gets the same requests, all waiting from the start: prompts of
random length (`serve_prompt_min` to `serve_prompt_max` tokens) that each
want `serve_new` tokens, greedily and without stopping early, with at most
`serve_concurrency` in flight. What is recorded:

- `serve_tok_s`: generated tokens per second over the whole run;
- `serve_ttft_ms`: the median time from the start of the run to a request's
  first token -- mostly time spent waiting for a free row, which is what a
  user of a loaded server waits;
- `serve_s`: the run's wall time.

The engines differ in how they batch. vLLM and Linnet's `linnet.serve`
batch continuously (a request joins as soon as a row frees); transformers'
`generate_batch` does too, with paged attention; KerasHub has only static
batches, which wait for their longest member. Triton Inference Server's vLLM
backend is vLLM behind an HTTP server, streamed, so its first-token times
are the ones a client sees.
"""

from __future__ import annotations

import json
import os
import random
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from bench.harness import Result

SAFE_TOKENS = (100, 20000)


def settings(workload: dict[str, Any]) -> tuple[int, int, int, int, int]:
    """`(requests, concurrency, shortest prompt, longest prompt, new tokens)`."""
    concurrency = int(workload.get("serve_concurrency", 64))
    return (
        int(workload.get("serve_requests", 4 * concurrency)),
        concurrency,
        int(workload.get("serve_prompt_min", 128)),
        int(workload.get("serve_prompt_max", 512)),
        int(workload.get("serve_new", 128)),
    )


def serve_max_seq(workload: dict[str, Any]) -> int:
    """The cache length the fixed-slot engines are compiled for: the longest
    prompt and its completion, rounded up."""
    _, _, _, longest, new = settings(workload)
    return ((longest + new + 127) // 128) * 128


def prompts(data: dict[str, Any], workload: dict[str, Any]) -> list[list[int]]:
    requests, _, shortest, longest, _ = settings(workload)
    vocab = int(data["generics"].get("Vocab", SAFE_TOKENS[1]))
    rng = random.Random(f"serve:{workload.get('seed', 0)}")  # noqa: S311 - a reproducible load
    high = min(vocab, SAFE_TOKENS[1])
    return [
        [rng.randrange(SAFE_TOKENS[0], high) for _ in range(rng.randint(shortest, longest))]
        for _ in range(requests)
    ]


def _result(
    name: str,
    kind: str,
    tokens: int,
    seconds: float,
    ttfts: list[float] | None,
    load_s: float,
    notes: str,
) -> Result:
    metrics: dict[str, float | None] = {
        "serve_tok_s": tokens / seconds,
        "serve_s": seconds,
        "load_s": load_s,
    }
    if ttfts:
        metrics["serve_ttft_ms"] = statistics.median(ttfts) * 1e3
    return Result(name, kind, metrics, notes=notes)


def _describe(workload: dict[str, Any]) -> str:
    requests, concurrency, shortest, longest, new = settings(workload)
    return (
        f"{requests} requests of {shortest}-{longest} prompt tokens and {new} new tokens, "
        f"at most {concurrency} in flight"
    )


# ------------------------------------------------------------------ vLLM


def vllm(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    from vllm import LLM, SamplingParams  # type: ignore[import-not-found]

    _, concurrency, _, _, new = settings(workload)
    start = time.perf_counter()
    llm = LLM(
        model=data["weights"]["repo"],
        dtype="bfloat16",
        max_model_len=serve_max_seq(workload),
        max_num_seqs=concurrency,
        gpu_memory_utilization=float(workload.get("vllm_memory", 0.85)),
        enable_prefix_caching=False,
    )
    load_s = time.perf_counter() - start
    params = SamplingParams(max_tokens=new, temperature=0.0, ignore_eos=True)
    inputs = [{"prompt_token_ids": ids} for ids in prompts(data, workload)]
    llm.generate(inputs[: min(8, len(inputs))], params, use_tqdm=False)  # warm-up
    begin = time.perf_counter()
    outputs = llm.generate(inputs, params, use_tqdm=False)
    seconds = time.perf_counter() - begin
    tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    ttfts: list[float] = []
    for output in outputs:
        # V1 reports per-request timestamps where it can; offline it may not.
        metrics = getattr(output, "metrics", None)
        first = getattr(metrics, "first_token_time", None) if metrics is not None else None
        arrival = getattr(metrics, "arrival_time", None) if metrics is not None else None
        if first and arrival:
            ttfts.append(first - arrival)
    return _result(
        "vLLM (offline, continuous batching)",
        "reference",
        tokens,
        seconds,
        ttfts,
        load_s,
        _describe(workload) + "; KV-cache pool at gpu_memory_utilization "
        f"{workload.get('vllm_memory', 0.85)}",
    )


# ---------------------------------------------------------- transformers


def transformers(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    """`generate_batch`, transformers' own continuous batching over a paged
    cache, where this version has it; otherwise static batches."""
    import torch
    from transformers import AutoModelForCausalLM, GenerationConfig

    _, concurrency, _, _, new = settings(workload)
    device = workload.get("device", "cuda")
    start = time.perf_counter()
    # Paged attention over SDPA where the architecture has SDPA, else over
    # its eager attention.
    for attention in ("paged|sdpa", "paged|eager"):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                data["weights"]["repo"], dtype=torch.bfloat16, attn_implementation=attention
            ).to(device)
            break
        except ValueError:
            continue
    else:
        raise RuntimeError("this transformers has no paged attention for the architecture")
    model.eval()
    load_s = time.perf_counter() - start
    requests = prompts(data, workload)
    config = GenerationConfig(
        max_new_tokens=new,
        min_new_tokens=new,
        do_sample=False,
        eos_token_id=None,
        pad_token_id=0,
        max_batch_tokens=concurrency * 256,
        num_blocks=max(64, concurrency * serve_max_seq(workload) // 32 + 64),
        block_size=32,
        scheduler="fifo",
    )
    with torch.no_grad():
        model.generate_batch(inputs=requests[:4], generation_config=config, progress_bar=False)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        outputs = model.generate_batch(
            inputs=requests, generation_config=config, progress_bar=False
        )
        torch.cuda.synchronize()
    seconds = time.perf_counter() - begin
    tokens = sum(len(o.generated_tokens) for o in outputs.values())
    if tokens == 0:
        raise RuntimeError("generate_batch returned no tokens for any request")
    return _result(
        "transformers (generate_batch, continuous batching)",
        "reference",
        tokens,
        seconds,
        None,
        load_s,
        _describe(workload) + f"; {attention} attention; reserves a paged KV-cache pool up "
        "front, so its memory is a setting, not a need; no per-request timestamps",
    )


# --------------------------------------------------------------- KerasHub


def keras_hub(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    """KerasHub on JAX: static batches of `concurrency` requests, padded to
    the longest prompt any request may have plus the new tokens, so one
    compiled program serves every batch; each runs until all its rows are
    done."""
    os.environ.setdefault("KERAS_BACKEND", "jax")
    import jax

    # Before KerasHub: it imports TensorFlow, whose CUDA 12 libraries, loaded
    # first, leave JAX's CUDA 13 plugin unable to start -- it then runs on
    # the CPU without saying so.
    jax.devices()
    import keras_hub  # type: ignore[import-not-found]
    import numpy as np

    _, concurrency, _, longest, new = settings(workload)
    start = time.perf_counter()
    lm = keras_hub.models.CausalLM.from_preset(f"hf://{data['weights']['repo']}", dtype="bfloat16")
    lm.preprocessor = None
    lm.compile(sampler="greedy")
    load_s = time.perf_counter() - start
    requests = prompts(data, workload)

    width = longest + new

    def batch(group: list[list[int]]) -> None:
        tokens = np.zeros((concurrency, width), dtype="int32")
        mask = np.zeros((concurrency, width), dtype=bool)
        for row, ids in enumerate(group):
            tokens[row, : len(ids)] = ids
            mask[row, : len(ids)] = True
        lm.generate({"token_ids": tokens, "padding_mask": mask}, stop_token_ids=None)

    groups = [requests[i : i + concurrency] for i in range(0, len(requests), concurrency)]
    batch(groups[0])  # compiles the one program
    begin = time.perf_counter()
    for group in groups:
        batch(group)
    seconds = time.perf_counter() - begin
    return _result(
        "KerasHub (JAX, static batches)",
        "reference",
        new * len(requests),
        seconds,
        None,
        load_s,
        _describe(workload) + "; every batch runs to the longest possible prompt plus the new "
        "tokens",
    )


# ------------------------------------------------------------------ Linnet


def _linnet(backend: str) -> Any:
    def run(data: dict[str, Any], workload: dict[str, Any]) -> Result:
        from linnet import nest
        from linnet.serve import Engine, Request

        _, concurrency, _, _, new = settings(workload)
        generics = {"Batch": concurrency, "MaxSeq": serve_max_seq(workload), "T": "bf16"}
        start = time.perf_counter()
        if backend == "torch":
            model = nest.load(
                data["directory"],
                backend="torch",
                device=workload.get("device", "cuda"),
                numerics="fast",
                compile=True,
                generics=generics,
                cast_dtype=True,
            )
        else:
            model = nest.load(
                data["directory"], backend="jax_model", generics=generics, cast_dtype=True
            )
        engine = Engine(model)
        load_s = time.perf_counter() - start
        requests = prompts(data, workload)
        engine.warmup(len(ids) for ids in requests)
        done, stats = engine.run([Request(prompt=ids, max_new_tokens=new) for ids in requests])
        label = "CUDA graphs" if backend == "torch" else "XLA"
        return _result(
            f"Linnet {backend} (linnet.serve, {label})",
            "linnet",
            stats.generated_tokens,
            stats.seconds,
            [c.ttft for c in done],
            load_s,
            _describe(workload) + f"; {concurrency} fixed cache rows of {serve_max_seq(workload)} "
            "positions",
        )

    return run


# ------------------------------------------------------ Triton + vLLM


VLLM_BACKEND_CONFIG = """backend: "vllm"
instance_group [{ count: 1, kind: KIND_MODEL }]
model_transaction_policy { decoupled: True }
"""


def triton_vllm(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    """vLLM as Triton Inference Server's vLLM backend, fed over HTTP with
    `concurrency` streams open. Prompts go as text (the backend takes text),
    decoded from the same token ids, so their lengths match only roughly."""
    import requests as http
    from transformers import AutoTokenizer

    from bench import triton

    _, concurrency, _, _, new = settings(workload)
    repo = data["weights"]["repo"]
    root = Path(tempfile.mkdtemp(prefix="nest-triton-"))
    model_dir = root / "llm" / "1"
    model_dir.mkdir(parents=True)
    (root / "llm" / "config.pbtxt").write_text(VLLM_BACKEND_CONFIG, encoding="utf-8")
    (model_dir / "model.json").write_text(
        json.dumps(
            {
                "model": repo,
                "dtype": "bfloat16",
                "max_model_len": serve_max_seq(workload) + 64,
                "max_num_seqs": concurrency,
                "gpu_memory_utilization": float(workload.get("vllm_memory", 0.85)),
                "enable_prefix_caching": False,
            }
        ),
        encoding="utf-8",
    )
    tokenizer = AutoTokenizer.from_pretrained(repo)
    texts = [tokenizer.decode(ids) for ids in prompts(data, workload)]
    start = time.perf_counter()
    with triton.server(root, python_path=os.environ.get("NEST_VLLM_SITE")) as url:
        load_s = time.perf_counter() - start
        endpoint = f"{url}/v2/models/llm/generate_stream"

        began = [0.0]  # when the measured run starts

        def call(index: int) -> tuple[float, int]:
            body = {
                "text_input": texts[index],
                "stream": True,
                "exclude_input_in_output": True,
                "sampling_parameters": json.dumps(
                    {"max_tokens": new, "temperature": 0.0, "ignore_eos": True}
                ),
            }
            first = 0.0
            chunks = 0
            with http.post(endpoint, json=body, stream=True, timeout=600) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith(b"data:"):
                        continue
                    chunks += 1
                    if not first:
                        first = time.perf_counter() - began[0]
            return first, chunks

        began[0] = time.perf_counter()
        triton.closed_loop(call, min(8, len(texts)), min(8, concurrency))  # warm-up
        began[0] = time.perf_counter()
        seconds, results = triton.closed_loop(call, len(texts), concurrency)
    return _result(
        "Triton Inference Server (vLLM backend)",
        "reference",
        new * len(texts),
        seconds,
        [first for first, _ in results],
        load_s,
        _describe(workload) + "; over HTTP, streamed, first tokens timed at the client",
    )


METHODS = {
    "serve-vllm": vllm,
    "serve-transformers": transformers,
    "serve-keras-hub": keras_hub,
    "serve-triton-vllm": triton_vllm,
    "serve-linnet-torch": _linnet("torch"),
    "serve-linnet-jax": _linnet("jax"),
}
