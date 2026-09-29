"""SGLang and Text Generation Inference, on each card's own checkpoint and on
Linnet's export of it.

SGLang runs in its own environment, as vLLM does, through its offline
`Engine`: token ids in, token ids out, continuous batching inside. TGI is a
server only; it runs as the launcher the pod's setup unpacks from TGI's image,
and takes text, so its prompts are the token ids decoded with the card's
tokenizer (TGI tokenizes them again, which can move a prompt's length by a
token or two). TGI has no way to ignore end-of-sequence tokens: a completion
may stop early, and throughput counts the tokens it actually produced.
"""

from __future__ import annotations

import json
import os
import socket
import statistics
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from bench.harness import Result
from bench.methods.serving import _describe, _result, prompts, serve_max_seq, settings

# ------------------------------------------------------------------ SGLang


def _sglang_engine(data: dict[str, Any], workload: dict[str, Any], length: int, rows: int) -> Any:
    import sglang  # type: ignore[import-not-found]

    return sglang.Engine(
        model_path=workload.get("model_path", data["weights"]["repo"]),
        dtype="bfloat16",
        context_length=length,
        max_running_requests=rows,
        mem_fraction_static=float(workload.get("sglang_memory", 0.85)),
        disable_radix_cache=True,
        log_level="error",
    )


def _sampling(new: int) -> dict[str, Any]:
    return {"max_new_tokens": new, "temperature": 0.0, "ignore_eos": True}


def sglang(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    """One request at a time: the first token alone, then the whole completion."""
    from bench.methods.llm import max_seq, prompt_ids  # llm imports this module

    new = int(workload["new_tokens"])
    start = time.perf_counter()
    engine = _sglang_engine(data, workload, max_seq(workload), 1)
    load_s = time.perf_counter() - start
    ids = prompt_ids(data, workload)

    def timed(count: int) -> float:
        begin = time.perf_counter()
        engine.generate(input_ids=[ids], sampling_params=_sampling(count))
        return (time.perf_counter() - begin) * 1e3

    try:
        for _ in range(int(workload["warmup"])):
            timed(new)
        ttft = statistics.median(timed(1) for _ in range(int(workload["iters"])))
        total = statistics.median(timed(new) for _ in range(int(workload["iters"])))
        output = engine.generate(input_ids=[ids], sampling_params=_sampling(1))[0]
    finally:
        engine.shutdown()
    metrics: dict[str, float | None] = {
        "ttft_ms": ttft,
        "decode_tok_s": (new - 1) / ((total - ttft) / 1e3),
        "load_s": load_s,
    }
    if output.get("output_ids"):
        metrics["first_token"] = float(output["output_ids"][0])
    return Result(
        "SGLang",
        "reference",
        metrics,
        notes="reserves a KV-cache pool up front (mem_fraction_static "
        f"{workload.get('sglang_memory', 0.85)}), so its memory is a setting, not a need",
    )


def serve_sglang(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    _, concurrency, _, _, new = settings(workload)
    start = time.perf_counter()
    engine = _sglang_engine(data, workload, serve_max_seq(workload), concurrency)
    load_s = time.perf_counter() - start
    inputs = prompts(data, workload)
    try:
        engine.generate(input_ids=inputs[: min(8, len(inputs))], sampling_params=_sampling(new))
        begin = time.perf_counter()
        outputs = engine.generate(input_ids=inputs, sampling_params=_sampling(new))
        seconds = time.perf_counter() - begin
    finally:
        engine.shutdown()
    tokens = sum(int(o["meta_info"]["completion_tokens"]) for o in outputs)
    return _result(
        "SGLang (offline, continuous batching)",
        "reference",
        tokens,
        seconds,
        None,
        load_s,
        _describe(workload) + "; KV-cache pool at mem_fraction_static "
        f"{workload.get('sglang_memory', 0.85)}",
    )


# --------------------------------------------------------------------- TGI


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@contextmanager
def _tgi(model: str, workload: dict[str, Any], total: int, rows: int) -> Iterator[str]:
    """TGI's launcher on `model`, stopped on the way out; yields its URL."""
    import requests as http

    launcher = os.environ.get("NEST_TGI_LAUNCHER")
    if not launcher:
        raise RuntimeError("TGI is not set up here (NEST_TGI_LAUNCHER)")
    port = _free_port()
    command = [launcher, "--model-id", model, "--port", str(port), "--dtype", "bfloat16"]
    command += ["--max-input-tokens", str(total - 1), "--max-total-tokens", str(total)]
    command += ["--max-batch-prefill-tokens", str(max(total, 8192))]
    command += ["--max-concurrent-requests", str(max(rows * 4, 128))]
    command += ["--cuda-memory-fraction", str(workload.get("tgi_memory", 0.85))]
    log = Path(workload["workdir"]) / f"tgi-{port}.log"
    with log.open("w") as sink:
        process = subprocess.Popen(command, stdout=sink, stderr=subprocess.STDOUT)  # noqa: S603
        url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + float(workload.get("timeout", 3600))
            while True:
                if process.poll() is not None:
                    tail = log.read_text(errors="replace").strip().splitlines()[-5:]
                    raise RuntimeError("TGI exited: " + " | ".join(tail))
                try:
                    if http.get(f"{url}/health", timeout=2).status_code == 200:
                        break
                except http.RequestException:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("TGI did not become healthy")
                time.sleep(2)
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()


def _texts(data: dict[str, Any], model: str, batches: list[list[int]]) -> list[str]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    return [tokenizer.decode(ids) for ids in batches]


def _generate(url: str, text: str, new: int) -> tuple[float, list[float]]:
    """One streamed request: its start and each token's arrival time."""
    import requests as http

    body = {"inputs": text, "parameters": {"max_new_tokens": new, "do_sample": False}}
    begin = time.perf_counter()
    arrivals: list[float] = []
    with http.post(f"{url}/generate_stream", json=body, stream=True, timeout=600) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith(b"data:"):
                event = json.loads(line[5:])
                if event.get("token") is not None:
                    arrivals.append(time.perf_counter())
    return begin, arrivals


def tgi(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    """One request at a time, streamed: the first token's time, then the
    rate of the rest."""
    from bench.methods.llm import max_seq, prompt_ids  # llm imports this module

    model = workload.get("model_path", data["weights"]["repo"])
    new = int(workload["new_tokens"])
    text = _texts(data, model, [prompt_ids(data, workload)])[0]
    start = time.perf_counter()
    with _tgi(model, workload, max_seq(workload), 1) as url:
        load_s = time.perf_counter() - start
        for _ in range(int(workload["warmup"])):
            _generate(url, text, new)
        firsts: list[float] = []
        rates: list[float] = []
        for _ in range(int(workload["iters"])):
            begin, arrivals = _generate(url, text, new)
            firsts.append((arrivals[0] - begin) * 1e3)
            if len(arrivals) > 1:
                rates.append((len(arrivals) - 1) / (arrivals[-1] - arrivals[0]))
    return Result(
        "Text Generation Inference",
        "reference",
        {
            "ttft_ms": statistics.median(firsts),
            "decode_tok_s": statistics.median(rates) if rates else None,
            "load_s": load_s,
        },
        notes="over HTTP, streamed; the prompt is the token ids decoded and tokenized "
        "again; reserves a KV-cache pool up front (cuda-memory-fraction "
        f"{workload.get('tgi_memory', 0.85)}), so its memory is a setting, not a need",
    )


def serve_tgi(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    from bench import triton

    model = workload.get("model_path", data["weights"]["repo"])
    _, concurrency, _, _, new = settings(workload)
    texts = _texts(data, model, prompts(data, workload))
    start = time.perf_counter()
    with _tgi(model, workload, serve_max_seq(workload), concurrency) as url:
        load_s = time.perf_counter() - start
        # First tokens are timed from the run's start, as for the other
        # serving rows: a request waiting for a free slot is part of it.
        began = [0.0]
        produced = [0]
        lock = threading.Lock()

        def call(index: int) -> float:
            _, arrivals = _generate(url, texts[index], new)
            with lock:
                produced[0] += len(arrivals)
            return arrivals[0] - began[0] if arrivals else 0.0

        began[0] = time.perf_counter()
        triton.closed_loop(call, min(8, len(texts)), min(8, concurrency))  # warm-up
        produced[0] = 0
        began[0] = time.perf_counter()
        seconds, firsts = triton.closed_loop(call, len(texts), concurrency)
    return _result(
        "Text Generation Inference (continuous batching)",
        "reference",
        produced[0],
        seconds,
        [f for f in firsts if f > 0],
        load_s,
        _describe(workload) + "; over HTTP, streamed; stops at end-of-sequence (TGI cannot "
        "ignore it), so throughput counts the tokens produced; KV-cache pool at "
        f"cuda-memory-fraction {workload.get('tgi_memory', 0.85)}",
    )


METHODS = {
    "sglang": sglang,
    "serve-sglang": serve_sglang,
    "tgi": tgi,
    "serve-tgi": serve_tgi,
}
