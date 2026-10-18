"""Task accuracy and long-run stability through the serving stacks.

`gsm8k-*`: the first 250 problems of GSM8K's test split, each a chat turn
answered greedily, served the way the stack serves (continuous batching),
scored on the number after `####` (or the last number). The same prompts,
tokenizer and scoring for every stack, so the rows differ only in how each
stack computes: what a reader of throughput numbers needs to know is
whether the fast path still answers the same.

`soak-*` (opt-in, long): a serving loop held at full load for
`soak_minutes`, a new request taking each finished one's place. Reports
throughput in the first and last minute, how memory grew, and failures:
what a single timed run does not show.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

from bench.harness import Result, memory_share
from bench.methods.serving import _layout, prompts, serve_max_seq, settings

PROBLEMS = 250
ANSWER_TOKENS = 512
INSTRUCTION = "Solve the problem. End with the final answer as a number after '####'."


def _problems(data: dict[str, Any]) -> tuple[list[list[int]], list[str], frozenset[int]]:
    """The chat prompts as token ids, the gold answers, and the stop tokens."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(data["weights"]["repo"])
    split = load_dataset("openai/gsm8k", "main", split="test").select(range(PROBLEMS))
    ids: list[list[int]] = []
    for question in split["question"]:
        messages = [{"role": "user", "content": f"{question}\n{INSTRUCTION}"}]
        # Thinking models answer within the token budget only without it.
        templated = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False
        )
        if hasattr(templated, "keys"):
            templated = templated["input_ids"]
        ids.append([int(t) for t in templated])
    gold = [answer.split("####")[-1].strip().replace(",", "") for answer in split["answer"]]
    eos = tokenizer.eos_token_id
    stops = frozenset([eos] if isinstance(eos, int) else list(eos or []))
    return ids, gold, stops


def _number(text: str) -> str | None:
    """The answer a completion gives: after `####`, else its last number."""
    tail = text.split("####")[-1] if "####" in text else text
    found = re.findall(r"-?\d[\d,]*\.?\d*", tail)
    return found[-1].replace(",", "").rstrip(".") if found else None


def _score(data: dict[str, Any], outputs: list[list[int]], gold: list[str]) -> float:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(data["weights"]["repo"])
    right = 0
    for tokens, answer in zip(outputs, gold, strict=True):
        got = _number(tokenizer.decode(tokens, skip_special_tokens=True))
        try:
            right += got is not None and float(got) == float(answer)
        except ValueError:
            continue
    return right / len(gold)


def _gsm8k_result(
    name: str,
    kind: str,
    data: dict[str, Any],
    outputs: list[list[int]],
    gold: list[str],
    seconds: float,
    notes: str,
) -> Result:
    tokens = sum(len(o) for o in outputs)
    return Result(
        name,
        kind,
        {
            "gsm8k_acc": _score(data, outputs, gold),
            "gsm8k_s": seconds,
            "gsm8k_tok_s": tokens / seconds,
        },
        notes=f"GSM8K test, first {PROBLEMS} problems, greedy, up to {ANSWER_TOKENS} tokens; "
        + notes,
    )


def gsm8k_vllm(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    from vllm import LLM, SamplingParams  # type: ignore[import-not-found]

    ids, gold, stops = _problems(data)
    _, concurrency, _, _, _ = settings(workload)
    llm = LLM(
        model=data["weights"]["repo"],
        dtype="bfloat16",
        max_model_len=max(len(p) for p in ids) + ANSWER_TOKENS,
        max_num_seqs=concurrency,
        gpu_memory_utilization=memory_share(workload, "vllm_memory"),
        enable_prefix_caching=False,
    )
    params = SamplingParams(max_tokens=ANSWER_TOKENS, temperature=0.0, stop_token_ids=sorted(stops))
    begin = time.perf_counter()
    done = llm.generate([{"prompt_token_ids": p} for p in ids], params, use_tqdm=False)
    seconds = time.perf_counter() - begin
    outputs = [list(o.outputs[0].token_ids) for o in done]
    return _gsm8k_result("vLLM", "reference", data, outputs, gold, seconds, "offline")


def gsm8k_sglang(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    from bench.methods.engines import _sglang_engine

    ids, gold, stops = _problems(data)
    _, concurrency, _, _, _ = settings(workload)
    engine = _sglang_engine(
        data,
        workload,
        max(len(p) for p in ids) + ANSWER_TOKENS,
        concurrency,
        memory_share(workload, "sglang_memory"),
    )
    params = {
        "temperature": 0.0,
        "max_new_tokens": ANSWER_TOKENS,
        "stop_token_ids": sorted(stops),
    }
    try:
        begin = time.perf_counter()
        done = engine.generate(input_ids=ids, sampling_params=params)
        seconds = time.perf_counter() - begin
    finally:
        engine.shutdown()
    outputs = [list(o["output_ids"]) for o in done]
    return _gsm8k_result("SGLang", "reference", data, outputs, gold, seconds, "offline")


def gsm8k_linnet(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    from linnet import nest
    from linnet.serve import Engine, Request

    ids, gold, stops = _problems(data)
    longest = max(len(p) for p in ids) + ANSWER_TOKENS
    sized = {**workload, "serve_prompt_max": longest - workload.get("serve_new", 128)}
    generics, options, layout = _layout(data, sized, "torch")
    model = nest.load(
        data["directory"],
        backend="torch",
        device=workload.get("device", "cuda"),
        numerics="fast",
        compile=True,
        generics=generics,
        cast_dtype=True,
    )
    engine = Engine(model, **options)
    engine.warmup(len(p) for p in ids)
    begin = time.perf_counter()
    done, _ = engine.run([Request(prompt=p, max_new_tokens=ANSWER_TOKENS, eos=stops) for p in ids])
    seconds = time.perf_counter() - begin
    outputs = [[t for t in c.tokens if t not in stops] for c in done]
    return _gsm8k_result(
        "Linnet torch (linnet.serve, CUDA graphs)", "linnet", data, outputs, gold, seconds, layout
    )


# ------------------------------------------------------------------- soak


def _memory_gib() -> float:
    """Memory in use: the driver's count where the GPU has its own, the
    host's where it shares it (a GB10), where a GPU's allocations land."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return float(pynvml.nvmlDeviceGetMemoryInfo(handle).used) / 2**30
    except Exception:  # noqa: BLE001 - no NVML count (shared memory): the host's
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            fields = dict(line.split(":", 1) for line in meminfo)
        total = int(fields["MemTotal"].split()[0])
        available = int(fields["MemAvailable"].split()[0])
        return (total - available) / 2**20


def _soak(
    name: str,
    kind: str,
    workload: dict[str, Any],
    submit: Callable[[list[int]], None],
    step: Callable[[], tuple[int, int]],
    pending: Callable[[], int],
    load_s: float,
    notes: str,
    data: dict[str, Any],
) -> Result:
    """Keeps the serving loop full for `soak_minutes`: `submit` adds a
    request, `step` advances and returns (tokens generated, requests
    finished), `pending` counts requests in flight."""
    minutes = float(workload.get("soak_minutes", 30))
    _, concurrency, _, _, _ = settings(workload)
    pool = prompts(data, workload)
    next_prompt = [0]

    def refill() -> None:
        while pending() < concurrency:
            submit(pool[next_prompt[0] % len(pool)])
            next_prompt[0] += 1

    per_minute: list[int] = []
    memory: list[float] = []
    failures = 0
    begin = time.perf_counter()
    mark = begin
    tokens = 0
    while time.perf_counter() - begin < minutes * 60:
        refill()
        try:
            made, _ = step()
        except Exception:  # noqa: BLE001 - counted, the loop goes on
            failures += 1
            continue
        tokens += made
        if time.perf_counter() - mark >= 60:
            per_minute.append(tokens)
            memory.append(_memory_gib())
            tokens, mark = 0, time.perf_counter()
    first = per_minute[0] / 60 if per_minute else None
    last = per_minute[-1] / 60 if per_minute else None
    return Result(
        name,
        kind,
        {
            "soak_tok_s_first": first,
            "soak_tok_s_last": last,
            "soak_drift": (last / first - 1) if first and last else None,
            "soak_mem_growth_gib": (memory[-1] - memory[0]) if len(memory) > 1 else None,
            "soak_failures": float(failures),
            "load_s": load_s,
        },
        notes=f"{minutes:g} minutes at {concurrency} in flight, a new request for each one "
        f"finished; memory sampled each minute; {notes}",
    )


def soak_vllm(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    from vllm import LLM, SamplingParams  # type: ignore[import-not-found]

    _, concurrency, _, _, new = settings(workload)
    start = time.perf_counter()
    llm = LLM(
        model=data["weights"]["repo"],
        dtype="bfloat16",
        max_model_len=serve_max_seq(workload),
        max_num_seqs=concurrency,
        gpu_memory_utilization=memory_share(workload, "vllm_memory"),
        enable_prefix_caching=False,
    )
    load_s = time.perf_counter() - start
    engine = llm.llm_engine
    params = SamplingParams(max_tokens=new, temperature=0.0, ignore_eos=True)
    counter = [0]

    def submit(ids: list[int]) -> None:
        engine.add_request(str(counter[0]), {"prompt_token_ids": ids}, params)
        counter[0] += 1

    def step() -> tuple[int, int]:
        outputs = engine.step()
        done = [o for o in outputs if o.finished]
        return sum(len(o.outputs[0].token_ids) for o in done), len(done)

    return _soak(
        "vLLM",
        "reference",
        workload,
        submit,
        step,
        engine.get_num_unfinished_requests,
        load_s,
        "LLMEngine steps",
        data,
    )


def soak_linnet(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    from linnet import nest
    from linnet.serve import Engine, Request

    _, _, _, _, new = settings(workload)
    generics, options, layout = _layout(data, workload, "torch")
    start = time.perf_counter()
    model = nest.load(
        data["directory"],
        backend="torch",
        device=workload.get("device", "cuda"),
        numerics="fast",
        compile=True,
        generics=generics,
        cast_dtype=True,
    )
    engine = Engine(model, **options)
    load_s = time.perf_counter() - start
    engine.warmup(len(ids) for ids in prompts(data, workload))
    engine.reset()
    flight = [0]

    def submit(ids: list[int]) -> None:
        engine.submit(Request(prompt=ids, max_new_tokens=new))
        flight[0] += 1

    def step() -> tuple[int, int]:
        finished = engine.step()
        flight[0] -= len(finished)
        return sum(len(c.tokens) for c in finished), len(finished)

    return _soak(
        "Linnet torch (linnet.serve, CUDA graphs)",
        "linnet",
        workload,
        submit,
        step,
        lambda: flight[0],
        load_s,
        layout,
        data,
    )


METHODS = {
    "gsm8k-vllm": gsm8k_vllm,
    "gsm8k-sglang": gsm8k_sglang,
    "gsm8k-linnet-torch": gsm8k_linnet,
    "soak-vllm": soak_vllm,
    "soak-linnet-torch": soak_linnet,
}

# Long, run only when named (`--methods`).
OPT_IN = {"soak-vllm", "soak-linnet-torch"}
