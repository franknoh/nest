"""Text generation: how long until the first token, and how fast the rest
arrive, for the reference stacks people use and for each Linnet backend.

Every method gets the same prompt (random token ids from a fixed seed: the
content does not change the arithmetic) and generates greedily. Time to
first token is the prompt's pass through the model plus choosing a token.
Decode throughput is new tokens per second after the first, batch 1.

Each method also saves the logits it computed for the first token, so the
report can show how far every stack is from the reference -- a fast wrong
answer should not look like a win.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bench.harness import Result, median_ms

SAFE_TOKENS = (100, 20000)  # ordinary vocabulary, clear of special tokens


def prompt_ids(data: dict[str, Any], workload: dict[str, Any]) -> list[int]:
    import random

    vocab = int(data["generics"].get("Vocab", SAFE_TOKENS[1]))
    rng = random.Random(workload.get("seed", 0))  # noqa: S311 - a reproducible prompt, not a secret
    high = min(vocab, SAFE_TOKENS[1])
    return [rng.randrange(SAFE_TOKENS[0], high) for _ in range(int(workload["prompt_tokens"]))]


def max_seq(workload: dict[str, Any]) -> int:
    """The cache length a static-shape stack is compiled for: the prompt and
    the new tokens, rounded up. Recorded with the results, since every static
    engine (TensorRT-LLM, XLA) is configured the same way."""
    needed = int(workload["prompt_tokens"]) + int(workload["new_tokens"])
    return ((needed + 255) // 256) * 256


def cache_generics(data: dict[str, Any], length: int) -> dict[str, int | str]:
    """The cache size a workload needs, for the generics this card has: a
    card without a KV cache (gpt-oss here) takes neither."""
    declared = data.get("generics", {})
    # bf16, like every reference row: a card that defaults to f32 (GPT-2)
    # otherwise runs twice the bytes beside them. Loads pass `cast_dtype`.
    wanted: dict[str, int | str] = {"MaxSeq": length, "Batch": 1, "T": "bf16"}
    return {name: value for name, value in wanted.items() if name in declared}


def save_logits(workload: dict[str, Any], method: str, logits: Any) -> None:
    import numpy as np

    directory = Path(workload["workdir"])
    directory.mkdir(parents=True, exist_ok=True)
    array = logits.float().cpu().numpy() if hasattr(logits, "float") else np.asarray(logits)
    np.save(directory / f"{method}.npy", array.astype("float32").reshape(-1))


def plain_greedy(model: Any, new: int, minimum: int | None = None) -> Any:
    """A `GenerationConfig` that is greedy and nothing else. `generate`
    otherwise applies the checkpoint's own generation settings, and some of
    them survive `do_sample=False`: Qwen2.5-Instruct's `repetition_penalty`
    of 1.05, for one, penalizes every token already in the prompt, so "the
    sky is blue" in the question turns the answer's greedy "is" into
    "appears". Linnet's argmax has no such processor, so the comparison
    would be between two different decoding rules."""
    from transformers import GenerationConfig

    eos = model.generation_config.eos_token_id
    return GenerationConfig(
        max_new_tokens=new,
        min_new_tokens=minimum,
        do_sample=False,
        eos_token_id=eos,
        pad_token_id=eos[0] if isinstance(eos, list) else eos,
    )


# ------------------------------------------------------------ reference stacks


class _Stamps:
    """A `generate` streamer that only records when each token arrived.
    `transformers` hands the streamer `next_tokens.cpu()`, so each stamp is
    taken after the token actually exists."""

    def __init__(self) -> None:
        self.times: list[float] = []

    def put(self, _value: Any) -> None:
        self.times.append(time.perf_counter())

    def end(self) -> None:
        return None


def _transformers(compiled: bool) -> Callable[[dict[str, Any], dict[str, Any]], Result]:
    def run(data: dict[str, Any], workload: dict[str, Any]) -> Result:
        import torch
        from transformers import AutoModelForCausalLM

        name = "transformers (torch.compile, static cache)" if compiled else "transformers (eager)"
        method = "transformers-compile" if compiled else "transformers-eager"
        start = time.perf_counter()
        attention = "sdpa"
        try:
            loaded = AutoModelForCausalLM.from_pretrained(
                data["weights"]["repo"], dtype=torch.bfloat16, attn_implementation=attention
            )
        except ValueError:  # an architecture without SDPA support (gpt-oss)
            attention = "eager"
            loaded = AutoModelForCausalLM.from_pretrained(
                data["weights"]["repo"], dtype=torch.bfloat16, attn_implementation=attention
            )
        model = loaded.to(workload.get("device", "cuda")).eval()
        if compiled:
            # Default inductor mode: with `reduce-overhead` its CUDA graphs
            # overwrite outputs `generate` still holds, in this transformers.
            model.generation_config.cache_implementation = "static"
            model.forward = torch.compile(model.forward, fullgraph=True)
        load_s = time.perf_counter() - start
        ids = torch.tensor([prompt_ids(data, workload)], device=workload.get("device", "cuda"))
        new = int(workload["new_tokens"])

        def generate() -> _Stamps:
            stamps = _Stamps()
            stamps.times.append(time.perf_counter())
            with torch.no_grad():
                model.generate(
                    ids, generation_config=plain_greedy(model, new, minimum=new), streamer=stamps
                )
            return stamps

        for _ in range(int(workload["warmup"])):
            generate()
        ttfts: list[float] = []
        rates: list[float] = []
        for _ in range(int(workload["iters"])):
            stamps = generate()
            # [start, prompt echoed, token 1, token 2, ...]
            first = stamps.times[2]
            ttfts.append((first - stamps.times[0]) * 1e3)
            rates.append((new - 1) / (stamps.times[-1] - first))
        with torch.no_grad():
            save_logits(workload, method, model(ids).logits[0, -1])
        import statistics

        return Result(
            name,
            "reference",
            notes="" if attention == "sdpa" else "eager attention: no SDPA for this architecture",
            metrics={
                "ttft_ms": statistics.median(ttfts),
                "decode_tok_s": statistics.median(rates),
                "load_s": load_s,
            },
        )

    return run


def vllm(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    import statistics

    from vllm import LLM, SamplingParams  # type: ignore[import-not-found]

    new = int(workload["new_tokens"])
    start = time.perf_counter()
    llm = LLM(
        model=data["weights"]["repo"],
        dtype="bfloat16",
        max_model_len=max_seq(workload),
        gpu_memory_utilization=float(workload.get("vllm_memory", 0.85)),
    )
    load_s = time.perf_counter() - start
    prompt = {"prompt_token_ids": prompt_ids(data, workload)}
    first = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    whole = SamplingParams(max_tokens=new, temperature=0.0, ignore_eos=True)

    def timed(params: Any) -> float:
        begin = time.perf_counter()
        llm.generate([prompt], params, use_tqdm=False)
        return (time.perf_counter() - begin) * 1e3

    for _ in range(int(workload["warmup"])):
        timed(whole)
    ttft = statistics.median(timed(first) for _ in range(int(workload["iters"])))
    total = statistics.median(timed(whole) for _ in range(int(workload["iters"])))
    token = llm.generate([prompt], first, use_tqdm=False)[0].outputs[0].token_ids[0]
    return Result(
        "vLLM",
        "reference",
        {
            "ttft_ms": ttft,
            "decode_tok_s": (new - 1) / ((total - ttft) / 1e3),
            "load_s": load_s,
            "first_token": float(token),
        },
        notes="reserves a KV-cache pool up front (gpu_memory_utilization "
        f"{workload.get('vllm_memory', 0.85)}), so its memory is a setting, not a need",
    )


# ------------------------------------------------------------------- Linnet


def _linnet_torch(
    compile: bool | str, placement: str | None = None
) -> Callable[[dict[str, Any], dict[str, Any]], Result]:
    """Linnet's PyTorch backend. `placement` spreads the model over every GPU
    (`"gpus"`) or caps the one GPU so that part of the model is streamed in
    from the host (`"offload"`, the cap from `workload["offload_gib"]`)."""
    label = {True: "generated source", "reduce-overhead": "CUDA graphs"}[compile]
    method = {True: "linnet-torch", "reduce-overhead": "linnet-cudagraphs"}[compile]
    if placement == "gpus":
        label, method = "all GPUs", "linnet-gpus"
    elif placement == "offload":
        label, method = "offloaded", "linnet-offload"

    def run(data: dict[str, Any], workload: dict[str, Any]) -> Result:
        import os
        import statistics

        if placement == "offload":
            # One GPU and the host: with a second GPU visible the planner would
            # put the overflow there, which is a two-GPU split, not offloading.
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(
                ","
            )[0]

        import torch
        from linnet import nest

        from bench.harness import torch_sync

        device = workload.get("device", "cuda")
        sync = torch_sync()
        start = time.perf_counter()
        placed: dict[str, Any] = {}
        if placement == "gpus":
            placed = {"device_map": "auto"}
        elif placement == "offload":
            placed = {
                "device_map": "auto",
                "max_memory": {0: f"{workload.get('offload_gib', 8)}GiB"},
            }
        model = nest.load(
            data["directory"],
            backend="torch",
            device=device,
            numerics="fast",
            compile=compile,
            generics=cache_generics(data, max_seq(workload)),
            cast_dtype=True,
            **placed,
        )
        load_s = time.perf_counter() - start
        where = getattr(model, "placement", None)
        ids = torch.tensor([prompt_ids(data, workload)], dtype=torch.int32, device=device)
        new = int(workload["new_tokens"])
        length = ids.shape[1]
        zero = torch.tensor(0, dtype=torch.int32, device=device)
        if "prefill" not in model.entries:
            # No KV-cache entries (gpt-oss here): the first token is one pass
            # over the prompt, and there is no decode to time -- repeating that
            # pass per token would measure recomputation, not decoding.
            def whole() -> None:
                model.run_entry("next_token", [ids]).argmax(-1).item()

            for _ in range(int(workload["warmup"])):
                whole()
            ttft = median_ms(whole, sync, 0, int(workload["iters"]))
            save_logits(workload, method, model.run_entry("next_token", [ids])[0])
            return Result(
                f"Linnet torch ({label})",
                "linnet",
                {"ttft_ms": ttft, "load_s": load_s},
                notes="the card has no KV-cache entries, so only the first token is timed",
            )

        def prefill() -> torch.Tensor:
            return model.run_entry("prefill", [ids, zero])

        def generate() -> float:
            logits = prefill()
            token = logits.argmax(-1).reshape(1, 1).to(torch.int32)
            sync()
            begin = time.perf_counter()
            for step in range(new - 1):
                pos = torch.tensor(length + step, dtype=torch.int32, device=device)
                token = model.run_entry("decode", [token, pos]).argmax(-1).reshape(1, 1)
                token = token.to(torch.int32)
            sync()
            return (new - 1) / (time.perf_counter() - begin)

        def first_token() -> None:
            prefill().argmax(-1).item()

        # Warmup also compiles: one shape for the prompt, one for a step.
        for _ in range(int(workload["warmup"])):
            generate()
        ttft = median_ms(first_token, sync, 0, int(workload["iters"]))
        rate = statistics.median(generate() for _ in range(max(3, int(workload["iters"]) // 3)))
        save_logits(workload, method, prefill()[0])
        notes = f"KV cache compiled for {max_seq(workload)} positions"
        if where is not None and not where.trivial:
            notes += "; " + where.describe().replace("\n", "; ")
        return Result(
            f"Linnet torch ({label})",
            "linnet",
            {"ttft_ms": ttft, "decode_tok_s": rate, "load_s": load_s},
            notes=notes,
        )

    return run


def linnet_jax(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    """The StableHLO export under XLA. Both entries share one copy of the
    weights on the device: `linnet.jax` takes parameters as arguments."""
    import statistics

    import jax
    import jax.numpy as jnp
    from linnet import nest
    from linnet.jax import load as load_jax

    generics = cache_generics(data, max_seq(workload))
    start = time.perf_counter()
    prefill = nest.load(
        data["directory"], backend="jax", entry="prefill", generics=generics, cast_dtype=True
    )
    # The second entry reuses the first one's arrays, already bound by path
    # and on the device, so the weights are not read or placed twice.
    decode = load_jax(
        Path(data["directory"]) / data["source"]["path"],
        generics=prefill.generics,
        weights=prefill.weights,
        root=prefill.root,
        entry="decode",
        cast_dtype=True,
    )
    load_s = time.perf_counter() - start
    ids = jnp.asarray([prompt_ids(data, workload)], dtype=jnp.int32)
    new = int(workload["new_tokens"])
    length = ids.shape[1]

    def first() -> tuple[Any, Any]:
        logits, state = prefill(ids, jnp.int32(0))
        return logits, state

    def generate() -> float:
        logits, state = first()
        token = jnp.argmax(logits, -1).reshape(1, 1).astype(jnp.int32)
        jax.block_until_ready(token)
        begin = time.perf_counter()
        for step in range(new - 1):
            logits, state = decode(token, jnp.int32(length + step), state=state)
            token = jnp.argmax(logits, -1).reshape(1, 1).astype(jnp.int32)
        jax.block_until_ready(token)
        return (new - 1) / (time.perf_counter() - begin)

    for _ in range(int(workload["warmup"])):
        generate()
    ttft = median_ms(
        lambda: jax.block_until_ready(jnp.argmax(first()[0], -1)),
        lambda: None,
        0,
        int(workload["iters"]),
    )
    rate = statistics.median(generate() for _ in range(max(3, int(workload["iters"]) // 3)))
    save_logits(workload, "linnet-jax", jax.device_get(first()[0])[0])
    return Result(
        f"Linnet JAX (XLA, {jax.default_backend()})",
        "linnet",
        {"ttft_ms": ttft, "decode_tok_s": rate, "load_s": load_s},
        notes=f"KV cache compiled for {max_seq(workload)} positions",
    )


# ------------------------------------------------------------------- sample

CHAT = "Explain in two sentences why the sky is blue."
CONTINUATION = "The printing press changed Europe because"
SAMPLE_TOKENS = 96


def _greedy_linnet(model: Any, ids: list[int], eos: set[int], device: str) -> list[int]:
    """Greedy continuation through the card's own entries: `prefill` and
    `decode` where it has them, otherwise `next_token` over the growing
    sequence (a card without a cache, such as gpt-oss here)."""
    import torch

    out: list[int] = []
    if "prefill" in model.entries:
        tokens = torch.tensor([ids], dtype=torch.int32, device=device)
        logits = model.run_entry(
            "prefill", [tokens, torch.tensor(0, dtype=torch.int32, device=device)]
        )
        for step in range(SAMPLE_TOKENS):
            token = int(logits.argmax(-1).item())
            out.append(token)
            if token in eos:
                break
            position = torch.tensor(len(ids) + step, dtype=torch.int32, device=device)
            one = torch.tensor([[token]], dtype=torch.int32, device=device)
            logits = model.run_entry("decode", [one, position])
        return out
    sequence = list(ids)
    for _ in range(SAMPLE_TOKENS):
        tokens = torch.tensor([sequence], dtype=torch.int32, device=device)
        token = int(model.run_entry("next_token", [tokens]).argmax(-1).item())
        out.append(token)
        sequence.append(token)
        if token in eos:
            break
    return out


def sample(data: dict[str, Any], workload: dict[str, Any]) -> dict[str, Any]:
    """A chat answer (or, for a base model, a continuation) from the card
    through Linnet, with `transformers`' greedy answer beside it.

    Both are greedy, but in bf16 two stacks can round a near-tie differently
    and part ways after a while; the sample records how many leading tokens
    agree rather than pretending they must all match."""
    import torch
    from linnet import nest
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = workload.get("device", "cuda")
    repo = data["weights"]["repo"]
    tokenizer = AutoTokenizer.from_pretrained(repo)
    if tokenizer.chat_template:
        prompt = CHAT
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": CHAT}], add_generation_prompt=True
        )
        # Older `transformers` return the ids; newer ones a `BatchEncoding`.
        if hasattr(templated, "keys"):
            templated = templated["input_ids"]
        ids = [int(t) for t in templated]
    else:
        prompt = CONTINUATION
        ids = [int(t) for t in tokenizer(CONTINUATION)["input_ids"]]
    eos = {int(t) for t in ([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [])}

    model = nest.load(
        data["directory"],
        backend="torch",
        device=device,
        numerics="fast",
        compile=True,
        generics=cache_generics(data, ((len(ids) + SAMPLE_TOKENS + 255) // 256) * 256),
        cast_dtype=True,
    )
    with torch.no_grad():
        ours = _greedy_linnet(model, ids, eos, device)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # The reference is `transformers`' own forward with its KV cache and a
    # bare argmax, not `generate`: `generate` layers the checkpoint's logits
    # processors on top, and on Qwen2.5 that alone turns "The sky is blue"
    # into "The sky appears blue" at a 0.6 logit margin, even with a plain
    # generation config.
    reference = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).to(device)
    generated: list[int] = []
    with torch.no_grad():
        out = reference(torch.tensor([ids], device=device), use_cache=True)
        for _ in range(SAMPLE_TOKENS):
            token = int(out.logits[0, -1].argmax())
            generated.append(token)
            if token in eos:
                break
            out = reference(
                torch.tensor([[token]], device=device),
                past_key_values=out.past_key_values,
                use_cache=True,
            )
    agree = 0
    for a, b in zip(ours, generated, strict=False):
        if a != b:
            break
        agree += 1
    return {
        "kind": "text",
        "prompt": prompt,
        "chat": bool(tokenizer.chat_template),
        "output": tokenizer.decode(ours, skip_special_tokens=True).strip(),
        "reference": tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "reference_stack": "transformers (KV cache, argmax, bf16)",
        "tokens": len(ours),
        "agreeing_prefix": agree,
    }


METHODS: dict[str, Callable[[dict[str, Any], dict[str, Any]], Result]] = {
    "transformers-eager": _transformers(compiled=False),
    "transformers-compile": _transformers(compiled=True),
    "vllm": vllm,
    "linnet-torch": _linnet_torch(True),
    "linnet-cudagraphs": _linnet_torch("reduce-overhead"),
    "linnet-gpus": _linnet_torch(True, placement="gpus"),
    "linnet-offload": _linnet_torch(True, placement="offload"),
    "linnet-jax": linnet_jax,
}

REFERENCE = "transformers-eager"


def methods_for(data: dict[str, Any]) -> list[str]:
    """Every method, less the ones a card cannot take part in: without
    KV-cache entries (gpt-oss here) the JAX path, which is built around
    `prefill` and `decode`, has nothing to run."""
    source = Path(data["directory"]) / data["source"]["path"]
    sources = [source] if source.is_file() else sorted(source.parent.glob("*.linnet"))
    text = "".join(path.read_text(encoding="utf-8") for path in sources)
    cached = "entry prefill" in text
    # Placement rows are opt-in (`--methods`): they need a second GPU or a
    # deliberately starved one, and say something about Linnet rather than
    # about each model.
    optional = {"linnet-gpus", "linnet-offload"}
    return [name for name in METHODS if name not in optional and (cached or name != "linnet-jax")]
