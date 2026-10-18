"""Accuracy for text generation: how close each stack's numbers come to the
model's own, on real text, so a fast row cannot hide a wrong one.

The reference is `transformers` in f32 (`acc-reference`). It runs first and
leaves its log-probabilities, token windows and greedy continuations in the
run's workdir; every other `acc-` row reads them and reports, beside its own
perplexity (`ppl`):

- `ppl_ratio`: its perplexity over the reference's;
- `top1`: how often its most likely next token is the reference's;
- `kl`: mean KL(reference || row) per position, where the row gives full
  logits (transformers, Linnet); engines only return the top few;
- `greedy_agree`: the agreeing prefix of its greedy continuations with the
  reference's, as a fraction of their length.

The text is WikiText-2's test split, joined with blank lines and tokenized
whole with the card's tokenizer, then cut into `acc_windows` consecutive
windows of `acc_length` tokens, the first token of each the BOS where the
tokenizer starts sequences with one. That is how llama.cpp's
`llama-perplexity` cuts the same text, so its row scores (nearly) the same
tokens. Greedy prompts are each window's first tokens; continuations run a
fixed length and take the bare argmax, with no logits processors (see
`llm.plain_greedy`).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.harness import Result, memory_share

PROMPT = 64  # greedy prompt: each window's first tokens
NEW = 64  # greedy tokens per prompt

TOKENS = "acc-tokens.npy"  # [windows, length] int32
LOGPROBS = "acc-reference-logprobs.npy"  # [windows * (length - 1), vocab] float16
ARGMAX = "acc-reference-argmax.npy"  # [windows, length - 1] int32, from f32
GREEDY = "acc-reference-greedy.npy"  # [windows, NEW] int32
SUMMARY = "acc-reference.json"

NO_REFERENCE = "no accuracy reference in this run (acc-reference), so perplexity alone"


def _settings(workload: dict[str, Any]) -> tuple[int, int]:
    return int(workload.get("acc_windows", 8)), int(workload.get("acc_length", 512))


def acc_seq(workload: dict[str, Any]) -> int:
    """The cache length a static stack is set up for: a window plus one
    generated token (vLLM scores a prompt only by generating), or a greedy
    prompt and its continuation, rounded up like `llm.max_seq`."""
    _, length = _settings(workload)
    return ((max(length + 1, PROMPT + NEW) + 255) // 256) * 256


def _text() -> str:
    from datasets import load_dataset  # type: ignore[import-untyped]

    rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(rows["text"])


def _bos(tokenizer: Any) -> int | None:
    """The BOS id where the tokenizer starts a sequence with one (Llama,
    Mistral), as llama.cpp's `add_bos` does; None where it does not (Qwen)."""
    bos = tokenizer.bos_token_id
    if bos is None:
        return None
    probe = tokenizer("a")["input_ids"]
    return int(bos) if probe and probe[0] == bos else None


def _windows(data: dict[str, Any], workload: dict[str, Any]) -> Any:
    """The token windows: the reference's where it left them, otherwise cut
    from the text here, the way `llama-perplexity` cuts it."""
    import numpy as np
    from transformers import AutoTokenizer

    windows, length = _settings(workload)
    saved = Path(workload["workdir"]) / TOKENS
    if saved.exists():
        tokens = np.load(saved)
        if tokens.shape == (windows, length):
            return tokens
    tokenizer = AutoTokenizer.from_pretrained(data["weights"]["repo"])
    bos = _bos(tokenizer)
    ids = tokenizer(_text(), add_special_tokens=False)["input_ids"]
    stream = ([bos] if bos is not None else []) + [int(t) for t in ids]
    if len(stream) < windows * length:
        raise RuntimeError(f"WikiText-2 has {len(stream)} tokens, fewer than {windows}x{length}")
    tokens = np.asarray(stream[: windows * length], dtype=np.int32).reshape(windows, length)
    if bos is not None:
        tokens[:, 0] = bos
    return tokens


def _prompts(tokens: Any) -> list[list[int]]:
    return [[int(t) for t in row[:PROMPT]] for row in tokens]


@dataclass
class _Reference:
    ppl: float
    ppl_half: float  # over the second half of each window, as llama-perplexity scores
    argmax: Any  # [windows, length - 1]
    greedy: Any  # [windows, NEW]
    logprobs: Any  # memory-mapped [windows * (length - 1), vocab]


def _reference(workload: dict[str, Any], tokens: Any) -> _Reference | None:
    import numpy as np

    directory = Path(workload["workdir"])
    if not (directory / SUMMARY).exists() or not (directory / TOKENS).exists():
        return None
    if not np.array_equal(np.load(directory / TOKENS), tokens):
        return None
    summary = json.loads((directory / SUMMARY).read_text(encoding="utf-8"))
    return _Reference(
        ppl=float(summary["ppl"]),
        ppl_half=float(summary["ppl_half"]),
        argmax=np.load(directory / ARGMAX),
        greedy=np.load(directory / GREEDY),
        logprobs=np.load(directory / LOGPROBS, mmap_mode="r"),
    )


class _Tally:
    """One row's scores, window by window, against the reference."""

    def __init__(self, tokens: Any, reference: _Reference | None) -> None:
        self.tokens = tokens
        self.reference = reference
        self.nll: list[Any] = []
        self.agree = 0
        self.positions = 0
        self.kl: list[float] = []

    def full(self, window: int, logprobs: Any) -> None:
        """A window's log-probabilities over the whole vocabulary,
        [length - 1, vocab] float32, row i predicting token i + 1."""
        import numpy as np

        targets = self.tokens[window, 1:].astype(np.int64)
        picked = np.take_along_axis(logprobs, targets[:, None], axis=-1)[:, 0]
        self.partial(window, picked, logprobs.argmax(-1))
        if self.reference is None:
            return
        rows = logprobs.shape[0]
        expected = np.asarray(
            self.reference.logprobs[window * rows : (window + 1) * rows], dtype=np.float32
        )
        # A padded vocabulary may be cut differently; the columns past the
        # narrower one are padding, with no probability to speak of.
        width = min(expected.shape[-1], logprobs.shape[-1])
        expected, got = expected[:, :width], logprobs[:, :width]
        p = np.exp(expected)
        terms = np.where(p > 0, p * (expected - got), 0.0)
        self.kl.append(float(terms.sum(-1).mean()))

    def partial(self, window: int, token_logprobs: Any, argmax: Any | None) -> None:
        """A window's log-probability of each actual token and, where the
        stack gives it, its most likely token, for positions 1..length-1."""
        import numpy as np

        self.nll.append(-np.asarray(token_logprobs, dtype=np.float64))
        if argmax is not None and self.reference is not None:
            self.agree += int((np.asarray(argmax) == self.reference.argmax[window]).sum())
            self.positions += len(argmax)

    def metrics(self, greedy: list[list[int]] | None) -> dict[str, float | None]:
        import numpy as np

        ppl = float(np.exp(np.concatenate(self.nll).mean()))
        out: dict[str, float | None] = {"ppl": ppl}
        if self.reference is None:
            return out
        out["ppl_ratio"] = ppl / self.reference.ppl
        out["top1"] = self.agree / self.positions if self.positions else None
        out["kl"] = float(np.mean(self.kl)) if self.kl else None
        out["greedy_agree"] = _greedy_agree(greedy, self.reference) if greedy else None
        return out


def _greedy_agree(greedy: list[list[int]], reference: _Reference) -> float:
    fractions: list[float] = []
    for ours, theirs in zip(greedy, reference.greedy, strict=True):
        agree = 0
        for a, b in zip(ours, theirs, strict=False):
            if int(a) != int(b):
                break
            agree += 1
        fractions.append(agree / NEW)
    return sum(fractions) / len(fractions)


def _notes(tokens: Any, extra: str = "", missing: bool = False) -> str:
    windows, length = tokens.shape
    notes = f"WikiText-2 test, {windows} windows of {length} tokens; {PROMPT}+{NEW} greedy"
    if missing:
        notes += "; " + NO_REFERENCE
    return notes + (f"; {extra}" if extra else "")


# ---------------------------------------------------------------- transformers


def _hf_model(data: dict[str, Any], workload: dict[str, Any], dtype: Any) -> tuple[Any, str]:
    """The checkpoint in `dtype`, placed by `accelerate` on a GPU run: an f32
    copy of a large model may not fit one GPU, and spills to the host."""
    from transformers import AutoModelForCausalLM

    device = workload.get("device", "cuda")
    placed: dict[str, Any] = {"device_map": "auto"} if device == "cuda" else {}
    attention = "sdpa"
    repo = data["weights"]["repo"]
    try:
        model = AutoModelForCausalLM.from_pretrained(
            repo, dtype=dtype, attn_implementation=attention, **placed
        )
    except ValueError:  # an architecture without SDPA support (gpt-oss)
        attention = "eager"
        model = AutoModelForCausalLM.from_pretrained(
            repo, dtype=dtype, attn_implementation=attention, **placed
        )
    if not placed:
        model = model.to(device)
    return model.eval(), attention


def _hf_logprobs(model: Any, window: Any) -> Any:
    """Log-probabilities in f32 at every position but the last, [length - 1,
    vocab], on the model's device."""
    import torch

    ids = torch.tensor(window[None].astype("int64"), device=model.device)
    with torch.no_grad():
        logits = model(ids, use_cache=False).logits[0, :-1]
    return logits.float().log_softmax(-1)


def _hf_greedy(model: Any, prompt: list[int]) -> list[int]:
    """`transformers`' forward with its KV cache and a bare argmax, as
    `llm.sample` does, for `NEW` tokens whatever they are."""
    import torch

    out: list[int] = []
    with torch.no_grad():
        step = model(torch.tensor([prompt], device=model.device), use_cache=True)
        while True:
            token = int(step.logits[0, -1].argmax())
            out.append(token)
            if len(out) == NEW:
                return out
            step = model(
                torch.tensor([[token]], device=model.device),
                past_key_values=step.past_key_values,
                use_cache=True,
            )


def reference(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    """`transformers` in f32: the numbers every other row is held to."""
    import numpy as np
    import torch

    tokens = _windows(data, workload)
    windows, length = tokens.shape
    directory = Path(workload["workdir"])
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / TOKENS, tokens)
    model, attention = _hf_model(data, workload, torch.float32)
    store: Any = None
    nll: list[Any] = []
    argmax = np.zeros((windows, length - 1), dtype=np.int32)
    for window in range(windows):
        logprobs = _hf_logprobs(model, tokens[window])
        if store is None:
            # Written as it is computed: a large vocabulary's whole array is
            # most of a GiB.
            store = np.lib.format.open_memmap(
                directory / LOGPROBS,
                mode="w+",
                dtype=np.float16,
                shape=(windows * (length - 1), logprobs.shape[-1]),
            )
        targets = torch.tensor(tokens[window, 1:].astype("int64"), device=logprobs.device)
        nll.append(-logprobs.gather(-1, targets[:, None])[:, 0].double().cpu().numpy())
        argmax[window] = logprobs.argmax(-1).cpu().numpy()
        rows = slice(window * (length - 1), (window + 1) * (length - 1))
        store[rows] = logprobs.to(torch.float16).cpu().numpy()
    store.flush()
    del store
    greedy = np.asarray([_hf_greedy(model, prompt) for prompt in _prompts(tokens)], np.int32)
    np.save(directory / ARGMAX, argmax)
    np.save(directory / GREEDY, greedy)
    every = np.concatenate(nll)
    # llama-perplexity scores the predictions of tokens length/2 + 1 onwards.
    half = np.concatenate([row[length // 2 :] for row in nll])
    ppl, ppl_half = float(np.exp(every.mean())), float(np.exp(half.mean()))
    summary = {"ppl": ppl, "ppl_half": ppl_half, "windows": windows, "length": length}
    (directory / SUMMARY).write_text(json.dumps(summary), encoding="utf-8")
    extra = "" if attention == "sdpa" else "eager attention: no SDPA for this architecture"
    return Result(
        "transformers (f32, accuracy reference)",
        "reference",
        {"ppl": ppl},
        notes=_notes(tokens, extra),
    )


def transformers(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    """`transformers` in bf16, the stack people actually run."""
    import torch

    tokens = _windows(data, workload)
    known = _reference(workload, tokens)
    model, attention = _hf_model(data, workload, torch.bfloat16)
    tally = _Tally(tokens, known)
    for window in range(tokens.shape[0]):
        tally.full(window, _hf_logprobs(model, tokens[window]).cpu().numpy())
    greedy = [_hf_greedy(model, prompt) for prompt in _prompts(tokens)]
    extra = "" if attention == "sdpa" else "eager attention: no SDPA for this architecture"
    return Result(
        "transformers (bf16, accuracy)",
        "reference",
        tally.metrics(greedy),
        notes=_notes(tokens, extra, known is None),
    )


# ----------------------------------------------------------------- engines


def vllm(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    """vLLM offline in bf16: each actual token's log-probability and the
    most likely one per position from `prompt_logprobs`, then greedy
    continuations at temperature 0."""
    import numpy as np
    from vllm import LLM, SamplingParams  # type: ignore[import-not-found]

    tokens = _windows(data, workload)
    known = _reference(workload, tokens)
    share = memory_share(workload, "vllm_memory")
    llm = LLM(
        model=workload.get("model_path", data["weights"]["repo"]),
        dtype="bfloat16",
        max_model_len=acc_seq(workload),
        gpu_memory_utilization=share,
        # Greedy prompts are the windows' own beginnings; from a prefix cache
        # they would be the scoring pass's numbers, not a fresh prefill's.
        enable_prefix_caching=False,
    )
    windows = [{"prompt_token_ids": [int(t) for t in row]} for row in tokens]
    scored = llm.generate(
        windows, SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1), use_tqdm=False
    )
    tally = _Tally(tokens, known)
    for window, output in enumerate(scored):
        # Position 0 has nothing before it; each later one holds the actual
        # token and the most likely one (the same entry when they agree).
        positions = output.prompt_logprobs[1:]
        picked = np.zeros(len(positions))
        argmax = np.zeros(len(positions), dtype=np.int64)
        for i, entry in enumerate(positions):
            picked[i] = entry[int(tokens[window, i + 1])].logprob
            argmax[i] = min(entry.items(), key=lambda item: item[1].rank)[0]
        tally.partial(window, picked, argmax)
    prompts = [{"prompt_token_ids": prompt} for prompt in _prompts(tokens)]
    continued = llm.generate(
        prompts, SamplingParams(max_tokens=NEW, temperature=0.0, ignore_eos=True), use_tqdm=False
    )
    greedy = [list(output.outputs[0].token_ids) for output in continued]
    return Result(
        "vLLM (accuracy)",
        "reference",
        tally.metrics(greedy),
        notes=_notes(
            tokens, "top-1 and log-probabilities from prompt_logprobs; no KL", known is None
        ),
    )


def sglang(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    """SGLang's offline engine, built as its timing rows build it: input
    log-probabilities with the top token per position, then greedy
    continuations."""
    import numpy as np

    from bench.methods.engines import _attention_note, _sampling, _sglang_engine

    tokens = _windows(data, workload)
    known = _reference(workload, tokens)
    windows, length = tokens.shape
    share = memory_share(workload, "sglang_memory")
    engine = _sglang_engine(data, workload, acc_seq(workload), windows, share)
    try:
        scored = engine.generate(
            input_ids=[[int(t) for t in row] for row in tokens],
            sampling_params={"max_new_tokens": 1, "temperature": 0.0},
            return_logprob=True,
            logprob_start_len=0,
            top_logprobs_num=1,
        )
        continued = engine.generate(input_ids=_prompts(tokens), sampling_params=_sampling(NEW))
    finally:
        engine.shutdown()
    tally = _Tally(tokens, known)
    for window, output in enumerate(scored):
        meta = output["meta_info"]
        # Entries are (logprob, token id, text); position 0's log-probability
        # is None, so the last length - 1 are the ones predicted.
        given = meta["input_token_logprobs"][-(length - 1) :]
        tops = meta["input_top_logprobs"][-(length - 1) :]
        ids = [int(entry[1]) for entry in given]
        if ids != [int(t) for t in tokens[window, 1:]]:
            raise RuntimeError("SGLang's input log-probabilities are not aligned with the window")
        picked = np.asarray([float(entry[0]) for entry in given])
        argmax = np.asarray([int(top[0][1]) for top in tops])
        tally.partial(window, picked, argmax)
    greedy = [[int(t) for t in output["output_ids"]] for output in continued]
    return Result(
        "SGLang (accuracy)",
        "reference",
        tally.metrics(greedy),
        notes=_notes(
            tokens,
            "top-1 and log-probabilities from input logprobs; no KL" + _attention_note(),
            known is None,
        ),
    )


def llamacpp(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    """llama.cpp's `llama-perplexity` on Linnet's GGUF export (f16), over the
    same text. It takes text, not ids, and tokenizes it with the GGUF's own
    tokenizer, so its windows match the others only as far as the two
    tokenizers agree; and it scores the second half of each window (the
    first half is context), so its ratio is to the reference's perplexity
    over those same halves. It has no greedy or top-1 to compare."""
    from linnet import gguf

    converter = os.environ.get("NEST_LLAMA_CONVERTER")
    perplexity = os.environ.get("NEST_LLAMA_PERPLEXITY")
    if not converter or not perplexity:
        raise RuntimeError(
            "llama.cpp is not set up here (NEST_LLAMA_CONVERTER, NEST_LLAMA_PERPLEXITY)"
        )
    windows, length = _settings(workload)
    directory = Path(workload["workdir"])
    exported = gguf.export(data["directory"], directory / "linnet-gguf", converter=converter)
    text = directory / "acc-wikitext.txt"
    text.write_text(_text(), encoding="utf-8")
    command = [perplexity, "-m", str(exported.gguf), "-f", str(text), "-c", str(length)]
    command += ["-b", str(length), "--chunks", str(windows), "-ngl", "99"]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    output = completed.stdout + completed.stderr
    found = re.search(r"Final estimate: PPL = ([0-9.]+)", output)
    if completed.returncode != 0 or found is None:
        tail = output.strip().splitlines()[-3:]
        raise RuntimeError("llama-perplexity failed: " + " | ".join(tail))
    ppl = float(found.group(1))
    metrics: dict[str, float | None] = {"ppl": ppl}
    notes = (
        f"linnet.gguf.export, then llama-perplexity on WikiText-2 test, {windows} windows of "
        f"{length} tokens as it tokenizes the text; it scores each window's second half, and "
        "the ratio is to the reference over the same halves; no top-1, KL or greedy"
    )
    summary = directory / SUMMARY
    if summary.exists():
        metrics["ppl_ratio"] = ppl / float(json.loads(summary.read_text())["ppl_half"])
    else:
        notes += "; " + NO_REFERENCE
    return Result("Linnet -> llama.cpp (GGUF f16, accuracy)", "linnet", metrics, notes=notes)


# ------------------------------------------------------------------- Linnet


def linnet_torch(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    """Linnet's PyTorch backend as `linnet-torch` loads it: the card's
    `forward` entry for every window's logits, greedy continuations through
    `prefill` and `decode`."""
    import torch
    from linnet import nest

    from bench.methods.llm import _greedy_linnet, cache_generics  # llm imports this module

    device = workload.get("device", "cuda")
    tokens = _windows(data, workload)
    known = _reference(workload, tokens)
    model = nest.load(
        data["directory"],
        backend="torch",
        device=device,
        numerics="fast",
        compile=True,
        generics=cache_generics(data, acc_seq(workload)),
        cast_dtype=True,
    )
    tally = _Tally(tokens, known)
    with torch.no_grad():
        for window in range(tokens.shape[0]):
            ids = torch.tensor(tokens[window : window + 1], dtype=torch.int32, device=device)
            logits = model.run_entry("forward", [ids])[0, :-1]
            tally.full(window, logits.float().log_softmax(-1).cpu().numpy())
        greedy = [_greedy_linnet(model, prompt, set(), device, NEW) for prompt in _prompts(tokens)]
    return Result(
        "Linnet torch (accuracy)",
        "linnet",
        tally.metrics(greedy),
        notes=_notes(tokens, missing=known is None),
    )


def linnet_jax(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    """Linnet's JAX backend as `linnet-jax` loads it (`jax_model`, the
    StableHLO export): `forward` for the logits, `prefill` and `decode` for
    greedy continuations."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from linnet import nest

    from bench.methods.llm import cache_generics  # llm imports this module

    tokens = _windows(data, workload)
    known = _reference(workload, tokens)
    model = nest.load(
        data["directory"],
        backend="jax_model",
        generics=cache_generics(data, acc_seq(workload)),
        cast_dtype=True,
        generated=workload.get("jax_generated", False),
    )
    scores = jax.jit(lambda logits: jax.nn.log_softmax(logits[0, :-1].astype(jnp.float32), -1))
    pick = jax.jit(lambda logits: jnp.argmax(logits, -1).reshape(1, 1).astype(jnp.int32))
    tally = _Tally(tokens, known)
    for window in range(tokens.shape[0]):
        ids = jnp.asarray(tokens[window : window + 1], dtype=jnp.int32)
        tally.full(window, np.asarray(jax.device_get(scores(model.run_entry("forward", [ids])))))
    greedy: list[list[int]] = []
    for prompt in _prompts(tokens):
        ids = jnp.asarray([prompt], dtype=jnp.int32)
        token = pick(model.run_entry("prefill", [ids, np.int32(0)]))
        steps = [token]
        for step in range(NEW - 1):
            token = pick(model.run_entry("decode", [token, np.int32(len(prompt) + step)]))
            steps.append(token)
        greedy.append([int(np.asarray(t).reshape(-1)[0]) for t in jax.device_get(steps)])
    return Result(
        f"Linnet JAX (XLA, {jax.default_backend()}; accuracy)",
        "linnet",
        tally.metrics(greedy),
        notes=_notes(tokens, missing=known is None),
    )


METHODS = {
    "acc-reference": reference,
    "acc-transformers": transformers,
    "acc-vllm": vllm,
    "acc-sglang": sglang,
    "acc-llamacpp": llamacpp,
    "acc-linnet-torch": linnet_torch,
    "acc-linnet-jax": linnet_jax,
}

REFERENCE = "acc-reference"
