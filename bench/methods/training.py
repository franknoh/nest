"""Training: Linnet's trainers (`linnet.train`) beside TRL's on the same card,
data and hyperparameters, so a faster step is not bought with a different
batch or a worse model.

Each pair of rows shares its data, tokenized once and the same way for both:
SFT on Alpaca packed into 4096-token rows, 4 a step; DPO on UltraFeedback, 32
pairs a step; GRPO on GSM8K, 16 prompts x 8 completions of up to 128 tokens,
scored by one reward. Both sides take torch's AdamW at the same learning rate
and schedule (`linnet.train.cosine_schedule`), gradients clipped to norm 1,
bf16 weights and adapters, no gradient checkpointing. TRL reads the rows
already tokenized (SFT's already packed) and in order.

Steps are timed alike: `step_s` is the median step after the first two,
`first_step_s` the first (compilation included), `train_tok_s` the median of
each step's real tokens (no padding) over its time. `heldout_loss` comes from
one evaluator for both, after training: the token-weighted mean NLL of 64
held-out examples' learned tokens, every side's logits through the same
log-softmax. TRL's own eval loss averages differently, so it is not used.

These rows run only when named (`--methods train-sft-lora-linnet,...`): each
takes minutes, and an 8B full fine-tune does not fit one GPU at all.
"""

from __future__ import annotations

import functools
import itertools
import re
import statistics
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from bench.harness import Result

Method = Callable[[dict[str, Any], dict[str, Any]], Result]
Measured = tuple[dict[str, float | None], str]

TOKENS = 4096  # positions in a packed row
ROWS = 4  # SFT rows a step
HELD = 64  # held-out examples (DPO: pairs) each evaluation scores
RANK, ALPHA = 16, 32
# Every attention and MLP projection. Only 2-D weights take adapters, so the
# norms these also match are left alone; a fused `qkv` (Phi-3) is one weight.
LORA_PATTERNS = ["layers.*.attention.*.weight", "layers.*.mlp.*.weight"]
LR = {"sft-lora": 2e-4, "sft-full": 1e-5, "dpo": 2e-5, "grpo": 5e-5}
WARMUP = 3  # steps of linear warmup before the cosine decay
PAIRS, BETA = 32, 0.1
# TRL's micro-batches: on an H100 with an 8B model, larger ones ran out of
# memory. Linnet packs a step's pairs or completions into 4096-token batches.
DPO_PER_PASS = 2
GRPO_PER_PASS = 8
PROMPTS, GROUP, NEW_TOKENS = 16, 8, 128
GRPO_PROMPT_LIMIT = 384
GRPO_CORRECTION_CAP = 2.0
GRAD_TOKENS = 128
QUANTIZED = {"fp8", "int8", "w8a8", "int4", "mxfp4"}


class DoesNotFit(Exception):  # noqa: N818 - reads as the row's error
    """The run would not fit on one GPU; the row says so instead."""


# ------------------------------------------------------------------- data


def tokenizer_for(data: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    weights = data["weights"]
    tokenizer = AutoTokenizer.from_pretrained(weights["repo"], revision=weights.get("revision"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prompt_ids(tokenizer: Any, conversation: list[dict[str, str]]) -> list[int]:
    """A conversation as the model's prompt, the assistant's turn opened. The
    template's text tokenized without added specials, which is what
    `apply_chat_template(tokenize=True)` (and so TRL's GRPO) does."""
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])
    text = "\n\n".join(message["content"] for message in conversation) + "\n\n"
    ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    return [tokenizer.bos_token_id, *ids] if tokenizer.bos_token_id is not None else ids


def chat_pair(
    tokenizer: Any, conversation: list[dict[str, str]], answer: str
) -> tuple[list[int], list[int]]:
    """A prompt and an answer to it as token ids, each tokenized on its own:
    the answer as the template writes it, end-of-turn token included. A
    template that rewrites earlier turns once an answer follows (no common
    prefix) gets the answer and the EOS token."""
    prompt = prompt_ids(tokenizer, conversation)
    eos = tokenizer.eos_token or ""
    reply = answer + eos
    if tokenizer.chat_template:
        opened = tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        whole = tokenizer.apply_chat_template(
            [*conversation, {"role": "assistant", "content": answer}], tokenize=False
        )
        if whole.startswith(opened):
            reply = whole[len(opened) :]
    return prompt, list(tokenizer(reply, add_special_tokens=False)["input_ids"])


def sft_data(tokenizer: Any, steps: int) -> tuple[list[Any], list[Any]]:
    """Alpaca as chat turns, the answer learned: 64 held-out examples, and
    `steps * ROWS` packed rows (`linnet.packing.Packed`) of the rest."""
    from datasets import load_dataset
    from linnet.packing import Example, pack

    rows = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=0)

    def example(row: dict[str, str]) -> Example:
        user = row["instruction"] + (f"\n\n{row['input']}" if row["input"] else "")
        prompt, answer = chat_pair(tokenizer, [{"role": "user", "content": user}], row["output"])
        prompt = prompt[-1024:]
        return Example.prompted(prompt, answer[: 2048 - len(prompt)])

    held = [example(rows[i]) for i in range(HELD)]
    rest = (example(rows[i]) for i in range(HELD, len(rows)))
    return held, list(itertools.islice(pack(rest, TOKENS), steps * ROWS))


def real_tokens(packed: Any) -> int:
    """A packed row's positions that are not padding (the last segment)."""
    return int((packed.segments < packed.sequences).sum())


def sft_rows(packed: Sequence[Any]) -> list[dict[str, list[int]]]:
    """The packed rows as TRL's padding-free rows: the real tokens, labels
    (-100 where nothing is learned; transformers shifts them), and each
    sequence's length, from which the collator makes `position_ids`."""
    import numpy as np

    rows: list[dict[str, list[int]]] = []
    for row in packed:
        real = real_tokens(row)
        tokens = [int(t) for t in row.tokens[:real]]
        # Position j - 1 predicting token j is what `mask[j - 1]` learns.
        labels = [-100] + [t if row.mask[j - 1] > 0 else -100 for j, t in enumerate(tokens) if j]
        lengths = np.bincount(row.segments[:real], minlength=row.sequences).tolist()
        rows.append({"input_ids": tokens, "labels": labels, "seq_lengths": lengths})
    return rows


def dpo_data(tokenizer: Any, steps: int) -> tuple[list[Any], list[Any]]:
    """UltraFeedback (binarized) pairs: 64 from the test split, held out, and
    `steps * PAIRS` from the shuffled train split. Prompts keep their last
    512 tokens, answers their first 512."""
    from datasets import load_dataset
    from linnet.packing import Pair

    def pairs(split: str, count: int, shuffle: bool) -> list[Pair]:
        rows = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
        if shuffle:
            rows = rows.shuffle(seed=0)
        chosen_pairs: list[Pair] = []
        for row in rows:
            conversation = row["chosen"][:-1]
            if not conversation or conversation[-1]["role"] != "user":
                continue
            prompt, chosen = chat_pair(tokenizer, conversation, row["chosen"][-1]["content"])
            _, rejected = chat_pair(tokenizer, conversation, row["rejected"][-1]["content"])
            chosen_pairs.append(Pair(prompt[-512:], chosen[:512], rejected[:512]))
            if len(chosen_pairs) == count:
                break
        return chosen_pairs

    return pairs("test_prefs", HELD, False), pairs("train_prefs", steps * PAIRS, True)


def pair_examples(pairs: Sequence[Any]) -> list[Any]:
    """Each pair's chosen answer, then its rejected one, after its prompt."""
    from linnet.packing import Example

    return [Example.prompted(p.prompt, answer) for p in pairs for answer in (p.chosen, p.rejected)]


GSM8K_ASK = "\n\nThink briefly, then end with the final answer as a number."
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def gsm8k_reward(text: str, answer: str) -> float:
    """1 when the completion's last number is the answer, else 0: the one
    reward both GRPO rows score with, on the decoded completion."""
    found = NUMBER.findall(text)
    if not found:
        return 0.0
    try:
        return float(float(found[-1].replace(",", "")) == float(answer.replace(",", "")))
    except ValueError:
        return 0.0


def grpo_data(
    tokenizer: Any, steps: int
) -> tuple[list[Any], list[tuple[list[dict[str, str]], list[int], str]]]:
    """GSM8K: 64 test questions with their worked answers (for the held-out
    loss), and `steps * PROMPTS` train questions as (conversation, prompt
    ids, final answer), prompts of at most 384 tokens."""
    from datasets import load_dataset
    from linnet.packing import Example

    def ask(question: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": question + GSM8K_ASK}]

    test = load_dataset("openai/gsm8k", "main", split="test")
    held: list[Example] = []
    for row in test.select(range(HELD)):
        prompt, answer = chat_pair(tokenizer, ask(row["question"]), row["answer"])
        held.append(Example.prompted(prompt, answer))
    train = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=0)
    prompts: list[tuple[list[dict[str, str]], list[int], str]] = []
    for row in train:
        conversation = ask(row["question"])
        ids = prompt_ids(tokenizer, conversation)
        if len(ids) > GRPO_PROMPT_LIMIT:
            continue
        prompts.append((conversation, ids, row["answer"].split("####")[-1].strip()))
        if len(prompts) == steps * PROMPTS:
            break
    return held, prompts


def eos_ids(data: dict[str, Any], tokenizer: Any) -> list[int]:
    """Every token that ends a turn: the tokenizer's EOS and the
    checkpoint's generation config's (Llama 3's `<|eot_id|>`, ...)."""
    from transformers import GenerationConfig

    found = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set[int]()
    try:
        weights = data["weights"]
        config = GenerationConfig.from_pretrained(weights["repo"], revision=weights.get("revision"))
        listed = config.eos_token_id
        found |= set(listed if isinstance(listed, list) else [listed] if listed is not None else [])
    except OSError:  # no generation config in the checkpoint
        pass
    return sorted(found)


# -------------------------------------------------------------- evaluator


Scorer = Callable[[Any], list[float]]


def scores(examples: Sequence[Any], scorer: Scorer) -> list[tuple[float, int]]:
    """Each example's summed NLL over its learned tokens, and their count:
    the examples packed into 4096-token rows (`linnet.packing.pack`), and
    `scorer` giving each packed sequence's sum. The one evaluator both
    stacks are scored by; only the forward pass differs."""
    from linnet.packing import pack

    found = [(0.0, 0)] * len(examples)
    for packed in pack(examples, TOKENS):
        sums = scorer(packed)
        for sequence, item in enumerate(packed.items):
            count = int(packed.mask[packed.segments == sequence].sum())
            found[item] = (sums[sequence], count)
    return found


def mean_nll(scored: Sequence[tuple[float, int]]) -> float:
    """The token-weighted mean: every learned token counts once."""
    return sum(nll for nll, _ in scored) / max(1, sum(count for _, count in scored))


def pair_accuracy(before: Sequence[tuple[float, int]], after: Sequence[tuple[float, int]]) -> float:
    """The share of held-out pairs whose chosen answer's implicit reward,
    `log p(answer) - log p_ref(answer)`, beats the rejected one's, the
    reference being the model before training."""
    right = 0
    for k in range(0, len(after), 2):
        chosen = before[k][0] - after[k][0]  # NLLs: the log-probability gained
        rejected = before[k + 1][0] - after[k + 1][0]
        right += chosen > rejected
    return right / max(1, len(after) // 2)


def linnet_scorer(model: Any) -> Scorer:
    """Sums through the card's `log_probs_packed` entry, a packed row at a
    time (the shape it trained at)."""
    import torch

    device = next(model.parameters()).device

    def score(packed: Any) -> list[float]:
        inputs = [
            torch.as_tensor(values).to(device)
            for values in (packed.tokens, packed.positions, packed.segments, packed.targets)
        ]
        with torch.no_grad():
            per_token = model.run_entry("log_probs_packed", inputs).float()
        mask = torch.as_tensor(packed.mask, device=device)
        sums = per_token.new_zeros(packed.sequences + 1)
        sums = sums.scatter_add(0, inputs[2].long(), -per_token * mask)
        return [float(value) for value in sums[: packed.sequences]]

    return score


def hf_scorer(model: Any) -> Scorer:
    """Sums from a transformers model's logits, one sequence at a time (no
    packing to trust), through the same f32 log-softmax."""
    import torch

    device = next(model.parameters()).device

    def score(packed: Any) -> list[float]:
        sums: list[float] = []
        for sequence in range(packed.sequences):
            where = packed.segments == sequence
            ids = torch.as_tensor(packed.tokens[where], dtype=torch.long, device=device)
            targets = torch.as_tensor(packed.targets[where], dtype=torch.long, device=device)
            mask = torch.as_tensor(packed.mask[where], device=device)
            with torch.no_grad():
                logits = model(input_ids=ids[None], use_cache=False).logits[0].float()
            chosen = torch.log_softmax(logits, -1).gather(-1, targets[:, None])[:, 0]
            sums.append(float(-(chosen * mask).sum()))
        return sums

    return score


# ---------------------------------------------------------------- helpers


def _steps(workload: dict[str, Any]) -> int:
    return int(workload.get("train_steps", 40))


def _source_text(data: dict[str, Any]) -> str:
    source = Path(data["directory"]) / data["source"]["path"]
    sources = [source] if source.is_file() else sorted(source.parent.glob("*.linnet"))
    return "".join(path.read_text(encoding="utf-8") for path in sources)


def unsupported(data: dict[str, Any], entries: Sequence[str]) -> str | None:
    """Why a card cannot take part, or None. A card Linnet cannot train gets
    no TRL row either: there would be nothing to compare it with."""
    quantized = sorted(set(data["model"].get("tags", [])) & QUANTIZED)
    if quantized:
        return f"{', '.join(quantized)} weights do not train here; the bf16 card does"
    text = _source_text(data)
    missing = [entry for entry in entries if f"entry {entry}" not in text]
    if missing:
        return f"the card has no {', '.join(missing)} entry to train through"
    return None


def _generics(
    data: dict[str, Any], length: int, batch: int = 1, dtype: str = "bf16"
) -> dict[str, int | str]:
    declared = data.get("generics", {})
    wanted: dict[str, int | str] = {"MaxSeq": length, "Batch": batch, "T": dtype}
    return {name: value for name, value in wanted.items() if name in declared}


def _linnet(data: dict[str, Any], workload: dict[str, Any], *, full: bool) -> Any:
    """The card for training: one packed row, Inductor; LoRA adapters on
    every projection unless `full`."""
    from linnet import nest

    model = nest.load(
        data["directory"],
        backend="torch",
        device=workload.get("device", "cuda"),
        numerics="fast",
        compile="inductor",
        generics=_generics(data, TOKENS),
        cast_dtype=True,
        trainable=full,
    )
    if full:
        _check_fits(model)
    else:
        model.add_lora(LORA_PATTERNS, rank=RANK, alpha=ALPHA)
    return model


def _attention() -> str:
    """FlashAttention 2 where installed; SDPA otherwise, which takes TRL's
    padding-free rows apart by their `position_ids` (a block-diagonal mask)."""
    from importlib.util import find_spec

    return "flash_attention_2" if find_spec("flash_attn") is not None else "sdpa"


def _hf(data: dict[str, Any], workload: dict[str, Any], *, full: bool, cache: bool = True) -> Any:
    """The checkpoint in transformers, bf16, with PEFT LoRA adapters on every
    linear layer but the head unless `full`. The adapters stay bf16
    (`autocast_adapter_dtype=False`), as Linnet's do. Without `cache`, the
    model makes no KV cache in training, which would hide packed rows'
    boundaries from the mask."""
    import torch
    from transformers import AutoModelForCausalLM

    weights = data["weights"]
    model = AutoModelForCausalLM.from_pretrained(
        weights["repo"],
        revision=weights.get("revision"),
        dtype=torch.bfloat16,
        attn_implementation=_attention(),
    ).to(workload.get("device", "cuda"))
    model.config.use_cache = cache
    if full:
        _check_fits(model)
        return model
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        lora_dropout=0.0,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config, autocast_adapter_dtype=False)


def _check_fits(model: Any) -> None:
    """A full fine-tune holds weights, gradients and AdamW's two moments, 8
    bytes a parameter in bf16, before any activation."""
    import torch

    if not torch.cuda.is_available():
        return
    count = sum(p.numel() for p in model.parameters())
    needed = 8 * count
    total = torch.cuda.get_device_properties(0).total_memory
    if needed > 0.9 * total:
        raise DoesNotFit(
            f"a full fine-tune of {count / 1e9:.1f}B parameters needs {needed / 2**30:.0f} GiB "
            f"for weights, gradients and AdamW state alone; the GPU has {total / 2**30:.0f} GiB"
        )


def _optimizer(model: Any, lr: float, steps: int) -> tuple[Any, Any]:
    """The same optimizer and schedule on both sides: AdamW without weight
    decay, linear warmup then cosine to zero."""
    import torch
    from linnet.train import cosine_schedule

    trained = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trained, lr=lr, weight_decay=0.0)
    return optimizer, cosine_schedule(optimizer, WARMUP, steps, floor=0.0)


def _reset_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _peak_gib() -> float | None:
    """The allocator's peak since `_reset_peak`, before training: weights
    included, the evaluator not. The driver's peak for the whole process is
    `peak_vram_mib`."""
    import torch

    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None


def _timing(seconds: Sequence[float], tokens: Sequence[float]) -> dict[str, float | None]:
    """The median step after the first two (which compile), the median of
    each such step's real tokens per second, and the first step."""
    steps = list(zip(seconds, tokens, strict=False))
    timed = steps[2:] or steps
    if not timed:
        return {"step_s": None, "train_tok_s": None, "first_step_s": None}
    return {
        "step_s": statistics.median(s for s, _ in timed),
        "train_tok_s": statistics.median(t / s for s, t in timed),
        "first_step_s": seconds[0],
    }


def _losses(losses: Sequence[float]) -> dict[str, float | None]:
    return {
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
    }


def _clock() -> Any:
    """A Trainer callback timing each optimizer step, generation included,
    synchronized at both ends: the batches are fetched before it starts."""
    import torch
    from transformers import TrainerCallback

    class Clock(TrainerCallback):
        def __init__(self) -> None:
            self.seconds: list[float] = []
            self.begin = 0.0

        def on_step_begin(self, *_: Any, **__: Any) -> None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.begin = time.perf_counter()

        def on_step_end(self, *_: Any, **__: Any) -> None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.seconds.append(time.perf_counter() - self.begin)

    return Clock()


def _trl_common(
    workload: dict[str, Any], steps: int, per_pass: int, passes: int, lr: float
) -> dict[str, Any]:
    """The arguments every TRL row shares: the rows in order, a log entry
    every step, bf16, no gradient checkpointing (TRL turns it on by
    default), the same clipping, nothing saved."""
    return {
        "output_dir": tempfile.mkdtemp(prefix="nest-trl-"),
        "max_steps": steps,
        "per_device_train_batch_size": per_pass,
        "gradient_accumulation_steps": passes,
        "learning_rate": lr,
        "max_grad_norm": 1.0,
        "bf16": True,
        "gradient_checkpointing": False,
        "train_sampling_strategy": "sequential",
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "none",
        "disable_tqdm": True,
        "dataloader_num_workers": 0,
        "seed": 0,
        "torch_compile": bool(workload.get("trl_compile", False)),
    }


def _logged(trainer: Any, key: str) -> list[float]:
    """Each step's `key` from the Trainer's log (the run's summary has
    `train_loss`, not `loss`)."""
    return [float(entry[key]) for entry in trainer.state.log_history if key in entry]


def _trl_notes(workload: dict[str, Any]) -> str:
    compiled = "torch.compile" if workload.get("trl_compile") else "eager"
    return f"transformers {compiled}, {_attention()} attention"


def _row(name: str, kind: str, entries: Sequence[str], measure: Callable[..., Measured]) -> Method:
    """A method that reports a card it cannot train, or a run that does not
    fit the GPU, as the row's error rather than a crash."""

    def run(data: dict[str, Any], workload: dict[str, Any]) -> Result:
        problem = unsupported(data, entries)
        if problem is not None:
            return Result(name, kind, error=problem)
        import torch

        try:
            metrics, notes = measure(data, workload)
        except DoesNotFit as error:
            return Result(name, kind, error=str(error))
        except torch.OutOfMemoryError as error:
            return Result(name, kind, error=f"out of memory on one GPU: {error}"[:400])
        return Result(name, kind, metrics, notes=notes)

    return run


# -------------------------------------------------------------------- SFT


def _sft_notes(full: bool, steps: int) -> str:
    lr = LR["sft-full" if full else "sft-lora"]
    adapted = "every weight" if full else f"LoRA rank {RANK}, alpha {ALPHA}, every projection"
    return (
        f"Alpaca, {ROWS} packed rows of {TOKENS} tokens a step, {steps} steps; {adapted}; "
        f"AdamW lr {lr:g}, {WARMUP} warmup steps then cosine; bf16; no gradient checkpointing"
    )


def _sft_tokens(packed: Sequence[Any]) -> list[float]:
    return [
        float(sum(real_tokens(row) for row in packed[start : start + ROWS]))
        for start in range(0, len(packed), ROWS)
    ]


def _sft_linnet(full: bool, data: dict[str, Any], workload: dict[str, Any]) -> Measured:
    from linnet.train import Batch, train

    steps = _steps(workload)
    held, packed = sft_data(tokenizer_for(data), steps)
    model = _linnet(data, workload, full=full)
    optimizer, schedule = _optimizer(model, LR["sft-full" if full else "sft-lora"], steps)
    _reset_peak()
    history = train(
        model,
        [Batch.of(row) for row in packed],
        optimizer=optimizer,
        steps=steps,
        accumulate=ROWS,
        schedule=schedule,
    )
    peak = _peak_gib()
    metrics = {
        **_timing([step.seconds for step in history.steps], _sft_tokens(packed)),
        "peak_gib": peak,
        **_losses(history.losses),
        "heldout_loss": mean_nll(scores(held, linnet_scorer(model))),
    }
    return metrics, _sft_notes(full, steps) + '; compile="inductor"'


def _sft_trl(full: bool, data: dict[str, Any], workload: dict[str, Any]) -> Measured:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    steps = _steps(workload)
    lr = LR["sft-full" if full else "sft-lora"]
    tokenizer = tokenizer_for(data)
    held, packed = sft_data(tokenizer, steps)
    model = _hf(data, workload, full=full, cache=False)
    optimizer, schedule = _optimizer(model, lr, steps)
    clock = _clock()
    args = SFTConfig(
        **_trl_common(workload, steps, 1, ROWS, lr),
        # The rows Linnet trains on, packed already: padding-free, each
        # padded to the row's 4096 positions as Linnet's are.
        packing=False,
        padding_free=True,
        max_length=None,
        pad_to_multiple_of=TOKENS,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(sft_rows(packed)),
        processing_class=tokenizer,
        callbacks=[clock],
        optimizers=(optimizer, schedule),
    )
    _reset_peak()
    trainer.train()
    peak = _peak_gib()
    model.eval()
    metrics = {
        **_timing(clock.seconds, _sft_tokens(packed)),
        "peak_gib": peak,
        **_losses(_logged(trainer, "loss")),
        "heldout_loss": mean_nll(scores(held, hf_scorer(model))),
    }
    notes = (
        _sft_notes(full, steps) + "; the same packed rows, padding-free; " + _trl_notes(workload)
    )
    return metrics, notes + ("" if full else "; PEFT")


# -------------------------------------------------------------------- DPO


def _dpo_notes(steps: int) -> str:
    return (
        f"UltraFeedback, {PAIRS} pairs a step, {steps} steps, beta {BETA}; LoRA rank {RANK}, "
        f"alpha {ALPHA}; AdamW lr {LR['dpo']:g}, {WARMUP} warmup steps then cosine; reference "
        "log-probabilities computed before training; heldout_loss is the held-out chosen "
        "answers' NLL"
    )


def _dpo_tokens(pairs: Sequence[Any]) -> list[float]:
    return [
        float(sum(2 * len(p.prompt) + len(p.chosen) + len(p.rejected) for p in step))
        for step in (pairs[start : start + PAIRS] for start in range(0, len(pairs), PAIRS))
    ]


def _dpo_metrics(
    seconds: Sequence[float],
    pairs: Sequence[Any],
    losses: Sequence[float],
    before: Sequence[tuple[float, int]],
    after: Sequence[tuple[float, int]],
    peak: float | None,
) -> dict[str, float | None]:
    return {
        **_timing(seconds, _dpo_tokens(pairs)),
        "peak_gib": peak,
        **_losses(losses),
        "heldout_loss": mean_nll(after[0::2]),
        "reward_accuracy": pair_accuracy(before, after),
    }


def _dpo_linnet(data: dict[str, Any], workload: dict[str, Any]) -> Measured:
    from linnet.train.dpo import dpo

    steps = _steps(workload)
    held, pairs = dpo_data(tokenizer_for(data), steps)
    model = _linnet(data, workload, full=False)
    examples = pair_examples(held)
    before = scores(examples, linnet_scorer(model))
    optimizer, schedule = _optimizer(model, LR["dpo"], steps)
    _reset_peak()
    history = dpo(
        model,
        pairs,
        optimizer=optimizer,
        steps=steps,
        pairs_per_step=PAIRS,
        beta=BETA,
        tokens=TOKENS,
        schedule=schedule,
    )
    peak = _peak_gib()
    after = scores(examples, linnet_scorer(model))
    seconds = [step.seconds for step in history]
    metrics = _dpo_metrics(seconds, pairs, [s.loss for s in history], before, after, peak)
    return metrics, _dpo_notes(
        steps
    ) + f'; pairs packed {TOKENS} tokens a batch; compile="inductor"'


def _dpo_trl(data: dict[str, Any], workload: dict[str, Any]) -> Measured:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    class Pretokenized(DPOTrainer):
        """DPOTrainer on pairs tokenized already (`prompt_ids`, `chosen_ids`,
        `rejected_ids`, what its own tokenization makes), so its ids are
        Linnet's rather than the text tokenized again."""

        def _prepare_dataset(self, dataset: Any, *_: Any, **__: Any) -> Any:
            return dataset

    steps = _steps(workload)
    tokenizer = tokenizer_for(data)
    held, pairs = dpo_data(tokenizer, steps)
    model = _hf(data, workload, full=False)
    examples = pair_examples(held)
    before = scores(examples, hf_scorer(model))
    optimizer, schedule = _optimizer(model, LR["dpo"], steps)
    clock = _clock()
    args = DPOConfig(
        **_trl_common(workload, steps, DPO_PER_PASS, PAIRS // DPO_PER_PASS, LR["dpo"]),
        beta=BETA,
        max_length=None,  # cut already, as Linnet's are
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=4 * DPO_PER_PASS,
    )
    rows = [
        {
            "prompt_ids": list(p.prompt),
            "chosen_ids": list(p.chosen),
            "rejected_ids": list(p.rejected),
        }
        for p in pairs
    ]
    trainer = Pretokenized(
        model=model,
        ref_model=None,  # PEFT: the model with its adapters off
        args=args,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        callbacks=[clock],
        optimizers=(optimizer, schedule),
    )
    _reset_peak()
    trainer.train()
    peak = _peak_gib()
    model.eval()
    after = scores(examples, hf_scorer(model))
    metrics = _dpo_metrics(clock.seconds, pairs, _logged(trainer, "loss"), before, after, peak)
    notes = _dpo_notes(steps) + f"; {DPO_PER_PASS} pairs a pass, padded; " + _trl_notes(workload)
    return metrics, notes + "; PEFT"


# ------------------------------------------------------------------- GRPO


def _grpo_notes(steps: int) -> str:
    return (
        f"GSM8K, {PROMPTS} prompts x {GROUP} completions of up to {NEW_TOKENS} tokens a step, "
        f"{steps} steps, temperature 1; reward 1 when the last number is the answer; loss the "
        f"mean over every completion token, advantages scaled by the group's deviation, no KL; "
        f"LoRA rank {RANK}, alpha {ALPHA}; AdamW lr {LR['grpo']:g}, {WARMUP} warmup steps then "
        "cosine; step_s includes sampling; heldout_loss is the NLL of held-out worked answers"
    )


def _grpo_metrics(
    seconds: Sequence[float],
    tokens: Sequence[float],
    losses: Sequence[float],
    rewards: Sequence[float],
    heldout: float,
    peak: float | None,
) -> dict[str, float | None]:
    return {
        **_timing(seconds, tokens),
        "peak_gib": peak,
        **_losses(losses),
        "heldout_loss": heldout,
        "reward_first5": statistics.fmean(rewards[:5]) if rewards else None,
        "reward_last5": statistics.fmean(rewards[-5:]) if rewards else None,
    }


def _grpo_tokens(
    prompts: Sequence[tuple[Any, list[int], str]], lengths: Sequence[float]
) -> list[float]:
    """Each step's prompts (once per completion) and its completions' tokens."""
    return [
        GROUP * sum(len(ids) for _, ids, _ in prompts[k * PROMPTS : (k + 1) * PROMPTS])
        + length * PROMPTS * GROUP
        for k, length in enumerate(lengths)
    ]


def _grpo_linnet(data: dict[str, Any], workload: dict[str, Any]) -> Measured:
    from linnet import nest
    from linnet.packing import Prompt
    from linnet.serve import Engine
    from linnet.train.grpo import grpo

    steps = _steps(workload)
    tokenizer = tokenizer_for(data)
    held, prompts = grpo_data(tokenizer, steps)
    policy = _linnet(data, workload, full=False)
    # The engine samples from a second copy compiled for serving: 64 cache
    # rows, each step a CUDA graph; it takes the policy's weights each step.
    engine = Engine(
        nest.load(
            data["directory"],
            backend="torch",
            device=workload.get("device", "cuda"),
            numerics="fast",
            compile=True,
            generics=_generics(data, 1024, batch=64),
            cast_dtype=True,
        )
    )

    def reward(prompt: Prompt, completion: list[int]) -> float:
        return gsm8k_reward(
            tokenizer.decode(completion, skip_special_tokens=True), str(prompt.data)
        )

    optimizer, schedule = _optimizer(policy, LR["grpo"], steps)
    _reset_peak()
    history = grpo(
        policy,
        engine,
        [Prompt(ids, answer) for _, ids, answer in prompts],
        reward,
        optimizer=optimizer,
        steps=steps,
        group=GROUP,
        prompts_per_step=PROMPTS,
        max_new_tokens=NEW_TOKENS,
        temperature=1.0,
        eos=eos_ids(data, tokenizer),
        tokens=TOKENS,
        schedule=schedule,
        correction_cap=GRPO_CORRECTION_CAP,
        seed=0,
    )
    peak = _peak_gib()
    metrics = _grpo_metrics(
        [step.sample_seconds + step.train_seconds for step in history],
        _grpo_tokens(prompts, [step.length for step in history]),
        [step.loss for step in history],
        [step.reward for step in history],
        mean_nll(scores(held, linnet_scorer(policy))),
        peak,
    )
    notes = _grpo_notes(steps) + (
        f"; sampled by linnet.serve.Engine (bf16, CUDA graphs) with truncated importance "
        f'sampling (cap {GRPO_CORRECTION_CAP:g}); policy compile="inductor"; peak includes the '
        "engine's copy"
    )
    return metrics, notes


def _grpo_trl(data: dict[str, Any], workload: dict[str, Any]) -> Measured:
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    steps = _steps(workload)
    tokenizer = tokenizer_for(data)
    held, prompts = grpo_data(tokenizer, steps)
    model = _hf(data, workload, full=False)

    def correct(completion_ids: list[list[int]], answer: list[str], **_: Any) -> list[float]:
        return [
            gsm8k_reward(tokenizer.decode(ids, skip_special_tokens=True), expected)
            for ids, expected in zip(completion_ids, answer, strict=True)
        ]

    optimizer, schedule = _optimizer(model, LR["grpo"], steps)
    clock = _clock()
    args = GRPOConfig(
        **_trl_common(workload, steps, GRPO_PER_PASS, PROMPTS * GROUP // GRPO_PER_PASS, LR["grpo"]),
        num_generations=GROUP,
        max_completion_length=NEW_TOKENS,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        generation_kwargs={"eos_token_id": eos_ids(data, tokenizer)},
        beta=0.0,
        epsilon=0.2,
        num_iterations=1,
        loss_type="dapo",  # the mean over every completion token of the step, as Linnet's
        scale_rewards="group",
        shuffle_dataset=False,
        use_vllm=False,
    )
    dataset = Dataset.from_list(
        [{"prompt": conversation, "answer": answer} for conversation, _, answer in prompts]
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=correct,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[clock],
        optimizers=(optimizer, schedule),
    )
    _reset_peak()
    trainer.train()
    peak = _peak_gib()
    model.eval()
    metrics = _grpo_metrics(
        clock.seconds,
        _grpo_tokens(prompts, _logged(trainer, "completions/mean_length")),
        _logged(trainer, "loss"),
        _logged(trainer, "reward"),
        mean_nll(scores(held, hf_scorer(model))),
        peak,
    )
    notes = _grpo_notes(steps) + (
        f"; sampled by transformers generate (vLLM lives in another environment here), "
        f"{GRPO_PER_PASS} completions a pass; " + _trl_notes(workload) + "; PEFT"
    )
    return metrics, notes


# ------------------------------------------------------------- gradients


def _match(theirs: dict[str, Any], name: str) -> Any:
    """The transformers parameter for a checkpoint tensor name: the same
    name, or the one name that ends with it (or it with)."""
    if name in theirs:
        return theirs[name]
    found = [p for key, p in theirs.items() if key.endswith("." + name) or name.endswith("." + key)]
    return found[0] if len(found) == 1 else None


def _grad_check(data: dict[str, Any], workload: dict[str, Any]) -> Measured:
    """One sequence of 128 tokens through the card (`loss_packed`, generated
    source) and transformers, both in f32 from the same checkpoint, and every
    parameter's gradient of the mean next-token loss compared. Linnet's run
    first, its gradients kept on the host by checkpoint name (a tied tensor's
    paths summed), so only one model is on the GPU at a time."""
    import gc
    import random

    import torch
    from linnet import nest
    from linnet.packing import Example
    from linnet.train import pack
    from transformers import AutoModelForCausalLM

    device = workload.get("device", "cuda")
    draws = random.Random(0)  # noqa: S311 - a reproducible sequence, not a secret
    high = min(int(data["generics"].get("Vocab", 20000)), 20000)
    ids = [draws.randrange(100, high) for _ in range(GRAD_TOKENS)]
    batch = next(pack([Example(ids)], GRAD_TOKENS))

    model = nest.load(
        data["directory"],
        backend="torch",
        device=device,
        compile=True,
        generics=_generics(data, GRAD_TOKENS, dtype="f32"),
        cast_dtype=True,
        trainable=True,
    )
    loss = model.run_entry("loss_packed", batch.inputs(batch.count, device))
    loss.backward()
    ours: dict[str, torch.Tensor] = {}
    for path, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        local = path.removeprefix("root.")
        name = model.weight_names.get(local, local)
        grad = parameter.grad.detach().float().cpu()
        ours[name] = ours[name] + grad if name in ours else grad
    linnet_loss = float(loss)
    del model, loss
    gc.collect()
    torch.cuda.empty_cache()

    weights = data["weights"]
    reference = AutoModelForCausalLM.from_pretrained(
        weights["repo"],
        revision=weights.get("revision"),
        dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)
    reference.eval()  # no dropout; gradients still flow
    tokens = torch.tensor([ids], device=device)
    out = reference(input_ids=tokens, labels=tokens, use_cache=False)
    out.loss.backward()
    theirs = dict(reference.named_parameters())
    errors: list[float] = []
    cosines: list[float] = []
    unmatched: list[str] = []
    for name, grad in ours.items():
        parameter = _match(theirs, name)
        if parameter is None or parameter.grad is None or parameter.grad.shape != grad.shape:
            unmatched.append(name)
            continue
        a = grad.to(device, torch.float64).flatten()
        b = parameter.grad.to(torch.float64).flatten()
        errors.append(float((a - b).abs().max() / b.abs().max().clamp_min(1e-30)))
        cosines.append(float(torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-30)))
    if not errors:
        raise RuntimeError("no Linnet gradient matched a transformers parameter")
    metrics: dict[str, float | None] = {
        "grad_max_rel": max(errors),
        "grad_cos": statistics.fmean(cosines),
        "grad_cos_min": min(cosines),
        "grad_params": float(len(errors)),
        "loss_abs_diff": abs(linnet_loss - float(out.loss)),
    }
    notes = (
        f"one sequence of {GRAD_TOKENS} random tokens, f32, transformers eager attention; "
        "a parameter's error is its largest absolute difference over its largest reference "
        "magnitude"
    )
    if unmatched:
        notes += f"; {len(unmatched)} unmatched, e.g. {unmatched[0]}"
    return metrics, notes


# ---------------------------------------------------------------- methods


def _both(key: str, title: str, entries: Sequence[str], linnet: Any, trl: Any) -> dict[str, Method]:
    trainer = {"sft": "SFTTrainer", "dpo": "DPOTrainer", "grpo": "GRPOTrainer"}[key.split("-")[0]]
    return {
        f"train-{key}-linnet": _row(f"Linnet torch ({title})", "linnet", entries, linnet),
        f"train-{key}-trl": _row(f"TRL {trainer} ({title})", "reference", entries, trl),
    }


_PACKED = ("loss_packed", "log_probs_packed")

METHODS: dict[str, Method] = {
    **_both(
        "sft-lora",
        "SFT, LoRA",
        _PACKED,
        functools.partial(_sft_linnet, False),
        functools.partial(_sft_trl, False),
    ),
    **_both(
        "sft-full",
        "SFT, full",
        _PACKED,
        functools.partial(_sft_linnet, True),
        functools.partial(_sft_trl, True),
    ),
    **_both("dpo", "DPO, LoRA", ("log_probs_packed",), _dpo_linnet, _dpo_trl),
    **_both(
        "grpo",
        "GRPO, LoRA",
        ("log_probs_packed", "prefill_slots", "decode_rows"),
        _grpo_linnet,
        _grpo_trl,
    ),
    "train-grad-check": _row(
        "Linnet torch against transformers (f32 gradients)",
        "linnet",
        ("loss_packed",),
        _grad_check,
    ),
}
