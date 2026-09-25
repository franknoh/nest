"""Text encoders: one forward pass, no autoregression. Latency at batch 1
and throughput at batch 64, for the reference stack and every Linnet
backend, plus a semantic-similarity sample.

Every card in this family compiles its attention from `std.nn.attention::
attention`, which takes an unbatched `[Q, K]` mask -- there is no way to
express a per-sequence padding mask. Every method here is therefore run on
unpadded batches of exactly the workload's sequence length (each sequence
real, none of them padding), which is also the only input shape these
cards accept exactly: a padded batch would attend onto the padding (or, for
ModernBERT, has no mask input to attend selectively at all). The sequence
length is 128, except for ModernBERT-base, whose sliding-window layers only
do anything interesting past a window (256, twice its 128-position window).

Each method also saves the last hidden state it computed at batch 1, so the
report can show how far every stack is from the reference (`bench.run`
compares whatever shape two methods agree on; a method computing something
else, such as sentence-transformers' pooled embedding, simply has nothing to
compare against).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bench.harness import Result, median_ms
from bench.methods import stacks
from bench.methods.stacks import Adapter

LATENCY_BATCH = 1
THROUGHPUT_BATCH = 64
DEFAULT_SEQ = 128
WINDOW_SEQ = 256  # long enough to exercise a sliding-window layer twice over
SAFE_TOKENS = (100, 20000)  # clear of the low ids every one of these vocabularies reserves


def seq_len(data: dict[str, Any]) -> int:
    """128 positions, or 256 for a card with a sliding-window layer (only
    ModernBERT here), so its local-window attention actually has more
    context than global attention would give it anyway."""
    return WINDOW_SEQ if "Window" in data["generics"] else DEFAULT_SEQ


def has_token_types(data: dict[str, Any]) -> bool:
    """Whether the card's `forward` takes a segment-id tensor (every card
    here except ModernBERT, which has no token-type embedding at all)."""
    return "TypeVocab" in data["generics"]


def is_sentence_embedder(data: dict[str, Any]) -> bool:
    """Whether the card publishes a trained sentence-embedding `embed`
    entry (only all-MiniLM-L6-v2); the others are encoders whose pooled
    output nobody trained to mean anything in particular."""
    return "sentence-similarity" in data["model"].get("tags", [])


def dtype_generic(device: str) -> str:
    return "bf16" if device == "cuda" else "f32"


def random_ids(
    data: dict[str, Any], workload: dict[str, Any], batch: int, seq: int
) -> list[list[int]]:
    """`batch` unpadded sequences of `seq` real token ids, from a fixed seed
    per shape: content does not change the arithmetic, only shape does."""
    import random

    vocab = int(data["generics"].get("Vocab", SAFE_TOKENS[1]))
    high = min(vocab, SAFE_TOKENS[1])
    rng = random.Random(f"{workload.get('seed', 0)}:{batch}:{seq}")  # noqa: S311 - reproducible, not a secret
    return [[rng.randrange(SAFE_TOKENS[0], high) for _ in range(seq)] for _ in range(batch)]


def save_output(workload: dict[str, Any], method: str, tensor: Any) -> None:
    import numpy as np

    directory = Path(workload["workdir"])
    directory.mkdir(parents=True, exist_ok=True)
    array = tensor.float().cpu().numpy() if hasattr(tensor, "float") else np.asarray(tensor)
    np.save(directory / f"{method}.npy", array.astype("float32").reshape(-1))


# ------------------------------------------------------------ reference stacks


def _transformers(compiled: bool) -> Callable[[dict[str, Any], dict[str, Any]], Result]:
    def run(data: dict[str, Any], workload: dict[str, Any]) -> Result:
        import torch
        from transformers import AutoModel

        from bench.harness import torch_sync

        name = "transformers (torch.compile)" if compiled else "transformers (eager)"
        method = "transformers-compile" if compiled else "transformers-eager"
        device = workload.get("device", "cuda")
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        seq = seq_len(data)
        types = has_token_types(data)
        warmup = int(workload["warmup"])
        iters = int(workload["iters"])
        sync = torch_sync()

        start = time.perf_counter()
        model = (
            AutoModel.from_pretrained(
                data["weights"]["repo"], torch_dtype=dtype, attn_implementation="eager"
            )
            .to(device)
            .eval()
        )
        if compiled:
            model.forward = torch.compile(model.forward)
        load_s = time.perf_counter() - start

        ids_lat = torch.tensor(random_ids(data, workload, LATENCY_BATCH, seq), device=device)
        ids_thr = torch.tensor(random_ids(data, workload, THROUGHPUT_BATCH, seq), device=device)
        tt_lat = torch.zeros_like(ids_lat) if types else None
        tt_thr = torch.zeros_like(ids_thr) if types else None

        def call(ids: torch.Tensor, tt: torch.Tensor | None) -> torch.Tensor:
            with torch.no_grad():
                kwargs = {} if tt is None else {"token_type_ids": tt}
                return model(ids, **kwargs).last_hidden_state

        latency_ms = median_ms(lambda: call(ids_lat, tt_lat), sync, warmup, iters)
        throughput_per_s = (
            THROUGHPUT_BATCH
            * 1000.0
            / median_ms(lambda: call(ids_thr, tt_thr), sync, warmup, iters)
        )
        save_output(workload, method, call(ids_lat, tt_lat)[0])
        return Result(
            name,
            "reference",
            {"latency_ms": latency_ms, "throughput_per_s": throughput_per_s, "load_s": load_s},
            notes=f"unpadded batches of exactly {seq} tokens (this card takes no padding mask)",
        )

    return run


FILLER_TEXT = (
    "The history of computing is a long chain of small, deliberate improvements: "
    "faster transistors, better compilers, smarter caches, and algorithms tuned to "
    "the machines that run them, each building on the last so that today's models "
    "can read a sentence and place it in a space where meaning determines distance."
)


def sentence_transformers_reference(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    """The library this checkpoint is actually published for. Its own
    tokenizer, padding, and pooling do the work, so it is not run on the
    fixed random-token workload the other rows share -- only offered for
    all-MiniLM-L6-v2, the one card with a trained sentence embedding."""
    name = "sentence-transformers (SentenceTransformer.encode)"
    if not is_sentence_embedder(data):
        return Result(
            name,
            "reference",
            notes="offered only for all-MiniLM-L6-v2, the sentence-transformers checkpoint",
        )

    import torch
    from sentence_transformers import SentenceTransformer

    from bench.harness import torch_sync

    device = workload.get("device", "cuda")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    warmup = int(workload["warmup"])
    iters = int(workload["iters"])
    sync = torch_sync()

    start = time.perf_counter()
    model = SentenceTransformer(
        data["weights"]["repo"], device=device, model_kwargs={"torch_dtype": dtype}
    )
    load_s = time.perf_counter() - start
    tokens = len(model.tokenizer(FILLER_TEXT)["input_ids"])

    def call(batch: int) -> Any:
        return model.encode(
            [FILLER_TEXT] * batch, batch_size=batch, convert_to_numpy=True, show_progress_bar=False
        )

    latency_ms = median_ms(lambda: call(LATENCY_BATCH), sync, warmup, iters)
    throughput_per_s = (
        THROUGHPUT_BATCH * 1000.0 / median_ms(lambda: call(THROUGHPUT_BATCH), sync, warmup, iters)
    )
    save_output(workload, "sentence-transformers", call(LATENCY_BATCH)[0])
    return Result(
        name,
        "reference",
        {"latency_ms": latency_ms, "throughput_per_s": throughput_per_s, "load_s": load_s},
        notes=(
            f"natural text tokenizing to {tokens} tokens, identical across the batch (no padding "
            "wasted); not the fixed-length random-token workload the other rows use, since this "
            "is the library's own string-in interface"
        ),
    )


# ------------------------------------------------------------------- Linnet


def _linnet_torch(compile: bool | str) -> Callable[[dict[str, Any], dict[str, Any]], Result]:
    label = {True: "generated source", "reduce-overhead": "CUDA graphs"}[compile]
    method = {True: "linnet-torch", "reduce-overhead": "linnet-cudagraphs"}[compile]

    def run(data: dict[str, Any], workload: dict[str, Any]) -> Result:
        import torch
        from linnet import nest

        from bench.harness import torch_sync

        name = f"Linnet torch ({label})"
        device = workload.get("device", "cuda")
        if compile == "reduce-overhead" and device != "cuda":
            return Result(name, "linnet", notes="CUDA graphs need a GPU (skip on CPU)")

        dtype = dtype_generic(device)
        seq = seq_len(data)
        types = has_token_types(data)
        warmup = int(workload["warmup"])
        iters = int(workload["iters"])
        sync = torch_sync()

        start = time.perf_counter()
        model = nest.load(
            data["directory"],
            backend="torch",
            device=device,
            numerics="fast",
            compile=compile,
            generics={**data["generics"], "T": dtype},
            cast_dtype=dtype != "f32",
        )
        load_s = time.perf_counter() - start

        ids_lat = torch.tensor(
            random_ids(data, workload, LATENCY_BATCH, seq), dtype=torch.int32, device=device
        )
        ids_thr = torch.tensor(
            random_ids(data, workload, THROUGHPUT_BATCH, seq), dtype=torch.int32, device=device
        )
        tt_lat = torch.zeros_like(ids_lat) if types else None
        tt_thr = torch.zeros_like(ids_thr) if types else None

        def call(ids: torch.Tensor, tt: torch.Tensor | None) -> torch.Tensor:
            return model(ids, tt) if tt is not None else model(ids)

        latency_ms = median_ms(lambda: call(ids_lat, tt_lat), sync, warmup, iters)
        throughput_per_s = (
            THROUGHPUT_BATCH
            * 1000.0
            / median_ms(lambda: call(ids_thr, tt_thr), sync, warmup, iters)
        )
        save_output(workload, method, call(ids_lat, tt_lat)[0])
        return Result(
            name,
            "linnet",
            {"latency_ms": latency_ms, "throughput_per_s": throughput_per_s, "load_s": load_s},
            notes=f"unpadded batches of exactly {seq} tokens; {dtype} on {device}",
        )

    return run


def linnet_jax(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    import jax
    import jax.numpy as jnp
    from linnet import nest

    device = workload.get("device", "cuda")
    seq = seq_len(data)
    types = has_token_types(data)
    warmup = int(workload["warmup"])
    iters = int(workload["iters"])

    dtype = dtype_generic(device)
    start = time.perf_counter()
    model = nest.load(
        data["directory"],
        backend="jax",
        numerics="fast",
        generics={**data["generics"], "T": dtype},
        cast_dtype=dtype != "f32",
    )
    load_s = time.perf_counter() - start

    ids_lat = jnp.asarray(random_ids(data, workload, LATENCY_BATCH, seq), dtype=jnp.int32)
    ids_thr = jnp.asarray(random_ids(data, workload, THROUGHPUT_BATCH, seq), dtype=jnp.int32)
    tt_lat = jnp.zeros_like(ids_lat) if types else None
    tt_thr = jnp.zeros_like(ids_thr) if types else None

    def call(ids: Any, tt: Any) -> Any:
        out = model(ids, tt) if tt is not None else model(ids)
        return jax.block_until_ready(out)

    latency_ms = median_ms(lambda: call(ids_lat, tt_lat), lambda: None, warmup, iters)
    throughput_per_s = (
        THROUGHPUT_BATCH
        * 1000.0
        / median_ms(lambda: call(ids_thr, tt_thr), lambda: None, warmup, iters)
    )
    save_output(workload, "linnet-jax", jax.device_get(call(ids_lat, tt_lat))[0])
    return Result(
        f"Linnet JAX (XLA, {jax.default_backend()})",
        "linnet",
        {"latency_ms": latency_ms, "throughput_per_s": throughput_per_s, "load_s": load_s},
        notes=f"unpadded batches of exactly {seq} tokens; {dtype} on {device}",
    )


def linnet_onnx_export(data: dict[str, Any], batch: int) -> tuple[Any, str]:
    """The card's `forward` as ONNX at one batch size, and how its optional
    biases were compiled. The Q/K/V/output and MLP `Linear` blocks declare
    their bias optional (`std.nn.linear::Linear`); a static ONNX graph picks
    one branch at export time. Every card here has the biases in its
    checkpoint (ModernBERT's source never declares them, so this is a no-op
    for it), so "present" comes first and "absent" only if the checkpoint
    disagrees."""
    from linnet.compiler import LinnetError
    from linnet.nest import Card, download_weights
    from linnet.onnx import export_model

    card = Card.read(data["directory"])
    kwargs: dict[str, Any] = {
        "generics": {**data["generics"], "B": batch, "S": seq_len(data)},
        "weights": download_weights(card),
        "entry": card.entry,
        "root": card.root,
        "bindings": card.bindings_path,
        "numerics": "fast",
    }
    try:
        return export_model(card.source_path, optionals="present", **kwargs), "present"
    except LinnetError:
        return export_model(card.source_path, optionals="absent", **kwargs), "absent"


def linnet_onnx(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    import numpy as np
    import onnxruntime as ort

    device = workload.get("device", "cuda")
    seq = seq_len(data)
    types = has_token_types(data)
    warmup = int(workload["warmup"])
    iters = int(workload["iters"])
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )

    optionals = "present"

    def build(batch: int) -> tuple[Any, list[str]]:
        nonlocal optionals
        exported, optionals = linnet_onnx_export(data, batch)
        session = ort.InferenceSession(exported.model.SerializeToString(), providers=providers)
        return session, [port.name for port in exported.inputs]

    start = time.perf_counter()
    session_lat, names_lat = build(LATENCY_BATCH)
    session_thr, names_thr = build(THROUGHPUT_BATCH)
    load_s = time.perf_counter() - start

    def feed(names: list[str], batch: int) -> dict[str, np.ndarray]:
        ids = np.asarray(random_ids(data, workload, batch, seq), dtype=np.int32)
        arrays = [ids, np.zeros_like(ids)] if types else [ids]
        return dict(zip(names, arrays, strict=True))

    feed_lat = feed(names_lat, LATENCY_BATCH)
    feed_thr = feed(names_thr, THROUGHPUT_BATCH)

    latency_ms = median_ms(lambda: session_lat.run(None, feed_lat), lambda: None, warmup, iters)
    throughput_per_s = (
        THROUGHPUT_BATCH
        * 1000.0
        / median_ms(lambda: session_thr.run(None, feed_thr), lambda: None, warmup, iters)
    )
    save_output(workload, "linnet-onnx", session_lat.run(None, feed_lat)[0])
    return Result(
        f"Linnet ONNX -> ONNX Runtime ({providers[0]})",
        "linnet",
        {"latency_ms": latency_ms, "throughput_per_s": throughput_per_s, "load_s": load_s},
        notes=(
            f"unpadded batches of exactly {seq} tokens; runs in f32, the published checkpoint's "
            "own dtype -- export embeds its tensors as initializers as they are, with no cast; "
            f"optional Linear biases compiled as {optionals} (checkpoint-detected)"
        ),
    )


# ------------------------------------------------- ONNX, Triton, KerasHub


def adapter(data: dict[str, Any], workload: dict[str, Any]) -> Adapter:
    """This family for the shared rows in `bench.methods.stacks`: the
    reference's last hidden state, from unpadded token ids (and zero segment
    ids where the card has them)."""
    import numpy as np

    seq = seq_len(data)
    types = has_token_types(data)

    def reference(device: str) -> Any:
        import torch
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            data["weights"]["repo"], torch_dtype=torch.float32, attn_implementation="eager"
        )

        class Hidden(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = model

            def forward(self, ids: Any, segments: Any = None) -> Any:
                kwargs = {} if segments is None else {"token_type_ids": segments.long()}
                return self.model(ids.long(), **kwargs).last_hidden_state

        return Hidden().to(device).eval()

    def inputs(batch: int) -> list[Any]:
        ids = np.asarray(random_ids(data, workload, batch, seq), dtype=np.int32)
        return [ids, np.zeros_like(ids)] if types else [ids]

    return Adapter(
        reference=reference,
        inputs=inputs,
        linnet=lambda batch: linnet_onnx_export(data, batch)[0],
        save=lambda method, out: save_output(workload, method, np.asarray(out)[0]),
        batches=(LATENCY_BATCH, THROUGHPUT_BATCH),
        note=f"unpadded batches of exactly {seq} tokens",
    )


def _keras_load(keras_hub: Any, data: dict[str, Any]) -> Any:
    return keras_hub.models.Backbone.from_preset(
        f"hf://{data['weights']['repo']}", dtype="bfloat16"
    )


def _keras_call(model: Any, arrays: list[Any]) -> Any:
    import numpy as np

    ids = arrays[0]
    feed = {"token_ids": ids, "padding_mask": np.ones_like(ids, dtype=bool)}
    if "segment_ids" in getattr(model, "input", {}):
        feed["segment_ids"] = arrays[1] if len(arrays) > 1 else np.zeros_like(ids)
    return model.predict_on_batch(feed)["sequence_output"]


# Card families KerasHub converts from a Transformers checkpoint (BERT and
# RoBERTa, the MiniLM sentence encoder among them); ModernBERT it does not.
KERAS_FAMILIES = {"bert"}


METHODS: dict[str, Callable[[dict[str, Any], dict[str, Any]], Result]] = {
    "transformers-eager": _transformers(compiled=False),
    "transformers-compile": _transformers(compiled=True),
    "sentence-transformers": sentence_transformers_reference,
    "linnet-torch": _linnet_torch(True),
    "linnet-cudagraphs": _linnet_torch("reduce-overhead"),
    "linnet-jax": linnet_jax,
    "linnet-onnx": linnet_onnx,
    "keras-hub": stacks.keras_hub(adapter, _keras_load, _keras_call),
    "onnx-reference": stacks.onnx_reference(adapter),
    "triton-onnx": stacks.triton(adapter, linnet=False),
    "triton-linnet-onnx": stacks.triton(adapter, linnet=True),
}


def methods_for(data: dict[str, Any]) -> list[str]:
    """Every method but KerasHub where it has no converter for the card, and
    sentence-transformers where the checkpoint is not one of its models."""
    chosen = []
    for name in METHODS:
        if name == "keras-hub" and data["model"].get("family") not in KERAS_FAMILIES:
            continue
        if name == "sentence-transformers" and not is_sentence_embedder(data):
            continue
        chosen.append(name)
    return chosen


REFERENCE = "transformers-eager"


# ------------------------------------------------------------------- sample

SAMPLE_SENTENCES = [
    "The cat curled up on the warm windowsill and fell asleep in the sun.",
    "Our new kitten refuses to eat anything except wet food.",
    "The central bank raised interest rates again to cool inflation.",
    "She rebalanced her portfolio after the market's sharp swings.",
    "The function threw an exception when the array index ran out of bounds.",
    "He refactored the module to remove three copies of the same logic.",
]


def _sentence_vector(
    model: Any, tokenizer: Any, sentence: str, *, types: bool, embed: bool, device: str
) -> list[float]:
    """One sentence through the card, batch 1 so its own real length needs
    no padding: `embed` where the card has it (already unit norm), otherwise
    the last hidden state mean-pooled over its tokens and L2-normalized."""
    import torch

    ids = torch.tensor([tokenizer(sentence)["input_ids"]], dtype=torch.int32, device=device)
    tt = torch.zeros_like(ids) if types or embed else None
    with torch.no_grad():
        if embed:
            vector = model.embed(ids, tt)[0]
        else:
            hidden = (model(ids, tt) if tt is not None else model(ids))[0]
            pooled = hidden.float().mean(dim=0)
            vector = pooled / pooled.norm()
    return vector.float().cpu().tolist()


def _cosine_matrix(vectors: list[list[float]]) -> list[list[float]]:
    import numpy as np

    unit = np.asarray(vectors, dtype=np.float64)
    matrix = unit @ unit.T
    return [[round(float(value), 6) for value in row] for row in matrix]


def sample(data: dict[str, Any], workload: dict[str, Any]) -> dict[str, Any]:
    """A semantic-similarity matrix over six sentences (two cats, two
    finance, two programming), from the card's own embedding through
    Linnet's generated-source backend, with the sentence-transformers
    library's matrix beside it for the one card trained to produce one."""
    from linnet import nest
    from transformers import AutoTokenizer

    device = workload.get("device", "cuda")
    types = has_token_types(data)
    embed = is_sentence_embedder(data)
    dtype = dtype_generic(device)

    tokenizer = AutoTokenizer.from_pretrained(data["weights"]["repo"])
    model = nest.load(
        data["directory"],
        backend="torch",
        device=device,
        numerics="fast",
        compile=True,
        generics={**data["generics"], "T": dtype},
        cast_dtype=dtype != "f32",
    )
    vectors = [
        _sentence_vector(model, tokenizer, sentence, types=types, embed=embed, device=device)
        for sentence in SAMPLE_SENTENCES
    ]

    result: dict[str, Any] = {
        "kind": "similarity",
        "sentences": SAMPLE_SENTENCES,
        "matrix": _cosine_matrix(vectors),
        "caption": (
            "The card's own `embed` entry: mean-pooled, L2-normalized sentence embeddings."
            if embed
            else (
                "Not a trained sentence embedder: the last hidden state, mean-pooled over "
                "tokens and L2-normalized by hand, since this checkpoint has no embedding head."
            )
        ),
    }
    if embed:
        from sentence_transformers import SentenceTransformer

        reference = SentenceTransformer(data["weights"]["repo"], device=device).encode(
            SAMPLE_SENTENCES, normalize_embeddings=True, show_progress_bar=False
        )
        result["reference_matrix"] = _cosine_matrix(reference.tolist())
        result["reference_stack"] = "sentence-transformers (SentenceTransformer.encode)"
    return result
