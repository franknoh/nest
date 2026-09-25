"""Speech recognition: how long the encoder takes over one 30-second window
of audio, and how long greedy transcription of a real clip takes end to end,
for the reference stack and for each Linnet backend.

Every method reads the same real audio -- the first clip of
`hf-internal-testing/librispeech_asr_dummy` -- through `WhisperProcessor`, so
`encode_ms` times the same log-mel window everywhere and `transcribe_ms`
times the same sentence everywhere; only the number of decode steps is
content-driven, never chosen by us.

Whisper's `decode` entry has no state to cache (see the card's README): it
recomputes its whole prefix at every step. The reference decodes through
`generate`, which keeps a KV cache. That asymmetry is real, it is exactly
what `transcribe_ms` is measuring on the Linnet side, and every Linnet
method's `notes` says so rather than leaving it to be discovered in the
numbers.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bench.harness import Result, median_ms, torch_sync

CLIP = "hf-internal-testing/librispeech_asr_dummy"
NO_CACHE = (
    "Linnet's decode has no KV cache: it recomputes the whole token prefix "
    "every step, unlike the reference's generate()"
)


def _fixture(data: dict[str, Any]) -> tuple[Any, list[int], int, Any]:
    """The real clip's log-mel features, Whisper's transcription prompt, the
    end-of-transcript id, and the processor that built them -- shared by
    every method and by `sample`, so every stack times and transcribes
    exactly the same audio.

    The clip's bytes are decoded with `soundfile` directly rather than
    through `datasets`' own `Audio` feature, which now reaches for
    `torchcodec` -- a CUDA-built dependency that has no reason to be part of
    a CPU dry run's requirements."""
    import io

    import numpy as np
    import soundfile as sf
    from datasets import Audio, load_dataset
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained(data["weights"]["repo"])
    clips = load_dataset(CLIP, "clean", split="validation").cast_column(
        "audio", Audio(decode=False)
    )
    raw = clips[0]["audio"]
    array, sampling_rate = sf.read(io.BytesIO(raw["bytes"]))
    features = processor(array, sampling_rate=sampling_rate, return_tensors="np").input_features
    forced = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    start = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    prompt = [start] + [token for _, token in forced]
    eot = processor.tokenizer.eos_token_id
    return features.astype(np.float32), prompt, int(eot), processor


def _max_steps(data: dict[str, Any], prompt_len: int, workload: dict[str, Any]) -> int:
    """A safety cap on decode steps: the workload's `new_tokens`, or the
    card's `MaxTokens` less the prompt, whichever is smaller. Real content
    is short enough that the cap is not expected to bind."""
    cap = int(workload.get("new_tokens", 128))
    limit = int(data["generics"]["MaxTokens"]) - prompt_len
    return max(1, min(cap, limit))


def _feature_dtype(data: dict[str, Any], device: str) -> Any:
    import torch
    from linnet.torch.dtypes import TORCH_DTYPES

    if device == "cpu":
        return torch.float32
    return TORCH_DTYPES[data["generics"]["T"]]


def save_output(workload: dict[str, Any], method: str, tensor: Any) -> None:
    """The batch-1 encoder states, so `compare` can put every method's
    distance from the reference's beside its speed."""
    import numpy as np

    directory = Path(workload["workdir"])
    directory.mkdir(parents=True, exist_ok=True)
    array = tensor.float().cpu().numpy() if hasattr(tensor, "float") else np.asarray(tensor)
    np.save(directory / f"{method}.npy", array.astype("float32").reshape(-1))


# ------------------------------------------------------------ reference stack


def _transformers(compiled: bool) -> Callable[[dict[str, Any], dict[str, Any]], Result]:
    def run(data: dict[str, Any], workload: dict[str, Any]) -> Result:
        import torch
        from transformers import WhisperForConditionalGeneration

        device = workload.get("device", "cuda")
        name = "transformers (torch.compile, static cache)" if compiled else "transformers (eager)"
        method = "transformers-compile" if compiled else "transformers-eager"
        mel, prompt, _eot, _processor = _fixture(data)
        sync = torch_sync()
        dtype = torch.float32 if device == "cpu" else _feature_dtype(data, device)

        start = time.perf_counter()
        model = (
            WhisperForConditionalGeneration.from_pretrained(
                data["weights"]["repo"], torch_dtype=dtype, attn_implementation="sdpa"
            )
            .to(device)
            .eval()
        )
        model.generation_config.forced_decoder_ids = None
        if compiled:
            model.generation_config.cache_implementation = "static"
            # Default inductor mode, for the reason the LLM family gives.
            model.forward = torch.compile(model.forward, fullgraph=True)
            encoder = model.get_encoder()
            encoder.forward = torch.compile(encoder.forward, fullgraph=True)
        load_s = time.perf_counter() - start

        features = torch.tensor(mel, dtype=dtype, device=device)
        decoder_prompt = torch.tensor([prompt], device=device)
        steps = _max_steps(data, len(prompt), workload)

        def encode() -> torch.Tensor:
            with torch.no_grad():
                return model.get_encoder()(features).last_hidden_state

        def transcribe() -> torch.Tensor:
            with torch.no_grad():
                return model.generate(
                    input_features=features,
                    decoder_input_ids=decoder_prompt,
                    max_new_tokens=steps,
                    do_sample=False,
                    num_beams=1,
                )

        encode_ms = median_ms(encode, sync, int(workload["warmup"]), int(workload["iters"]))
        transcribe_ms = median_ms(transcribe, sync, int(workload["warmup"]), int(workload["iters"]))
        save_output(workload, method, encode()[0])
        return Result(
            name,
            "reference",
            {"encode_ms": encode_ms, "transcribe_ms": transcribe_ms, "load_s": load_s},
        )

    return run


# ------------------------------------------------------------------- Linnet


def _linnet_torch(compile: bool | str) -> Callable[[dict[str, Any], dict[str, Any]], Result]:
    label = {True: "generated source", "reduce-overhead": "CUDA graphs"}[compile]
    method = {True: "linnet-torch", "reduce-overhead": "linnet-cudagraphs"}[compile]

    def run(data: dict[str, Any], workload: dict[str, Any]) -> Result:
        import torch
        from linnet import nest

        device = workload.get("device", "cuda")
        if compile == "reduce-overhead" and device == "cpu":
            return Result(
                f"Linnet torch ({label})", "linnet", notes="CUDA graphs need a GPU; skipped on cpu"
            )
        sync = torch_sync()
        mel, prompt, eot, _processor = _fixture(data)
        dtype = _feature_dtype(data, device)

        start = time.perf_counter()
        model = nest.load(
            data["directory"], backend="torch", device=device, numerics="fast", compile=compile
        )
        load_s = time.perf_counter() - start

        features = torch.tensor(mel, dtype=dtype, device=device)
        prompt_ids = torch.tensor([prompt], dtype=torch.int32, device=device)
        steps = _max_steps(data, len(prompt), workload)

        def encode() -> torch.Tensor:
            return model.run_entry("encode", [features])

        def transcribe() -> torch.Tensor:
            states = model.run_entry("encode", [features])
            tokens = prompt_ids
            for _ in range(steps):
                logits = model.run_entry("decode", [tokens, states])
                next_token = logits[:, -1].argmax(-1, keepdim=True).to(torch.int32)
                tokens = torch.cat([tokens, next_token], dim=1)
                if int(next_token.item()) == eot:
                    break
            return tokens

        encode_ms = median_ms(encode, sync, int(workload["warmup"]), int(workload["iters"]))
        transcribe_ms = median_ms(transcribe, sync, int(workload["warmup"]), int(workload["iters"]))
        save_output(workload, method, encode()[0])
        return Result(
            f"Linnet torch ({label})",
            "linnet",
            {"encode_ms": encode_ms, "transcribe_ms": transcribe_ms, "load_s": load_s},
            notes=NO_CACHE,
        )

    return run


def linnet_jax(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    """`encode` and `decode` as two StableHLO exports sharing one copy of the
    weights on the device, the way `bench.methods.llm.linnet_jax` shares
    `prefill`'s and `decode`'s."""
    import jax
    import jax.numpy as jnp
    from linnet import nest
    from linnet.jax import load as load_jax

    mel, prompt, eot, _processor = _fixture(data)
    generics = dict(data["generics"])

    start = time.perf_counter()
    encode = nest.load(data["directory"], backend="jax", entry="encode", generics=generics)
    decode = load_jax(
        Path(data["directory"]) / data["source"]["path"],
        generics=encode.generics,
        weights=encode.weights,
        root=encode.root,
        entry="decode",
    )
    load_s = time.perf_counter() - start

    dtype = {"f32": jnp.float32, "f16": jnp.float16, "bf16": jnp.bfloat16}[data["generics"]["T"]]
    features = jnp.asarray(mel, dtype=dtype)
    prompt_ids = jnp.asarray([prompt], dtype=jnp.int32)
    steps = _max_steps(data, len(prompt), workload)

    def encode_call() -> Any:
        return encode(features)

    def transcribe() -> Any:
        states = encode_call()
        tokens = prompt_ids
        for _ in range(steps):
            logits = decode(tokens, states)
            next_token = jnp.argmax(logits[:, -1], axis=-1).reshape(1, 1).astype(jnp.int32)
            tokens = jnp.concatenate([tokens, next_token], axis=1)
            if int(next_token[0, 0]) == eot:
                break
        return tokens

    encode_ms = median_ms(
        lambda: jax.block_until_ready(encode_call()),
        lambda: None,
        int(workload["warmup"]),
        int(workload["iters"]),
    )
    # Transcription is timed only where every prefix length's program fits:
    # without a KV cache each new length is a new XLA program and its own
    # buffers, which exhausted a 94 GB GPU on large-v3's 32-layer decoder.
    metrics: dict[str, float | None] = {"encode_ms": encode_ms, "load_s": load_s}
    notes = NO_CACHE
    if int(data["generics"].get("DecoderLayers", 0)) <= 4:
        metrics["transcribe_ms"] = median_ms(
            lambda: jax.block_until_ready(transcribe()),
            lambda: None,
            int(workload["warmup"]),
            int(workload["iters"]),
        )
    else:
        notes += "; transcription not timed: one XLA program per prefix length without a cache"
    save_output(workload, "linnet-jax", jax.device_get(encode_call())[0])
    return Result(f"Linnet JAX (XLA, {jax.default_backend()})", "linnet", metrics, notes=notes)


def linnet_onnx(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    """`encode` alone: a plain forward pass over a fixed shape, unlike
    `decode`, whose shape grows by one position every step with no cache to
    keep it static, so only the encoder is exported."""
    import numpy as np
    import onnxruntime as ort
    from linnet.nest import Card, download_weights
    from linnet.onnx import export_model

    device = workload.get("device", "cuda")
    mel, _prompt, _eot, _processor = _fixture(data)
    card = Card.read(data["directory"])
    generics = {**dict(card.generics), **dict(card.check)}

    start = time.perf_counter()
    weights = download_weights(card)
    exported = export_model(
        card.source_path,
        generics=generics,
        weights=weights,
        entry="encode",
        root=card.root,
        bindings=card.bindings_path,
    )
    onnx_path = Path(workload["workdir"]) / f"{data['model']['name']}-encode.onnx"
    exported.save(onnx_path)
    providers = ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    load_s = time.perf_counter() - start

    input_name = exported.inputs[0].name
    onnx_dtype = np.float16 if data["generics"]["T"] == "f16" else np.float32
    features = mel.astype(onnx_dtype)

    def encode() -> Any:
        return session.run(None, {input_name: features})[0]

    encode_ms = median_ms(encode, lambda: None, int(workload["warmup"]), int(workload["iters"]))
    save_output(workload, "linnet-onnx", encode()[0])
    return Result(
        "Linnet ONNX -> ONNX Runtime",
        "linnet",
        {"encode_ms": encode_ms, "load_s": load_s},
        notes="encoder only: decode's shape grows every step with no cache, so it is not exported",
    )


METHODS: dict[str, Callable[[dict[str, Any], dict[str, Any]], Result]] = {
    "transformers-eager": _transformers(compiled=False),
    "transformers-compile": _transformers(compiled=True),
    "linnet-torch": _linnet_torch(True),
    "linnet-cudagraphs": _linnet_torch("reduce-overhead"),
    "linnet-jax": linnet_jax,
    "linnet-onnx": linnet_onnx,
}

REFERENCE = "transformers-eager"


# --------------------------------------------------------------------- sample


def sample(card: dict[str, Any], workload: dict[str, Any]) -> dict[str, Any]:
    """Transcribes the real clip with the best Linnet backend and with the
    reference stack, so the page can show they agree."""
    import torch
    from linnet import nest
    from transformers import WhisperForConditionalGeneration

    device = workload.get("device", "cuda")
    mel, prompt, eot, processor = _fixture(card)
    dtype = _feature_dtype(card, device)

    model = nest.load(card["directory"], backend="torch", device=device)
    features = torch.tensor(mel, dtype=dtype, device=device)
    tokens = torch.tensor([prompt], dtype=torch.int32, device=device)
    states = model.run_entry("encode", [features])
    steps = _max_steps(card, len(prompt), workload)
    for _ in range(steps):
        logits = model.run_entry("decode", [tokens, states])
        next_token = logits[:, -1].argmax(-1, keepdim=True).to(torch.int32)
        tokens = torch.cat([tokens, next_token], dim=1)
        if int(next_token.item()) == eot:
            break
    text = processor.tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True).strip()

    reference = (
        WhisperForConditionalGeneration.from_pretrained(
            card["weights"]["repo"], torch_dtype=torch.float32
        )
        .to(device)
        .eval()
    )
    reference.generation_config.forced_decoder_ids = None
    with torch.no_grad():
        reference_ids = reference.generate(
            input_features=torch.tensor(mel, dtype=torch.float32, device=device),
            decoder_input_ids=torch.tensor([prompt], device=device),
            max_new_tokens=steps,
            do_sample=False,
            num_beams=1,
        )
    reference_text = processor.batch_decode(reference_ids, skip_special_tokens=True)[0].strip()

    return {
        "kind": "transcription",
        "audio": f"{CLIP}, clean/validation[0]",
        "text": text,
        "reference_text": reference_text,
    }
