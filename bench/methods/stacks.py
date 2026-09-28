"""The rows every encoder family shares: ONNX Runtime on the reference
implementation's own ONNX export, Triton Inference Server over both that
export and Linnet's, and KerasHub on JAX where it can load the checkpoint.

A family describes a model with an `Adapter`, built per run from the card
and the workload: the reference module (whose output is what the family
compares), the inputs at a batch size, and Linnet's ONNX export at a batch
size. Both ONNX graphs run in f32 -- the
checkpoints are published in f32 and Linnet's exporter embeds them as they
are -- so the ONNX rows compare like with like.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.harness import Result, median_ms


@dataclass
class Adapter:
    reference: Callable[[str], Any]  # device -> an f32 torch module
    inputs: Callable[[int], list[Any]]  # batch -> numpy arrays, in order
    linnet: Callable[[int], Any]  # batch -> `linnet.onnx.export_model` result
    save: Callable[[str, Any], None]  # (method, output) -> saved for comparison
    batches: tuple[int, int]  # (latency batch, throughput batch)
    note: str = ""
    # For Triton's Python backend: the entry and the generics at a batch size.
    entry: Callable[[int], tuple[str, dict[str, int | str]]] | None = None


def _export_reference(adapter: Adapter, batch: int, path: Path) -> list[str]:
    """The reference module as ONNX at one batch size, f32; returns its
    input names."""
    import torch

    module = adapter.reference("cpu")
    arrays = adapter.inputs(batch)
    examples = tuple(torch.from_numpy(a) for a in arrays)
    names = [f"input_{i}" for i in range(len(arrays))]
    with torch.no_grad():
        try:
            torch.onnx.export(
                module, examples, str(path), input_names=names, output_names=["output"], dynamo=True
            )
        except Exception:  # noqa: BLE001 - the TorchScript exporter takes what dynamo cannot
            torch.onnx.export(
                module,
                examples,
                str(path),
                input_names=names,
                output_names=["output"],
                opset_version=18,
                dynamo=False,
            )
    return names


def _providers(device: str) -> list[str]:
    return ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]


Make = Callable[[dict[str, Any], dict[str, Any]], Adapter]


def onnx_reference(make: Make) -> Callable[[dict[str, Any], dict[str, Any]], Result]:
    def run(data: dict[str, Any], workload: dict[str, Any]) -> Result:
        import onnxruntime

        adapter = make(data, workload)
        device = workload.get("device", "cuda")
        low, high = adapter.batches
        work = Path(tempfile.mkdtemp(prefix="nest-onnx-"))
        start = time.perf_counter()
        sessions = []
        for batch in (low, high):
            path = work / f"b{batch}.onnx"
            names = _export_reference(adapter, batch, path)
            session = onnxruntime.InferenceSession(str(path), providers=_providers(device))
            sessions.append((session, [s.name for s in session.get_inputs()] or names))
        load_s = time.perf_counter() - start
        feeds = [
            dict(zip(names, adapter.inputs(batch), strict=True))
            for (_, names), batch in zip(sessions, (low, high), strict=True)
        ]
        warmup, iters = int(workload["warmup"]), int(workload["iters"])
        latency = median_ms(lambda: sessions[0][0].run(None, feeds[0]), lambda: None, warmup, iters)
        wide = median_ms(lambda: sessions[1][0].run(None, feeds[1]), lambda: None, warmup, iters)
        adapter.save("onnx-reference", sessions[0][0].run(None, feeds[0])[0])
        return Result(
            "transformers -> torch.onnx -> ONNX Runtime",
            "reference",
            {"latency_ms": latency, "throughput_per_s": high / (wide / 1e3), "load_s": load_s},
            notes="the reference model's own ONNX export, f32 like Linnet's; " + adapter.note,
        )

    return run


TRITON_ONNX = """platform: "onnxruntime_onnx"
max_batch_size: 0
instance_group [{ count: 1, kind: KIND_GPU }]
"""


PYTHON_INSTANCE = "\ninstance_group [{ count: 1, kind: KIND_GPU }]\n"


def triton(
    make: Make, linnet: bool, python: bool = False
) -> Callable[[dict[str, Any], dict[str, Any]], Result]:
    """Triton Inference Server over a model per batch size: its ONNX Runtime
    backend over the reference's export or Linnet's, or (`python`) its Python
    backend running Linnet's generated PyTorch (`linnet.triton.export`, bf16,
    CUDA graphs). Latency is one request at a time at the small batch;
    throughput keeps four requests at the large batch in flight, which is
    what a client of a busy server does."""

    def run(data: dict[str, Any], workload: dict[str, Any]) -> Result:
        import numpy as np
        import tritonclient.http as httpclient  # type: ignore[import-not-found]

        from bench import triton as server

        adapter = make(data, workload)
        low, high = adapter.batches
        root = Path(tempfile.mkdtemp(prefix="nest-triton-"))
        names: dict[int, list[str]] = {}
        for batch in (low, high):
            directory = root / f"b{batch}" / "1"
            if python:
                from linnet import triton as linnet_triton

                if adapter.entry is None:
                    raise RuntimeError("this family has no entry for Triton's Python backend")
                entry, generics = adapter.entry(batch)
                repository = linnet_triton.export(
                    data["directory"],
                    root,
                    name=f"b{batch}",
                    entry=entry,
                    generics=generics,
                    backend="python",
                    numerics="fast",
                    options={"cast_dtype": True, "compile": "reduce-overhead"},
                )
                with repository.config.open("a", encoding="utf-8") as config:
                    config.write(PYTHON_INSTANCE)
                names[batch] = [port.name for port in repository.inputs]
                continue
            directory.mkdir(parents=True)
            (root / f"b{batch}" / "config.pbtxt").write_text(TRITON_ONNX, encoding="utf-8")
            if linnet:
                exported = adapter.linnet(batch)
                exported.save(directory / "model.onnx")
                names[batch] = [port.name for port in exported.inputs]
            else:
                names[batch] = _export_reference(adapter, batch, directory / "model.onnx")
        start = time.perf_counter()
        # The Python backend imports Linnet from the benchmark's environment.
        site = os.environ.get("NEST_MAIN_SITE") if python else None
        with server.server(root, python_path=site) as url:
            load_s = time.perf_counter() - start
            host = url.removeprefix("http://")
            # One client per thread: the HTTP client runs on gevent, whose
            # connections cannot be handed from one thread to another.
            local = threading.local()

            def client_here() -> Any:
                if not hasattr(local, "client"):
                    local.client = httpclient.InferenceServerClient(host)
                return local.client

            def request(_unused: Any, batch: int) -> Any:
                client = client_here()
                arrays = adapter.inputs(batch)
                inputs = []
                for name, array in zip(names[batch], arrays, strict=True):
                    port = httpclient.InferInput(
                        name, list(array.shape), _triton_dtype(np.asarray(array).dtype)
                    )
                    port.set_data_from_numpy(np.ascontiguousarray(array), binary_data=True)
                    inputs.append(port)
                return client.infer(f"b{batch}", inputs).as_numpy(
                    client.get_model_metadata(f"b{batch}")["outputs"][0]["name"]
                )

            warmup, iters = int(workload["warmup"]), int(workload["iters"])
            latency = median_ms(lambda: request(None, low), lambda: None, warmup, iters)
            server.closed_loop(lambda i: request(None, high), 8, 4)
            count = max(16, 4 * iters)
            seconds, _ = server.closed_loop(lambda i: request(None, high), count, 4)
            output = request(None, low)
        key = (
            "triton-linnet-python" if python else "triton-linnet-onnx" if linnet else "triton-onnx"
        )
        adapter.save(key, output)
        if python:
            name = "Triton Inference Server (Python backend, Linnet torch)"
            precision = "bf16, CUDA graphs"
        else:
            which = "Linnet ONNX" if linnet else "torch.onnx export"
            name = f"Triton Inference Server (ONNX Runtime backend, {which})"
            precision = "f32"
        return Result(
            name,
            "linnet" if linnet or python else "reference",
            {
                "latency_ms": latency,
                "throughput_per_s": high * count / seconds,
                "load_s": load_s,
            },
            notes=f"over HTTP; {precision}; throughput with 4 requests of batch {high} in flight",
        )

    return run


def _triton_dtype(dtype: Any) -> str:
    return {"float32": "FP32", "int32": "INT32", "int64": "INT64", "float16": "FP16"}[str(dtype)]


def keras_hub(
    make: Make, load: Callable[[Any, dict[str, Any]], Any], call: Callable[[Any, list[Any]], Any]
) -> Callable[[dict[str, Any], dict[str, Any]], Result]:
    """KerasHub on its JAX backend, bf16 like the torch rows. `load(keras_hub,
    card)` builds the model; `call(model, arrays)` runs it on the adapter's
    inputs (through `predict_on_batch`, which is jitted) and returns the
    output the family compares."""

    def run(data: dict[str, Any], workload: dict[str, Any]) -> Result:
        os.environ.setdefault("KERAS_BACKEND", "jax")
        import jax

        # Before KerasHub: it imports TensorFlow, whose CUDA 12 libraries,
        # loaded first, leave JAX's CUDA 13 plugin unable to start -- it then
        # runs on the CPU without saying so.
        jax.devices()
        import keras_hub  # type: ignore[import-not-found]

        adapter = make(data, workload)
        low, high = adapter.batches
        start = time.perf_counter()
        model = load(keras_hub, data)
        load_s = time.perf_counter() - start
        small, large = adapter.inputs(low), adapter.inputs(high)

        def timed(arrays: list[Any]) -> Any:
            return call(model, arrays)  # predict_on_batch returns host arrays

        warmup, iters = int(workload["warmup"]), int(workload["iters"])
        latency = median_ms(lambda: timed(small), lambda: None, warmup, iters)
        wide = median_ms(lambda: timed(large), lambda: None, warmup, iters)
        adapter.save("keras-hub", timed(small))
        return Result(
            "KerasHub (JAX)",
            "reference",
            {"latency_ms": latency, "throughput_per_s": high / (wide / 1e3), "load_s": load_s},
            notes="bf16, under jax.jit; " + adapter.note,
        )

    return run


# ------------------------------------------------ Linnet on ONNX Runtime

ONNX_DTYPES = ("f32", "f16", "bf16")
ONNX_PROVIDERS = ("cuda", "trt")


def onnx_providers(provider: str, dtype: str, workload: dict[str, Any]) -> list[Any]:
    """ONNX Runtime's execution providers for a row. TensorRT builds an
    engine per graph, cached in the run's work directory; it runs the graph's
    own dtype (f16 graphs with its f16 kernels on)."""
    if workload.get("device", "cuda") == "cpu":
        return ["CPUExecutionProvider"]
    if provider == "trt":
        cache = Path(workload["workdir"]) / "trt-cache"
        cache.mkdir(parents=True, exist_ok=True)
        options = {
            "trt_engine_cache_enable": "True",
            "trt_engine_cache_path": str(cache),
            "trt_fp16_enable": "True" if dtype == "f16" else "False",
            "trt_bf16_enable": "True" if dtype == "bf16" else "False",
        }
        return [
            ("TensorrtExecutionProvider", options),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def onnx_key(dtype: str, provider: str) -> str:
    return f"linnet-onnx-{dtype}" + ("-trt" if provider == "trt" else "")


def linnet_onnx(
    run: Callable[[Any, dict[str, Any], dict[str, Any]], tuple[dict[str, float], Any]],
    save: Callable[[dict[str, Any], str, Any], None],
    dtype: str,
    provider: str,
    generics: Callable[[dict[str, Any], dict[str, Any]], dict[str, int | str]] | None = None,
    note: str = "",
) -> Callable[[dict[str, Any], dict[str, Any]], Result]:
    """Linnet's ONNX export on ONNX Runtime (`linnet.onnx.load_model`): the
    checkpoint cast to `dtype`, one copy of the weights on the GPU shared by
    every entry's session, state kept there. `run(model, card, workload)`
    times the family's entries and returns its metrics and the output the
    family compares; `generics` adds the card's own (a cache length)."""

    def method(data: dict[str, Any], workload: dict[str, Any]) -> Result:
        from linnet import nest

        extra = generics(data, workload) if generics is not None else {}
        start = time.perf_counter()
        model = nest.load(
            data["directory"],
            backend="onnx_model",
            generics={**extra, "T": dtype},
            cast_dtype=True,
            numerics="fast",
            providers=onnx_providers(provider, dtype, workload),
        )
        load_s = time.perf_counter() - start
        metrics, output = run(model, data, workload)
        # ONNX Runtime falls back to the next provider without saying so; a
        # TensorRT row that ran on CUDA would be a CUDA row under a wrong name.
        wanted = "TensorrtExecutionProvider" if provider == "trt" else "CUDAExecutionProvider"
        used = {s.session.get_providers()[0] for s in model._sessions.values()}
        if workload.get("device", "cuda") != "cpu" and used != {wanted}:
            raise RuntimeError(f"ONNX Runtime ran on {sorted(used)}, not {wanted}")
        metrics["load_s"] = load_s
        save(workload, onnx_key(dtype, provider), output)
        where = "TensorRT" if provider == "trt" else "CUDA"
        notes = (
            f"{dtype}; ONNX Runtime's {where} execution provider; first calls build the sessions"
        )
        return Result(
            f"Linnet ONNX {dtype} -> ONNX Runtime ({where})",
            "linnet",
            metrics,
            notes=notes + ("; " + note if note else ""),
        )

    return method


def onnx_methods(
    run: Callable[[Any, dict[str, Any], dict[str, Any]], tuple[dict[str, float], Any]],
    save: Callable[[dict[str, Any], str, Any], None],
    generics: Callable[[dict[str, Any], dict[str, Any]], dict[str, int | str]] | None = None,
    note: str = "",
) -> dict[str, Callable[[dict[str, Any], dict[str, Any]], Result]]:
    """The six ONNX rows: f32, f16, bf16, each on CUDA and on TensorRT."""
    return {
        onnx_key(dtype, provider): linnet_onnx(run, save, dtype, provider, generics, note)
        for dtype in ONNX_DTYPES
        for provider in ONNX_PROVIDERS
    }


def jax_variant(
    method: Callable[[dict[str, Any], dict[str, Any]], Result], label: str, key: str, **flags: Any
) -> Callable[[dict[str, Any], dict[str, Any]], Result]:
    """A family's JAX row on one of Linnet's two XLA paths: the StableHLO
    export (`linnet.jax.load`) or generated JAX source (`load_source`,
    `load_model`), chosen by `flags` the family's method reads from the
    workload."""

    def run(data: dict[str, Any], workload: dict[str, Any]) -> Result:
        result = method(data, {**workload, **flags, "jax_key": key})
        result.method = f"Linnet JAX ({label})"
        return result

    return run
