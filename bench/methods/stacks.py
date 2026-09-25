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


def triton(make: Make, linnet: bool) -> Callable[[dict[str, Any], dict[str, Any]], Result]:
    """Triton Inference Server's ONNX Runtime backend over one of the two
    exports, a model per batch size. Latency is one request at a time at the
    small batch; throughput keeps four requests at the large batch in
    flight, which is what a client of a busy server does."""

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
            directory.mkdir(parents=True)
            (root / f"b{batch}" / "config.pbtxt").write_text(TRITON_ONNX, encoding="utf-8")
            if linnet:
                exported = adapter.linnet(batch)
                exported.save(directory / "model.onnx")
                names[batch] = [port.name for port in exported.inputs]
            else:
                names[batch] = _export_reference(adapter, batch, directory / "model.onnx")
        start = time.perf_counter()
        with server.server(root) as url:
            load_s = time.perf_counter() - start
            host = url.removeprefix("http://")
            clients = [httpclient.InferenceServerClient(host) for _ in range(4)]

            def request(client: Any, batch: int) -> Any:
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
            latency = median_ms(lambda: request(clients[0], low), lambda: None, warmup, iters)
            server.closed_loop(lambda i: request(clients[i % 4], high), 8, 4)
            count = max(16, 4 * iters)
            seconds, _ = server.closed_loop(lambda i: request(clients[i % 4], high), count, 4)
            output = request(clients[0], low)
        adapter.save("triton-linnet-onnx" if linnet else "triton-onnx", output)
        which = "Linnet ONNX" if linnet else "torch.onnx export"
        return Result(
            f"Triton Inference Server (ONNX Runtime backend, {which})",
            "linnet" if linnet else "reference",
            {
                "latency_ms": latency,
                "throughput_per_s": high * count / seconds,
                "load_s": load_s,
            },
            notes=f"over HTTP; f32; throughput with 4 requests of batch {high} in flight",
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
