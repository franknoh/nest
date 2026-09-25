"""One method, one model, one process: `python -m bench.worker <model>
<method> <workload json>`. Prints a single `RESULT {...}` line.

The family of methods comes from the card's tags, so a text-generation card
is measured as text generation whatever its architecture."""

from __future__ import annotations

import json
import sys
import traceback
from importlib import import_module
from typing import Any

from bench.harness import MODELS, Result, card, emit

FAMILIES = {
    "text-generation": "bench.methods.llm",
    "diffusion": "bench.methods.diffusion",
    "text-encoder": "bench.methods.text_encoder",
    "feature-extraction": "bench.methods.text_encoder",
    "image-classification": "bench.methods.vision",
    "image-feature-extraction": "bench.methods.vision",
    "image-text": "bench.methods.vision",
    "automatic-speech-recognition": "bench.methods.speech",
    "image-segmentation": "bench.methods.segmentation",
}


def family_of(data: dict[str, Any]) -> str:
    tags = set(data["model"].get("tags", []))
    for tag, module in FAMILIES.items():
        if tag in tags:
            return module
    raise SystemExit(f"no benchmark family for tags {sorted(tags)}")


def main() -> None:
    model, method, workload_text = sys.argv[1], sys.argv[2], sys.argv[3]
    workload: dict[str, Any] = json.loads(workload_text)
    data = card(model)
    module = import_module(family_of(data))
    if method == "sample":
        # What the model produces, for the page: written next to the card,
        # in its own process like any method so its memory is its own.
        shown = module.sample(data, workload)
        path = MODELS / model / "samples.json"
        path.write_text(json.dumps(shown, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        emit(Result("sample", "sample", notes=str(path.relative_to(MODELS.parent))))
        return
    methods: dict[str, Any] = module.METHODS
    if "onnx" in method:
        # onnxruntime-gpu is built for CUDA 12 and this environment's PyTorch
        # for CUDA 13; the CUDA 12 libraries come from pip packages, which the
        # runtime only finds when asked. Without them it quietly uses the CPU.
        try:
            import onnxruntime  # type: ignore[import-untyped]

            onnxruntime.preload_dlls()
        except Exception:  # noqa: BLE001, S110 - a CPU dry run has nothing to load
            pass
    if method not in methods:
        raise SystemExit(f"{module.__name__} has no method {method!r}")
    try:
        result: Result = methods[method](data, workload)
    except Exception as error:  # noqa: BLE001 - one failing method is a row, not a crash
        traceback.print_exc()
        result = Result(method, "unknown", error=f"{type(error).__name__}: {error}"[:400])
    if result.error is None:
        allocator_peak(result)
    emit(result)


def allocator_peak(result: Result) -> None:
    """For a method that ran on JAX, the peak its allocator had in use.

    Without preallocation JAX's allocator grows its pool in large regions and
    keeps them, so what the driver reports -- the peak every other row shows --
    can be twice what the model needed, in steps of whole regions. The
    allocator's own peak is the comparable number; the driver's is kept
    beside it (`driver_vram_mib`)."""
    if "jax" not in sys.modules:
        return
    import jax

    try:
        peaks = [d.memory_stats().get("peak_bytes_in_use", 0) for d in jax.local_devices()]
    except Exception:  # noqa: BLE001 - a CPU run has no memory stats
        return
    if any(peaks) and jax.default_backend() == "gpu":
        result.metrics["peak_vram_mib"] = round(sum(peaks) / 2**20, 1)
        result.notes = (result.notes + "; " if result.notes else "") + (
            "peak memory is JAX's allocator peak; the driver shows its pool, which grows in "
            "whole regions"
        )


if __name__ == "__main__":
    main()
