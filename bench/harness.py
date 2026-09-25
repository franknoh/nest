"""The measuring half of the Nest benchmarks: how a method is timed, how its
GPU memory is read, and the shape of what gets written down.

Every method runs in its own process. Frameworks disagree about memory --
PyTorch caches freed blocks, JAX and vLLM reserve pools up front -- so the
only fair number is what the driver says the process holds. The parent
process polls NVML for the child's usage and keeps the peak, the same way
for every framework. Running each method alone also keeps one framework's
caches, compiled kernels and allocator state out of the next one's numbers.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent  # the registry checkout
MODELS = ROOT / "models"


@dataclass
class Result:
    """One method's numbers for one model. Metrics a method cannot report
    stay `None` and are drawn as absent, never as zero."""

    method: str  # "transformers (eager)", "Linnet torch (CUDA graphs)", ...
    kind: str  # "reference" or "linnet"
    metrics: dict[str, float | None] = field(default_factory=dict)
    max_abs_diff: float | None = None  # against the reference's output
    notes: str = ""
    error: str | None = None
    key: str = ""  # the method's key in its family's METHODS, which `reference` names


def median_ms(fn: Callable[[], Any], sync: Callable[[], None], warmup: int, iters: int) -> float:
    """Median wall time of `fn` in milliseconds, synchronized after every
    call so asynchronous launches are not mistaken for finished work."""
    for _ in range(warmup):
        fn()
    sync()
    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def torch_sync() -> Callable[[], None]:
    import torch

    if torch.cuda.is_available():
        return torch.cuda.synchronize
    return lambda: None


class GpuPeak:
    """Peak GPU memory a worker holds, read from the driver every 10 ms.

    By process where the driver can name it; inside a container it cannot
    (NVML reports host PIDs, the worker knows its container PID), so there
    the peak is the growth of the memory in use on the GPUs the worker sees,
    over what they held when it started. Each method runs alone on its GPUs,
    so that growth is the worker's own, CUDA context included. Without a
    driver (a CPU dry run) the peak is unknown."""

    def __init__(self, pid: int, visible: str | None, interval: float = 0.01) -> None:
        self.pid = pid
        self.visible = visible
        self.interval = interval
        self.peak_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> GpuPeak:
        try:
            import pynvml  # type: ignore[import-untyped]

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            indices = (
                [int(i) for i in self.visible.split(",") if i.strip()]
                if self.visible
                else list(range(count))
            )
            handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in indices if i < count]
            baseline = sum(int(pynvml.nvmlDeviceGetMemoryInfo(h).used) for h in handles)
        except Exception:  # noqa: BLE001 - no driver, no numbers
            return self
        self._thread = threading.Thread(
            target=self._poll, args=(pynvml, handles, baseline), daemon=True
        )
        self._thread.start()
        return self

    def _poll(self, pynvml: Any, handles: list[Any], baseline: int) -> None:
        while not self._stop.is_set():
            by_pid = 0
            for handle in handles:
                for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                    if process.pid == self.pid and process.usedGpuMemory is not None:
                        by_pid += int(process.usedGpuMemory)
            growth = sum(int(pynvml.nvmlDeviceGetMemoryInfo(h).used) for h in handles) - baseline
            used = by_pid or growth
            if used > 0 and (self.peak_bytes is None or used > self.peak_bytes):
                self.peak_bytes = used
            time.sleep(self.interval)

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


def run_isolated(
    python: str, model: str, method: str, workload: dict[str, Any], timeout: float
) -> Result:
    """Runs one method in a fresh interpreter and returns its result, with
    the process's peak GPU memory filled in from the outside."""
    command = [python, "-m", "bench.worker", model, method, json.dumps(workload)]
    env = dict(os.environ)
    # JAX reserves 75% of the GPU at start unless told not to, which would
    # make its "peak" a setting rather than a measurement.
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # CUDA numbers GPUs fastest first by default, NVML by bus; the memory
    # reading maps one to the other, so both count by bus.
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    process = subprocess.Popen(  # noqa: S603 - our own interpreter and module
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    with GpuPeak(process.pid, env.get("CUDA_VISIBLE_DEVICES")) as peak:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            return Result(method, "unknown", error=f"timed out after {timeout:.0f} s")
    lines = [line for line in stdout.splitlines() if line.startswith("RESULT ")]
    if process.returncode != 0 or not lines:
        tail = (stderr.strip().splitlines() or ["no output"])[-1]
        return Result(method, "unknown", error=f"exit {process.returncode}: {tail[:300]}")
    result = Result(**json.loads(lines[-1][len("RESULT ") :]))
    if peak.peak_bytes is not None:
        result.metrics["peak_vram_mib"] = round(peak.peak_bytes / 2**20, 1)
    return result


def emit(result: Result) -> None:
    """How a worker hands its result to the parent: one tagged line."""
    print("RESULT " + json.dumps(asdict(result)), flush=True)


def environment() -> dict[str, str]:
    """What the numbers were measured on, recorded with them."""
    from importlib import metadata

    env = {"python": platform.python_version(), "machine": platform.machine()}
    distributions = {
        "torch": "torch",
        "jax": "jax",
        "transformers": "transformers",
        "vllm": "vllm",
        "diffusers": "diffusers",
        "onnxruntime": "onnxruntime-gpu",
        "linnet": "linnet-lang",
    }
    for name, distribution in distributions.items():
        try:
            env[name] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    try:
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        env["gpu"] = name.decode() if isinstance(name, bytes) else str(name)
        env["gpus"] = str(pynvml.nvmlDeviceGetCount())
        env["driver"] = str(pynvml.nvmlSystemGetDriverVersion())
    except Exception:  # noqa: BLE001 - a CPU dry run
        env["gpu"] = "none"
    return env


def card(model: str) -> dict[str, Any]:
    """The model's card, as `linnet.nest` reads it, plus its directory."""
    import tomllib

    directory = MODELS / model
    data = tomllib.loads((directory / "nest.toml").read_text(encoding="utf-8"))
    data["directory"] = str(directory)
    return data


def write(
    model: str, workload: dict[str, Any], results: list[Result], extra: dict[str, Any]
) -> Path:
    from datetime import UTC, datetime

    path = MODELS / model / "bench.json"
    document = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": environment(),
        "workload": workload,
        "methods": [asdict(result) for result in results],
        **extra,
    }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def python_for(method: str) -> str:
    """vLLM pins its own PyTorch, so it lives in its own environment; the
    pod's setup script exports where. Everything else runs here."""
    if method.startswith("vllm") and "NEST_VLLM_PYTHON" in os.environ:
        return os.environ["NEST_VLLM_PYTHON"]
    return sys.executable
