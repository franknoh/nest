"""Triton Inference Server for the benchmarks: a server started over a model
repository for one method and stopped after it, and a client that keeps a
fixed number of requests in flight.

The pod image is NVIDIA's `tritonserver` (see `setup-pod.sh`), which puts the
server at `/opt/tritonserver/bin/tritonserver`. Python-backend models import
from whatever environment `PYTHONPATH` names, so each method says which one:
vLLM's for the vLLM backend, ours for anything Linnet.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SERVER = os.environ.get("NEST_TRITON", "/opt/tritonserver/bin/tritonserver")
HTTP_PORT = 18000


def available() -> bool:
    return Path(SERVER).exists() or shutil.which("tritonserver") is not None


@contextmanager
def server(repository: Path, python_path: str | None = None, timeout: float = 900.0) -> Any:
    """A running server over `repository`, ready when this yields."""
    if not available():
        raise RuntimeError("no tritonserver here (the benchmark pod's image provides it)")
    env = dict(os.environ)
    if python_path is not None:
        env["PYTHONPATH"] = python_path
    log = open(repository / "server.log", "w", encoding="utf-8")  # noqa: SIM115 - closed below
    process = subprocess.Popen(  # noqa: S603 - the server binary, our own arguments
        [
            SERVER if Path(SERVER).exists() else "tritonserver",
            f"--model-repository={repository}",
            f"--http-port={HTTP_PORT}",
            f"--grpc-port={HTTP_PORT + 1}",
            f"--metrics-port={HTTP_PORT + 2}",
            "--log-verbose=0",
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + timeout
        while True:
            if process.poll() is not None:
                tail = (repository / "server.log").read_text(encoding="utf-8")[-800:]
                raise RuntimeError(f"tritonserver exited: {tail}")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{HTTP_PORT}/v2/health/ready", timeout=2
                ) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("tritonserver did not become ready")
            time.sleep(1.0)
        yield f"http://127.0.0.1:{HTTP_PORT}"
    finally:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()


def closed_loop(
    call: Callable[[int], Any], total: int, concurrency: int
) -> tuple[float, list[Any]]:
    """Runs `call(i)` for i in range(total) with `concurrency` calls in
    flight, each worker taking the next index as it finishes. Returns the
    wall time and the results in index order."""
    results: list[Any] = [None] * total
    lock = threading.Lock()
    counter = iter(range(total))
    errors: list[BaseException] = []

    def worker() -> None:
        while True:
            with lock:
                index = next(counter, None)
            if index is None or errors:
                return
            try:
                results[index] = call(index)
            except BaseException as error:  # noqa: BLE001 - reported below
                errors.append(error)
                return

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return time.perf_counter() - start, results
