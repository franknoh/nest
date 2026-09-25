"""Benchmarks a model against the stacks people already use, and writes
`models/<name>/bench.json` for the site to draw.

    python -m bench.run qwen2.5-0.5b-instruct
    python -m bench.run qwen3-8b --methods linnet-torch,transformers-eager
    python -m bench.run tinyllama-1.1b-chat --device cpu --prompt 16 --new 4   # a dry run

Methods run one at a time, each in its own process (see `bench.harness`).
After all of them, every saved first-token output is compared with the
reference method's, so each row carries its distance from the reference
beside its speed.
"""

from __future__ import annotations

import argparse
import tempfile
from importlib import import_module
from pathlib import Path
from typing import Any

from bench.harness import Result, card, python_for, run_isolated, write
from bench.worker import family_of


def compare(results: list[Result], workdir: Path, reference: str, keys: dict[str, str]) -> None:
    """Fills in each result's `max_abs_diff` from the outputs its method
    saved, against the reference method's."""
    import numpy as np

    expected_path = workdir / f"{reference}.npy"
    if not expected_path.exists():
        return
    expected = np.load(expected_path)
    for result in results:
        path = workdir / f"{keys[result.method]}.npy"
        if path.exists() and result.error is None:
            got = np.load(path)
            if got.shape == expected.shape:
                result.max_abs_diff = float(np.abs(got - expected).max())


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m bench.run")
    parser.add_argument("models", nargs="+")
    parser.add_argument("--methods", help="comma-separated subset, in order")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", type=int, default=512, help="prompt tokens (text generation)")
    parser.add_argument("--new", type=int, default=128, help="new tokens (text generation)")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=3600.0, help="seconds per method")
    parser.add_argument(
        "--offload-gib",
        type=float,
        default=8.0,
        help="the GPU cap for `linnet-offload`, so part of the model streams from the host",
    )
    parser.add_argument(
        "--samples",
        choices=("also", "only", "skip"),
        default="also",
        help="run the model's sample for its page as well, instead, or not",
    )
    args = parser.parse_args()

    for model in args.models:
        data = card(model)
        family = import_module(family_of(data))
        # A family may leave out, per card, the methods that cannot apply to
        # it, so the page shows fewer rows rather than rows that only fail.
        if hasattr(family, "methods_for"):
            applicable: list[str] = family.methods_for(data)
        else:
            applicable = list(family.METHODS)
        methods: list[str] = args.methods.split(",") if args.methods else applicable
        workdir = Path(tempfile.mkdtemp(prefix=f"nest-bench-{model}-"))
        workload: dict[str, Any] = {
            "device": args.device,
            "prompt_tokens": args.prompt,
            "new_tokens": args.new,
            "batch": args.batch,
            "warmup": args.warmup,
            "iters": args.iters,
            "workdir": str(workdir),
            "seed": 0,
            "offload_gib": args.offload_gib,
        }
        results: list[Result] = []
        keys: dict[str, str] = {}
        if args.samples != "skip":
            print(f"[{model}] sample ...", flush=True)
            shown = run_isolated(python_for("sample"), model, "sample", workload, args.timeout)
            print(f"[{model}] sample: {shown.error or shown.notes}", flush=True)
        if args.samples == "only":
            continue
        for method in methods:
            print(f"[{model}] {method} ...", flush=True)
            result = run_isolated(python_for(method), model, method, workload, args.timeout)
            result.key = method
            if result.kind == "unknown":  # a crash reports no kind of its own
                result.kind = "linnet" if method.startswith("linnet") else "reference"
            keys[result.method] = method
            status = result.error or ", ".join(
                f"{key}={value:.4g}" for key, value in result.metrics.items() if value is not None
            )
            print(f"[{model}] {method}: {status}", flush=True)
            results.append(result)
        compare(results, workdir, family.REFERENCE, keys)
        public = {key: value for key, value in workload.items() if key != "workdir"}
        path = write(model, public, results, {"reference": family.REFERENCE})
        print(f"[{model}] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
