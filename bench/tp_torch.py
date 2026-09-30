"""One process of Linnet torch's tensor-parallel row: `torchrun
--nproc-per-node N -m bench.tp_torch <model> <workload json>`. Every process
loads the card split across the GPUs (`tensor_parallel=`), runs the same
prompt and decoding steps, and the first prints the result line the harness
reads."""

from __future__ import annotations

import gc
import json
import os
import statistics
import sys
import time
from dataclasses import asdict

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from bench.harness import Result, card, median_ms
from bench.methods.llm import cache_generics, max_seq, prompt_ids


def main() -> None:
    model_name, workload = sys.argv[1], json.loads(sys.argv[2])
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    mesh = init_device_mesh("cuda", (world,))
    data = card(model_name)
    from linnet import nest

    start = time.perf_counter()
    model = nest.load(
        data["directory"],
        backend="torch",
        device=f"cuda:{rank}",
        numerics="fast",
        # Each step captured as one CUDA graph, its all-reduces inside:
        # eager generated code would run every op through DTensor's Python
        # dispatch, and inductor alone launches each kernel from the host.
        compile="reduce-overhead",
        generics=cache_generics(data, max_seq(workload)),
        cast_dtype=True,
        tensor_parallel=mesh,
    )
    load_s = time.perf_counter() - start
    device = f"cuda:{rank}"
    ids = torch.tensor([prompt_ids(data, workload)], dtype=torch.int32, device=device)
    new, length = int(workload["new_tokens"]), ids.shape[1]
    zero = torch.tensor(0, dtype=torch.int32, device=device)

    def sync() -> None:
        torch.cuda.synchronize()
        dist.barrier()

    def prefill() -> torch.Tensor:
        return model.run_entry("prefill", [ids, zero])

    def generate() -> float:
        token = prefill().argmax(-1).reshape(1, 1).to(torch.int32)
        sync()
        begin = time.perf_counter()
        for step in range(new - 1):
            pos = torch.tensor(length + step, dtype=torch.int32, device=device)
            token = model.run_entry("decode", [token, pos]).argmax(-1).reshape(1, 1)
            token = token.to(torch.int32)
        sync()
        return (new - 1) / (time.perf_counter() - begin)

    with torch.no_grad():
        for _ in range(int(workload["warmup"])):
            generate()
        ttft = median_ms(lambda: prefill().argmax(-1).item(), sync, 0, int(workload["iters"]))
        rate = statistics.median(generate() for _ in range(max(3, int(workload["iters"]) // 3)))
    if rank == 0:
        result = Result(
            f"Linnet torch (tensor parallel on {world} GPUs, DTensor)",
            "linnet",
            {"ttft_ms": ttft, "decode_tok_s": rate, "load_s": load_s},
            notes=f"one process per GPU under torchrun, NCCL, CUDA graphs; KV cache compiled for "
            f"{max_seq(workload)} positions",
            key="linnet-tp-torch",
        )
        print("RESULT " + json.dumps(asdict(result)), flush=True)
    # The captured graphs hold the process group's communicators: let go of
    # them first, or tearing the group down waits on them.
    model = None
    gc.collect()
    torch.cuda.synchronize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
