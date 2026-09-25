"""Diffusion: the Stable Diffusion VAE decoder and the SDXL denoising UNet,
against `diffusers` and for each Linnet backend.

Two cards share this family and are told apart by their `family` field:

- `sd-vae` (`sd-vae-ft-mse`, `decode`): decode latency for a 64x64 latent (a
  512-pixel image) at batch 1, and images per second at batch 8.
- `sdxl` (`sdxl-base-unet`, `forward`): one denoising step at SDXL's native
  128x128 latent (a 1024-pixel image), batch 2 -- the classifier-free
  guidance pair -- with 77-token text embeddings, as `step_ms`.

Both checkpoints are published in f32; on a GPU every method runs them in
bf16 (Linnet converts them as they are read, `cast_dtype=True`), on a CPU dry
run in f32 unless the workload says otherwise. Inputs are drawn from a fixed
seed on the host, so every method sees the same numbers, and each method
saves its first output for `compare`.

The SDXL card takes the timestep and SDXL's added conditioning as sinusoidal
embeddings, not as numbers; `sdxl_embeddings` computes them the way
`diffusers` does before entering its own UNet, and `LinnetUNet` wraps the
card in the call signature `StableDiffusionXLPipeline` expects, which is how
the sample runs the stock pipeline with the Linnet network inside.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bench.harness import Result, median_ms, torch_sync

VAE_REPO = "stabilityai/sd-vae-ft-mse"
SDXL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_SCALE = 0.18215  # the SD 1.x pipeline's latent scaling (the caller's job)
SDXL_TEXT = 77  # tokens of text context: the pipeline pads every prompt to this
SDXL_TIMESTEP = 500.0  # a mid-schedule step; the arithmetic does not depend on it
COCO_IMAGE = "http://images.cocodataset.org/val2017/000000039769.jpg"
SAMPLE_PROMPT = (
    "A red fox curled up asleep on a mossy boulder in a misty pine forest at dawn, "
    "soft golden light through the trees, dew on the moss, detailed fur, "
    "shallow depth of field, wildlife photograph"
)
SAMPLE_SEED = 1234
MAX_PNG_BYTES = 300_000


# ----------------------------------------------------------------- workload


def is_vae(data: dict[str, Any]) -> bool:
    return data["model"].get("family") == "sd-vae"


def device_of(workload: dict[str, Any]) -> str:
    return str(workload.get("device", "cuda"))


def dtype_name(workload: dict[str, Any]) -> str:
    """bf16 on a GPU, f32 on a CPU dry run, unless the workload names one."""
    if "dtype" in workload:
        return str(workload["dtype"])
    return "f32" if device_of(workload) == "cpu" else "bf16"


def torch_dtype(name: str) -> Any:
    import torch

    return {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[name]


def vae_side(workload: dict[str, Any]) -> int:
    """The latent's side: 64 (a 512-pixel image), 16 on a CPU dry run."""
    return int(workload.get("latent", 16 if device_of(workload) == "cpu" else 64))


def vae_batch(workload: dict[str, Any]) -> int:
    """The batch throughput is measured at: 8, 2 on a CPU dry run."""
    return int(workload.get("throughput_batch", 2 if device_of(workload) == "cpu" else 8))


def sdxl_mid(workload: dict[str, Any]) -> int:
    """The coarsest grid, a quarter of the latent: 32 is the native 128x128
    latent; 4 (a 16x16 latent) on a CPU dry run. Not 2: at a 4x4 grid the
    torch backend's `conv2d` reads the stride-2 downsampler's 4x4 -> 2x2 as
    stride 1 without padding (it recovers the window from the shapes), and
    the result is wrong."""
    return int(workload.get("mid", 4 if device_of(workload) == "cpu" else 32))


def sdxl_batch(workload: dict[str, Any]) -> int:
    return int(workload.get("step_batch", 2))


def vae_workload_note(workload: dict[str, Any]) -> str:
    side = vae_side(workload)
    return (
        f"{side}x{side} latent ({side * 8}-pixel image), {dtype_name(workload)}; "
        f"throughput at batch {vae_batch(workload)}"
    )


def sdxl_workload_note(workload: dict[str, Any]) -> str:
    side = sdxl_mid(workload) * 4
    return (
        f"one step, {side}x{side} latent ({side * 8}-pixel image), batch "
        f"{sdxl_batch(workload)}, {SDXL_TEXT}-token context, {dtype_name(workload)}"
    )


def save_output(workload: dict[str, Any], method: str, tensor: Any) -> None:
    """The batch-1 output of a method, flattened, for `compare`."""
    import numpy as np

    directory = Path(workload["workdir"])
    directory.mkdir(parents=True, exist_ok=True)
    if hasattr(tensor, "float"):
        array = tensor.detach().float().cpu().numpy()
    else:
        array = np.asarray(tensor, dtype=np.float32)
    np.save(directory / f"{method}.npy", array.astype("float32").reshape(-1))


# ------------------------------------------------------------------- inputs


def vae_latent(batch: int, side: int, seed: int) -> Any:
    """A latent in the pipeline's convention, already divided by the scaling
    factor as the pipeline does before decoding: `decode` takes it as is."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 4, side, side, generator=generator) / VAE_SCALE


def sdxl_inputs(batch: int, mid: int, seed: int) -> dict[str, Any]:
    """One step's inputs as the pipeline hands them to its UNet (f32, host)."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    side = mid * 4
    size = side * 8
    return {
        "latent": torch.randn(batch, 4, side, side, generator=generator),
        "timestep": torch.full((batch,), SDXL_TIMESTEP),
        "text": torch.randn(batch, SDXL_TEXT, 2048, generator=generator),
        "pooled": torch.randn(batch, 1280, generator=generator),
        "time_ids": torch.tensor([[size, size, 0, 0, size, size]] * batch, dtype=torch.float32),
    }


def sinusoidal(values: Any, dim: int) -> Any:
    """`diffusers.models.embeddings.get_timestep_embedding(values, dim,
    flip_sin_to_cos=True, downscale_freq_shift=0)`: cosines first, then
    sines, at frequencies 10000 ** (-i / (dim / 2))."""
    import torch

    half = dim // 2
    exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32, device=values.device)
    frequencies = torch.exp(exponent / half)
    angles = values.float().reshape(-1, 1) * frequencies.reshape(1, -1)
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


def sdxl_embeddings(timesteps: Any, pooled: Any, time_ids: Any) -> tuple[Any, Any]:
    """What the card's `forward` takes besides the latent and the text: the
    timestep's 320-wide embedding, and the pooled text embedding followed by
    the 256-wide embedding of each of the six micro-conditioning numbers
    (2816 wide). Computed in f32, as `diffusers` does."""
    import torch

    batch = pooled.shape[0]
    timestep_embedding = sinusoidal(timesteps, 320)
    time_embeds = sinusoidal(time_ids.flatten(), 256).reshape(batch, -1)
    added = torch.cat([pooled.float(), time_embeds], dim=-1)
    return timestep_embedding, added


def linnet_sdxl_inputs(inputs: dict[str, Any], dtype: Any, device: str) -> list[Any]:
    timestep_embedding, added = sdxl_embeddings(
        inputs["timestep"], inputs["pooled"], inputs["time_ids"]
    )
    values = [inputs["latent"], timestep_embedding, inputs["text"], added]
    return [value.to(device=device, dtype=dtype) for value in values]


# ------------------------------------------------------------------ loading


def linnet_generics(data: dict[str, Any], workload: dict[str, Any]) -> dict[str, int | str]:
    if is_vae(data):
        side = vae_side(workload)
        return {"H": side, "W": side, "T": dtype_name(workload)}
    mid = sdxl_mid(workload)
    return {"MidH": mid, "MidW": mid, "T": dtype_name(workload)}


def load_linnet_torch(
    data: dict[str, Any],
    workload: dict[str, Any],
    compile: bool | str | None,
    **generics: int | str,
) -> Any:
    from linnet import nest

    return nest.load(
        data["directory"],
        backend="torch",
        device=device_of(workload),
        numerics="fast",
        compile=compile,
        generics={**linnet_generics(data, workload), **generics},
        cast_dtype=True,
    )


def checkpoint(data: dict[str, Any]) -> Path:
    from linnet import nest

    return Path(nest.download_weights(nest.resolve(data["directory"])))


def numpy_weights(data: dict[str, Any], name: str) -> dict[str, Any]:
    """The checkpoint as host arrays in the run's dtype, converted one tensor
    at a time (the JAX loader takes weights as they are; it does not cast)."""
    import ml_dtypes  # type: ignore[import-untyped]
    import numpy as np
    from safetensors import safe_open  # type: ignore[import-untyped]

    target = {"f32": np.float32, "bf16": ml_dtypes.bfloat16, "f16": np.float16}[name]
    weights: dict[str, Any] = {}
    with safe_open(str(checkpoint(data)), framework="numpy") as handle:
        for key in handle.keys():  # noqa: SIM118 - a safetensors handle, not a dict
            array = handle.get_tensor(key)
            weights[key] = array.astype(target) if array.dtype.kind == "f" else array
    return weights


# ------------------------------------------------------- reference stacks


def _diffusers(compiled: bool) -> Callable[[dict[str, Any], dict[str, Any]], Result]:
    name = "diffusers (torch.compile)" if compiled else "diffusers (eager)"
    method = "diffusers-compile" if compiled else "diffusers-eager"

    def run(data: dict[str, Any], workload: dict[str, Any]) -> Result:
        import torch

        device = device_of(workload)
        dtype = torch_dtype(dtype_name(workload))
        sync = torch_sync()
        warmup, iters = int(workload["warmup"]), int(workload["iters"])
        # The documented recipe: CUDA graphs over a fully captured graph.
        mode = "reduce-overhead" if device != "cpu" else None
        start = time.perf_counter()
        if is_vae(data):
            from diffusers import AutoencoderKL

            vae = AutoencoderKL.from_pretrained(VAE_REPO, torch_dtype=dtype).to(device).eval()

            def decode(z: Any) -> Any:
                return vae.decode(z, return_dict=False)[0]

            fn = torch.compile(decode, mode=mode, fullgraph=True) if compiled else decode
            load_s = time.perf_counter() - start
            side, batch = vae_side(workload), vae_batch(workload)
            one = vae_latent(1, side, workload.get("seed", 0)).to(device, dtype)
            many = vae_latent(batch, side, workload.get("seed", 0)).to(device, dtype)
            with torch.inference_mode():
                latency = median_ms(lambda: fn(one), sync, warmup, iters)
                per_batch = median_ms(lambda: fn(many), sync, warmup, iters)
                save_output(workload, method, fn(one)[0].clone())
            return Result(
                name,
                "reference",
                {
                    "latency_ms": latency,
                    "throughput_per_s": batch / (per_batch / 1e3),
                    "load_s": load_s,
                },
                notes=vae_workload_note(workload),
            )

        from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained(SDXL_REPO, subfolder="unet", torch_dtype=dtype)
        unet = unet.to(device).eval()
        if compiled:
            unet = torch.compile(unet, mode=mode, fullgraph=True)
        load_s = time.perf_counter() - start
        inputs = sdxl_inputs(sdxl_batch(workload), sdxl_mid(workload), workload.get("seed", 0))
        latent = inputs["latent"].to(device, dtype)
        timestep = inputs["timestep"].to(device)
        text = inputs["text"].to(device, dtype)
        added = {
            "text_embeds": inputs["pooled"].to(device, dtype),
            "time_ids": inputs["time_ids"].to(device, dtype),
        }

        def step() -> Any:
            return unet(
                latent,
                timestep,
                encoder_hidden_states=text,
                added_cond_kwargs=added,
                return_dict=False,
            )[0]

        with torch.inference_mode():
            step_ms = median_ms(step, sync, warmup, iters)
            save_output(workload, method, step()[0].clone())
        return Result(
            name,
            "reference",
            {"step_ms": step_ms, "load_s": load_s},
            notes=sdxl_workload_note(workload),
        )

    return run


# ------------------------------------------------------------------- Linnet


def _linnet_torch(compile: bool | str) -> Callable[[dict[str, Any], dict[str, Any]], Result]:
    label = {True: "generated source", "reduce-overhead": "CUDA graphs"}[compile]
    method = {True: "linnet-torch", "reduce-overhead": "linnet-cudagraphs"}[compile]
    name = f"Linnet torch ({label})"

    def run(data: dict[str, Any], workload: dict[str, Any]) -> Result:
        import torch

        device = device_of(workload)
        if compile == "reduce-overhead" and device == "cpu":
            return Result(name, "linnet", notes="not run: CUDA graphs need a GPU")
        dtype = torch_dtype(dtype_name(workload))
        sync = torch_sync()
        warmup, iters = int(workload["warmup"]), int(workload["iters"])
        start = time.perf_counter()
        model = load_linnet_torch(data, workload, compile)
        load_s = time.perf_counter() - start
        with torch.inference_mode():
            if is_vae(data):
                side, batch = vae_side(workload), vae_batch(workload)
                one = vae_latent(1, side, workload.get("seed", 0)).to(device, dtype)
                many = vae_latent(batch, side, workload.get("seed", 0)).to(device, dtype)
                # Warmup also compiles: one program per batch size.
                latency = median_ms(lambda: model.run_entry("decode", [one]), sync, warmup, iters)
                save_output(workload, method, model.run_entry("decode", [one])[0])
                metrics: dict[str, float | None] = {"latency_ms": latency, "load_s": load_s}
                notes = vae_workload_note(workload)
                try:
                    per_batch = median_ms(
                        lambda: model.run_entry("decode", [many]), sync, warmup, iters
                    )
                    metrics["throughput_per_s"] = batch / (per_batch / 1e3)
                except torch.OutOfMemoryError:
                    notes += f"; batch {batch} ran out of GPU memory, so no throughput"
                return Result(name, "linnet", metrics, notes=notes)
            inputs = sdxl_inputs(sdxl_batch(workload), sdxl_mid(workload), workload.get("seed", 0))
            arguments = linnet_sdxl_inputs(inputs, dtype, device)
            step_ms = median_ms(lambda: model.run_entry("forward", arguments), sync, warmup, iters)
            save_output(workload, method, model.run_entry("forward", arguments)[0])
        return Result(
            name,
            "linnet",
            {"step_ms": step_ms, "load_s": load_s},
            notes=sdxl_workload_note(workload),
        )

    return run


# What a real convolution looks like in each export's text: without one, the
# export spells `conv2d` as its canonical body, a gather whose index tensor
# has B * Cout * H * W * Cin * 9 entries (2.4 G of them for the VAE's first
# 3x3 convolution at an 8x8 latent), which no runtime can hold.
NATIVE_CONV = {"stablehlo": "stablehlo.convolution", "onnx": "= Conv "}


def conv_gap(data: dict[str, Any], target: str) -> str | None:
    """Why the `target` export cannot run this card, or None if it can:
    emits the entry at the smallest grid and looks for a convolution op."""
    from linnet.compiler import run_compiler
    from linnet.nest import resolve

    card = resolve(data["directory"])
    small = {"H": 1, "W": 1, "B": 1} if is_vae(data) else {"MidH": 1, "MidW": 1, "B": 1, "S": 1}
    binds = {**card.generics, **small}
    arguments = [target, "--root", card.root, "--entry", "decode" if is_vae(data) else "forward"]
    for key, value in binds.items():
        arguments += ["--bind", f"{key}={value}"]
    text = run_compiler(*arguments, str(card.source_path))
    if NATIVE_CONV[target] in text:
        return None
    return (
        f"not run: Linnet's {target} export has no native convolution yet and spells conv2d "
        "as a gather over B*Cout*H*W*Cin*9 indices (2.4 G at an 8x8 latent), past any memory"
    )


def linnet_jax(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    """The StableHLO export under XLA, with the checkpoint converted to the
    run's dtype on the host first."""
    gap = conv_gap(data, "stablehlo")
    if gap is not None:
        return Result("Linnet JAX (XLA)", "linnet", notes=gap)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from linnet import nest

    name = f"Linnet JAX (XLA, {jax.default_backend()})"
    dtype = {"f32": jnp.float32, "bf16": jnp.bfloat16, "f16": jnp.float16}[dtype_name(workload)]
    warmup, iters = int(workload["warmup"]), int(workload["iters"])
    entry = "decode" if is_vae(data) else "forward"
    start = time.perf_counter()
    function = nest.load(
        data["directory"],
        backend="jax",
        entry=entry,
        generics=linnet_generics(data, workload),
        weights=numpy_weights(data, dtype_name(workload)),
    )
    load_s = time.perf_counter() - start

    def call(*inputs: Any) -> Any:
        result = function(*inputs)
        return result[0] if isinstance(result, tuple | list) else result

    def timed(*inputs: Any) -> float:
        return median_ms(lambda: jax.block_until_ready(call(*inputs)), lambda: None, warmup, iters)

    if is_vae(data):
        side, batch = vae_side(workload), vae_batch(workload)
        one = jnp.asarray(vae_latent(1, side, workload.get("seed", 0)).numpy(), dtype=dtype)
        many = jnp.asarray(vae_latent(batch, side, workload.get("seed", 0)).numpy(), dtype=dtype)
        latency = timed(one)
        per_batch = timed(many)
        save_output(workload, "linnet-jax", np.asarray(call(one)[0], dtype=np.float32))
        return Result(
            name,
            "linnet",
            {
                "latency_ms": latency,
                "throughput_per_s": batch / (per_batch / 1e3),
                "load_s": load_s,
            },
            notes=vae_workload_note(workload),
        )
    import torch

    inputs = sdxl_inputs(sdxl_batch(workload), sdxl_mid(workload), workload.get("seed", 0))
    arguments = [
        jnp.asarray(value.numpy(), dtype=dtype)
        for value in linnet_sdxl_inputs(inputs, torch.float32, "cpu")
    ]
    step_ms = timed(*arguments)
    save_output(workload, "linnet-jax", np.asarray(call(*arguments)[0], dtype=np.float32))
    return Result(
        name,
        "linnet",
        {"step_ms": step_ms, "load_s": load_s},
        notes=sdxl_workload_note(workload),
    )


def linnet_onnx(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    """`linnet.onnx.export_model` under ONNX Runtime. The export embeds the
    checkpoint in the dtype it is published in, so on a GPU the decoder's
    tensors are first converted to f16 (ONNX Runtime's CUDA kernels cover f16
    far better than bf16) in a scratch copy; on a CPU the f32 checkpoint is
    used as it is."""
    name = "Linnet ONNX -> ONNX Runtime"
    gap = conv_gap(data, "onnx")
    if gap is not None:
        return Result(name, "linnet", notes=gap)
    if not is_vae(data):
        return Result(
            name,
            "linnet",
            notes="not run: the export embeds the f32 checkpoint as published, a 10 GB graph "
            "at twice the other rows' precision, so it would not be a like-for-like row",
        )
    import numpy as np
    import onnxruntime as ort  # type: ignore[import-untyped]
    from linnet import nest
    from linnet.onnx import export_model

    device = device_of(workload)
    precision = "f32" if device == "cpu" else "f16"
    providers = (
        ["CPUExecutionProvider"]
        if device == "cpu"
        else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    warmup, iters = int(workload["warmup"]), int(workload["iters"])
    side, batch = vae_side(workload), vae_batch(workload)
    card = nest.resolve(data["directory"])
    start = time.perf_counter()
    weights = checkpoint(data)
    if precision != "f32":
        from safetensors.numpy import load_file, save_file  # type: ignore[import-untyped]

        converted = {
            key: value.astype(np.float16)
            for key, value in load_file(str(weights)).items()
            if key.startswith(("decoder.", "post_quant_conv."))
        }
        weights = Path(workload["workdir"]) / "onnx-f16.safetensors"
        save_file(converted, str(weights))

    def session(size: int) -> Any:
        exported = export_model(
            card.source_path,
            generics={"H": side, "W": side, "T": precision, "B": size},
            weights=weights,
            entry="decode",
            root=card.root,
            numerics="fast",
            bindings=card.bindings_path,
        )
        return ort.InferenceSession(exported.model.SerializeToString(), providers=providers)

    one_session, many_session = session(1), session(batch)
    load_s = time.perf_counter() - start
    input_type = np.float32 if precision == "f32" else np.float16
    one = vae_latent(1, side, workload.get("seed", 0)).numpy().astype(input_type)
    many = vae_latent(batch, side, workload.get("seed", 0)).numpy().astype(input_type)

    def run(active: Any, value: Any) -> Any:
        return active.run(None, {active.get_inputs()[0].name: value})[0]

    latency = median_ms(lambda: run(one_session, one), lambda: None, warmup, iters)
    per_batch = median_ms(lambda: run(many_session, many), lambda: None, warmup, iters)
    save_output(workload, "linnet-onnx", run(one_session, one)[0])
    provider = one_session.get_providers()[0].removesuffix("ExecutionProvider")
    return Result(
        f"{name} ({provider})",
        "linnet",
        {"latency_ms": latency, "throughput_per_s": batch / (per_batch / 1e3), "load_s": load_s},
        notes=f"{side}x{side} latent ({side * 8}-pixel image), {precision} (the export does not "
        f"cast, so the checkpoint was converted first); throughput at batch {batch}",
    )


METHODS: dict[str, Callable[[dict[str, Any], dict[str, Any]], Result]] = {
    "diffusers-eager": _diffusers(compiled=False),
    "diffusers-compile": _diffusers(compiled=True),
    "linnet-torch": _linnet_torch(True),
    "linnet-cudagraphs": _linnet_torch("reduce-overhead"),
    "linnet-jax": linnet_jax,
    "linnet-onnx": linnet_onnx,
}

REFERENCE = "diffusers-eager"


# ------------------------------------------------------------------ samples


def sample_backend(device: str) -> str:
    """Samples run the torch backend its own way: generated source on a GPU,
    the interpreter on a CPU (it frees each intermediate as it goes)."""
    return "Linnet torch (interpreter)" if device == "cpu" else "Linnet torch (generated source)"


def samples_dir(data: dict[str, Any], workload: dict[str, Any]) -> tuple[Path, str]:
    """Where images go and how `samples.json` names them. A dry run can send
    them elsewhere with `sample_dir`, so its images never reach the card."""
    if "sample_dir" in workload:
        directory = Path(workload["sample_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        return directory, str(directory)
    directory = Path(data["directory"]) / "samples"
    directory.mkdir(parents=True, exist_ok=True)
    return directory, "samples"


def save_pngs(images: dict[str, Any], directory: Path, prefix: str) -> dict[str, str]:
    """Writes PIL images (filename -> image) as PNGs of one common size, each
    under the size budget: all are shrunk by an eighth of a side at a time until
    every one fits, so a pair stays comparable pixel for pixel. Returns the
    paths the sample dict names."""
    from PIL import Image

    side = max(image.width for image in images.values())
    while True:
        fits = True
        for filename, image in images.items():
            shown = image
            if image.width != side:
                height = side * image.height // image.width
                shown = image.resize((side, height), Image.LANCZOS)
            shown.save(directory / filename, format="PNG", optimize=True)
            fits = fits and (directory / filename).stat().st_size <= MAX_PNG_BYTES
        if fits or side <= 128:
            return {filename: f"{prefix}/{filename}" for filename in images}
        side = (side * 7) // 8


def to_pil(tensor: Any) -> Any:
    """[3, H, W] in [-1, 1] -> an 8-bit RGB image."""
    import numpy as np
    from PIL import Image

    array = ((tensor.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round()
    return Image.fromarray(array.permute(1, 2, 0).numpy().astype(np.uint8))


def psnr(a: Any, b: Any) -> float:
    """Peak signal-to-noise ratio of two [-1, 1] images, on the [0, 1] scale."""
    first = (a.detach().float().cpu().clamp(-1, 1) + 1) / 2
    second = (b.detach().float().cpu().clamp(-1, 1) + 1) / 2
    mse = float(((first - second) ** 2).mean())
    return round(10 * math.log10(1.0 / mse), 2) if mse > 0 else float("inf")


def fetch_image(url: str) -> Any:
    import io
    import urllib.request

    from PIL import Image

    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - a fixed https/http dataset URL
        return Image.open(io.BytesIO(response.read())).convert("RGB")


def sample_vae(data: dict[str, Any], workload: dict[str, Any]) -> dict[str, Any]:
    """The COCO image through `diffusers`' encoder and Linnet's decoder."""
    import numpy as np
    import torch
    from diffusers import AutoencoderKL
    from PIL import Image

    device = device_of(workload)
    dtype = torch_dtype(dtype_name(workload))
    size = int(workload.get("sample_size", 512))
    photo = fetch_image(COCO_IMAGE)
    crop = min(photo.size)
    left, top = (photo.width - crop) // 2, (photo.height - crop) // 2
    photo = photo.crop((left, top, left + crop, top + crop)).resize((size, size), Image.LANCZOS)
    pixels = torch.from_numpy(np.asarray(photo).copy()).permute(2, 0, 1)
    image = (pixels.float() / 127.5 - 1).unsqueeze(0)

    vae = AutoencoderKL.from_pretrained(VAE_REPO, torch_dtype=torch.float32).to(device).eval()
    with torch.inference_mode():
        # The encoder's mean, which `decode` reverses directly: no scaling,
        # that belongs to the diffusion pipeline, not to the autoencoder.
        latent = vae.encode(image.to(device)).latent_dist.mean
        vae = vae.to(dtype)
        reference = vae.decode(latent.to(dtype), return_dict=False)[0][0]
    del vae
    workload = {**workload, "latent": size // 8}
    model = load_linnet_torch(data, workload, None)
    with torch.inference_mode():
        output = model.run_entry("decode", [latent.to(device, dtype)])[0]

    directory, prefix = samples_dir(data, workload)
    original = image[0]
    saved = save_pngs({"input.png": photo, "output.png": to_pil(output)}, directory, prefix)
    return {
        "kind": "image",
        "input": saved["input.png"],
        "output": saved["output.png"],
        "caption": (
            f"A COCO val2017 photo ({size}x{size}) encoded to a {size // 8}x{size // 8}x4 latent "
            "by diffusers' AutoencoderKL encoder (Linnet implements only the decoder), then "
            f"decoded by this card with Linnet torch in {dtype_name(workload)}."
        ),
        "source": COCO_IMAGE,
        "backend": f"{sample_backend(device)}, {dtype_name(workload)}, {device}",
        "psnr_db": {
            "linnet_vs_input": psnr(output, original),
            "diffusers_vs_input": psnr(reference, original),
            "linnet_vs_diffusers": psnr(output, reference),
        },
        "reference": f"diffusers AutoencoderKL.decode, {dtype_name(workload)}",
    }


def unet_class() -> type:
    """`LinnetUNet`, defined on first use so importing this module does not
    import PyTorch."""
    import torch

    class LinnetUNet(torch.nn.Module):
        """The SDXL card in the call signature `StableDiffusionXLPipeline`
        uses for its `UNet2DConditionModel`: the pipeline passes the timestep
        and the added conditioning as numbers, the card takes them embedded,
        so the wrapper embeds them (`sdxl_embeddings`) and calls `forward`.
        It answers the attributes the pipeline reads off its UNet: the stock
        UNet's `config`, `dtype`, and `add_embedding.linear_1.in_features`."""

        def __init__(self, model: Any, config: Any, dtype: Any) -> None:
            from types import SimpleNamespace

            super().__init__()
            self.linnet = model
            self.config = config
            self._dtype = dtype
            self.add_embedding = SimpleNamespace(linear_1=SimpleNamespace(in_features=2816))

        @property
        def dtype(self) -> Any:
            return self._dtype

        @property
        def device(self) -> Any:
            return next(self.linnet.parameters()).device

        def forward(
            self,
            sample: Any,
            timestep: Any,
            encoder_hidden_states: Any,
            timestep_cond: Any = None,
            cross_attention_kwargs: Any = None,
            added_cond_kwargs: Any = None,
            return_dict: bool = True,
            **_: Any,
        ) -> Any:
            if timestep_cond is not None or cross_attention_kwargs:
                raise ValueError("the SDXL card takes no timestep_cond or cross-attention kwargs")
            if added_cond_kwargs is None:
                raise ValueError("SDXL needs added_cond_kwargs (text_embeds, time_ids)")
            batch = sample.shape[0]
            timesteps = torch.as_tensor(timestep, device=sample.device).float().reshape(-1)
            timesteps = timesteps.expand(batch)
            timestep_embedding, added = sdxl_embeddings(
                timesteps, added_cond_kwargs["text_embeds"], added_cond_kwargs["time_ids"]
            )
            inputs = [sample, timestep_embedding, encoder_hidden_states, added]
            noise = self.linnet.run_entry(
                "forward", [value.to(device=sample.device, dtype=self._dtype) for value in inputs]
            ).to(sample.dtype)
            if return_dict:
                from types import SimpleNamespace

                return SimpleNamespace(sample=noise)
            return (noise,)

    return LinnetUNet


def sample_sdxl(data: dict[str, Any], workload: dict[str, Any]) -> dict[str, Any]:
    """One image from `StableDiffusionXLPipeline` with the Linnet UNet in
    place of its own, and one from the stock pipeline with the same seed.
    Text encoders, scheduler and VAE are the pipeline's in both."""
    import gc

    import torch
    from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
    from diffusers.configuration_utils import FrozenDict

    device = device_of(workload)
    dtype = torch_dtype(dtype_name(workload))
    # A CPU dry run defaults to a 16x16 latent and 2 steps: it proves the
    # wiring, not the picture.
    cpu = device == "cpu"
    size = int(workload.get("sample_size", 128 if cpu else 1024))
    steps = int(workload.get("sample_steps", 2 if cpu else 25))
    seed = int(workload.get("sample_seed", SAMPLE_SEED))
    # The Linnet UNet is loaded first, so the pipeline never holds a second copy.
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_REPO, unet=None, torch_dtype=dtype, variant="fp16", use_safetensors=True
    )
    pipe.set_progress_bar_config(disable=True)
    config = FrozenDict(UNet2DConditionModel.load_config(SDXL_REPO, subfolder="unet"))
    linnet_workload = {**workload, "mid": size // 32}
    model = load_linnet_torch(data, linnet_workload, None)
    pipe.register_modules(unet=unet_class()(model, config, dtype))
    pipe = pipe.to(device)

    def generate() -> Any:
        generator = torch.Generator("cpu").manual_seed(seed)
        with torch.inference_mode():
            return pipe(
                SAMPLE_PROMPT,
                height=size,
                width=size,
                num_inference_steps=steps,
                generator=generator,
                output_type="pt",
            ).images[0]

    start = time.perf_counter()
    ours = generate()
    linnet_s = time.perf_counter() - start
    pipe.register_modules(unet=None)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    stock = UNet2DConditionModel.from_pretrained(SDXL_REPO, subfolder="unet", torch_dtype=dtype)
    pipe.register_modules(unet=stock.to(device).eval())
    start = time.perf_counter()
    theirs = generate()
    stock_s = time.perf_counter() - start

    directory, prefix = samples_dir(data, workload)
    # `output_type="pt"` images are [3, H, W] in [0, 1]; to [-1, 1] for `to_pil`.
    ours, theirs = ours * 2 - 1, theirs * 2 - 1
    saved = save_pngs(
        {"linnet.png": to_pil(ours), "diffusers.png": to_pil(theirs)}, directory, prefix
    )
    return {
        "kind": "text-to-image",
        "prompt": SAMPLE_PROMPT,
        "output": saved["linnet.png"],
        "reference_output": saved["diffusers.png"],
        "caption": (
            f"StableDiffusionXLPipeline, {steps} steps, {size}x{size}, seed {seed}, guidance 5.0: "
            "once with this card's UNet (Linnet torch) in place of the pipeline's own, once "
            "stock. The text encoders, scheduler and VAE are the pipeline's in both."
        ),
        "backend": f"{sample_backend(device)}, {dtype_name(workload)}, {device}",
        "reference": f"diffusers StableDiffusionXLPipeline (stock UNet), {dtype_name(workload)}",
        "steps": steps,
        "size": [size, size],
        "seed": seed,
        "psnr_db": {"linnet_vs_diffusers": psnr(ours, theirs)},
        "seconds": {"linnet": round(linnet_s, 2), "diffusers": round(stock_s, 2)},
    }


def sample(data: dict[str, Any], workload: dict[str, Any]) -> dict[str, Any]:
    return sample_vae(data, workload) if is_vae(data) else sample_sdxl(data, workload)
