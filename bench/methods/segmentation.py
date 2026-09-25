"""Segmentation: how long SAM's image encoder takes over a 1024-pixel image,
for the reference stack and for each Linnet backend.

The image encoder is segmentation's expensive part -- the card's README
calls it out by name -- so it is the only thing benchmarked here. Decoding a
prompt against the resulting embedding costs a fraction of the encode and is
exercised once, unbenchmarked, in `sample`.

Every method reads the same real image -- the COCO val2017 image of two cats
the `transformers` docs use -- through `SamProcessor`, so `encode_ms` times
the same 1024x1024 tensor everywhere.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bench.harness import Result, median_ms, torch_sync

IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
# Squarely on the larger, right-hand cat's back, clear of both remotes -- in
# the original image's own pixel space, as `SamProcessor` expects.
POINT = (490.0, 170.0)


def _fixture(data: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    """The real COCO image, preprocessed to `[1, 3, 1024, 1024]` the way
    `SamProcessor` does it, the sizes needed to undo the padding, and the
    processor itself -- shared by every method and by `sample`."""
    import requests
    from PIL import Image
    from transformers import SamProcessor

    processor = SamProcessor.from_pretrained(data["weights"]["repo"])
    image = Image.open(requests.get(IMAGE_URL, stream=True, timeout=30).raw).convert("RGB")
    encoded = processor(image, return_tensors="np")
    return (
        encoded["pixel_values"],
        encoded["original_sizes"],
        encoded["reshaped_input_sizes"],
        processor,
    )


def save_output(workload: dict[str, Any], method: str, tensor: Any) -> None:
    """The batch-1 image embedding, so `compare` can put every method's
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
        from transformers import SamModel

        device = workload.get("device", "cuda")
        name = "transformers (torch.compile)" if compiled else "transformers (eager)"
        method = "transformers-compile" if compiled else "transformers-eager"
        pixel_values, *_rest = _fixture(data)
        sync = torch_sync()

        start = time.perf_counter()
        model = SamModel.from_pretrained(data["weights"]["repo"]).to(device).eval()
        if compiled:
            model.vision_encoder.forward = torch.compile(
                model.vision_encoder.forward, mode="reduce-overhead", fullgraph=True
            )
        load_s = time.perf_counter() - start

        pixels = torch.tensor(pixel_values, device=device)

        def encode() -> torch.Tensor:
            with torch.no_grad():
                return model.get_image_embeddings(pixels)

        encode_ms = median_ms(encode, sync, int(workload["warmup"]), int(workload["iters"]))
        save_output(workload, method, encode()[0])
        return Result(name, "reference", {"encode_ms": encode_ms, "load_s": load_s})

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
        pixel_values, *_rest = _fixture(data)

        start = time.perf_counter()
        model = nest.load(
            data["directory"], backend="torch", device=device, numerics="fast", compile=compile
        )
        load_s = time.perf_counter() - start

        pixels = torch.tensor(pixel_values, dtype=torch.float32, device=device)

        def encode() -> torch.Tensor:
            return model.run_entry("encode_image", [pixels])

        encode_ms = median_ms(encode, sync, int(workload["warmup"]), int(workload["iters"]))
        save_output(workload, method, encode()[0])
        return Result(
            f"Linnet torch ({label})", "linnet", {"encode_ms": encode_ms, "load_s": load_s}
        )

    return run


def linnet_jax(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    import jax
    import jax.numpy as jnp
    from linnet import nest

    pixel_values, *_rest = _fixture(data)

    start = time.perf_counter()
    model = nest.load(
        data["directory"], backend="jax", entry="encode_image", generics=dict(data["generics"])
    )
    load_s = time.perf_counter() - start

    pixels = jnp.asarray(pixel_values, dtype=jnp.float32)

    def encode() -> Any:
        return model(pixels)

    encode_ms = median_ms(
        lambda: jax.block_until_ready(encode()),
        lambda: None,
        int(workload["warmup"]),
        int(workload["iters"]),
    )
    save_output(workload, "linnet-jax", jax.device_get(encode())[0])
    return Result(
        f"Linnet JAX (XLA, {jax.default_backend()})",
        "linnet",
        {"encode_ms": encode_ms, "load_s": load_s},
    )


def linnet_onnx(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    """The image encoder is a plain forward pass over a fixed shape, exactly
    what `linnet.onnx.export_model` is for."""
    import numpy as np
    import onnxruntime as ort
    from linnet.nest import Card, download_weights
    from linnet.onnx import export_model

    device = workload.get("device", "cuda")
    pixel_values, *_rest = _fixture(data)
    card = Card.read(data["directory"])
    generics = {**dict(card.generics), **dict(card.check)}

    start = time.perf_counter()
    weights = download_weights(card)
    exported = export_model(
        card.source_path,
        generics=generics,
        weights=weights,
        entry="encode_image",
        root=card.root,
        bindings=card.bindings_path,
    )
    onnx_path = Path(workload["workdir"]) / f"{data['model']['name']}-encode.onnx"
    exported.save(onnx_path)
    providers = ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    load_s = time.perf_counter() - start

    input_name = exported.inputs[0].name
    pixels = pixel_values.astype(np.float32)

    def encode() -> Any:
        return session.run(None, {input_name: pixels})[0]

    encode_ms = median_ms(encode, lambda: None, int(workload["warmup"]), int(workload["iters"]))
    save_output(workload, "linnet-onnx", encode()[0])
    return Result(
        "Linnet ONNX -> ONNX Runtime", "linnet", {"encode_ms": encode_ms, "load_s": load_s}
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


def _overlay(image: Any, mask: Any, color: tuple[int, int, int], point: tuple[float, float]) -> Any:
    import numpy as np
    from PIL import Image, ImageDraw

    base = image.convert("RGBA")
    tint = Image.new("RGBA", base.size, (*color, 0))
    tint.putalpha(Image.fromarray((np.asarray(mask).astype("uint8")) * 130))
    combined = Image.alpha_composite(base, tint)
    draw = ImageDraw.Draw(combined)
    x, y = point
    radius = 6
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius], outline=(255, 255, 0, 255), width=3
    )
    return combined.convert("RGB")


def _save_side_by_side(image: Any, linnet_mask: Any, reference_mask: Any, path: Path) -> None:
    left = _overlay(image, linnet_mask, (30, 144, 255), POINT)
    right = _overlay(image, reference_mask, (255, 69, 0), POINT)
    max_side = 300
    left.thumbnail((max_side, max_side))
    right.thumbnail((max_side, max_side))
    from PIL import Image

    gap = 8
    canvas = Image.new(
        "RGB", (left.width + right.width + gap, max(left.height, right.height)), "white"
    )
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    canvas.save(path, format="PNG", optimize=True)


def sample(card: dict[str, Any], workload: dict[str, Any]) -> dict[str, Any]:
    """The mask for a point prompt on one of the cats, from Linnet and from
    `transformers.SamModel`, as translucent overlays side by side, with the
    IoU between them and each stack's own predicted IoU."""
    import numpy as np
    import torch
    from linnet import nest
    from transformers import SamModel

    device = workload.get("device", "cuda")
    directory = Path(card["directory"]) / "samples"
    directory.mkdir(parents=True, exist_ok=True)

    _pixel_values, _orig, _reshaped, processor = _fixture(card)
    image = _fetch_image()
    encoded = processor(
        image,
        input_points=[[[list(POINT)]]],
        input_labels=[[[1]]],
        return_tensors="pt",
    )
    pixel_values = encoded["pixel_values"].to(device)
    input_points = encoded["input_points"].to(torch.float32).to(device)
    input_labels_i32 = encoded["input_labels"].to(torch.int32).to(device)
    original_sizes = encoded["original_sizes"]
    reshaped_sizes = encoded["reshaped_input_sizes"]

    model = nest.load(card["directory"], backend="torch", device=device)
    embeddings = model.run_entry("encode_image", [pixel_values])
    masks, iou = model.run_entry("decode_points", [embeddings, input_points, input_labels_i32])
    linnet_mask = processor.image_processor.post_process_masks(
        [masks[0].detach().cpu()], original_sizes, reshaped_sizes
    )[0][0, 0].numpy()
    linnet_score = float(iou[0, 0, 0])

    reference = SamModel.from_pretrained(card["weights"]["repo"]).to(device).eval()
    with torch.no_grad():
        reference_embeddings = reference.get_image_embeddings(pixel_values)
        outputs = reference(
            image_embeddings=reference_embeddings,
            input_points=input_points,
            input_labels=encoded["input_labels"].to(device),
            multimask_output=False,
        )
    reference_mask = processor.image_processor.post_process_masks(
        [outputs.pred_masks[0].detach().cpu()], original_sizes, reshaped_sizes
    )[0][0, 0].numpy()
    reference_score = float(outputs.iou_scores[0, 0, 0])

    intersection = float(np.logical_and(linnet_mask, reference_mask).sum())
    union = float(np.logical_or(linnet_mask, reference_mask).sum())
    agreement = intersection / union if union else 1.0

    thumbnail = image.copy()
    thumbnail.thumbnail((400, 400))
    thumbnail.save(directory / "input.png", format="PNG", optimize=True)
    _save_side_by_side(image, linnet_mask, reference_mask, directory / "masks.png")

    return {
        "kind": "image",
        "input": "samples/input.png",
        "output": "samples/masks.png",
        "caption": (
            f"A point prompt at {POINT} on the right cat: Linnet's mask (left, blue) vs "
            f"transformers.SamModel's (right, orange). IoU between the two masks: "
            f"{agreement:.3f}; predicted IoU, Linnet {linnet_score:.3f}, "
            f"reference {reference_score:.3f}."
        ),
    }


def _fetch_image() -> Any:
    import requests
    from PIL import Image

    return Image.open(requests.get(IMAGE_URL, stream=True, timeout=30).raw).convert("RGB")
