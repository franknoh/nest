"""Vision: image classification (ResNet, ViT), dense self-supervised
features (DINOv2), and image-text similarity (SigLIP), benchmarked against
their `transformers` reference.

Every method gets the same synthetic image (`torch.randn` from the
workload's seed: a convolution or an attention layer does not care what the
picture shows) at batch 1 for `latency_ms` and batch 32 for
`throughput_per_s`, images per second. The reference stack and Linnet's
torch backend both run in bf16 on a GPU and f32 on a CPU dry run; the JAX
and ONNX paths bind the checkpoint's own tensors directly (the JAX loader
has no `cast_dtype`, and an ONNX export embeds the checkpoint's tensors as
initializers, dtype and all), so both stay at the published f32 regardless
of device -- noted on their `Result` rather than faked.

SigLIP is two towers that only meet in `similarity`, which this family does
not time: `image_features` alone is a plain forward pass through the vision
tower, exactly like the other cards here, and is what every method below
measures for that card.

Each method also saves its batch-1 output, so the report can show how far
every stack's numbers are from the reference.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bench.harness import Result, median_ms

LATENCY_BATCH = 1
THROUGHPUT_BATCH = 32

COCO_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
SIGLIP_LABELS = [
    "a photo of two cats",
    "a photo of a dog",
    "a photo of a remote control",
    "a photo of a person",
    "a photo of a pizza",
]


def image_shape(data: dict[str, Any]) -> tuple[int, int, int]:
    """`(Channels, Height, Width)`; the convolutional cards fix Channels at
    3 in their source rather than declaring it as a generic."""
    generics = data["generics"]
    return int(generics.get("Channels", 3)), int(generics["Height"]), int(generics["Width"])


def precision(workload: dict[str, Any]) -> tuple[Any, str]:
    """bf16 on a GPU, f32 on a CPU dry run: the torch dtype and the Linnet
    generic name that selects it."""
    import torch

    if workload.get("device", "cuda") == "cuda":
        return torch.bfloat16, "bf16"
    return torch.float32, "f32"


def canonical_images(data: dict[str, Any], workload: dict[str, Any], batch: int) -> Any:
    """The same synthetic image, from the workload's seed, as f32 on the
    host; every method casts it to its own precision and device."""
    import torch

    channels, height, width = image_shape(data)
    generator = torch.Generator().manual_seed(int(workload.get("seed", 0)))
    return torch.randn(batch, channels, height, width, generator=generator)


def save_output(workload: dict[str, Any], method: str, tensor: Any) -> None:
    import numpy as np

    directory = Path(workload["workdir"])
    directory.mkdir(parents=True, exist_ok=True)
    array = tensor.float().cpu().numpy() if hasattr(tensor, "float") else np.asarray(tensor)
    np.save(directory / f"{method}.npy", array.astype("float32"))


def linnet_entry(data: dict[str, Any]) -> str:
    """`image_features` for SigLIP (its own main entry, `similarity`, also
    runs the text tower, which this family does not time); `forward` for
    every plain image encoder."""
    return "image_features" if data["model"].get("family") == "siglip" else "forward"


# ------------------------------------------------------------ reference stack


def _reference_model(data: dict[str, Any], device: str, dtype: Any) -> Any:
    from transformers import AutoModel, AutoModelForImageClassification

    repo = data["weights"]["repo"]
    family = data["model"].get("family")
    if family in ("resnet", "vit"):
        model = AutoModelForImageClassification.from_pretrained(repo, torch_dtype=dtype)
    elif family == "dinov2":
        model = AutoModel.from_pretrained(repo, torch_dtype=dtype)
    elif family == "siglip":
        from transformers import SiglipModel

        model = SiglipModel.from_pretrained(repo, torch_dtype=dtype)
    else:
        raise ValueError(f"vision has no reference stack for family `{family}`")
    return model.to(device).eval()


def _reference_output(data: dict[str, Any], model: Any, images: Any) -> Any:
    family = data["model"].get("family")
    if family in ("resnet", "vit"):
        return model(pixel_values=images).logits
    if family == "dinov2":
        return model(pixel_values=images).last_hidden_state
    if family == "siglip":
        features = model.get_image_features(pixel_values=images)
        # A tensor in older transformers, an output object in newer ones.
        return getattr(features, "pooler_output", features)
    raise ValueError(f"vision has no reference stack for family `{family}`")


def _transformers(compiled: bool) -> Callable[[dict[str, Any], dict[str, Any]], Result]:
    def run(data: dict[str, Any], workload: dict[str, Any]) -> Result:
        import torch

        from bench.harness import torch_sync

        name = "transformers (torch.compile)" if compiled else "transformers (eager)"
        method = "transformers-compile" if compiled else "transformers-eager"
        device = workload.get("device", "cuda")
        dtype, _ = precision(workload)
        sync = torch_sync()

        start = time.perf_counter()
        model = _reference_model(data, device, dtype)
        load_s = time.perf_counter() - start

        def call(images: Any) -> Any:
            with torch.no_grad():
                return _reference_output(data, model, images)

        if compiled:
            call = torch.compile(call)

        images1 = canonical_images(data, workload, LATENCY_BATCH).to(device=device, dtype=dtype)
        images32 = canonical_images(data, workload, THROUGHPUT_BATCH).to(device=device, dtype=dtype)
        warmup = int(workload["warmup"])
        iters = int(workload["iters"])
        latency_ms = median_ms(lambda: call(images1), sync, warmup, iters)
        throughput_ms = median_ms(lambda: call(images32), sync, warmup, iters)
        save_output(workload, method, call(images1))
        throughput = THROUGHPUT_BATCH / (throughput_ms / 1e3)
        return Result(
            name,
            "reference",
            {"latency_ms": latency_ms, "throughput_per_s": throughput, "load_s": load_s},
        )

    return run


# ------------------------------------------------------------------- Linnet


def _linnet_torch(compile: bool | str) -> Callable[[dict[str, Any], dict[str, Any]], Result]:
    label = {True: "generated source", "reduce-overhead": "CUDA graphs"}[compile]
    method = {True: "linnet-torch", "reduce-overhead": "linnet-cudagraphs"}[compile]

    def run(data: dict[str, Any], workload: dict[str, Any]) -> Result:
        from linnet import nest

        from bench.harness import torch_sync

        device = workload.get("device", "cuda")
        if compile == "reduce-overhead" and device != "cuda":
            return Result(
                f"Linnet torch ({label})", "linnet", error="CUDA graphs need a CUDA device"
            )
        dtype, dtype_name = precision(workload)
        sync = torch_sync()
        entry = linnet_entry(data)

        start = time.perf_counter()
        model = nest.load(
            data["directory"],
            backend="torch",
            device=device,
            numerics="fast",
            compile=compile,
            generics={"T": dtype_name},
            cast_dtype=True,
        )
        load_s = time.perf_counter() - start

        def call(images: Any) -> Any:
            return model.run_entry(entry, [images])

        images1 = canonical_images(data, workload, LATENCY_BATCH).to(device=device, dtype=dtype)
        images32 = canonical_images(data, workload, THROUGHPUT_BATCH).to(device=device, dtype=dtype)
        warmup = int(workload["warmup"])
        iters = int(workload["iters"])
        latency_ms = median_ms(lambda: call(images1), sync, warmup, iters)
        throughput_ms = median_ms(lambda: call(images32), sync, warmup, iters)
        save_output(workload, method, call(images1))
        throughput = THROUGHPUT_BATCH / (throughput_ms / 1e3)
        return Result(
            f"Linnet torch ({label})",
            "linnet",
            {"latency_ms": latency_ms, "throughput_per_s": throughput, "load_s": load_s},
        )

    return run


def linnet_jax(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    import jax
    import jax.numpy as jnp
    from linnet import nest

    entry = linnet_entry(data)
    _dtype, dtype_name = precision(workload)
    jax_dtype = {"bf16": jnp.bfloat16, "f32": jnp.float32}[dtype_name]

    start = time.perf_counter()
    model = nest.load(
        data["directory"],
        backend="jax",
        entry=entry,
        generics={"T": dtype_name},
        cast_dtype=True,
    )
    load_s = time.perf_counter() - start

    def call(images: Any) -> Any:
        output = model(jnp.asarray(images.float().numpy()).astype(jax_dtype))
        jax.block_until_ready(output)
        return output

    images1 = canonical_images(data, workload, LATENCY_BATCH)
    images32 = canonical_images(data, workload, THROUGHPUT_BATCH)
    warmup = int(workload["warmup"])
    iters = int(workload["iters"])
    latency_ms = median_ms(lambda: call(images1), lambda: None, warmup, iters)
    throughput_ms = median_ms(lambda: call(images32), lambda: None, warmup, iters)
    save_output(workload, "linnet-jax", jax.device_get(call(images1)))
    throughput = THROUGHPUT_BATCH / (throughput_ms / 1e3)
    return Result(
        f"Linnet JAX (XLA, {jax.default_backend()})",
        "linnet",
        {"latency_ms": latency_ms, "throughput_per_s": throughput, "load_s": load_s},
        notes=f"{dtype_name}, like the torch rows",
    )


def linnet_onnx(data: dict[str, Any], workload: dict[str, Any]) -> Result:
    import onnxruntime
    from linnet import nest
    from linnet.onnx import export_model

    device = workload.get("device", "cuda")
    entry = linnet_entry(data)
    card = nest.Card.read(data["directory"])
    weights = nest.download_weights(card)
    bindings = str(card.bindings_path) if card.bindings_path is not None else None
    providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]

    def build(batch: int) -> tuple[Any, str]:
        exported = export_model(
            card.source_path,
            generics={**card.generics, "B": batch},
            weights=weights,
            entry=entry,
            root=card.root,
            bindings=bindings,
        )
        session = onnxruntime.InferenceSession(
            exported.model.SerializeToString(), providers=providers
        )
        return session, exported.inputs[0].name

    start = time.perf_counter()
    session1, input_name = build(LATENCY_BATCH)
    session32, _ = build(THROUGHPUT_BATCH)
    load_s = time.perf_counter() - start

    images1 = canonical_images(data, workload, LATENCY_BATCH).numpy()
    images32 = canonical_images(data, workload, THROUGHPUT_BATCH).numpy()

    def call(session: Any, images: Any) -> Any:
        (output,) = session.run(None, {input_name: images})
        return output

    warmup = int(workload["warmup"])
    iters = int(workload["iters"])
    latency_ms = median_ms(lambda: call(session1, images1), lambda: None, warmup, iters)
    throughput_ms = median_ms(lambda: call(session32, images32), lambda: None, warmup, iters)
    save_output(workload, "linnet-onnx", call(session1, images1))
    throughput = THROUGHPUT_BATCH / (throughput_ms / 1e3)
    return Result(
        "Linnet ONNX -> ONNX Runtime",
        "linnet",
        {"latency_ms": latency_ms, "throughput_per_s": throughput, "load_s": load_s},
        notes="the exporter embeds the checkpoint's own f32 tensors, so this always runs f32",
    )


# ------------------------------------------------------------------- sample


def _fetch_coco_image() -> Any:
    """The COCO validation image the `transformers` docs use for these
    cards, fetched fresh so the sample runs on a real picture."""
    import urllib.request
    from io import BytesIO

    from PIL import Image

    with urllib.request.urlopen(COCO_IMAGE_URL, timeout=30) as response:
        raw = response.read()
    return Image.open(BytesIO(raw)).convert("RGB")


def _save_image(image: Any, path: Path, limit: int = 300_000) -> None:
    """PNG, shrinking until it is under the page's size budget."""
    path.parent.mkdir(parents=True, exist_ok=True)
    picture = image
    while True:
        picture.save(path, format="PNG", optimize=True)
        if path.stat().st_size <= limit or min(picture.size) <= 64:
            return
        picture = picture.resize((picture.width * 3 // 4, picture.height * 3 // 4))


def _topk(logits: Any, id2label: dict[int, str], k: int = 5) -> list[dict[str, Any]]:
    import torch

    probabilities = torch.softmax(logits.float(), dim=-1)
    values, indices = probabilities.topk(k)
    return [
        {"label": id2label[int(index)], "p": float(value)}
        for value, index in zip(values.tolist(), indices.tolist(), strict=True)
    ]


def _pixel_values(repo: str, image: Any, **overrides: Any) -> Any:
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(repo)
    return processor(images=image, return_tensors="pt", **overrides).pixel_values.float()


def _sample_classification(data: dict[str, Any], workload: dict[str, Any]) -> dict[str, Any]:
    import torch
    from linnet import nest
    from transformers import AutoConfig, AutoModelForImageClassification

    device = workload.get("device", "cuda")
    repo = data["weights"]["repo"]

    image = _fetch_coco_image()
    pixel_values = _pixel_values(repo, image)
    id2label = AutoConfig.from_pretrained(repo).id2label

    model = nest.load(data["directory"], backend="torch", device=device, numerics="fast")
    with torch.no_grad():
        logits = model(pixel_values.to(device))
    ours = _topk(logits[0], id2label)

    reference = AutoModelForImageClassification.from_pretrained(repo).to(device).eval()
    with torch.no_grad():
        reference_logits = reference(pixel_values=pixel_values.to(device)).logits
    reference_top = _topk(reference_logits[0], id2label)

    samples_dir = Path(data["directory"]) / "samples"
    _save_image(image, samples_dir / "input.png")

    return {
        "kind": "classification",
        "image": "samples/input.png",
        "top": ours,
        "reference": reference_top,
        "reference_stack": "transformers (eager, fp32)",
    }


def _sample_dinov2(data: dict[str, Any], workload: dict[str, Any]) -> dict[str, Any]:
    import torch
    from linnet import nest
    from PIL import Image

    device = workload.get("device", "cuda")
    repo = data["weights"]["repo"]
    _, height, width = image_shape(data)
    patch = int(data["generics"]["Patch"])

    image = _fetch_coco_image()
    size = {"height": height, "width": width}
    pixel_values = _pixel_values(repo, image, size=size, crop_size=size, do_center_crop=False)

    model = nest.load(data["directory"], backend="torch", device=device, numerics="fast")
    with torch.no_grad():
        features = model(pixel_values.to(device))
    patch_tokens = features[0, 1:, :].float().cpu()

    grid = height // patch
    _, _, v = torch.pca_lowrank(patch_tokens, q=3)
    projected = patch_tokens @ v[:, :3]
    projected = projected - projected.min(dim=0).values
    projected = projected / projected.max(dim=0).values.clamp_min(1e-6)
    rgb = (projected.reshape(grid, grid, 3) * 255).round().clamp(0, 255).byte().numpy()
    pca_image = Image.fromarray(rgb, mode="RGB").resize((width, height), Image.NEAREST)

    samples_dir = Path(data["directory"]) / "samples"
    _save_image(image, samples_dir / "input.png")
    _save_image(pca_image, samples_dir / "output.png")

    return {
        "kind": "image",
        "input": "samples/input.png",
        "output": "samples/output.png",
        "caption": "a PCA of the patch tokens to 3 components, mapped to RGB and upsampled",
    }


def _sample_siglip(data: dict[str, Any], workload: dict[str, Any]) -> dict[str, Any]:
    import torch
    from linnet import nest
    from transformers import AutoProcessor, SiglipModel

    device = workload.get("device", "cuda")
    repo = data["weights"]["repo"]
    max_positions = int(data["generics"]["MaxPositions"])

    image = _fetch_coco_image()
    processor = AutoProcessor.from_pretrained(repo)
    encoded = processor(
        text=SIGLIP_LABELS,
        images=image,
        padding="max_length",
        max_length=max_positions,
        truncation=True,
        return_tensors="pt",
    )
    pixel_values = encoded.pixel_values.float().to(device)
    input_ids = encoded.input_ids.to(torch.int32).to(device)

    model = nest.load(data["directory"], backend="torch", device=device, numerics="fast")
    with torch.no_grad():
        logits_per_text = model.similarity(pixel_values, input_ids)
    ours = torch.sigmoid(logits_per_text.float().cpu())[:, 0].tolist()

    reference = SiglipModel.from_pretrained(repo).to(device).eval()
    with torch.no_grad():
        reference_logits = reference(pixel_values=pixel_values, input_ids=input_ids).logits_per_text
    reference_probabilities = torch.sigmoid(reference_logits.float().cpu())[:, 0].tolist()

    samples_dir = Path(data["directory"]) / "samples"
    _save_image(image, samples_dir / "input.png")

    return {
        "kind": "similarity",
        "images": ["samples/input.png"],
        "sentences": SIGLIP_LABELS,
        "matrix": [ours],
        "reference": [reference_probabilities],
        "reference_stack": "transformers SiglipModel (eager, fp32)",
    }


def sample(data: dict[str, Any], workload: dict[str, Any]) -> dict[str, Any]:
    """What the card produces, through the best Linnet backend, with the
    reference stack's answer beside it -- see the per-family helpers above
    for what each `kind` means."""
    family = data["model"].get("family")
    if family in ("resnet", "vit"):
        return _sample_classification(data, workload)
    if family == "dinov2":
        return _sample_dinov2(data, workload)
    if family == "siglip":
        return _sample_siglip(data, workload)
    raise ValueError(f"vision has no sample for family `{family}`")


METHODS: dict[str, Callable[[dict[str, Any], dict[str, Any]], Result]] = {
    "transformers-eager": _transformers(compiled=False),
    "transformers-compile": _transformers(compiled=True),
    "linnet-torch": _linnet_torch(True),
    "linnet-cudagraphs": _linnet_torch("reduce-overhead"),
    "linnet-jax": linnet_jax,
    "linnet-onnx": linnet_onnx,
}

REFERENCE = "transformers-eager"
