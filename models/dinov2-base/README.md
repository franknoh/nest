# DINOv2-Base

Meta's self-supervised Vision Transformer: images are trained on with no
labels at all, and the resulting features are used directly for retrieval,
segmentation, and depth without fine-tuning. This checkpoint cuts a
518-pixel image into 14-pixel patches (37x37 = 1369 of them), embeds them to
width 768, prefixes a class token, and runs 12 pre-norm encoder layers with
12 heads.

The encoder is the same pre-norm block ViT uses (`models/vit-base-patch16-224`
is the style reference), with two additions:

- **LayerScale.** Each residual branch — the attention output and the MLP
  output — is multiplied by a learned per-channel vector (`layer_scale1`,
  `layer_scale2`) before it is added back to the stream. This is what lets a
  plain transformer of this depth train stably without the fine-tuning
  supervision a classifier loss would otherwise provide.
- **A fixed-resolution position table.** DINOv2 stores its position
  embeddings at the checkpoint's native grid (`image_size` / `patch_size`
  from `config.json`: 518 / 14 = 37 per side) and bicubically interpolates
  them for any other input size. Bicubic interpolation has no Linnet
  expression today — `std.nn.resize` only has `upsample_nearest2d` — so
  **this source only supports the native 518x518 resolution**, where the
  stored table is used exactly as the checkpoint has it (the reference
  implementation's own interpolation is a no-op at that resolution too, so
  the two agree). Do not read this model at another resolution by resizing
  the position table with nearest-neighbour or any other substitute; that
  would compute a different function from the reference model. A future
  bicubic-interpolation primitive would let a generic-resolution version of
  this card exist.

The checkpoint also carries `embeddings.mask_token`, used during
self-supervised pretraining to replace masked patches. Inference never reads
it, so this source declares no parameter for it; the tensor is simply left
unbound in the checkpoint.

## The patch embedding

As in `vit-base-patch16-224`, a convolution with stride equal to its kernel
is a contraction over each patch, so the source keeps the checkpoint's
four-axis weight and reads it directly:

```linnet
let embedded[b, p, d] = sum[c, i, j] windows[b, p, c, i, j] * patch_weight[d, c, i, j]
```

## Loading

```python
from linnet import nest

model = nest.load("dinov2-base", backend="torch")
features = model(images)   # Tensor[B, 3, 518, 518; f32] -> Tensor[B, 1370, 768; f32]
```

Images are ImageNet-normalized (mean `[0.485, 0.456, 0.406]`, standard
deviation `[0.229, 0.224, 0.225]`) in `[B, C, H, W]`; `preprocessor_config.json`
in the Hub repository has these values, but its default `size`/`crop_size`
(224, for other members of the DINOv2 family) does not match this card:
resize to 518x518 instead, since that is the resolution its position table
is stored at and the only one this source supports, for the reason above.

The output is the sequence of the class token followed by the patch tokens,
after the final layer norm — exactly `transformers`'s `last_hidden_state`.
The class token (`features[:, 0, :]`) is the image-level embedding used for
retrieval; the patch tokens (`features[:, 1:, :]`) are the dense per-patch
features used for segmentation and depth.

The activation is `std.nn.activations::gelu_erf`, the error-function GELU
upstream uses. It matters more here than in a classifier: DINOv2 has no
final head to average an activation error away, so with the tanh
approximation instead the features drift by 9e-2. On a fixed random image
at 518x518 the maximum absolute difference against `transformers`'s
`last_hidden_state` is 1.7e-4, and 9.8e-6 on the class token.

Attention runs full, unmasked, over every token: there is no padding mask to
apply, since every input is one square image rather than variable-length
text.

## Entries

| Entry | |
| --- | --- |
| `forward<B>(image)` | class token followed by patch tokens, after the final layer norm |

## Provenance

- Weights: [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base), Apache-2.0.
- Code: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2).
- Paper: [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193).
