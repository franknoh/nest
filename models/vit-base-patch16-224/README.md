# ViT-Base/16 224

Google's Vision Transformer, fine-tuned on ImageNet-1k: a 224-pixel image is
cut into 196 patches of 16 pixels, embedded to width 768, prefixed with a
class token, and run through 12 pre-norm encoder layers with 12 heads. The
classifier reads the class token.

This is the first model in Nest that is not a language model, and it shows
what the type system does with image shapes: `Height % Patch == 0` is a
`where` clause on the block, so the patch grid `(Height / Patch) * (Width /
Patch)` is checked arithmetic rather than a number written by hand, and the
position table's length follows from it.

## The patch embedding

A convolution with stride equal to its kernel is a contraction over each
patch, so the source keeps the checkpoint's four-axis weight and reads it
directly:

```linnet
let embedded[b, p, d] = sum[c, i, j] windows[b, p, c, i, j] * patch_weight[d, c, i, j]
```

Nothing reshapes the checkpoint: `patch_weight` is declared
`Tensor[D, Channels, Patch, Patch; T]`, the shape `Conv2d` stores, and the
window axes are cut from the image with `reshape` and `permute` whose element
counts the checker proves.

## Loading

```python
from linnet import nest

model = nest.load("vit-base-patch16-224", backend="torch")
logits = model(images)      # Tensor[B, 3, 224, 224; f32] -> Tensor[B, 1000; f32]
```

Images are the usual ImageNet normalization (mean and standard deviation
0.5), in `[B, C, H, W]`; `preprocessor_config.json` in the Hub repository has
the values.

## Entries

| Entry | |
| --- | --- |
| `forward<B>(image)` | class logits |

## Provenance

- Weights: [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224), Apache-2.0.
- Code: [google-research/vision_transformer](https://github.com/google-research/vision_transformer).
- Paper: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929).
