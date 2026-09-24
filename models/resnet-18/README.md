# ResNet-18

The 18-layer residual network, fine-tuned on ImageNet-1k: a 7x7 stem at
stride 2, max pooling, four stages of two residual blocks each, spatial
averaging, and a linear classifier. Widths double and resolution halves at
each stage, so the first block of a stage projects its shortcut with a 1x1
convolution.

This is the first convolutional model in Nest, and the first to use
`std.nn.conv`. A convolution in Linnet is a contraction over a window:

```linnet
let mixed[b, co, oh, ow] = sum[ci, kh, kw]
    padded[b, ci, rows[oh, kh], columns[ow, kw]] * weight[co, ci, kh, kw]
```

`rows` and `columns` are index tensors built from `iota`, so the window
arithmetic is ordinary indexing that the checker verifies. That body is the
normative meaning; a backend with `F.conv2d` (or `Conv` in ONNX) is free to
call it instead, and the compiler recovers the stride and padding from the
shapes to do so.

Batch normalization appears as what inference actually computes — an affine
transform along the channel axis from the statistics the checkpoint stores —
so `running_mean` and `running_var` bind like any other parameter.

## Loading

```python
from linnet import nest

model = nest.load("resnet-18", backend="torch")
logits = model(images)      # Tensor[B, 3, 224, 224; f32] -> Tensor[B, 1000; f32]
```

Images are ImageNet-normalized in `[B, C, H, W]`; `preprocessor_config.json`
in the Hub repository has the mean and standard deviation. Other input sizes
work as long as both sides are multiples of 32, which the block's `where`
clause states.

## Entries

| Entry | |
| --- | --- |
| `forward<B>(image)` | class logits |

## Provenance

- Weights: [microsoft/resnet-18](https://huggingface.co/microsoft/resnet-18), Apache-2.0.
- Code: [KaimingHe/deep-residual-networks](https://github.com/KaimingHe/deep-residual-networks).
- Paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
