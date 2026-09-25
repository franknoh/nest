# ResNet-50

The 50-layer residual network, fine-tuned on ImageNet-1k: a 7x7 stem at
stride 2, max pooling, four stages of bottleneck blocks three, four, six,
and three deep, spatial averaging, and a linear classifier. A bottleneck
reduces to a quarter of its stage's width with a 1x1 convolution, convolves
3x3 there, and expands back with another 1x1; the first bottleneck of a
stage also projects its shortcut with a 1x1 convolution (plus its own
normalization) whenever the width or the resolution changes, which is true
of every stage's first block since the width always changes there.

This is **v1.5**, the variant `microsoft/resnet-50` publishes and the one
most current ResNet-50 checkpoints use: the stride that halves the
resolution sits on the stage's 3x3 convolution rather than its first 1x1
(the original v1 placement), so more of the stage's work happens at full
resolution before the channel count grows. Only the first stage keeps
stride 1 throughout, since max pooling has already halved the resolution
once; the other three each halve it again on their first block.

Reuses `resnet-18`'s `ConvNorm` (a convolution plus the affine batch
normalization inference computes) and stem, adapted for `std.nn.conv`'s
convolution as a contraction over an index-built window. `Stage` is generic
over the input and output width, the stride, and the depth, so all four
stages share one block definition; the per-stage counts come from the
checkpoint's `depths` config (`[3, 4, 6, 3]`).

## Loading

```python
from linnet import nest

model = nest.load("resnet-50", backend="torch")
logits = model(images)      # Tensor[B, 3, 224, 224; f32] -> Tensor[B, 1000; f32]
```

Images are ImageNet-normalized in `[B, C, H, W]`; `preprocessor_config.json`
in the Hub repository has the mean and standard deviation. Other input
sizes work as long as both sides are multiples of 32, which the block's
`where` clause states.

## Entries

| Entry | |
| --- | --- |
| `forward<B>(image)` | class logits |

## Provenance

- Weights: [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50), Apache-2.0.
- Code: [KaimingHe/deep-residual-networks](https://github.com/KaimingHe/deep-residual-networks).
- Paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
