# SigLIP Base/16 224

Google's image-text model: one checkpoint holding two independent encoder
towers that never mix until their pooled outputs meet in a single dot
product. This is Nest's first multimodal model, and the sigmoid in the name
is exactly why that separation is possible: CLIP's contrastive loss is a
softmax over a whole batch of pairs, so every image needs every text's
embedding (and vice versa) to normalize against. SigLIP's loss scores each
image-text pair independently with a sigmoid, so nothing about training or
inference requires the towers to know about each other beyond agreeing on
the width of the vector they hand to that dot product. In this source that
agreement is a single `where VisionD == TextD` clause on the `similarity`
entry -- everything else about the two towers is free to differ.

## The vision tower

A 224-pixel image is cut into 196 patches of 16 pixels and embedded to width
768 with learned position embeddings, exactly like the ViT already in Nest,
**except there is no class token**. Instead, the image embedding comes from
an attention pooling head (`vision_model.head.*` in the checkpoint): a
single learned probe vector attends over the 196 patch tokens as the query,
with the patch tokens as key and value, followed by a LayerNorm and an MLP
added back as a residual around the probe's own attention output. PyTorch's
`nn.MultiheadAttention`, which the reference implementation builds this head
from, stores its query/key/value projections as one fused `[3D, D]` matrix;
the source keeps that layout and recovers each projection by slicing, the
same way GPT-2's fused QKV projection is split in the model already in Nest.

## The text tower

A bidirectional encoder (no causal mask) over a fixed 64-token window with
learned positions -- SigLIP always pads or truncates to exactly this length,
never a shorter sequence. The sentence embedding is a linear projection
(`text_model.head.*`) of the encoder's **last** position, position 63,
unconditionally. This is not a class token and not an end-of-sequence
position picked out with a padding mask: because the sequence length never
varies, "last position" is simply a fixed index, and `text_features` takes
tokens shaped to exactly `MaxPositions` rather than a free sequence length.

Neither tower masks anything internally: images are one fixed grid of
patches and text is always the model's full 64-token window, so both run
plain full attention over every position. `std.nn.attention::attention`
takes an unbatched `Tensor[Q, K; bool]` mask, which cannot express
per-sequence padding; SigLIP's own fixed-length convention sidesteps the
question rather than needing that mask.

## Activation

Both towers use `hidden_act = "gelu_pytorch_tanh"`, which is exactly the
tanh approximation `std.nn.activations::gelu` computes. Unlike BERT-style
checkpoints whose config specifies plain (erf) `gelu`, there is no
approximation gap to budget for here -- the tolerance in this model's
validation comes only from floating-point accumulation order, not from a
different activation function.

## Loading

```python
from linnet import nest

model = nest.load("siglip-base-patch16-224", backend="torch")
image_embeds = model.image_features(images)        # Tensor[B, 3, 224, 224; f32] -> Tensor[B, 768; f32]
text_embeds = model.text_features(tokens)           # Tensor[B, 64; i32] -> Tensor[B, 768; f32]
logits_per_text = model.similarity(images, tokens)  # the card's main entry
```

Images use SigLIP's own normalization (mean and standard deviation 0.5,
like ViT) in `[B, C, H, W]`; tokens are the SigLIP tokenizer's output padded
to length 64 with `padding="max_length"`, since that is how the model was
trained. `preprocessor_config.json` and the tokenizer files in the Hub
repository have the exact values.

## Entries

| Entry | |
| --- | --- |
| `image_features<B>(image)` | the pooled image embedding, before normalization |
| `text_features<B>(tokens)` | the pooled sentence embedding, before normalization |
| `similarity<Bi, Bt>(image, tokens)` | `logits_per_text`: each text's dot product against each image, L2-normalized, scaled by `exp(logit_scale)` and shifted by `logit_bias` |

`similarity` is the card's main entry, since it is the only one that
exercises both towers and the constraint tying their widths together.
`logits_per_image` (as the reference implementation names it) is the same
matrix transposed; nothing else changes. Applying `sigmoid` to an entry of
`similarity`'s output gives the probability that a given image and text
match, the way the reference model's own usage example does.

## Provenance

- Weights: [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224), Apache-2.0.
- Code: [google-research/big_vision](https://github.com/google-research/big_vision).
- Paper: [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343).
