# SAM ViT-Base

Meta's Segment Anything Model, and the first segmentation model in Nest.
Segmentation here is shaped very differently from the classifiers and
language models already in the zoo: an image is run through a heavy ViT
encoder **once** to a 64x64 grid of 256-channel embeddings, and then any
number of cheap prompts -- points, boxes -- are decoded against that same
encoding by a small transformer, each decode costing a fraction of the
encode. That split is why the source has three entries instead of one:
`encode_image` is the expensive part, `decode_points` and `decode_boxes` are
not.

## The image encoder

A 1024-pixel image is cut into 16-pixel patches onto a 64x64 grid and run
through 12 ViT-Det layers of width 768. Ten of the twelve restrict attention
to non-overlapping 14x14 windows; the four the checkpoint's
`global_attn_indexes` names (layers 2, 5, 8, 11) attend over the full 64x64
grid instead. 64 does not divide evenly by 14, so a windowed layer pads the
grid to 70x70 (five windows per side) before cutting it apart and crops back
to 64x64 afterward -- `window_partition`/`window_unpartition` in the source,
built from nothing but `reshape`, `permute`, and a bottom/right zero-pad,
since the padding only ever needs to happen on the high side.

Every layer, windowed or global, adds a **decomposed relative position
bias** to its attention scores before the softmax, from `rel_pos_h` and
`rel_pos_w`. This is the part of SAM most likely to need something the
language does not have, and it turns out not to: because every query and
key range in this checkpoint are the same size (14 for a window, 64 for a
global layer), the reference's `get_rel_pos` never has to resize the table,
so the relative offset between query position `qi` and key position `ki` is
a plain `iota` difference, `qi - ki + (Wh - 1)`. Building that as an index
*tensor* and gathering `rel_pos_h[idx[qh, kh], c]` through it is exactly
`std.nn.conv::conv2d`'s own `iota`-built-gather, just gathering a position
table instead of an input window. The bias itself is query-dependent (each
head's query vector dotted with the gathered table), so it cannot be
expressed as `std.nn.attention::attention`'s fixed `Tensor[Q, K; bool]` mask
-- the source writes the windowed attention out by hand instead, the same
way `conv2d` writes its own contraction out by hand.

The neck is two convolutions (1x1 then 3x3) with a channels-first LayerNorm
after each, taking the 768-channel tokens down to the 256-channel image
embedding every prompt is decoded against.

## The prompt encoder

Point and box prompts are embedded through a random-Fourier positional
encoding: `sin`/`cos` of `2 * pi * ((2 * coords - 1) @ pe)`, where `pe` is
the checkpoint's `shared_image_embedding.positional_embedding` tensor,
**read directly rather than regenerated** (it is a fixed random matrix
frozen at training time, not a computed table). A point's label then
selects a learned embedding added on top: `1` (foreground) and `0`
(background) each add their own vector, `-1` replaces the positional
encoding entirely with a separate `not_a_point` embedding. A box becomes two
corner points, tagged with their own pair of learned embeddings instead of
a label. The same 64x64-grid positional encoding is added to the image side
of the decoder every time, from the identical frequency table -- computed
once per decode from `iota`, not stored.

**Not implemented**: dense mask prompts (`prompt_encoder.mask_embed.*` in
the checkpoint). Every decode uses the reference's default dense embedding,
`no_mask_embed`, broadcast over the 64x64 grid. `attention_similarity` and
`target_embedding` (the PerSAM personalization inputs) are likewise not
exposed. The reference's `-10` padding label, which the *processor* uses
internally to mark a point invisible to the encoder, is not implemented
either -- only `1`, `0`, and `-1` are recognized here.

## The mask decoder

A two-layer two-way transformer: the sparse tokens (an IoU token, four mask
tokens, and the prompt embeddings) self-attend, cross-attend to the dense
image tokens, run an MLP, and the image tokens cross-attend back -- in that
order, in every layer. The cross-attentions run at half width
(`attention_downsample_rate = 2`); the self-attention does not. The block's
own MLP uses **ReLU**, per `mask_decoder_config.hidden_act`, unlike the
image encoder and prompt encoder, which use GELU.

The reference's first two-way layer skips adding the sparse positional
embedding before its self-attention and takes that attention's output
directly rather than as a residual (`skip_first_layer_pe`). Linnet has no
boolean generic to select that at compile time, so instead of a runtime
branch the block exposes two forward methods, `forward_first` and
`forward_rest`, sharing every parameter; `TwoWayTransformer` calls the first
on layer 0 and the second on layer 1.

The upscaling head is two transposed convolutions of kernel 2 and stride 2.
Kernel equal to stride means no overlap, so unlike a general transposed
convolution this needs no gather at all: each input pixel projects straight
to its own 2x2 output block, which is a contraction over channels
(`weight[Cin, Cout, kh, kw]`) followed by a `permute`/`reshape` rather than
anything windowed -- `deconv2x2` in the source. Mask logits come from a dot
product between the four mask tokens' own hypernetwork MLPs and the
upscaled image features; an IoU head reads the IoU token the same way.

The source always produces the full four mask candidates (index 0 is the
reference's `multimask_output=False` single mask, indices 1-3 are its
`multimask_output=True` triple) rather than slicing at a runtime flag,
since Linnet needs a static output shape; a caller picks the mask it wants
from the four.

## Loading

```python
from linnet import nest

model = nest.load("sam-vit-base", backend="torch")

image_embeddings = model.encode_image(pixel_values)  # [B, 3, 1024, 1024] -> [B, 256, 64, 64]

# input_labels: 1 foreground, 0 background, -1 padding/background-only point
masks, iou = model.decode_points(image_embeddings, input_points, input_labels)
# input_points: [B, Pts, N, 2], input_labels: [B, Pts, N] i32
# masks: [B, Pts, 4, 256, 256], iou: [B, Pts, 4]

masks, iou = model.decode_boxes(image_embeddings, input_boxes)
# input_boxes: [B, Nb, 4] as (x1, y1, x2, y2)
```

Pixel values are `[B, 3, 1024, 1024]`, resized and normalized the way
`SamProcessor` does it (`preprocessor_config.json` in the Hub repository has
the mean and standard deviation). Point and box coordinates are in that same
1024-pixel input space, not the original image's.

## Entries

| Entry | |
| --- | --- |
| `encode_image<B>(pixel_values)` | the 64x64 image embedding |
| `decode_points<B, Pts, N>(image_embeddings, input_points, input_labels)` | mask logits and IoU predictions for point prompts |
| `decode_boxes<B, Nb>(image_embeddings, input_boxes)` | mask logits and IoU predictions for box prompts |

## Numerics

SAM's configs all say plain `gelu` (the error-function form): `hidden_act`
is `"gelu"` for both the vision encoder and the prompt encoder.
`std.nn.activations::gelu_erf` is used throughout rather than the tanh
approximation `std.nn.activations::gelu`, which the language tour warns
differs by about 4.7e-4 per activation. The mask decoder's own two-way
transformer block uses **ReLU** instead (`mask_decoder_config.hidden_act` is
`"relu"`), and its upscaling head's activation is plain (erf) GELU too,
`gelu_erf` again.

Against `transformers.SamModel` exactly as published, in f32 on CPU, batch
1, with a fixed random image and a fixed point and box prompt:

| | max abs diff |
| --- | --- |
| image embeddings | 1.4e-6 |
| point-prompted mask logits (single mask) | 3.1e-5 |
| point-prompted mask logits (multi-mask) | 8.4e-5 |
| point-prompted IoU predictions | 6.9e-6 |
| box-prompted mask logits (single mask) | 7.8e-5 |
| box-prompted mask logits (multi-mask) | 1.9e-4 |
| box-prompted IoU predictions | 3.1e-6 |

One detail of the prompt encoder is load-bearing: when a prompt has points
and no box, SAM appends a padding point (label `-1`, embedded as
`not_a_point`) before decoding, and `decode_points` does the same. Without
it the mask still looks right -- about 0.97 IoU against the reference -- but
the mask logits are a full unit off, because the decoder was trained with
that extra token.

## Provenance

- Weights: [facebook/sam-vit-base](https://huggingface.co/facebook/sam-vit-base), Apache-2.0.
- Code: [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything).
- Paper: [Segment Anything](https://arxiv.org/abs/2304.02643).
