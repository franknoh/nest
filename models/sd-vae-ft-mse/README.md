# SD VAE ft-MSE (decoder)

Stability AI's MSE-finetuned autoencoder for Stable Diffusion. This is
Nest's first diffusion-family model, and the first that is not a classifier
or a language model: it turns a 4-channel latent grid into an 8x-larger RGB
image, and along the way it exercises group normalization, nearest-neighbour
upsampling, and self-attention over spatial positions instead of a token
sequence.

Only the **decoder** (`decode`) is implemented here. The encoder half of the
checkpoint (`encoder.*` and `quant_conv.*`, which turn an image into the
latent distribution `decode` reverses) is future work; this card's
`bindings.json` binds only the `decoder.*` and `post_quant_conv.*` tensors.

## Architecture

A 1x1 convolution (`post_quant_conv`) reads the 4-channel latent, then
`decoder.conv_in` widens it to 512 channels with a 3x3 convolution. A mid
block runs a residual block, one spatial self-attention, and a second
residual block, all at that width. Four up blocks follow, three of them
ending in a nearest-neighbour upsample and a 3x3 convolution that doubles
the resolution; the widths are 512, 512, 512, 256, 128, so only the last two
up blocks narrow the channel count (with a 1x1 convolution on their
shortcut) and only the first three upsample — the fourth is already at the
full 8x resolution. `conv_norm_out`, `SiLU`, and `conv_out` produce the
3-channel image.

A residual block is `group_norm(32 groups, eps 1e-6) -> SiLU -> conv3x3 ->
group_norm -> SiLU -> conv3x3`, added to its input (widening blocks project
the input through a 1x1 convolution first, since the addition needs matching
channel counts).

The mid-block attention is ordinary scaled dot-product self-attention over
the `H * W` spatial positions, with a single head spanning the full channel
width, preceded by its own group norm. This checkpoint predates the
`to_q`/`to_k`/`to_v`/`to_out.0` naming current `diffusers` checkpoints use;
its raw tensor names are `query`, `key`, `value`, `proj_attn` (all plain
`[C, C]` linear layers), and `bindings.json` binds to those directly rather
than to names the checkpoint does not have.

## Latent scaling

`decode` is the architecture only — the same way `diffusers.AutoencoderKL`'s
own `decode` method has no notion of a scaling factor. The Stable Diffusion
pipeline scales its latent by `1 / 0.18215` before calling the VAE decoder
(and the encoder side would multiply by `0.18215` going the other way).
That scaling belongs to the **caller**, not to this entry: apply it to your
latent before calling `decode` if you are reproducing the pipeline. If you
are decoding a latent that is not already in the pipeline's convention (for
example one taken directly from this checkpoint's own encoder, once that
side is implemented), no scaling is needed.

## Loading

```python
from linnet import nest

model = nest.load("sd-vae-ft-mse", backend="torch")
image = model.run_entry("decode", [latent / 0.18215])   # Tensor[B, 4, H, W; f32] -> Tensor[B, 3, H * 8, W * 8; f32]
```

The card's generics fix the latent at 16x16 (decoding to 128x128) so `check`
and export stay fast; pass `generics={"H": ..., "W": ...}` to `nest.load` to
decode a different size, for instance 64x64 for the usual 512x512 Stable
Diffusion image.

## Entries

| Entry | |
| --- | --- |
| `decode<B>(latent)` | `Tensor[B, 4, H, W; f32]` -> `Tensor[B, 3, H * 8, W * 8; f32]` |

## Provenance

- Weights: [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse), MIT.
- Code: [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion).
- Paper: [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752).
