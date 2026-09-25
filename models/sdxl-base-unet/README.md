# SDXL Base UNet

The denoising UNet of Stability AI's Stable Diffusion XL base model: 2.6
billion parameters, the largest model in Nest and by a wide margin the
largest diffusion model. One call is one denoising step -- the network
predicts the noise present in a latent, which a sampler subtracts before
calling it again.

It is the counterpart to [`sd-vae-ft-mse`](../sd-vae-ft-mse), which provides
the other end of the pipeline: this UNet cleans a 4-channel latent, and that
decoder turns the finished latent into an 8x-larger RGB image. The two
together are the whole image path of Stable Diffusion, with only the sampler
loop and the text encoders outside.

SDXL ships its own VAE under `vae/` in the same repository, and its decoder
is architecturally identical to `sd-vae-ft-mse`: the same 128/256/512/512
widths, 4 latent channels, 32 groups, SiLU. Two things differ if you want to
decode with the SDXL weights rather than the ft-MSE ones. The latent scaling
is 0.13025 instead of 0.18215 -- the caller's job in both cards, so a number
to get right rather than a change to the model. And the SDXL checkpoint
spells its mid-block attention with the current `diffusers` names
(`to_q`, `to_k`, `to_v`, `to_out.0`) where the older ft-MSE checkpoint uses
`query`, `key`, `value`, `proj_attn`, so the eight attention entries of that
card's `bindings.json` would need renaming. Everything else in it binds
as it stands.

Only the UNet is implemented here. The two CLIP text encoders that produce
this card's `text_embedding` and `added_embedding` inputs are separate models
in the same Hub repository (`text_encoder/` and `text_encoder_2/`) and are
not implemented; neither is the VAE under `vae/`.

## Architecture

`conv_in`, a 3x3 convolution, widens the 4-channel latent to 320 channels.
Three down stages, a mid stage, and three up stages follow, at three
resolutions and three widths:

| Stage | Grid | Channels | Residual blocks | Transformer depth |
| --- | --- | --- | --- | --- |
| down 0 | latent | 320 | 2 | none |
| down 1 | latent / 2 | 640 | 2 | 2 |
| down 2 | latent / 4 | 1280 | 2 | 10 |
| mid | latent / 4 | 1280 | 2 | 10 |
| up 0 | latent / 4 | 1280 | 3 | 10 |
| up 1 | latent / 2 | 640 | 3 | 2 |
| up 2 | latent | 320 | 3 | none |

A down stage runs each residual block followed, where it has them, by a
spatial transformer, and ends in a stride-2 convolution that halves the grid;
down stage 2 has no downsampler, being already at the coarsest grid. An up
stage does the same three times over and ends in a nearest-neighbour upsample
followed by a 3x3 convolution; up stage 2 has no upsampler, being already at
the latent's own resolution. The outermost stages, down 0 and up 2, carry no
attention at all: SDXL moved that capacity down to the coarser grids, where
attention over `H * W` positions is cheap, which is why the depths are 2 and
10 rather than 1 at every stage as in SD 1.5.

**Skip connections.** Nine hidden states from the down path are concatenated
along the channel axis into the nine residual blocks of the up path, in
reverse order: the output of `conv_in`, of each down-stage residual block,
and of each downsampler. This is why every up-path residual block needs a
1x1 convolution on its shortcut and why their input widths are 2560, 2560,
1920, then 1920, 1280, 960, then 960, 640, 640. `forward` names these states
`s0` through `s8` and consumes them visibly rather than hiding the
bookkeeping inside a block.

A residual block is `group_norm(32 groups, eps 1e-5) -> SiLU -> conv3x3`,
then the conditioning vector added as a per-channel shift, then
`group_norm -> SiLU -> conv3x3`, added to the input (through a 1x1
convolution when the width changes). The conditioning goes through SiLU
before its own linear projection.

A spatial transformer is a group norm (eps **1e-6** here, which `diffusers`
hard-codes in `Transformer2DModel` instead of reading `norm_eps`), the grid
flattened to a sequence of `H * W` positions, a linear projection in, `N`
transformer blocks, a linear projection out, and the whole thing added to
the input. This checkpoint sets `use_linear_projection`, so the two
projections are linear layers on the flattened sequence rather than 1x1
convolutions on the grid.

A transformer block is self-attention, then cross-attention over the text
embedding, then a gated feed-forward, each pre-normed by its own LayerNorm
and added back. Heads are 64 wide throughout: 10 of them at 640 channels, 20
at 1280. The query, key, and value projections carry no bias; the output
projection does. Cross-attention reads a 2048-wide context, which is the two
CLIP text encoders' hidden states concatenated.

The feed-forward is a GEGLU: the inner width is four times the channel
width, and one projection produces twice that, whose first half is the value
and whose second half is the gate. The gate goes through **`gelu_erf`**,
GELU in its exact error-function form, which is what this checkpoint was
trained with. The tanh approximation in `std.nn.activations::gelu` is off by
about 5e-4 per activation, and a gate multiplies that error by an unbounded
value rather than averaging it away, so the distinction is not cosmetic here.

`conv_norm_out`, SiLU, and `conv_out` turn the 320-channel result back into
a 4-channel noise prediction.

## Conditioning: what the caller computes

Two vectors reach every residual block, summed. This card takes both as
**already-projected sinusoidal tensors**, and owns only the two-linear MLP
that follows each:

- `timestep_embedding`, `Tensor[B, 320; f32]`: the sinusoidal embedding of
  the timestep.
- `added_embedding`, `Tensor[B, 2816; f32]`: the 1280-wide pooled output of
  the second text encoder, concatenated with the 256-wide sinusoidal
  embedding of each of the six micro-conditioning numbers SDXL was trained
  with -- original height and width, crop top and left, target height and
  width -- which is `1280 + 6 * 256 = 2816`.

The projection is deliberately outside the entry: it has no parameters, and
it depends on conventions (the frequency schedule, and whether sine or
cosine comes first) that belong with the pipeline rather than with the
network. Everything with weights in it is inside. The exact calls, which is
what `diffusers` itself does before entering the UNet:

```python
import torch
from diffusers.models.embeddings import get_timestep_embedding

timestep_embedding = get_timestep_embedding(
    torch.tensor([timestep]), 320, flip_sin_to_cos=True, downscale_freq_shift=0
)                                                       # [B, 320]

time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], dtype=torch.float32)
time_embeds = get_timestep_embedding(
    time_ids.flatten(), 256, flip_sin_to_cos=True, downscale_freq_shift=0
).reshape(time_ids.shape[0], -1)                        # [B, 1536]
# [B, 2816]
added_embedding = torch.cat([pooled_prompt_embeds, time_embeds], dim=-1)
```

`pooled_prompt_embeds` is the pooled (`text_embeds`) output of
`text_encoder_2`, and `text_embedding` is the two encoders' penultimate
hidden states concatenated on the last axis, 2048 wide -- both of which
`StableDiffusionXLPipeline.encode_prompt` returns.

Classifier-free guidance is likewise the caller's: run the batch twice, once
with the negative prompt's embeddings and once with the positive prompt's
(or stack them into `B = 2`) and combine the two noise predictions yourself.

## Attention masks

`std.nn.attention::attention` takes an unbatched `Tensor[Q, K; bool]` mask,
so a per-sequence padding mask cannot be expressed today. Cross-attention
here runs over all 77 context positions, which is what the SDXL pipeline
does anyway: it pads the prompt to the full context length and passes no
mask.

## Loading

```python
from linnet import nest

model = nest.load("sdxl-base-unet", backend="torch", numerics="fast")
noise = model.run_entry(
    "forward", [latent, timestep_embedding, text_embedding, added_embedding]
)
```

The card's generics `MidH` and `MidW` are the **coarsest** grid, the
resolution the mid stage runs at, which is a quarter of the latent in each
direction; the latent is `MidH * 4` by `MidW * 4`. They are named that way
because the shape solver is syntactic about division: from a latent dimension
`H` it cannot prove that halving twice and doubling twice returns to `H`, so
the network is parameterized from the bottom up, where every dimension is a
product.

The card fixes `MidH = MidW = 8`, a 32x32 latent -- a 256-pixel image through
the VAE -- so that `check` and the four exports stay quick. SDXL's native
resolution is a **128x128 latent** (a 1024-pixel image), which is
`generics={"MidH": 32, "MidW": 32}`. `Depth1` and `Depth2` are the
transformer depths at the 640- and 1280-channel stages, 2 and 10 in this
checkpoint; lowering them loads a shallower slice of the same weights, which
is how this card was validated against `diffusers` without holding two copies
of a 2.6-billion-parameter model in memory at once.

## Entries

| Entry | |
| --- | --- |
| `forward<B, S>(latent, timestep_embedding, text_embedding, added_embedding)` | `Tensor[B, 4, MidH * 4, MidW * 4; f32]`, `Tensor[B, 320; f32]`, `Tensor[B, S, 2048; f32]`, `Tensor[B, 2816; f32]` -> `Tensor[B, 4, MidH * 4, MidW * 4; f32]` |

## Provenance

- Weights: [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), `unet/diffusion_pytorch_model.safetensors`, CreativeML Open RAIL++-M.
- Code: [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models).
- Paper: [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952).
