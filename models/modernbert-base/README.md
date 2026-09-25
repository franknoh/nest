# ModernBERT-base

Answer.AI and LightOn's 2024 rebuild of the BERT encoder. Only the job is the
same: 22 layers of width 768 with 12 heads, reading a whole sequence at once
and producing one vector per token. Everything inside the layer changed.
Normalization moved in front of both residual branches, every bias was
deleted (`attention_bias`, `mlp_bias` and `norm_bias` are all false), the
absolute position table is gone, and the feed-forward network became a GeGLU.

The part with no analogue in BERT is the attention pattern. Only every third
layer attends over the whole sequence; the two layers between each of those
attend inside a sliding window 128 positions wide, so most of the depth costs
work linear in the sequence length instead of quadratic. The two kinds of
layer also use different rotary base frequencies -- 160000 for the global
layers, which have to resolve positions thousands of tokens apart, and 10000
for the local ones, which never see more than half a window. Missing that
second detail changes every attention score in the model.

## The alternation is in the block structure

A `static for` loop variable is an `i64` scalar, not a compile-time integer,
so it cannot index a sub-block array: there is no way to write "if this
layer's index is a multiple of three". The source encodes the cycle in the
types instead. `LayerCycle` holds the `Period - 1` windowed layers and the
global layer that closes the cycle, and `Model` holds an array of cycles:

```linnet
sub local: [EncoderLayer<D, Heads, Inner, T>; Period - 1]
sub global: EncoderLayer<D, Heads, Inner, T>
```

Each member is reached by name, so each gets its own mask and its own rotary
base with no index arithmetic, and `where (Layers - 1) % Period == 0` makes
the compiler check that the depth actually divides into cycles. Layer 0 is a
global layer too, but it is the one layer whose attention normalization
upstream replaces with an identity -- the embedding LayerNorm has just run --
so it is peeled off as `FirstLayer`, a block with no `attn_norm` at all
rather than an optional parameter that is absent.

## The sliding window

The window is a `[Q, K]` band, which is exactly the shape
`std.nn.attention::attention` takes, so the source writes its own mask in the
style of `causal_mask`:

```linnet
op band_mask<Q: Dim, K: Dim, Window: Dim>() -> Tensor[Q, K; bool] {
    let queries = iota<i32>(Q)
    let keys = iota<i32>(K)
    let mask[q, k] = abs(keys[k] - queries[q]) <= cast<i32>(Window / 2)
    return mask
}
```

`local_attention` counts the whole window, so 128 means 64 back and 64
forward, 129 positions in all. A custom mask costs nothing in the kernel
selection that matters: `linnet explain --numerics fast` still picks
`torch.nn.functional.scaled_dot_product_attention` for the attention itself,
because that registration only requires a boolean mask whose true values mean
attend. What a private op does give up is the mask's own kernel -- the stdlib
`causal_mask` lowers to `torch.tril`, while `band_mask` is built from two
`iota`s every time.

## The fused projections

Two weights in each layer hold two or three tensors' worth of rows, and both
are kept exactly as the checkpoint stores them:

- `attn.Wqkv` is `[3 * D, D]`, laid out as query, then key, then value, each
  already head by head, so the three parts are three slices of the last axis
  of its output.
- `mlp.Wi` is `[2 * Inner, D]`. The **first** half is the branch GELU runs
  on and the **second** half multiplies it (`input, gate = Wi(x).chunk(2)`
  upstream). Reading the halves the other way round is a silent error.

## Loading

```python
from linnet import nest

model = nest.load("modernbert-base", backend="torch")
hidden = model(tokens)      # Tensor[B, S; i32] -> Tensor[B, S, 768; f32]
```

The weights are the published `model.safetensors` (f32) of the masked-LM
checkpoint; `bindings.json` maps every Linnet parameter path to its tensor
name, including the shift between this source's cycle structure and the flat
`model.layers.N` of the checkpoint. The masked-LM head itself
(`head.dense`, `head.norm`, `decoder.bias`) is not part of this card.

Every position attends over every other position it is allowed to see. There
is no padding mask: `std.nn.attention::attention` takes an unbatched
`Tensor[Q, K; bool]`, so a per-sequence mask cannot be expressed today, and a
batch here has to be sequences of the same real length.

## Entries

| Entry | |
| --- | --- |
| `forward<B, S>(tokens)` | the last hidden state, after the final norm |

## Why the GELU matters

`hidden_activation` in the config is plain `gelu`, which to `transformers`
means the error-function form, so the source uses
`std.nn.activations::gelu_erf` rather than the tanh approximation `gelu`.
The two differ by at most 4.7e-4 per activation, which is normally an
acceptable rounding. It is not here, and the reason is worth recording: a
GeGLU multiplies the activated half by an unbounded gate, and twenty-two
pre-norm layers add the result straight into the residual stream. With the
tanh form the last hidden state came out 6.9e-1 from the reference, and
substituting the tanh GELU into `transformers` brought its own output back to
within 1.2e-4 of the Linnet one -- which is how the cause was pinned down
rather than guessed, and why the standard library now has `gelu_erf`.

## Provenance

- Weights: [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base), Apache-2.0.
- Code: [AnswerDotAI/ModernBERT](https://github.com/AnswerDotAI/ModernBERT).
- Paper: [Smarter, Better, Faster, Longer](https://arxiv.org/abs/2412.13663).

The last hidden state matches `transformers.AutoModel` to **1.6e-4** absolute
in f32, over a fixed 256-token sequence -- twice the local window, so the
windowed layers really do mask most of the sequence rather than quietly
degenerating to full attention.
