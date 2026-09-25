# BERT Base (uncased)

The original BERT: word, learned-position, and token-type embeddings summed
and normalized once, then 12 post-norm encoder layers of width 768 with 12
heads. Post-norm is the block order BERT and RoBERTa share and the
pre-norm ViT and GPT-2 already in Nest do not: the residual add happens
first and LayerNorm reads the sum, `h = LayerNorm(x + attention(x))`, then
`out = LayerNorm(h + ffn(h))`, rather than normalizing each branch's input.
There is no final normalization after the last layer, since every block
already ends with one.

The checkpoint's `LayerNorm` weight and bias are stored as `gamma` and
`beta` (an older naming convention this repository predates); `bindings.json`
maps Linnet's `weight`/`bias` names onto them.

Upstream's `hidden_act` is plain GELU, the error-function form, so the
source uses `std.nn.activations::gelu_erf` rather than the tanh
approximation `gelu`. The distinction is not cosmetic: over twelve post-norm
layers the tanh form moves some tokens' hidden states by 1.8e-1.

## Loading

```python
from linnet import nest

model = nest.load("bert-base-uncased", backend="torch")
hidden = model(tokens, token_types)        # Tensor[B, S; i32], Tensor[B, S; i32] -> Tensor[B, S, 768; f32]
pooled = model.pool(tokens, token_types)   # -> Tensor[B, 768; f32]
```

`token_types` is the segment-id tensor (0 for a single sentence); pass a
zero tensor of the same shape as `tokens` unless doing sentence-pair tasks.

Sequences longer than 512 positions are rejected at compile time
(`where S <= MaxPositions`). The encoder runs full attention over every
position: `std.nn.attention::attention` takes an unbatched `[Q, K]` mask, so
a per-sequence padding mask cannot be expressed here. Pad-free batches (or a
batch of one) give exact results; padded batches attend onto the padding.

## Entries

| Entry | |
| --- | --- |
| `forward<B, S>(tokens, token_types)` | last hidden state, `Tensor[B, S, 768; f32]` |
| `pool<B, S>(tokens, token_types)` | `tanh(dense(hidden[:, 0, :]))`, BERT's pooled output, `Tensor[B, 768; f32]` |

## Numerics

Validated against `transformers.BertModel` (`AutoModel.from_pretrained`,
`numerics="fast"`) on CPU in f32, comparing `last_hidden_state` for
`forward` and `pooler_output` for `pool` on pad-free sentences. The maximum
absolute difference is 3.7e-5 for `forward` and 1.1e-6 for `pool`.

## Provenance

- Weights: [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased), Apache-2.0.
- Code: [google-research/bert](https://github.com/google-research/bert).
- Paper: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).
