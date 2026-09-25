# all-MiniLM-L6-v2

The most downloaded model on the Hugging Face Hub: a BERT encoder distilled
down to 6 post-norm layers of width 384 with 12 heads, wrapped by
sentence-transformers into a general-purpose sentence embedding model. The
`Transformer` stage is exactly a small BERT -- word, learned-position, and
token-type embeddings summed and normalized, then post-norm blocks,
`h = LayerNorm(x + attention(x))` then `out = LayerNorm(h + ffn(h))`, the
same order `bert-base-uncased` and `roberta-base` use and the pre-norm ViT
and GPT-2 already in Nest do not.

The Hub repository's `modules.json` chains three stages: `Transformer`,
`1_Pooling` (`1_Pooling/config.json` sets `pooling_mode_mean_tokens: true`,
i.e. mean pooling over tokens, no CLS or max pooling), and `2_Normalize`
(unit L2 norm). Neither pooling stage has learned weights -- the
checkpoint's `pooler.dense` tensors belong to the underlying `BertModel`
architecture and go unused by the sentence-transformers pipeline, so this
card does not bind them. The `embed` entry below computes the two stages
directly.

Upstream's `hidden_act` is plain GELU, the error-function form, so the
source uses `std.nn.activations::gelu_erf` rather than the tanh
approximation `gelu`.

## Loading

```python
from linnet import nest

model = nest.load("all-MiniLM-L6-v2", backend="torch")
hidden = model(tokens, token_types)          # Tensor[B, S; i32], Tensor[B, S; i32] -> Tensor[B, S, 384; f32]
vectors = model.embed(tokens, token_types)   # -> Tensor[B, 384; f32], unit norm
```

`token_types` is the segment-id tensor; sentence-transformers always passes
zeros for it. Sequences longer than 512 positions are rejected at compile
time. The encoder runs full attention over every position and `embed`'s
mean pooling averages every position uniformly: `std.nn.attention::
attention` takes an unbatched `[Q, K]` mask, so a per-sequence padding mask
cannot be expressed here, and pooling likewise has no mask to exclude
padding from the mean. A pad-free batch (or a batch of one) matches
upstream's masked mean exactly, since every position is then real; a padded
batch does not.

## Entries

| Entry | |
| --- | --- |
| `forward<B, S>(tokens, token_types)` | last hidden state, `Tensor[B, S, 384; f32]` |
| `embed<B, S>(tokens, token_types)` | mean-pooled, L2-normalized sentence embedding, `Tensor[B, 384; f32]` |

## Numerics

Validated against `transformers.AutoModel.from_pretrained` (`numerics=
"fast"`) on CPU in f32 over pad-free sentences: the maximum absolute
difference is 1.8e-6 on `forward` (`last_hidden_state`) and 1.2e-7 on
`embed`, the mean-pooled, L2-normalized sentence embedding.

## Provenance

- Weights: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), Apache-2.0.
- Code: [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers).
- Paper: [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084).
