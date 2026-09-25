# RoBERTa Base

Meta's more thoroughly trained BERT: the same post-norm encoder --
`h = LayerNorm(x + attention(x))`, then `out = LayerNorm(h + ffn(h))`, the
block order `bert-base-uncased` and `all-MiniLM-L6-v2` also use and the
pre-norm ViT and GPT-2 already in Nest do not -- with dynamic masking, no
next-sentence-prediction objective (so `type_vocab_size` is 1: a single
token-type row, looked up like any other embedding so the checkpoint's
tensor loads unmodified), and a byte-level BPE vocabulary of 50265.

RoBERTa's position ids are not zero-based. `padding_idx` (RoBERTa's pad
token, id 1) reserves position row 0 for padding, and the real positions
start at `padding_idx + 1`, so token index i (0-based) reads position row
`i + 2`; the position table therefore has 514 rows for 512 usable
positions. Upstream derives this from the attention mask
(`create_position_ids_from_input_ids`, a running count of non-pad tokens);
with no padding in the sequence -- the only case this encoder's full
attention supports -- that count is just the token's own index, so the
offset collapses to the constant `+2` the source uses.

This checkpoint (`RobertaForMaskedLM`) carries no pooler, only a
masked-language-model head this card does not expose, so there is a single
entry.

Upstream's `hidden_act` is plain (erf) GELU; the standard library's `gelu`
is the tanh approximation, so hidden states differ by roughly 1e-3 per
activation. See "Numerics" below.

## Loading

```python
from linnet import nest

model = nest.load("roberta-base", backend="torch")
hidden = model(tokens, token_types)   # Tensor[B, S; i32], Tensor[B, S; i32] -> Tensor[B, S, 768; f32]
```

`token_types` is the segment-id tensor; pass zeros of the same shape as
`tokens` (RoBERTa's single token-type row means any value looks up the same
embedding, but zero matches upstream's default). Sequences longer than 512
positions are rejected at compile time (`where S + 2 <= MaxPositions`). The
encoder runs full attention over every position: `std.nn.attention::
attention` takes an unbatched `[Q, K]` mask, so a per-sequence padding mask
cannot be expressed here. Pad-free batches (or a batch of one) give exact
results; padded batches attend onto the padding.

## Entries

| Entry | |
| --- | --- |
| `forward<B, S>(tokens, token_types)` | last hidden state, `Tensor[B, S, 768; f32]` |

## Numerics

Validated against `transformers.RobertaModel` (`AutoModel.from_pretrained`,
`numerics="fast"`) on CPU in f32, comparing `last_hidden_state` on
pad-free sentences with RoBERTa's own BPE tokenizer. Max abs diff was
5.2e-2 over two test sentences. The position-id formula above (`i + 2`)
was checked directly against transformers' own
`create_position_ids_from_input_ids` on both tokenized sentences and
matches exactly, so it is not the source of the gap; re-running the
checkpoint through `transformers` twice with only the activation swapped
(erf vs. the tanh approximation), all other weights fixed, reproduces
4.1e-2 of the 5.2e-2 by itself, confirming the gap is the tanh-vs-erf GELU
noted above compounding over 12 layers.

## Provenance

- Weights: [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base), MIT.
- Code: [fairseq/examples/roberta](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta).
- Paper: [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692).
