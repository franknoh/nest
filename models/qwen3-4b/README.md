# Qwen3 4B

A 4B-parameter decoder with the Qwen3 architecture: 36 layers of width
2560, grouped-query attention (32 query heads, 8 key/value heads, head
dimension 128 -- not `H / Heads`, which would be 80; Qwen3 sizes its
projections by `Heads * HeadDim`, not by `H`), rotary positions with base
1,000,000, RMS normalization with epsilon 1e-6, and a SwiGLU MLP of width
9728. The query and key projections carry no bias (`attention_bias` is
false in config.json, unlike Qwen2.5); the change from Qwen2 that does
matter here is QK-norm, an RMS normalization of every head's
`head_dim`-wide slice of the query and key projections, applied after the
projection is split into heads but before rope turns it. The output head
is tied to the token embedding.

The Linnet source is a self-contained Qwen3 package (`src/lib.linnet`,
`src/attention.linnet`, `src/norm.linnet`, `src/rope.linnet`), the same one
`models/qwen3-8b` uses -- Qwen3-4B and Qwen3-8B are the same architecture at
two sizes, differing in width, MLP width, and whether the head is tied,
all of which are generic parameters or a `bindings.json` choice, not
structural. Three changes from the Qwen2 decoder `models/qwen2.5-0.5b-instruct`
uses:

- `HeadDim` is its own generic rather than `H / Heads`: Qwen3 sizes the
  query, key, and value projections by `Heads * HeadDim` and
  `KvHeads * HeadDim`, and the two do not agree with `H` here (`H` is 2560,
  `Heads * HeadDim` is 4096). `q_proj`/`k_proj`/`v_proj` therefore map `H`
  to `Heads * HeadDim` / `KvHeads * HeadDim`, and `o_proj` maps
  `Heads * HeadDim` back to `H`, not `H` to `H`.
- `q_norm` and `k_norm`, in `src/norm.linnet`'s `RmsNorm` block instantiated
  at width `HeadDim`, called on the `[B, Heads, S, HeadDim]` tensor
  `split_heads` produces (and the one-position slice `decode` uses), before
  rope. `RmsNorm::forward` normalizes only its last axis, so this
  normalizes each head's vector independently rather than the flattened
  `Heads * HeadDim` projection -- putting it before the `split_heads`
  reshape would normalize over all heads at once, which is a different
  (and wrong) computation.
- `attention_bias` is false, so `q_proj`, `k_proj`, and `v_proj` carry no
  bias -- `std.nn.linear::Linear`'s optional bias parameter already
  defaults to `none`, so this needs no structural change, only that
  `bindings.json` has no `*.bias` entries for them (nor did Qwen2's
  `o_proj` and MLP projections).

`bindings.json` maps both `embedding.weight` and `lm_head.weight` to the
checkpoint's single `model.embed_tokens.weight` (`tie_word_embeddings` is
true in config.json, unlike the 8B checkpoint of this family), which is how
tied embeddings are expressed without a language feature for them. The
epsilon Qwen3 uses for `RMSNorm` (1e-6) is tighter than the stdlib
`RmsNorm` block's default (1e-5), so `src/norm.linnet` defines its own
`RmsNorm` block that calls `std.nn.norm::rms_norm` directly with the
checkpoint's epsilon; the same block, instantiated at `H` and at `HeadDim`,
serves both the per-layer norms and QK-norm.

## Loading

```python
from linnet import nest

model = nest.load("qwen3-4b", backend="torch", numerics="fast", compile="inductor")
logits = model(tokens)                      # Tensor[1, S; i32] -> Tensor[1, S, 151936; bf16]
```

The weights are the published, sharded `model-0000N-of-00003.safetensors`
(bf16); `bindings.json` maps Linnet parameter paths to their tensor names.
Every name, shape, and dtype is checked before anything runs.

## Entries

| Entry | |
| --- | --- |
| `forward<B, S>(tokens)` | logits for a whole sequence |
| `next_token<B, S>(tokens)` | logits for the last position |
| `decode(token, pos)` | one token through the KV caches (`Batch = 1`, `MaxSeq = 40960`) |
| `prefill<S>(tokens, pos)` | a whole prompt through the KV caches in one pass; logits after its last token, `decode` continues at `pos + S` |
| `generate<Steps>(token, pos)` | greedy decoding in the graph |
| `sample<Steps>(token, pos, key, temperature)` | sampling with `std.random` |
| `generate_until<MaxNew>(token, pos, eos)` | decoding until an end token |

## Provenance

- Weights: [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), Apache-2.0.
- Code: [QwenLM/Qwen3](https://github.com/QwenLM/Qwen3).
- Paper: [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388).

The model is large enough that comparing the full checkpoint against
`transformers` in one pass on shared CPU hardware is impractical, so this
was validated on CPU with both sides truncated to the first 2 layers
(`num_hidden_layers = 2` on the `transformers` side, `generics =
{"Layers": 2}` on this one, both loaded in f32, real weights read from the
published checkpoint): on a chat-templated prompt, the Linnet forward pass
matches `transformers` logits to a maximum absolute difference of 5.0e-6
on the final position (well inside the usual 2e-3), with the top-1 next
token agreeing exactly; peak RSS for the comparison was 7.2 GiB. This
proves the architecture -- every weight this checkpoint has, read
correctly, through the right shapes -- but not the whole depth. A second
pass, comparing the full model in bf16 (top-1 agreement and the logits'
overall magnitude of difference), is deferred to a batched GPU run across
every Nest model and **has not been done here**; treat that pass as
outstanding, not as having passed.
