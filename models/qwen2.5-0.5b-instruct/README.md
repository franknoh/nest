# Qwen2.5 0.5B Instruct

A 0.5B-parameter decoder with the Qwen2 architecture, instruction-tuned. 24
layers of width 896, grouped-query attention (14 query heads, 2 key/value
heads, head dimension 64), rotary positions with base 1,000,000, RMS
normalization with epsilon 1e-6, and a SwiGLU MLP of width 4864. The query,
key, and value projections carry a bias; the output projection and the MLP
do not. The output head is tied to the token embedding.

The Linnet source is a self-contained Qwen2 package (`src/lib.linnet`,
`src/attention.linnet`, `src/rope.linnet`), adapted from the Llama-style
decoder the other Nest models use. `std.nn.linear::Linear` already carries
an optional bias, so the query/key/value bias needs no structural change --
only `bindings.json` entries for `q_proj.bias`, `k_proj.bias`, and
`v_proj.bias`, left unbound (and so `none`) for `o_proj` and the MLP
projections, which the checkpoint has none of. `bindings.json` maps both
`embedding.weight` and `lm_head.weight` to the checkpoint's single
`model.embed_tokens.weight` (`tie_word_embeddings` in config.json), which is
how tied embeddings are expressed without a language feature for them. The
epsilon Qwen2.5 uses for `RMSNorm` (1e-6) is tighter than the stdlib
`RmsNorm` block's default (1e-5), so this model defines its own `RmsNorm`
block that calls `std.nn.norm::rms_norm` directly with the checkpoint's
epsilon.

## Loading

```python
from linnet import nest

model = nest.load("qwen2.5-0.5b-instruct", backend="torch", numerics="fast", compile="inductor")
logits = model(tokens)                      # Tensor[1, S; i32] -> Tensor[1, S, 151936; bf16]
```

The weights are the published `model.safetensors` (bf16); `bindings.json`
maps Linnet parameter paths to its tensor names. Every name, shape, and
dtype is checked before anything runs.

## Entries

| Entry | |
| --- | --- |
| `forward<B, S>(tokens)` | logits for a whole sequence |
| `next_token<B, S>(tokens)` | logits for the last position |
| `decode(token, pos)` | one token through the KV caches (`Batch = 1`, `MaxSeq = 32768`) |
| `generate<Steps>(token, pos)` | greedy decoding in the graph |
| `sample<Steps>(token, pos, key, temperature)` | sampling with `std.random` |
| `generate_until<MaxNew>(token, pos, eos)` | decoding until an end token |

## Provenance

- Weights: [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), Apache-2.0.
- Code: [QwenLM/Qwen2.5](https://github.com/QwenLM/Qwen2.5).
- Paper: [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115).

The Linnet forward pass matches `transformers` logits on the final position
to within 2e-3 absolute in f32, with the top-1 next token agreeing, on a
prompt run through the model's chat template. The `decode` entry's cached
logits match a full forward pass over the same extended sequence to the same
tolerance.
