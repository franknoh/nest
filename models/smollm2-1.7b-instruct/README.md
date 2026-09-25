# SmolLM2 1.7B Instruct

A 1.7B-parameter decoder with the Llama architecture, trained on 11 trillion
tokens and instruction-tuned. 24 layers of width 2048, 32 query heads with no
grouping, a SwiGLU MLP of width 8192, rotary positions with base 130000, and
the output head tied to the token embedding.

The Linnet source is the same Llama package the other decoders use, with
`THETA` set to this model's base frequency: the architecture is a parameter
of the source, not a separate implementation. `bindings.json` maps both
`embedding.weight` and `lm_head.weight` to the checkpoint's single
`model.embed_tokens.weight`, which is how tied embeddings are expressed
without a language feature for them.

## Loading

```python
from linnet import nest

model = nest.load("smollm2-1.7b-instruct", backend="torch", compile="inductor")
logits = model(tokens)
```

## Entries

| Entry | |
| --- | --- |
| `forward<B, S>(tokens)` | logits for a whole sequence |
| `next_token<B, S>(tokens)` | logits for the last position |
| `decode(token, pos)` | one token through the KV caches (`Batch = 1`, `MaxSeq = 8192`) |
| `prefill<S>(tokens, pos)` | a whole prompt through the KV caches in one pass; logits after its last token, `decode` continues at `pos + S` |
| `prefill_slot<S>(tokens, slot, length)` | one request's prompt into row `slot` of the caches (padded to `S`, the first `length` tokens real), as it joins a batch being served |
| `decode_rows(tokens, positions)` | one token for every row of the caches, each at its own position: the step `linnet.serve` takes for continuous batching |
| `generate<Steps>(token, pos)` | greedy decoding in the graph |
| `sample<Steps>(token, pos, key, temperature)` | sampling with `std.random` |
| `generate_until<MaxNew>(token, pos, eos)` | decoding until an end token |

## Provenance

- Weights: [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct), Apache-2.0.
- Code: [huggingface/smollm](https://github.com/huggingface/smollm).
- Paper: [SmolLM2: When Smol Goes Big](https://arxiv.org/abs/2502.02737).
