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
| `prefill_slots<M, S>(tokens, slots, lengths)` | `M` requests' prompts into rows `slots` of the caches in one pass (each padded to `S`, its first `lengths[m]` tokens real), as they join a batch being served |
| `prefill_packed<P>(tokens, rows, positions, segments, last)` | several requests' prompts packed end to end into one pass of `P` tokens, each token given its cache row, its position, and its prompt: no padding between prompts, and each prompt sees only itself |
| `decode_rows(tokens, positions)` | one token for every row of the caches, each at its own position: the step `linnet.serve` takes for continuous batching |
| `prefill_paged<P, Rows, Pages>(tokens, positions, rows, slots, last, table)` | for serving from pages: prompts packed end to end, each token written at its place in the pool and attending over its row's pages up to its position (chunks, shared prefixes) |
| `decode_paged<Rows, Pages>(tokens, positions, table)` | `decode_rows` for serving from pages: loaded with `Batch = 1`, the caches' one row is a pool of `MaxSeq` positions in pages of `PageSize` (64), and row `b`'s positions lie in the pages `table[b]` lists |
| `step_paged<P, Rows, Pages>(...)` | `prefill_paged` and `decode_paged` in one pass |
| `generate<Steps>(token, pos)` | greedy decoding in the graph |
| `sample<Steps>(token, pos, key, temperature)` | sampling with `std.random` |
| `generate_until<MaxNew>(token, pos, eos)` | decoding until an end token |

## Shards

`Shards` (1 unless bound) splits the model across devices: one shard holds
`Heads / Shards` query heads, `KvHeads / Shards` key and value heads and
`Inner / Shards` hidden units, with the KV caches to match, and the
attention's and the MLP's output projections end in
`std.nn.parallel::all_reduce`, the sum over the shards. `lm_head` holds
`Vocab / Shards` rows of the vocabulary, and `std.nn.parallel::all_gather`
sets the shards' slices of the logits side by side. On one device the sum
and the gather are the value itself.
`linnet.torch.load(..., tensor_parallel=mesh)` binds `Shards` to the mesh
size and gives each process its part of the checkpoint.
The training entries train split as well: each split computation reads its
input through `std.nn.parallel::shared`, and `loss_packed` and
`log_probs_packed` run over the vocabulary's parts
(`std.nn.loss::split_cross_entropy`, `split_token_log_probs`).


## Provenance

- Weights: [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct), Apache-2.0.
- Code: [huggingface/smollm](https://github.com/huggingface/smollm).
- Paper: [SmolLM2: When Smol Goes Big](https://arxiv.org/abs/2502.02737).
