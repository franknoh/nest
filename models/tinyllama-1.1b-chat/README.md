# TinyLlama 1.1B Chat v1.0

A 1.1B-parameter decoder with the Llama 2 architecture, pretrained on
3 trillion tokens and chat-tuned. The Linnet source is the Llama package:
grouped-query attention (32 query heads, 4 key/value heads), rotary
positions with base 10000, RMS normalization, and a SwiGLU MLP, with a KV
cache for token-by-token decoding.

## Loading

```python
from linnet import nest

model = nest.load("tinyllama-1.1b-chat", backend="torch", numerics="fast", compile="inductor")
logits = model(tokens)                      # Tensor[1, S; i32] -> Tensor[1, S, 32000; bf16]
```

The weights are the published `model.safetensors` (bf16); `bindings.json`
maps Linnet parameter paths to its tensor names. Every name, shape, and
dtype is checked before anything runs.

## Entries

| Entry | |
| --- | --- |
| `forward<B, S>(tokens)` | logits for a whole sequence |
| `next_token<B, S>(tokens)` | logits for the last position |
| `decode(token, pos)` | one token through the KV caches (`Batch = 1`, `MaxSeq = 2048`) |
| `prefill<S>(tokens, pos)` | a whole prompt through the KV caches in one pass; logits after its last token, `decode` continues at `pos + S` |
| `prefill_slots<M, S>(tokens, slots, lengths)` | `M` requests' prompts into rows `slots` of the caches in one pass (each padded to `S`, its first `lengths[m]` tokens real), as they join a batch being served |
| `prefill_packed<P>(tokens, rows, positions, segments, last)` | several requests' prompts packed end to end into one pass of `P` tokens, each token given its cache row, its position, and its prompt: no padding between prompts, and each prompt sees only itself |
| `decode_rows(tokens, positions)` | one token for every row of the caches, each at its own position: the step `linnet.serve` takes for continuous batching |
| `generate<Steps>(token, pos)` | greedy decoding in the graph |
| `sample<Steps>(token, pos, key, temperature)` | sampling with `std.random` |
| `generate_until<MaxNew>(token, pos, eos)` | decoding until an end token |

## Shards

`Shards` (1 unless bound) splits the model across devices: one shard holds
`Heads / Shards` query heads, `KvHeads / Shards` key and value heads and
`Inner / Shards` hidden units, with the KV caches to match, and the
attention's and the MLP's output projections end in
`std.nn.parallel::all_reduce`, the sum over the shards. On one device the
sum is the value itself. `linnet.torch.load(..., tensor_parallel=mesh)`
binds `Shards` to the mesh size and gives each process its part of the
checkpoint.

## Provenance

- Weights: [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0), Apache-2.0.
- Code: [jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama).
- Paper: [TinyLlama: An Open-Source Small Language Model](https://arxiv.org/abs/2401.02385).

The Linnet forward pass matches `transformers` logits to 2e-2 absolute in
f32 (see `python/linnet/tests/torch/test_hf_checkpoints.py` in the Linnet
repository).
