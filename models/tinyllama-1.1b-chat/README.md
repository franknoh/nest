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
| `prefill_slot<S>(tokens, slot, length)` | one request's prompt into row `slot` of the caches (padded to `S`, the first `length` tokens real), as it joins a batch being served |
| `decode_rows(tokens, positions)` | one token for every row of the caches, each at its own position: the step `linnet.serve` takes for continuous batching |
| `generate<Steps>(token, pos)` | greedy decoding in the graph |
| `sample<Steps>(token, pos, key, temperature)` | sampling with `std.random` |
| `generate_until<MaxNew>(token, pos, eos)` | decoding until an end token |

## Provenance

- Weights: [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0), Apache-2.0.
- Code: [jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama).
- Paper: [TinyLlama: An Open-Source Small Language Model](https://arxiv.org/abs/2401.02385).

The Linnet forward pass matches `transformers` logits to 2e-2 absolute in
f32 (see `python/linnet/tests/torch/test_hf_checkpoints.py` in the Linnet
repository).
