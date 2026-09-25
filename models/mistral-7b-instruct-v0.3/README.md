# Mistral 7B Instruct v0.3

A 7.25B-parameter decoder with the Llama architecture, instruction-tuned:
grouped-query attention (32 query heads, 8 key/value heads, head dimension
128), rotary positions with base 1000000, RMS normalization, a SwiGLU MLP,
and an untied output head, with a KV cache for token-by-token decoding. The
context is 32768 tokens; v0.3 has no sliding window (`config.json` sets
`sliding_window` to `null`), so attention is full rather than windowed, and
its vocabulary of 32768 tokens is larger than v0.2's.

The Linnet source is the same Llama package the other decoders in this zoo
use (`tinyllama-1.1b-chat`, `smollm2-1.7b-instruct`), with `THETA` set to
this model's rotary base and the generics set to its width, head counts, and
depth. Unlike those two, the output head is not tied to the embedding:
`bindings.json` maps `lm_head.weight` to the checkpoint's own separate
`lm_head.weight` tensor.

## Loading

```python
from linnet import nest

model = nest.load("mistral-7b-instruct-v0.3", backend="torch", numerics="fast")
logits = model(tokens)                      # Tensor[1, S; i32] -> Tensor[1, S, 32768; bf16]
```

The weights are the published, sharded `model-0000N-of-00003.safetensors`
(bf16); `bindings.json` maps Linnet parameter paths to their tensor names.
Every name, shape, and dtype is checked before anything runs.

## Entries

| Entry | |
| --- | --- |
| `forward<B, S>(tokens)` | logits for a whole sequence |
| `next_token<B, S>(tokens)` | logits for the last position |
| `decode(token, pos)` | one token through the KV caches (`Batch = 1`, `MaxSeq = 32768`) |
| `prefill<S>(tokens, pos)` | a whole prompt through the KV caches in one pass; logits after its last token, `decode` continues at `pos + S` |
| `generate<Steps>(token, pos)` | greedy decoding in the graph |
| `sample<Steps>(token, pos, key, temperature)` | sampling with `std.random` |
| `generate_until<MaxNew>(token, pos, eos)` | decoding until an end token |

## Provenance

- Weights: [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3), Apache-2.0.
- Code: [mistralai/mistral-inference](https://github.com/mistralai/mistral-inference).
- Paper: [Mistral 7B](https://arxiv.org/abs/2310.06825) (the base architecture; v0.3 extends its tokenizer and drops the sliding window).

## Validation

Checked against `transformers` (`MistralForCausalLM`) on CPU, `torch.no_grad()`,
under the project's memory discipline for multi-billion-parameter models.

- **Truncated, f32, tight tolerance -- done.** Both sides built from only
  the first 2 of 32 layers' real weights (`generics={"Layers": 2, "T":
  "f32"}` on the Linnet side, `num_hidden_layers=2` on the reference, both
  reading the same 21 tensors, cast to f32), prompt `"The capital of
  France is"`. Max absolute difference on the logits: **4.3e-6**; the
  last-position argmax agrees. (A first attempt left the Linnet side at
  its native bf16 against the f32 reference and saw a 6e-2 max absolute
  difference with the same argmax match -- bf16 storage-rounding noise
  consistent with this zoo's other bf16-only checkpoints, not an
  architecture bug; the all-f32 comparison above is the one that isolates
  the architecture.)
- **Whole model, bf16, loose tolerance -- deferred, not run here.** Loading
  all 32 layers in bf16 on both sides needs roughly 14 GB of RSS per side,
  which this pass would hold on the shared CPU machine this card was
  written on. By instruction, that measurement is deferred to a batched
  GPU validation run instead of being run here; treat it as **not yet
  checked** until that run reports a top-1 token agreement and a bf16-scale
  (order 1e-2) logits max absolute difference for this model.
