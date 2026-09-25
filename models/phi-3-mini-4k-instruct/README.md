# Phi-3 Mini 4K Instruct

A 3.8B-parameter decoder: RMS normalization, full (non-grouped) multi-head
attention with rotary positions, a SwiGLU MLP, and an untied output head, no
bias anywhere. `config.json` sets `num_attention_heads` and
`num_key_value_heads` both to 32, so there is no grouped-query attention to
model. Two checkpoint tensors are fused and the Linnet source keeps them
fused, slicing the projected result instead of splitting the weight:
`self_attn.qkv_proj.weight` is one `[3 * H, H]` tensor (query, then key,
then value, along the output axis) and `mlp.gate_up_proj.weight` is one
`[2 * Inner, H]` tensor (gate, then up). Both keep the ordinary `nn.Linear`
`[Out, In]` layout, so `bindings.json` binds them with no transpose.

## Determining the gate/up order

`gate_up_proj`'s halves are not distinguishable from the shape alone. The
reference `Phi3MLP.forward` in `transformers/models/phi3/modeling_phi3.py`
reads:

```python
up_states = self.gate_up_proj(hidden_states)
gate, up_states = up_states.chunk(2, dim=-1)
up_states = up_states * self.activation_fn(gate)
```

`chunk(2, dim=-1)` splits the output axis in storage order, so the first
`Inner` rows of the weight produce `gate` (the half passed through SiLU) and
the second `Inner` rows produce the value `gate` multiplies. The same
ordering (query, key, value) holds for `qkv_proj`, read from
`Phi3Attention.forward`'s `qkv[..., :query_pos]` /
`qkv[..., query_pos : query_pos + num_key_value_heads * head_dim]` /
`qkv[..., query_pos + num_key_value_heads * head_dim :]`. A wrong gate/up
split still produces fluent-looking text, since SwiGLU is not symmetric but
both halves are learned linear projections of the same input -- see
"Validation" below for the check that actually catches it.

## Not modeled

`config.json` sets `sliding_window` to 2047, and recent `transformers`
applies a sliding-window causal mask throughout when it is set. The Linnet
source uses a plain causal mask instead. This only differs from the
reference beyond 2047 tokens of context, which the 4K instruction-tuned
checkpoint rarely reaches and the validation below does not exercise; a
sequence at or under the window sees identical attention either way.

## Loading

```python
from linnet import nest

model = nest.load("phi-3-mini-4k-instruct", backend="torch", numerics="fast")
logits = model(tokens)                      # Tensor[1, S; i32] -> Tensor[1, S, 32064; bf16]
```

The weights are the published checkpoint (bf16, two shards);
`bindings.json` maps Linnet parameter paths to its tensor names. Every name,
shape, and dtype is checked before anything runs.

## Entries

| Entry | |
| --- | --- |
| `forward<B, S>(tokens)` | logits for a whole sequence |
| `next_token<B, S>(tokens)` | logits for the last position |
| `decode(token, pos)` | one token through the KV caches (`Batch = 1`, `MaxSeq = 4096`) |
| `prefill<S>(tokens, pos)` | a whole prompt through the KV caches in one pass; logits after its last token, `decode` continues at `pos + S` |
| `prefill_slot<S>(tokens, slot, length)` | one request's prompt into row `slot` of the caches (padded to `S`, the first `length` tokens real), as it joins a batch being served |
| `decode_rows(tokens, positions)` | one token for every row of the caches, each at its own position: the step `linnet.serve` takes for continuous batching |
| `generate<Steps>(token, pos)` | greedy decoding in the graph |
| `sample<Steps>(token, pos, key, temperature)` | sampling with `std.random` |
| `generate_until<MaxNew>(token, pos, eos)` | decoding until an end token |

## Provenance

- Weights: [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct), MIT.
- Code: [transformers `models/phi3`](https://github.com/huggingface/transformers/tree/main/src/transformers/models/phi3).
- Paper: [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219).

## Validation

A 3.8B model in f32 is too large to compare whole against `transformers` on
this machine, so validation ran truncated, against `transformers` under
`torch.no_grad()` with the tokenizer's chat template:

- **Truncated, f32, tight tolerance.** Both sides read only the first two
  layers of the published checkpoint: the Linnet side through
  `nest.load(..., generics={"Layers": 2, "T": "f32"})` bound from a small
  local file holding only the 15 tensors the first two layers and the
  embedding/head need, cast to f32 from the published bf16 checkpoint and
  never touching the other 30 layers; the reference through
  `AutoModelForCausalLM.from_pretrained(..., config=<num_hidden_layers=2>,
  torch_dtype=torch.float32)`. Both were fed the same 10-token prompt built
  from `tokenizer.apply_chat_template`. This proves the fused-QKV slicing,
  the gate/up split, RoPE, and normalization are wired correctly, since a
  wrong gate/up split still produces fluent-looking logits but would not
  survive a 2e-3 tolerance here.

  **Max abs difference in logits: 7.15e-6.** Argmax (next-token prediction)
  matches exactly. Peak RSS for the whole run (reference built and released
  before the Linnet model, per the shared-machine memory discipline): 2.77
  GiB.

- **Whole model, bf16, top-1 agreement at full depth.** Deferred to a batch
  GPU run (RunPod) covering every model in this pass, alongside the other
  additions to this registry; not yet run for this model. Until that run
  lands, correctness at full depth (32 layers, e.g. rope behavior far past
  the truncated pass) is not directly checked, only implied by the
  truncated pass and by `check`'s shape/dtype verification against the full
  checkpoint's headers.

See `python/linnet/tests/torch/test_hf_checkpoints.py` in the Linnet
repository for the harness this follows.
