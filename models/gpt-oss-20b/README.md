# GPT-OSS 20B

A 21B-parameter mixture-of-experts decoder, of which 3.6B parameters are read
per token. 24 layers of width 2880: RMS normalization with epsilon 1e-5,
grouped-query attention with 64 query heads and 8 key/value heads of dimension
64, and a feed-forward block that is 32 experts wide with 4 chosen per token.
Every projection carries a bias. The output head is not tied to the embedding.

Three things separate it from a Llama-shaped decoder, and each is a module of
its own in `src/`:

- **Attention sinks** (`src/attention.linnet`). Each head owns one learned
  logit, `self_attn.sinks`, that joins its row of attention scores as an extra
  key, takes part in the softmax, and is then discarded. The probabilities over
  real keys therefore sum to less than one, and a head can decline to attend
  anywhere. That is a change to the softmax's normalization, not a mask or a
  bias, so no stdlib attention op can express it;
  `gpt_oss.attention::sink_attention` is a model-local `op` that does, by
  putting the sink's term in the denominator instead of concatenating a column
  and slicing it off again -- the same numbers without a `K + 1` axis.
- **MXFP4 expert weights** (`src/mxfp4.linnet`). See below.
- **YaRN rope and alternating windows** (`src/rope.linnet`, `src/lib.linnet`).
  `layer_types` in config.json alternates a 128-position sliding window with
  full causal attention, starting with sliding. A `sub` array holds one block
  type, so the repeating unit here is a `LayerPair` of one windowed layer and
  one full one; the window is a runtime bound, and the full layer is given a
  bound as wide as the sequence, which is the same mask as no window.
  Parameter paths are `pairs.<i>.sliding.*` and `pairs.<i>.full.*`, which
  `bindings.json` maps to checkpoint layers `2i` and `2i + 1`.

## MXFP4

The official repository publishes the expert weights in MXFP4, not bf16:
`quantization_config.quant_method` is `mxfp4`, and the checkpoint carries
`mlp.experts.gate_up_proj_blocks` and `..._scales` as `u8` tensors rather than
a float `gate_up_proj`. Attention, the routers, the embedding, and the head
stay bf16 (`modules_to_not_convert`).

This card binds those bytes as what they are and dequantizes them in the
source. Nothing pretends a dtype matches:

| Checkpoint tensor | Shape | Dtype | Linnet parameter |
| --- | --- | --- | --- |
| `...experts.gate_up_proj_blocks` | `[32, 5760, 90, 16]` | `u8` | `mlp.gate_up_blocks: Tensor[Experts, 2 * Inner, H / 32, 16; u8]` |
| `...experts.gate_up_proj_scales` | `[32, 5760, 90]` | `u8` | `mlp.gate_up_scales: Tensor[Experts, 2 * Inner, H / 32; u8]` |
| `...experts.down_proj_blocks` | `[32, 2880, 90, 16]` | `u8` | `mlp.down_blocks: Tensor[Experts, H, Inner / 32, 16; u8]` |
| `...experts.down_proj_scales` | `[32, 2880, 90]` | `u8` | `mlp.down_scales: Tensor[Experts, H, Inner / 32; u8]` |

A row of `N` weights is cut into `N / 32` blocks of 32. Each block is 16 bytes
holding two FP4 (E2M1) values each -- low nibble the even element, high nibble
the odd one -- plus one byte of E8M0 scale, an exponent biased by 127. A
weight is its FP4 value times `2 ** (scale - 127)`: 4.25 bits per weight, and
`90 * 32 = 2880`, the width of the projection's input.

`std.quant` cannot express this. Its `unpack_int4` splits bytes into nibbles
with exactly this interleaving, but reads each nibble as a two's-complement
integer in `-8..7`, where MXFP4's nibble is a sign, two exponent bits, and one
mantissa bit selecting one of `0, 0.5, 1, 1.5, 2, 3, 4, 6` and their negatives.
And `dequantize_int8`'s scale is one factor per row (`Tensor[*S; f32]`), where
MXFP4 has one per 32-element block, as a power-of-two exponent byte.
`gpt_oss.mxfp4::dequantize_mxfp4` is the model-local op that covers both
differences. It computes the FP4 values arithmetically rather than from a
lookup table, because the language has no tensor data in source: twice each
value is an integer in `0..12`, so the unpacking stays in one byte per weight
and the only float arithmetic is the final scaling. The block scale is built
from integer shifts rather than `exp`, so it is exact; the shifts saturate at
`2 ** 62` either way, and every scale byte in this checkpoint's 96 expert
scale tensors lies in 115..136, that is `2 ** -12` to `2 ** 9`, so the
saturation is never reached. It returns `[Experts, Rows, N]`, the
checkpoint's own layout -- `Rows` the projection's output width -- and
`src/experts.linnet` reads that layout rather than transposing it, as the
reference implementation does to suit its own parameter.

The dequantization is arithmetic in the graph, so it runs on every forward
pass and a full-depth pass materializes each layer's 530M-element gate/up
weight in turn. That is the price of reading the published bytes instead of a
dequantized mirror. A backend with a fused MXFP4 kernel can select it against
this op's body; `linnet explain` says whether one did.

## Dense routing

The reference gathers the tokens each expert was chosen for and runs the
experts one at a time. That needs indices whose *values* decide which rows are
read, which the language has no shapes for. `src/experts.linnet` instead
evaluates every expert on every token and weighs the results by the router's
probabilities, which are exactly zero outside the top four, so the sum is the
same and costs eight times the arithmetic.

Both expert projections go through `std.nn.linear::linear` with the weight
flattened rather than through a contraction with a free expert axis. That is
not cosmetic. `sum[h] x[b, s, h] * w[e, o, h]` is the natural way to write the
gate/up projection, but `[Experts, 2 * Inner, H]` reshapes to
`[Experts * 2 * Inner, H]` without moving an element, so the flattened form is
the same matrix product and a backend selects its kernel for it, where the
contraction form makes an interpreter materialize the whole
`[B, S, Experts, 2 * Inner, H]` outer product before reducing it -- 85 GiB at
160 positions. For the down projection the same trick needs one more step:
weighing each expert's activation by its router probability *before* the
projection rather than its output after folds the sum over experts into the
projection's own contraction, so that too is a single matrix product.

The top-`k` selection is written without a top-k primitive: count how many
experts beat each expert (ties to the lower index, as `torch.topk` orders
them), push everything ranked `TopK` or worse to `-1e30`, and softmax over all
32. The floored entries come out of the softmax as exactly zero, so the
surviving four probabilities are the reference's softmax over its four kept
logits.

## Loading

```python
from linnet import nest

model = nest.load("gpt-oss-20b", backend="torch", numerics="fast")
logits = model(tokens)                      # Tensor[1, S; i32] -> Tensor[1, S, 201088; bf16]
```

The weights are the published `model-0000N-of-00002.safetensors` shards, which
are numbered from zero, so there are three of them. `bindings.json` maps all
459 tensors to Linnet parameter paths; every name, shape, and dtype is checked
before anything runs.

## Entries

| Entry | |
| --- | --- |
| `forward<B, S>(tokens)` | logits for a whole sequence |
| `next_token<B, S>(tokens)` | logits for the last position |
| `states<B, S>(tokens)` | the last layer's hidden states, before the norm and the head |
| `prefill<S>(tokens, pos)` | a prompt into the KV caches, logits after its last token |
| `decode(token, pos)` | one token per row through the KV caches |
| `prefill_slot<S>(tokens, slot, length)` | one request's prompt into row `slot` of the caches |
| `decode_rows(tokens, positions)` | one token per row, each row at its own position |

The caches are `state` members of the attention blocks, `Batch` rows of
`MaxSeq` positions (the card binds 1 and 4096). A sliding-window layer still
caches every position and masks all but the last 128 of them; the full layers'
window is the whole cache. `prefill_slot` and `decode_rows` are what
`linnet.serve` batches continuously.

A decoded token reads the experts differently from a prompt. `decode` routes
through `MixtureOfExperts.forward_topk`: the four chosen experts' weights are
gathered while still MXFP4 and only they are dequantized, an eighth of the
dense form's work. `decode_rows` keeps the dense form, since in a batch of
rows every expert is chosen by someone and the dense form reads each once;
both still dequantize on every call, which a fused MXFP4 kernel would avoid.

## Validation

Two layers of the real checkpoint -- layer 0 sliding, layer 1 full -- with all
32 experts, run in f32 against `transformers` 5.17 at 160 positions, so the
128-position window is narrower than the sequence and the two layers really do
mask differently:

| | |
| --- | --- |
| max abs logit difference | 1.88e-05 |
| mean abs logit difference | 1.47e-06 |
| argmax agreement | 160 / 160 positions |

`gpt_oss.mxfp4::dequantize_mxfp4` is separately bitwise equal to
`transformers.integrations.mxfp4.convert_moe_packed_tensors` over random
blocks and scales, for every one of the sixteen FP4 codes.

One trap for anyone repeating this. `transformers` dequantizes MXFP4 to
**bf16 regardless of the dtype asked for** -- `convert_moe_packed_tensors`
defaults to `torch.bfloat16`, so `dtype=torch.float32` still leaves
`experts.gate_up_proj` in bf16 while the rest of the model is f32. Comparing
against that gives 9.5e-2, which looks like an architecture bug and is not
one: the dequantized *values* are exact in bf16 (an FP4 value needs three
mantissa bits), so only the accumulation differs. Casting the reference's
expert parameters to f32 after loading is what produces the numbers above.

Not measured here, and left for a GPU run: the whole 24 layers in bf16 against
the same reference, and any timing. Both are memory- rather than
correctness-bound. The two-layer f32 run above peaked at 15.8 GiB of RSS on
the Linnet side, almost all of it the dequantized expert weights and the
interpreter's copies of them, which is why full depth is not a CPU job.

## Provenance

- Weights: [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b), Apache-2.0.
- Code: [openai/gpt-oss](https://github.com/openai/gpt-oss).
- Paper: [gpt-oss-120b & gpt-oss-20b Model Card](https://arxiv.org/abs/2508.10925).
