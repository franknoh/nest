# Qwen3.8 27B

The text model of Qwen3.8-27B, a 27B-parameter hybrid decoder of the
Qwen3.5 architecture (`qwen3_5`), without its vision tower or its
multi-token-prediction layer. Its 64 layers come in fours: three of
linear attention (Gated DeltaNet) and one of gated full attention, each
followed by a SwiGLU MLP of width 17408.

- **Gated DeltaNet** (`src/deltanet.linnet`): one projection gives the
  queries, keys and values (16 query/key heads, 48 value heads, width 128
  each); a depthwise causal convolution of width 4 and `silu` run over them;
  queries and keys are L2-normalized. Each value head keeps a `[128, 128]`
  state that the gated delta rule (`std.nn.recurrent`) decays by
  `exp(-exp(A_log) * softplus(a + dt_bias))` and writes at rate
  `sigmoid(b)`. The heads' outputs are RMS-normalized and gated by
  `silu(z)`.
- **Gated attention** (`src/attention.linnet`): 24 query heads, 4 key/value
  heads, head dimension 256, a per-head RMS norm on queries and keys, rotary
  positions on the first quarter of each head (base 10,000,000), and the
  output scaled by `sigmoid` of a gate `q_proj` gives each head.
- **Norms** (`src/norm.linnet`): RMS norms scale by `1 + w` (zero-centered),
  in f32.

The full-attention layers keep key/value caches; the linear-attention
layers keep each row's convolution history and its recurrent state, in f32
as the checkpoint's `mamba_ssm_dtype` asks, unless the `State` generic says
bf16 (half the reads and writes per decoding step, some drift).
`prefill`/`decode` serve one batch at one position, `prefill_slots`/
`decode_rows` a server's rows each at its own length, and `prefill_packed`/
`step_packed` prompts packed into one pass, each prompt a sequence of its
own in the linear-attention layers (fixed rows; the card has no paged
entries yet).

`bindings.json` maps `groups.<g>.linear.<i>` to checkpoint layer `4g + i`
and `groups.<g>.full` to layer `4g + 3`.
