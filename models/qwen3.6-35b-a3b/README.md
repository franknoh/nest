# Qwen3.6 35B-A3B

The text model of Qwen3.6-35B-A3B, a 35B-parameter hybrid mixture of
experts of the Qwen3.5-MoE architecture (`qwen3_5_moe`), about 3B active
per token, without its vision tower or its multi-token-prediction layer.
Its 40 layers come in fours: three of linear attention (Gated DeltaNet) and
one of gated full attention, the same blocks as `qwen3.8-27b` at width
2048 (16 query/key heads and 32 value heads in the linear attention; 16
query heads and 2 key/value heads in the full attention).

Each layer's feed-forward block (`src/moe.linnet`) is a mixture of 256
experts: the router picks 8 per token by its softmax, renormalized over
the chosen, and each expert is a SwiGLU of width 512 whose gate and up
projections the checkpoint stores side by side (`experts.gate_up_proj`,
`[256, 1024, 2048]`). A shared expert of width 512 runs on every token,
scaled by its own sigmoid gate. The chosen experts go through
`std.nn.moe`: a grouped product on a GPU, the dense form elsewhere.

Caches, states and entries are as in `qwen3.8-27b`; `bindings.json` maps
the layers the same way.
