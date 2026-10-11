# Qwen3 8B FP8

Qwen3 8B with its decoder projections in FP8: Qwen's own
[FP8 checkpoint](https://huggingface.co/Qwen/Qwen3-8B-FP8). Each query,
key, value, output, gate, up and down projection holds E4M3 weights (one
byte each) with one bf16 scale per block of 128 by 128 weights, the scheme
DeepSeek-V3 uses; the embedding, the norms and the output head stay bf16.

The source is the `qwen3-8b` package with those projections as
`std.quant::Fp8BlockLinear`. Its blocks need every projection's widths to
be multiples of 128, which the `where` clauses state. `bindings.json` binds
each `scale` to the checkpoint's `weight_scale_inv` (a multiplier, despite
its name) and each FP8 `weight` to its bytes.

## Loading

```python
from linnet import nest

model = nest.load("qwen3-8b-fp8", backend="torch", numerics="fast")
```

On a GPU with compute capability 9 or later, the projections multiply the
FP8 weights as they are: one row in a kernel that reads each byte once, up
to 32 rows on tensor cores, and longer inputs rounded to FP8 128 inputs at
a time (`torch._scaled_mm` with block scales). Elsewhere the weights are
decoded to bf16 for each product.

## Against bf16

One H100, the same host; CUDA graphs for decoding, a 512-token prompt run
eagerly:

| | Perplexity, WikiText-2 | Time to first token | Decoding | Peak memory |
| --- | ---: | ---: | ---: | ---: |
| bf16 (`qwen3-8b`) | 12.71 | 21.6 ms | 157 tokens/s | 16.9 GiB |
| FP8 | 12.84 | 23.2 ms | 238 tokens/s | 10.4 GiB |

Under `numerics="equivalent"`, which never rounds the input, the perplexity
is 12.77. The prompt runs slower than bf16's: cuBLAS's block-scaled FP8
product is no faster than bf16 at these sizes.
