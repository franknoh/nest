# Llama 3.1 8B Instruct FP8

Llama 3.1 8B Instruct with its decoder projections in FP8: RedHatAI's
[FP8-dynamic](https://huggingface.co/RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic)
checkpoint, made with `llm-compressor`. Each query, key, value, output,
gate, up and down projection holds E4M3 weights (one byte each) with one
bf16 scale per output row; the embedding and the output head stay bf16.

The source is the `llama-3.1-8b-instruct` package with those projections as
`std.quant::Fp8Linear`. `bindings.json` binds each `scale` to the
checkpoint's `weight_scale` and each FP8 `weight` to its bytes.

## Loading

```python
from linnet import nest

model = nest.load("llama-3.1-8b-instruct-fp8", backend="torch", numerics="fast")
```

On a GPU with compute capability 9 or later, `numerics="fast"` multiplies
one sequence's decoding step by the FP8 weights directly and rounds longer
inputs to FP8 a row at a time (`torch._scaled_mm`). Elsewhere the weights
are decoded to bf16 for each product.

## Against bf16

One H100, the same host; CUDA graphs for decoding, a 512-token prompt run
eagerly:

| | Perplexity, WikiText-2 | Time to first token | Decoding | Peak memory |
| --- | ---: | ---: | ---: | ---: |
| bf16 (`llama-3.1-8b-instruct`) | 9.55 | 19.1 ms | 162 tokens/s | 16.3 GiB |
| FP8 | 9.62 | 20.3 ms | 197 tokens/s | 9.9 GiB |
