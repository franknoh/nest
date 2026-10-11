# Llama 3.1 8B Instruct W8A8

Llama 3.1 8B Instruct with its decoder projections in int8: RedHatAI's
[W8A8](https://huggingface.co/RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8)
checkpoint, made with `llm-compressor` (GPTQ). Each query,
key, value, output, gate, up and down projection holds int8 weights with
one bf16 scale per output row; the embedding and the output head stay
bf16. The checkpoint was calibrated for inputs rounded to int8 a token at a
time, as the products below run them.

The source is the `llama-3.1-8b-instruct` package with those projections
as `std.quant::Int8Linear`. `bindings.json` binds each `scale` to the
checkpoint's `weight_scale`.

## Loading

```python
from linnet import nest

model = nest.load("llama-3.1-8b-instruct-w8a8", backend="torch", numerics="fast")
```

On a CUDA GPU, the projections multiply the int8 weights as they are: one
row in a kernel that reads each byte once, up to 32 rows on tensor cores,
and longer inputs rounded to int8 a row at a time and multiplied in int8
(Linnet's kernel, summing in int32). Elsewhere the weights are dequantized
to bf16 for each product.

## Against bf16

One H100, the same host; CUDA graphs for decoding, a 512-token prompt run
eagerly:

| | Perplexity, WikiText-2 | Time to first token | Decoding | Peak memory |
| --- | ---: | ---: | ---: | ---: |
| bf16 (`llama-3.1-8b-instruct`) | 9.55 | 18.9 ms | 163 tokens/s | 16.3 GiB |
| W8A8 | 9.70 | 21.0 ms | 271 tokens/s | 9.9 GiB |

Under `numerics="equivalent"`, which never rounds the input, the perplexity
is 9.55. Run eagerly, the prompt's two kernels a projection cost the host
more than its int8 products save on the device.
