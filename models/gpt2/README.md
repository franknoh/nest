# GPT-2 (124M)

OpenAI's smallest GPT-2: 12 pre-norm blocks of width 768 with 12 heads,
learned absolute positions up to 1024, a fused QKV projection split by
slicing, the tanh GELU, and an output head tied to the token embedding. The
Linnet source keeps GPT-2's `[in, out]` projection layout (`Conv1D` in the
reference code), so the published checkpoint loads as it is.

## Loading

```python
from linnet import nest

model = nest.load("gpt2", backend="torch", numerics="equivalent")
logits = model(tokens)                      # Tensor[B, S; i32] -> Tensor[B, S, 50257; f32]
```

`bindings.json` maps Linnet parameter paths (`blocks.0.attn.qkv.weight`) to
the checkpoint's names (`h.0.attn.c_attn.weight`). Sequences longer than
1024 positions are rejected at compile time (`where S <= MaxPositions`).

## Entries

| Entry | |
| --- | --- |
| `forward<B, S>(tokens)` | logits for a whole sequence |

## Provenance

- Weights: [openai-community/gpt2](https://huggingface.co/openai-community/gpt2), MIT.
- Code: [openai/gpt-2](https://github.com/openai/gpt-2).
- Announcement: [Better language models and their implications](https://openai.com/research/better-language-models).

The Linnet forward pass matches `transformers` logits to 2e-2 absolute
(see `python/linnet/tests/torch/test_hf_checkpoints.py` in the Linnet
repository).
