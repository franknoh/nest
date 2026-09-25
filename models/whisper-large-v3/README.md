# Whisper large-v3

OpenAI's largest multilingual speech recognition model, and the biggest
re-parameterization of `whisper-tiny`'s architecture: an encoder that reads a
30-second window of audio as a **128-channel** log-mel spectrogram of 3000
frames, halves the frame rate with two convolutions, and runs **32** pre-norm
transformer layers of width **1280** with **20** attention heads over the
1500 positions that remain. The decoder is an ordinary causal language model
of the same width and 32 layers over a **51866-token** multilingual
vocabulary, with a cross-attention over those encoder states inserted
between its self-attention and its MLP, and an output head tied to its token
embedding.

Whisper large-v3's only architectural difference from `whisper-tiny` is the
mel filterbank: 128 bins instead of 80, per its `config.json`
(`num_mel_bins`). Width, head count, depth, and inner size differ as any two
sizes of a family do. The decoder's learned position table stays
`[448, 1280]` -- `max_target_positions` is 448 in both checkpoints, so
`MaxTokens` is unchanged -- and the tied head grows only because the
vocabulary does: large-v3 adds one token over tiny's 51865, all other
special-token ids besides that addition are shifted by it (`<|transcribe|>`
is 50360 here, not 50359). Everything else -- the stem, the attention block
serving both self- and cross-attention, the missing `k_proj.bias` -- is
`whisper-tiny`'s source with new generic values; see that card's README for
how each piece is built.

## The convolutional stem

The stem is two 1-D convolutions of kernel three over the mel channels, the
first at stride 1 and the second at stride 2, each followed by GELU. The
standard library has no `conv1d` -- `std.nn.conv::conv2d` takes a square
kernel -- so the source carries its own hand-built 1-D convolution, the way
`whisper-tiny`'s does: an index expression may not do arithmetic, so the tap
positions are a tensor made from `iota` and the padded signal is gathered
through it, then the convolution finishes as one `linear` against the
checkpoint's `[Cout, Cin, K]` kernel flattened over (channel, tap).

## The key projection has no bias

In both stacks, queries, values and the output projection carry a bias and
keys do not, exactly as in `whisper-tiny`. `k_proj.bias` is left out of
`bindings.json`, `Linear`'s bias is an optional parameter, and the absent
optional makes the `none` branch of `std.nn.linear::linear` run.

## Two entries

Transcription encodes once and decodes many times, so the two stacks are
separate entries rather than one forward pass, as in `whisper-tiny`.

```python
from linnet import nest

model = nest.load("whisper-large-v3", backend="torch")
states = model.run_entry("encode", [mel])            # [B, 128, 3000] -> [B, 1500, 1280]
logits = model.run_entry("decode", [tokens, states])  # -> [B, S, 51866]
```

`mel` is what `WhisperFeatureExtractor` produces for this checkpoint
(`input_features`): a log-mel spectrogram of 128 bins and 3000 frames,
padded to 30 seconds -- note that this is *not* the 80-bin extractor
`whisper-tiny` and earlier Whisper sizes use. `tokens` begins with Whisper's
prompt of special tokens, for example
`<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>`, which the
Hub repository's tokenizer supplies (`large-v3`'s ids for these differ
slightly from smaller Whisper checkpoints', since one extra vocabulary token
shifts everything after it; do not reuse another checkpoint's hardcoded
ids).

| Entry | |
| --- | --- |
| `encode<B>(mel)` | encoder states for a 30-second window |
| `decode<B, S, A>(tokens, audio)` | logits for every token position |

There is no KV cache: `decode` recomputes the prefix each step. The encoder
attends over all 1500 positions, including the silence a shorter clip was
padded with, exactly as the reference encoder does; `std.nn.attention::attention`
takes an unbatched `Tensor[Q, K; bool]` mask, so a per-sequence padding mask
could not be expressed even if the reference used one.

## Numerics

Whisper's `config.json` asks for plain `gelu`, the error-function form. The
source uses `std.nn.activations::gelu_erf`, which carries the same
Abramowitz-and-Stegun polynomial `whisper-tiny` hand-rolled before that
stdlib op existed (accurate to within 5e-7 of `F.gelu` in f32), not the tanh
approximation (`std.nn.activations::gelu`). Under `numerics="fast"` --
the default, what `nest.load` uses -- `gelu_erf` is not even evaluated as
that polynomial: `linnet explain` shows both GELU ops as registered stdlib
`op`s with a fused candidate, so the backend calls `torch.nn.functional.gelu`
directly, with `approximate='none'` for `gelu_erf` and `approximate='tanh'`
for `gelu`. A hand-rolled private `fn`, which is what `whisper-tiny` used
before `gelu_erf` existed, has no such registration and always falls back to
evaluating the canonical polynomial regardless of tier; moving the same math
into a stdlib `op` is not just deduplication, it lets `fast` numerics select
the exact PyTorch kernel instead.

`large-v3` has eight times `whisper-tiny`'s depth (32 encoder layers against
4) to compound whatever an activation gets wrong, so getting the form right
matters more here, not less. To measure what getting it wrong would have
cost, both GELU forms were run through the **same** truncated model (below)
and compared to the same truncated reference, real weights, f32:

| encoder states, truncated 2 encoder / 2 decoder layers | max abs diff |
| --- | --- |
| `gelu_erf` (shipped), mel from `N(0, 1)` | 1.9e-5 |
| `gelu_erf` (shipped), mel from `U(-1, 1)` | 1.5e-5 |
| tanh `gelu` (ablation, not shipped), mel from `N(0, 1)` | 5.5e-2 |
| tanh `gelu` (ablation, not shipped), mel from `U(-1, 1)` | 6.9e-2 |

At only 2 of the 32 encoder layers, the tanh approximation is already
roughly 2,900-4,600x worse than `gelu_erf` on this checkpoint, and on the
decoder side the ablation's end-to-end logits reach 2.6e-1 and 2.8e-1 max
abs diff and its **argmax disagrees with the reference at one of the two
inputs** -- a wrong transcription, not just a rounding difference, from two
layers alone. `gelu_erf` is the correct choice and the only one shipped.

### Validation

`check` only proves shapes and exports. Numeric validation for a 1.5B model
follows the two-pass scheme for large checkpoints: this card reports the
first pass only -- **truncated, f32, real weights, tight tolerance** -- and
defers the second pass, the whole model in bf16 for full-depth agreement, to
a batched GPU run alongside the rest of the registry's large models.

Both sides -- `WhisperForConditionalGeneration` and the Linnet model -- were
built with `EncoderLayers = DecoderLayers = 2`, so both read only the first
two layers of each stack out of the real `openai/whisper-large-v3`
checkpoint. In f32 on CPU, with a fixed random mel and the five-token prefix
above:

| | max abs diff |
| --- | --- |
| encoder states, mel from `N(0, 1)` | 1.9e-5 |
| encoder states, mel from `U(-1, 1)`, a real log-mel's range | 1.5e-5 |
| decoder logits, end to end, mel from `N(0, 1)` | 4.3e-6 |
| decoder logits, end to end, mel from `U(-1, 1)` | 4.2e-6 |
| decoder logits, on the reference's own encoder states, mel from `N(0, 1)` | 4.1e-6 |
| decoder logits, on the reference's own encoder states, mel from `U(-1, 1)` | 4.3e-6 |

The argmax agrees at every position in every case, well inside the usual
2e-3 tolerance -- tighter even than `whisper-tiny`'s own reported four-layer
figures (1.2e-4 to 2.4e-4 on encoder states), consistent with `gelu_erf`
costing nothing here (the fused kernel runs, not the polynomial) and with
only 2 of 32 layers having had the chance to accumulate ordinary f32
rounding.

Peak RSS for the whole pass (truncated reference, released, then the shipped
model, released, then the tanh-gelu ablation) was 1.5 GiB.

**The whole-model bf16 pass is deferred.** Per the current plan for this
registry, large-model whole-checkpoint agreement (full 32+32 depth, top-1
token agreement, logits max abs diff against bf16 rounding) runs once on a
GPU in a batch with every other large model's measurements, rather than on
this CPU machine per model. Only the truncated f32 pass above ran here.

## Provenance

- Weights: [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3), Apache-2.0.
- Code: [openai/whisper](https://github.com/openai/whisper).
- Paper: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356).
