# Whisper tiny

OpenAI's smallest multilingual speech recognition model, and the first
encoder-decoder model in Nest. The encoder reads a 30-second window of audio
as an 80-channel log-mel spectrogram of 3000 frames, halves the frame rate
with two convolutions, and runs four pre-norm transformer layers of width 384
over the 1500 positions that remain. The decoder is an ordinary causal
language model over the 51865-token multilingual vocabulary with a
cross-attention over those encoder states inserted between its self-attention
and its MLP, and an output head tied to its token embedding.

## The convolutional stem

The stem is two 1-D convolutions of kernel three over the mel channels, the
first at stride 1 and the second at stride 2, each followed by GELU. The
standard library has no `conv1d` -- `std.nn.conv::conv2d` takes a square
kernel -- so the source carries its own, built the way `conv2d` is built: an
index expression may not do arithmetic, so the tap positions are a tensor
made from `iota` and the padded signal is gathered through it.

```linnet
let taps[o, k] =
    iota<i32>((L + 2 * Pad - K) / Stride + 1)[o] * cast<i32>(Stride) + iota<i32>(K)[k]
let windows[b, o, ci, k] = padded[b, taps[o, k], ci]
```

Signals are `[B, Positions, Channels]` throughout the stem, which is the
layout the encoder layers want anyway, so the gathered window flattens over
(channel, tap) and the convolution finishes as one `linear` against the
checkpoint's `[Cout, Cin, K]` kernel flattened the same way. That is a
deliberate choice rather than a shortcut: there is no `conv1d` op for a
backend to recognize, so a contraction written out in index notation would
stay a five-axis product, while `linear` selects `F.linear`, a `dot_general`,
or an ONNX `Gemm` on every backend.

The output length is the form the shape solver derives rather than one
written by hand. For kernel three at stride two with one frame of padding,
`(Frames + 2 * 1 - 3) / 2 + 1` reduces to `(Frames - 1) / 2 + 1`, and that
expression is the length of `encoder_positions`: the position table's size is
checked arithmetic, not a constant that happens to be 1500.

Whisper's positions are sinusoidal in origin, but the checkpoint stores them
as an ordinary tensor, so the source simply reads them.

## The key projection has no bias

In both stacks, queries, values and the output projection carry a bias and
keys do not. That asymmetry is in the reference implementation and it is
real: a key bias shifts every key by the same vector, which a softmax over
dot products does not ignore. `k_proj.bias` is therefore left out of
`bindings.json`, `Linear`'s bias is an optional parameter, and the absent
optional makes the `none` branch of `std.nn.linear::linear` run.

## Two entries

Transcription encodes once and decodes many times, so the two stacks are
separate entries rather than one forward pass. The audio length the decoder
cross-attends over is its own generic, so encoder states shorter than a full
30-second window are accepted as they are.

```python
from linnet import nest

model = nest.load("whisper-tiny", backend="torch")
states = model.run_entry("encode", [mel])          # [B, 80, 3000] -> [B, 1500, 384]
logits = model.run_entry("decode", [tokens, states])  # -> [B, S, 51865]
```

`mel` is what `WhisperFeatureExtractor` produces (`input_features`): a
log-mel spectrogram of 80 bins and 3000 frames, padded to 30 seconds.
`tokens` begins with Whisper's prompt of special tokens, for example
`<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>`, which the
Hub repository's tokenizer supplies.

| Entry | |
| --- | --- |
| `encode<B>(mel)` | encoder states for a 30-second window |
| `decode<B, S, A>(tokens, audio)` | logits for every token position |

There is no KV cache: `decode` recomputes the prefix each step, which is
honest about what the source expresses today. The encoder attends over all
1500 positions, including the silence a shorter clip was padded with, exactly
as the reference encoder does; `std.nn.attention::attention` takes an
unbatched `Tensor[Q, K; bool]` mask, so a per-sequence padding mask could not
be expressed even if the reference used one.

## Numerics

Whisper's `config.json` asks for plain `gelu`, the error-function form, and
its encoder applies that activation twice in the convolutional stem before
any transformer layer runs. `std.nn.activations::gelu` is the tanh
approximation instead, and substituting it is not free here: it perturbs the
stem by 2.0e-3 per activation and the four encoder layers amplify that into
5.7e-2 on the encoder states -- thirty times the tolerance a card should
need. So the source carries its own GELU, `x * Phi(x)` with the normal
distribution function from Abramowitz and Stegun 26.2.17, since Linnet has no
`erf`. It agrees with `F.gelu` to 4.8e-7 in f32 over `[-60, 60]`, against
4.7e-4 for the tanh form.

Against `WhisperForConditionalGeneration` exactly as published, in f32 on CPU
with a fixed random mel and the five-token prefix above:

| | max abs diff |
| --- | --- |
| encoder states, mel from `N(0, 1)` | 1.2e-4 |
| encoder states, mel from `U(-1, 1)`, a real log-mel's range | 2.4e-4 |
| decoder logits, end to end | 2.1e-5 |
| decoder logits, on the reference's own encoder states | 2.1e-5 |

The argmax agrees at every position. What remains is f32 rounding, most of it
the difference between `scaled_dot_product_attention` over 1500 positions and
the reference's eager attention; feeding the decoder the reference's own
encoder states isolates it and leaves 2e-5.

## Provenance

- Weights: [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny), Apache-2.0.
- Code: [openai/whisper](https://github.com/openai/whisper).
- Paper: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356).
