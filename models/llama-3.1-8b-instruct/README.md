# Llama 3.1 8B Instruct

An 8.03B-parameter decoder with the Llama architecture, instruction-tuned:
grouped-query attention (32 query heads, 8 key/value heads, head dimension
128), rotary positions with base 500000 and `llama3`-style rope scaling
(context stretched from an 8192-token pretraining length to 131072), RMS
normalization, a SwiGLU MLP, and an untied output head, with a KV cache for
token-by-token decoding. The vocabulary is 128256 tokens.

The Linnet source is the same Llama package the other decoders in this zoo
use (`tinyllama-1.1b-chat`, `smollm2-1.7b-instruct`, `mistral-7b-instruct-v0.3`),
with `THETA` set to this model's rotary base and the generics set to its
width, head counts, and depth. What is new here is `crate.rope`'s rope
scaling: Llama 3.1 rescales each rotary frequency's *wavelength* in three
bands before computing angles, rather than using the raw base frequency
directly (`rope_scaling` in `config.json`, type `llama3`). Per frequency,
independent of position:

- a wavelength longer than `original_max_position_embeddings /
  low_freq_factor` (8192) is divided by `factor` (8);
- a wavelength shorter than `original_max_position_embeddings /
  high_freq_factor` (2048) is left alone;
- in between, the frequency is linearly blended between the unscaled and
  the divided-by-`factor` value.

This is ordinary elementwise scalar arithmetic on the `[D / 2]` frequency
table (two nested `select`s on the wavelength comparisons, mirroring
`transformers`' `torch.where` chain in `_compute_llama3_parameters`); the
language's existing arithmetic and `select` were enough, with no new index
arithmetic involved, since the rescaling depends only on which frequency
`i` is being computed, never on the sequence position.

## Loading

```python
from linnet import nest

model = nest.load("llama-3.1-8b-instruct", backend="torch", numerics="fast")
logits = model(tokens)                      # Tensor[1, S; i32] -> Tensor[1, S, 128256; bf16]
```

The weights are the published, sharded `model-0000N-of-00004.safetensors`
(bf16); `bindings.json` maps Linnet parameter paths to their tensor names.
Every name, shape, and dtype is checked before anything runs.

Every `DecoderLayer`'s KV cache (its `state` members) is allocated eagerly
at load time, sized by `Batch` and `MaxSeq`, whether or not `decode` is
ever called -- so `MaxSeq` sets a memory cost you pay just to build the
model, not only to decode at that length. The card's default is
`MaxSeq = 8192`, Llama 3.1's own pretraining context before rope scaling
stretches it further: at that default the caches add about 1 GiB in bf16
(32 layers x 2 caches x 8 key/value heads x 8192 positions x 128 head
dimension) on top of the ~15 GiB of bf16 weights, so the model fits a
24 GB GPU with room for activations. Raise it for a longer decoding
context with `generics={"MaxSeq": 131072}` for the full rope-scaled
context -- but that alone is about 16 GiB of cache in bf16 (twice the
model's own weight size), so plan for roughly 31 GiB total and a GPU
larger than 24 GB. Cache size scales linearly with `MaxSeq`, so a value
in between (`16384`, `32768`, ...) trades context length for memory
directly.

## Entries

| Entry | |
| --- | --- |
| `forward<B, S>(tokens)` | logits for a whole sequence |
| `next_token<B, S>(tokens)` | logits for the last position |
| `decode(token, pos)` | one token through the KV caches (`Batch = 1`, `MaxSeq = 8192` by default) |
| `prefill<S>(tokens, pos)` | a whole prompt through the KV caches in one pass; logits after its last token, `decode` continues at `pos + S` |
| `prefill_slot<S>(tokens, slot, length)` | one request's prompt into row `slot` of the caches (padded to `S`, the first `length` tokens real), as it joins a batch being served |
| `decode_rows(tokens, positions)` | one token for every row of the caches, each at its own position: the step `linnet.serve` takes for continuous batching |
| `generate<Steps>(token, pos)` | greedy decoding in the graph |
| `sample<Steps>(token, pos, key, temperature)` | sampling with `std.random` |
| `generate_until<MaxNew>(token, pos, eos)` | decoding until an end token |

## Provenance

- Weights: [NousResearch/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/NousResearch/Meta-Llama-3.1-8B-Instruct).
  Meta's own [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
  is the original release of the same weights, but it sits behind Meta's
  manual gated-access approval; this card points at NousResearch's ungated
  mirror instead so `check` can read the checkpoint headers and anyone can
  load it without requesting access. The license is Meta's own regardless
  of which repository serves the bytes: the Llama 3.1 Community License
  (SPDX-style identifier `llama3.1`), not a license NousResearch grants.
- Code: [meta-llama/llama-models](https://github.com/meta-llama/llama-models).
- Paper: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783).

## Validation

Checked against `transformers` (`LlamaForCausalLM`) on CPU, `torch.no_grad()`,
under the project's memory discipline for multi-billion-parameter models.

- **Rope-scaling table, directly, no model (done locally).** Before
  trusting a model-level comparison, `crate.rope::tables`' scaled
  `inv_freq` (all 64 frequencies, the entire wavelength range from a
  single-token to a 20-million-token period) was compared elementwise
  against `transformers.models.llama.modeling_llama.LlamaRotaryEmbedding(config).inv_freq`
  for this checkpoint's real `rope_scaling` -- no weights, no model, just
  the frequency table, so it costs nothing to check at every position
  band. This caught a real bug: the smooth-interpolation band (frequency
  indices 29-34 of 64, wavelengths 2048-8192) had its two blend weights
  swapped, `smooth[i]` multiplying the scaled term instead of
  `(1 - smooth[i])` -- confirmed by a boundary check (at
  `wavelen == high_freq_wavelen`, `smooth = 1`, and the correct blend must
  equal the *unscaled* frequency for continuity with the band just beyond
  it; the swapped version gave the scaled one instead). Fixed in
  `src/rope.linnet`; re-checked, `inv_freq` now matches to **5.96e-8**
  (float32's own precision floor) at every one of the 64 frequencies. The
  cos/sin tables built from it also match to 4.69e-7 at short positions,
  widening to 7.8e-3 at position 131071 -- but by then the worst-matching
  dimension is a high-frequency one the scaling never touches, so that
  residual is ordinary f32 rounding on a large `position x frequency`
  angle, present in any rope implementation at that length, not a second
  scaling bug.
- **Truncated, f32, tight tolerance (done locally).** Both sides built from
  the real checkpoint with only the first 2 of 32 layers
  (`generics={"Layers": 2, "MaxSeq": 16, "T": "f32"}` on the Linnet side,
  `num_hidden_layers=2` on the reference), prompt `"The capital of France
  is"`. `MaxSeq` is overridden down from the card's default -- the KV
  cache `state` members are allocated at load time regardless of `Layers`,
  so leaving `MaxSeq` at its default would still have cost several GiB of
  zeroed cache for no reason. Run under `flock` and a `systemd-run --user
  --scope -p MemoryMax=12G -p MemorySwapMax=0` cap, peak RSS 11.58 GiB
  (both sides, measured separately, well inside the cap). Before the rope
  fix above this gave 4.97e-3, about a thousand times larger than the
  sibling Llama-family cards' numbers on the same comparison (Mistral 7B
  4.3e-6, Qwen3-8B 5.25e-6, Phi-3 7.15e-6) -- a real difference, not f32
  noise, and the rope-table check above found exactly why. After the fix,
  max absolute difference on the logits: **6.68e-6**, in line with those
  siblings; top-1 token agrees exactly (token 105690, "Paris" continues
  correctly). This proves the non-positional parts of the architecture
  (GQA, SwiGLU, RMSNorm, the untied head) independent of rope scaling.
- **Whole model, bf16, loose tolerance, including a long-enough prompt for
  rope scaling to engage (deferred).** Not run on this machine: holding a
  32-layer, 8B-parameter model in bf16 (about 16 GiB) is exactly the kind
  of run the project's memory discipline asks multi-billion-parameter
  models to avoid on this shared 48 GiB CPU box, and the long prompt this
  pass needs to push `position x frequency` past where a wrong `llama3`
  scaling would diverge from a correct one makes it heavier still. This
  card's whole-model and long-position numbers are deferred to a batched
  GPU run (RunPod) that covers every model in the zoo together. The rope
  scaling itself does not have to wait for that run, though: the direct
  table check above already verifies it, including at long positions,
  without loading the model at all.

The piecewise rule itself is `crate.rope::tables`' two nested `select`s
described above, over the wavelength comparisons `wavelen[i] >
low_freq_wavelen` and `wavelen[i] < high_freq_wavelen`, with the smooth
blend `(1.0 - smooth[i]) * (base_inv_freq[i] / FACTOR) + smooth[i] *
base_inv_freq[i]` in between -- so the GPU run can check it directly
against `transformers`' `_compute_llama3_parameters` at whatever positions
it covers, not just re-derive it from the constants.

Numbers are recorded in the change that added this model rather than
restated here, since they come from one specific run on one specific
machine; rerun the comparison above to reproduce them.
