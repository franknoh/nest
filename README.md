# Nest

Nest is the model zoo for [Linnet](https://github.com/franknoh/Linnet): a
registry of models whose architecture is a checked `.linnet` source and
whose weights are a SafeTensors checkpoint on the Hugging Face Hub. Every
entry runs in PyTorch, JAX, XLA, and ONNX Runtime from the same file.

```python
from linnet import nest

model = nest.load("tinyllama-1.1b-chat", backend="torch")
```

Browse the models at [linnet.franknoh.dev/nest](https://linnet.franknoh.dev/nest/).

## What a model is

```text
models/<name>/
  nest.toml        the card: title, license, links, generics, checkpoint
  README.md        what the model is, how to load it, where it comes from
  bindings.json    Linnet parameter path -> checkpoint tensor name (when they differ)
  src/ or *.linnet the architecture, a Linnet package or file
  preview.svg      the main entry drawn by `linnet.diagram`
```

`nest.toml`:

```toml
[model]
name = "gpt2"                       # the directory name
title = "GPT-2 (124M)"
summary = "one sentence"
license = "MIT"
family = "gpt2"
tags = ["text-generation"]

[links]
huggingface = "https://huggingface.co/openai-community/gpt2"   # required
github = "..."
arxiv = "..."

[source]
path = "gpt2.linnet"                # or a package's src/lib.linnet
root = "Model"
entry = "forward"                   # the entry previews and checks use

[generics]                          # the root block's generics for this checkpoint
Vocab = 50257
H = 768
T = "f32"

[check]                             # entry generics the export check binds
B = 1
S = 8

[weights]
repo = "openai-community/gpt2"
files = ["model.safetensors"]
bindings = "bindings.json"
```

## Requirements

Every model must pass `python -m linnet.nest check models/<name>`:

- a README of at least a paragraph, and `title`, `summary`, `license`, and
  `links.huggingface` in the card;
- a source that compiles (`linnet check`) with the card's generics;
- a SafeTensors checkpoint on the Hub in which every parameter of the
  manifest has a tensor of the same shape and dtype, read from the file
  headers without downloading (optional parameters may be absent);
- the main entry exports to StableHLO, ONNX, PyTorch source, and JAX source.

`index.json` and the previews are generated, not written by hand:

```bash
export LINNET_BIN=/path/to/linnet LINNET_STD=/path/to/Linnet/stdlib
python -m linnet.nest check models/*
python -m linnet.nest preview models/<name> -o models/<name>/preview.svg
python -m linnet.nest index . -o index.json
```

## Adding a model

1. Write the architecture in Linnet, or start from an
   [example](https://linnet.franknoh.dev/examples/). Name parameters after
   the checkpoint where you can; map the rest in `bindings.json`.
2. Fill in `nest.toml` and the README, run `check`, render the preview,
   rebuild the index.
3. Open a pull request. The checks run again there.

## License

The registry (cards, sources, bindings) is MIT. Each checkpoint keeps the
license its card states.
