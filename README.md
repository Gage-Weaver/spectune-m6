# SpecTune M6

Qwen3.8-27B Q4 + DFlash2 with packed-Q4 kernels for Apple M6.
Requires an M6 Mac, 32 GB+ unified memory and macOS 26.2+.

From this directory:

```sh
./run
```

The first run installs the pinned runtime, downloads about 20 GB of weights
and checks the kernels. Then it opens a terminal chat. The model stays loaded
between replies. `/clear` resets the conversation; `/quit` exits.

Everything stays in this directory. To put downloads elsewhere:

```sh
SPECTUNE_CACHE=/path/to/cache ./run
```

Pass a prompt for a single reply: `./run "Write a Python LRU cache."`
Use `--max-tokens 8192` for longer replies, `--check` to check hardware without
loading models, or `--canonical` to use the canonical weight layout.

Model revisions, dependencies and block16/NAX settings are pinned. Other M6
variants run startup checks but have not been benchmarked. Runtime metadata
and timings go to `logs/`; prompts and generated code are not saved.
