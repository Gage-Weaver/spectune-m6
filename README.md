# SpecTune M6

Qwen3.8-27B Q4 + DFlash2 with packed-Q4 kernels for Apple M6.
Requires an M6 Mac, 32 GB+ unified memory and macOS 26.2+.

Install with Homebrew:

```sh
brew install Gage-Weaver/tap/spectune-m6
spectune-m6 serve
```

Or clone this repo and run from its directory:

```sh
./run
```

The first run installs the pinned runtime, downloads about 20 GB of weights
and checks the kernels. Then it opens a terminal chat. The model stays loaded
between replies. `/clear` resets the conversation; `/quit` exits.

By default, downloads and runtime files stay in this directory. To put downloads elsewhere:

```sh
SPECTUNE_CACHE=/path/to/cache ./run
```

Pass a prompt for a single reply: `./run "Write a Python LRU cache."`
Use `--max-tokens 8192` for longer replies, `--check` to check hardware without
loading models, or `--canonical` to use the canonical weight layout.

Model revisions, dependencies and block16/NAX settings are pinned. Other M6
variants run startup checks but have not been benchmarked. Runtime metadata
and timings go to `logs/`; prompts and generated code are not saved.

## Local server

```sh
./run serve
```

Connect an OpenAI-compatible **Chat Completions** client with:

- Base URL: `http://127.0.0.1:8000/v1`
- Model: `spectune-m6`
- Temperature: `0`
- API key: any placeholder if your client requires one (e.g. `local`)

Streaming, tool calls and tool results use the existing DFlash/MLX-LM server.
The model stays loaded; requests generate one at a time. The server listens
only on localhost. Use `--port 8080` to change the port.

For OpenCode v2, merge [this provider configuration](examples/opencode.json)
into your project's `opencode.json`. File reading and a follow-up reply were
tested with OpenCode v2.0.23. Its example uses a conservative 16K context budget.

Temperature zero uses DFlash. Sampling and some other generation options fall
back to target-only decoding upstream. Streaming Responses API clients are
not supported. Server compatibility is tested; the published speedup measures
the inference runner, not HTTP requests.
