# SpecTune M6

Run Qwen3.8-27B Q4 + DFlash2 locally with packed-Q4 kernels for Apple M6.
Requires an M6 Mac, 32 GB+ unified memory and macOS 26.2+.

## Install and serve

```sh
brew install Gage-Weaver/tap/spectune-m6
spectune-m6 serve
```

Or clone into any directory:

```sh
git clone https://github.com/Gage-Weaver/spectune-m6.git
cd spectune-m6
./run serve
```

The first run installs pinned Python and dependencies, downloads about 20 GB
of weights and checks the kernels. No separate Python setup is needed.
Downloads default to `.cache/` in the repo. To use another disk:

```sh
SPECTUNE_CACHE="/path/to/cache" ./run serve
```

Keep that variable set on subsequent runs to reuse the downloads. Homebrew
uses its own managed cache and environment. Models stay loaded until you stop
the server with Ctrl-C. Use `--port 8080` to change the port.

## Use with coding tools

Choose an **OpenAI-compatible Chat Completions** provider in your client:

| Setting | Value |
|---|---|
| Base URL | `http://127.0.0.1:8000/v1` |
| Model | `spectune-m6` |
| API key, if required | `local` |
| Temperature | `0` |

The server supports streaming, tool calls and tool results. Your client handles
file access and command execution. Requests generate one at a time; the server
listens only on localhost. Clients requiring the Responses or Anthropic API
cannot connect directly.

For **OpenCode v2**, merge [examples/opencode.json](examples/opencode.json)
into your coding project's `opencode.json`, keeping any existing settings, then run:

```sh
opencode run --model spectune/spectune-m6 "Explain this project."
```

File reading and follow-up replies were tested with OpenCode v2.0.23. The example
uses a conservative 16K context budget. Other clients can use the settings above;
configuration field names depend on the client and version.

Check the API from another terminal:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"spectune-m6","messages":[{"role":"user","content":"Write a Python LRU cache."}],"temperature":0,"max_tokens":2048,"stream":true}'
```

Temperature zero uses DFlash. Sampling and some other generation options fall
back to target-only decoding upstream.

## Terminal use

```sh
./run                                 # interactive chat
./run "Write a Python LRU cache."      # one reply
./run --max-tokens 8192 "Your prompt"   # longer output
./run --check                          # hardware/runtime check, no model load
```

In chat, `/clear` resets the conversation and `/quit` exits. Homebrew users can
replace `./run` with `spectune-m6`.

Model revisions, dependencies and block16/NAX settings are pinned. `--canonical`
uses the original weight layout for comparison. Other M6 variants run startup
checks but have not been benchmarked. Runtime metadata and timings go to `logs/`;
prompts and generated code are not saved.
