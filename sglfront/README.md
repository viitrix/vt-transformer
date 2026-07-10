# sglfront

Standalone **frontend** for the [Mini-SGLang](https://github.com/sgl-project/mini-sglang) inference framework.

`sglfront` runs the HTTP API server plus the tokenizer/detokenizer worker(s) — everything language-aware. The actual LLM inference (scheduler + GPU engine) is provided by an **external backend** that implements the same ZMQ + msgpack protocol. The backend can be written in any language; see [`docs/protocol.md`](./docs/protocol.md) for the wire spec.

```
                ┌─────────── sglfront (this package) ────────────┐                 external backend
   HTTP client  │                                                │                 (any language)
       ────────►│  API server  ──►  Tokenizer worker  ──UserMsg──┼──►  scheduler/engine  │
                │     ▲                  │                       │                       │
                │     │                  ◄──DetokenizeMsg────────┤                       │
                │     └─UserReply────────┘                       │                       │
                └────────────────────────────────────────────────┘
```

## What's included

- `sglfront.server.api_server` — FastAPI app exposing `/v1/chat/completions`, `/v1/models`, `/generate` (OpenAI-compatible).
- `sglfront.tokenizer` — multiprocessing worker that tokenizes prompts and detokenizes streamed token IDs.
- `sglfront.message` — msgpack-serialized message types shared with the backend over ZMQ.
- `sglfront.utils` — ZMQ wrappers, HF tokenizer loader, logger.

## Install

```bash
cd sglfront
uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e .
```

## Run

```bash
# Default: ipc:// addresses, expects the in-tree python/minisgl scheduler
# (or any compatible backend) running on the same machine.
python -m sglfront --tokenizer-path Qwen/Qwen3-0.6B --port 1919

# Cross-machine: point the four ZMQ links at a remote backend over TCP.
python -m sglfront \
    --tokenizer-path Qwen/Qwen3-0.6B \
    --port 1919 \
    --zmq-frontend-addr    tcp://0.0.0.0:4003 \
    --zmq-tokenizer-addr   tcp://0.0.0.0:4004 \
    --zmq-backend-addr     tcp://backend.host:4000 \
    --zmq-detokenizer-addr tcp://backend.host:4001

# Interactive shell
python -m sglfront --tokenizer-path Qwen/Qwen3-0.6B --shell-mode
```

Then call it like any OpenAI endpoint:

```bash
curl http://localhost:1919/v1/chat/completions \
    -d '{"model":"qwen","messages":[{"role":"user","content":"hello"}],"max_tokens":32,"stream":true}'
```

## CLI reference

| Flag | Default | Purpose |
|---|---|---|
| `--tokenizer-path` *(required)* | — | HF tokenizer dir or repo id. Used only for tokenization/detokenization. |
| `--host` | `127.0.0.1` | HTTP server bind address. |
| `--port` | `1919` | HTTP server port. |
| `--num-tokenizer` | `0` | Number of dedicated tokenizer workers. `0` = shared with detokenizer. |
| `--zmq-frontend-addr` | `ipc:///tmp/minisgl_3` | detokenizer → API server (UserReply). |
| `--zmq-tokenizer-addr` | *(empty → detokenizer_addr)* | API server → tokenizer (TokenizeMsg, AbortMsg). |
| `--zmq-backend-addr` | `ipc:///tmp/minisgl_0` | tokenizer → backend (UserMsg, AbortBackendMsg). |
| `--zmq-detokenizer-addr` | `ipc:///tmp/minisgl_1` | backend → detokenizer (DetokenizeMsg). |
| `--model-name` | `sglfront` | Reported via `/v1/models`. |
| `--shell-mode` | off | Run as interactive shell instead of HTTP server. |

Environment variables `SGLFRONT_SHELL_MAX_TOKENS`, `SGLFRONT_SHELL_TOP_K`, `SGLFRONT_SHELL_TOP_P`, `SGLFRONT_SHELL_TEMPERATURE` configure the shell's sampling defaults.

## Tests

```bash
cd sglfront && pytest tests/
```

## License

MIT.
