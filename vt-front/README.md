# vt-front

A standalone LLM inference HTTP frontend, extracted independently from
[mini-sglang](https://github.com/mini-sglang) (package `minisgl`) and renamed to
`vt_front`. It has **no dependency on `minisgl`** and contains no GPU engine
code — only the HTTP serving layer.

## What's included

- **HTTP API server** (`vt_front/server/api_server.py`)
  - `POST /generate` — raw prompt streaming generation (SSE)
  - `POST /v1/chat/completions` — OpenAI-compatible chat completions
    (streaming and non-streaming)
  - `GET /v1/models` — model list
  - Interactive shell mode (`--shell-mode`)
- **Message protocol** (`vt_front/message/`) — ZMQ + JSON wire protocol (human
  readable for debugging; 1D tensors are encoded as `{"__type__": "Tensor",
  "dtype": "torch.int32", "data": [...]}`) between frontend / tokenizer /
  backend (`TokenizeMsg`, `DetokenizeMsg`, `UserReply`, `AbortMsg`, ...).
- **Tokenizer / detokenizer workers** (`vt_front/tokenizer/`) — separate
  processes doing prompt tokenization (incl. chat template) and incremental
  streaming detokenization (CJK-aware).
- **Supporting utils** (`vt_front/utils/`) — ZMQ push/pull queues, logger,
  HuggingFace tokenizer loading.

## Architecture

```
HTTP client ── FastAPI (vt_front/server) ──ZMQ── tokenizer/detokenizer workers ──ZMQ── your backend
```

The frontend expects an inference backend that speaks the message protocol on
the ZMQ links:

| Link | Default IPC address |
|------|---------------------|
| backend (tokenizer → scheduler) | `ipc:///tmp/vt_front_0.pid=<pid>` |
| detokenizer (scheduler → detokenizer) | `ipc:///tmp/vt_front_1.pid=<pid>` |
| frontend (detokenizer → HTTP server) | `ipc:///tmp/vt_front_3.pid=<pid>` |
| tokenizer (HTTP server → tokenizer) | `ipc:///tmp/vt_front_4.pid=<pid>` (only when `--num-tokenizer > 0`) |

## Install

```bash
cd vt-front
pip install -e .
```

## Usage

```bash
# Start the frontend (tokenizers + HTTP server); backend must be running
# separately and connected over the ZMQ links above.
python -m vt_front.server.launch --model-path <hf-model-or-local-path> --port 1919

# Or via the console script
vt-front --model-path <hf-model-or-local-path>

# Interactive shell instead of HTTP server
python -m vt_front.server.launch --model-path <hf-model-or-local-path> --shell-mode
```

Then:

```bash
curl http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "any",
  "messages": [{"role": "user", "content": "Hello!"}],
  "max_tokens": 64,
  "stream": false
}'
```

## Environment variables

Prefixed with `VT_FRONT_` (e.g. `VT_FRONT_SHELL_MAX_TOKENS`, `VT_FRONT_SHELL_TEMPERATURE`).

## License

MIT, same as the original mini-sglang project.
