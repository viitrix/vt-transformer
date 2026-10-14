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

## Request validation and streaming

Chat requests must supply exactly one of a non-empty `messages` list or `prompt`.
`max_tokens` must be positive, `temperature` non-negative, `top_p` in `(0, 1]`,
and `top_k` either `-1` or positive. Only `n=1`, empty `stop`, and zero presence /
frequency penalties are supported; other values are rejected with HTTP 422
instead of being silently ignored.

Both streaming endpoints use SSE events separated by a blank line. `/generate`
returns plain text in `data:` fields (multiple lines for multiline output),
followed by `[DONE]`. Chat streaming uses `chat.completion.chunk` JSON events
with `id`, `created`, and `model`, followed by `[DONE]`.

The current backend protocol does not report token usage or the termination
reason: non-streaming usage is still a zero placeholder and the finish reason
is always `stop`. These fields should not be used for billing or to distinguish
EOS from reaching the token limit.

## Tests

```bash
pip install -e '.[dev]'
python -m pytest tests
python tests/e2e_smoke.py
```

Unit tests use fake tokenizers and queues. The integration smoke test creates a
small local tokenizer and uses real ZMQ links, spawned tokenizer/detokenizer
workers, a mock backend, and the HTTP handlers through an ASGI test client. It
checks shared and separate worker modes, errors, timeouts after killing the
backend, startup timeout cleanup, and child-process/owned-IPC cleanup. Neither
suite needs a GPU, pretrained model, network download, or running backend.


## Failure handling and lifecycle

`--startup-timeout` is a positive deadline for starting the entire worker group
(default: 300 seconds). Worker exit is detected during startup; partial startup
and HTTP/shell exit terminate and reap only the processes this launcher owns.
Explicit missing local paths (`/path`, `./path`, `../path`, expanded `~/path`)
are rejected before launch. A name such as `org/model` is allowed as a remote
repository ID and is validated by the tokenizer loader.

`--request-timeout` limits total inference time, including dispatch and waiting
for tokens (default: 300 seconds; `0` disables it). Non-streaming failures return
502 for backend/tokenizer errors or 504 for timeouts. Once streaming headers
have been sent, the HTTP status remains 200: failures are reported as
`data: {"error": {"message": "...", "type": "server_error", "code": 502}}`
(or code 504), followed by `[DONE]`. Abort delivery is best-effort and has its
own one-second bound, so a disconnected worker cannot indefinitely block cleanup.

Disconnected requests release their state and send an abort. While waiting for
tokens, both HTTP modes check for disconnection every 250ms. Malformed incoming
messages are logged and skipped; a failed frontend reply connection terminates
in-flight requests and rejects new dispatches. Tokenizer/model errors are
isolated per request; transport failures are not retried in a busy loop.

`/generate` also accepts `temperature`, `top_k`, and `top_p`, with the same
validation as chat requests. Its default temperature remains zero.

## Backend integration contract

- Send `ErrorReply(uid, message)` on the **frontend** link when a request fails
  (for example, context overflow or an internal error). Do not silently drop it.
  Tokenizer workers use the same reply for prompt/template/decoding failures.
- Stop generation on `AbortBackendMsg(uid)` and suppress subsequent token
  replies. In shared-worker mode the receiving worker clears its decode state
  immediately. In separate-worker mode the backend should forward the existing
  `AbortMsg(uid)` on the **detokenizer** link; the dedicated detokenizer clears
  state without forwarding the abort back to the backend.
- Do not invent a terminal token such as `next_token=-1` for cancellation: it is
  not a valid token and is not part of the protocol. No fields have been added
  to existing messages. For older backends that do not forward cancellation,
  decode state idle for one hour is reclaimed, including when no traffic arrives.
  Late tokens after cancellation can recreate state until this idle cleanup.
- Finish reasons and token counts still require a coordinated protocol upgrade.
  Adding defaulted dataclass fields alone is **not** enough for compatibility:
  old decoders construct messages with `cls(**kwargs)` and reject extra fields.


### Transient send failures and shell recovery

Worker sends retry `zmq.Again` (temporary backpressure) with 100ms backoff and
one 30-second retry budget, plus at most one socket send timeout (1 second).
Retries preserve order and cover normal replies, error replies and aborts.
Other transport errors, or exhausting the retry budget, still stop the worker;
runtime worker monitoring/restart is not implemented. Pending HTTP requests
remain protected by their configured request deadline.

The shell reports a failed request and continues accepting prompts. It consumes
text directly instead of parsing SSE; failed/partial turns are not appended to
conversation history. If a killed process cannot yet be closed during shutdown,
its handle is retained for another cleanup attempt while other workers are reaped.
