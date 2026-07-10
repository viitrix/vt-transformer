# sglfront MQ Protocol

This document specifies the wire protocol sglfront uses to talk to an inference backend. Any process that implements this protocol — in any language — can serve as the backend, including the in-tree `python/minisgl` scheduler (the reference implementation).

## Transport

Four ZMQ links, each a PUSH/PULL pair. Both `ipc://` (single-machine) and `tcp://` (cross-machine) are supported; the address strings are passed verbatim to ZMQ.

| Link | Default address | Direction | Socket on sglfront side |
|---|---|---|---|
| `zmq_tokenizer_addr` | `ipc:///tmp/minisgl_4` | API server → tokenizer | API server: PUSH |
| `zmq_backend_addr` | `ipc:///tmp/minisgl_0` | tokenizer → backend | tokenizer: PUSH |
| `zmq_detokenizer_addr` | `ipc:///tmp/minisgl_1` | backend → tokenizer | tokenizer: PULL |
| `zmq_frontend_addr` | `ipc:///tmp/minisgl_3` | tokenizer → API server | API server: PULL |

### Who binds, who connects

Two operating modes, controlled by `--num-tokenizer`:

**Shared mode (`--num-tokenizer 0`, the default).** A single worker handles both tokenize and detokenize.

| Link | Binder | Connector |
|---|---|---|
| `zmq_frontend_addr` | API server (PULL) | tokenizer worker (PUSH) |
| `zmq_tokenizer_addr` *(collapses onto detokenizer_addr)* | tokenizer worker (PULL) | API server (PUSH) |
| `zmq_backend_addr` | **backend** (PULL) | tokenizer worker (PUSH) |
| `zmq_detokenizer_addr` | tokenizer worker (PULL) | **backend** (PUSH) |

**Separate mode (`--num-tokenizer N>0`).** One detokenizer worker plus N dedicated tokenizer workers.

| Link | Binder | Connector |
|---|---|---|
| `zmq_frontend_addr` | API server (PULL) | tokenizer + detokenizer workers (PUSH) |
| `zmq_tokenizer_addr` | API server (PUSH) | N tokenizer workers (PULL) |
| `zmq_backend_addr` | **backend** (PULL) | tokenizer + detokenizer workers (PUSH) |
| `zmq_detokenizer_addr` | **backend** (PUSH) | detokenizer worker (PULL) |

For a remote backend the simplest setup is `--num-tokenizer 0` (shared) plus four `tcp://` addresses: backend binds `zmq_backend_addr` (PULL) and connects `zmq_detokenizer_addr` (PUSH); sglfront does the opposite.

## Serialization

Every ZMQ frame is a single msgpack object (`msgpack.packb(..., use_bin_type=True)`, unpacked with `raw=False`).

The encoded payload of a message is a flat dict: each dataclass field becomes a top-level key, plus a synthetic `__type__` key naming the dataclass. Nested dataclasses and lists are encoded recursively.

```
{
    "__type__": "TokenizeMsg",
    "uid": 7,
    "text": "hello",
    "sampling_params": {
        "__type__": "SamplingParams",
        "temperature": 0.0,
        "top_k": -1,
        "top_p": 1.0,
        "ignore_eos": False,
        "max_tokens": 1024
    }
}
```

### Tensor encoding

1D CPU tensors (currently the only shape used) are encoded as:

```
{
    "__type__": "Tensor",
    "buffer": <raw bytes from tensor.numpy().tobytes()>,
    "dtype": "torch.int32"   # any torch dtype; consumer must parse the trailing component
}
```

The reference decoder reads `buffer` as raw bytes and reinterprets via `numpy.frombuffer`. Other-language implementations should do the same.

The current protocol only sends one tensor: `UserMsg.input_ids` (1D `torch.int32`). All other fields are scalars or strings.

## Message types

### API server → tokenizer (`zmq_tokenizer_addr`)

```python
@dataclass
class TokenizeMsg:
    uid: int
    text: str | List[Dict[str, str]]   # raw prompt or OpenAI-style messages list
    sampling_params: SamplingParams

@dataclass
class AbortMsg:
    uid: int
```

### tokenizer → backend (`zmq_backend_addr`)

```python
@dataclass
class UserMsg:
    uid: int
    input_ids: torch.Tensor  # 1D CPU int32
    sampling_params: SamplingParams

@dataclass
class AbortBackendMsg:
    uid: int

@dataclass
class ExitMsg:
    pass

@dataclass
class BatchBackendMsg:
    data: List[BaseBackendMsg]   # any of the above; sent when batching >1 item
```

### backend → tokenizer (`zmq_detokenizer_addr`)

```python
@dataclass
class DetokenizeMsg:
    uid: int
    next_token: int
    finished: bool
```

A `DetokenizeMsg` is sent for **every** generated token (including the final one with `finished=True`). When `finished=True` and `next_token` equals the tokenizer's EOS id, the detokenizer drops that token before decoding.

### tokenizer → API server (`zmq_frontend_addr`)

```python
@dataclass
class UserReply:
    uid: int
    incremental_output: str   # only the new text since the previous reply for this uid
    finished: bool

@dataclass
class BatchFrontendMsg:
    data: List[UserReply]
```

## Sampling parameters

```python
@dataclass
class SamplingParams:
    temperature: float = 0.0   # <= 0 means greedy
    top_k: int = -1            # -1 disables
    top_p: float = 1.0         # 1.0 disables
    ignore_eos: bool = False   # if True, generate past EOS up to max_tokens
    max_tokens: int = 1024
```

A request is greedy iff `(temperature <= 0.0 or top_k == 1) and top_p == 1.0`.

## Request lifecycle

```
HTTP client          API server              tokenizer worker          backend
    │                    │                          │                      │
    │  POST /v1/chat/    │                          │                      │
    │ ─────────────────► │                          │                      │
    │                    │  TokenizeMsg(uid, text)  │                      │
    │                    │ ───────────────────────► │                      │
    │                    │                          │  UserMsg(uid, ids)   │
    │                    │                          │ ──────────────────── ►│
    │                    │                          │                      │ (run prefill/decode)
    │                    │                          │  DetokenizeMsg(uid,  │
    │                    │                          │      token, False)   │
    │                    │                          │ ◄──────────────────── │
    │                    │  UserReply(uid, chunk,   │                      │
    │                    │       False)             │                      │
    │                    │ ◄──────────────────────  │                      │
    │  data: chunk       │                          │                      │
    │ ◄────────────────  │                          │                      │
    │            ...repeat per generated token...   │                      │
    │                    │                          │  DetokenizeMsg(uid,  │
    │                    │                          │      eos, True)      │
    │                    │                          │ ◄──────────────────── │
    │                    │  UserReply(uid, "", True)│                      │
    │                    │ ◄──────────────────────  │                      │
    │  data: [DONE]      │                          │                      │
    │ ◄────────────────  │                          │                      │
```

Cancellation flows the other way: if the HTTP client disconnects, the API server sends `AbortMsg` to the tokenizer, which forwards it as `AbortBackendMsg` to the backend. The backend must stop generating for that `uid` and discard any pending `DetokenizeMsg`s for it.

## Implementing a backend in another language

1. Bind a PULL socket on `zmq_backend_addr`; connect a PUSH socket to `zmq_detokenizer_addr`.
2. Receive a frame; msgpack-decode it; dispatch on `__type__`:
   - `UserMsg` → run prefill + decode loop on `input_ids` with `sampling_params`. For each token, send `DetokenizeMsg(uid, token, False)`. On the final token, send `DetokenizeMsg(uid, eos_or_last, True)`.
   - `AbortBackendMsg` → abort the corresponding request.
   - `BatchBackendMsg` → unpack `data` and dispatch each entry.
3. Honor the `finished` semantics: when `next_token == eos_token_id` and the request is finished, the tokenizer drops that token, so send the EOS id (or any placeholder if you've already terminated) on the final message.

Tensor decode reference (Python):

```python
def decode_input_ids(field):
    arr = np.frombuffer(field["buffer"], dtype=np.dtype(field["dtype"].replace("torch.", "")))
    return arr  # 1D int32 array, length = prompt length
```

## Compatibility with `python/minisgl`

The in-tree `python/minisgl` scheduler uses the **same** message types and the **same** serialization. To use it as the backend for sglfront:

1. Decide which mode you're running in.
   - Shared mode (`--num-tokenizer 0`): the in-tree scheduler must be configured to **connect** `zmq_detokenizer_addr` (i.e. it must use its original shared-tokenizer defaults).
   - Separate mode: scheduler must **bind** `zmq_detokenizer_addr` (its original separate-tokenizer defaults).
2. Make sure both sides agree on the four address strings.
3. Start the scheduler first; then start sglfront. sglfront waits for the tokenizer workers to come up but does not wait for the backend — requests will queue in the PUSH sockets until the backend binds.
