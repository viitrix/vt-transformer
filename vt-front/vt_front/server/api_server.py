from __future__ import annotations

import asyncio
import json
import time
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Tuple

import uvicorn
import zmq
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from pydantic import BaseModel, Field, model_validator

from vt_front.core import SamplingParams
from vt_front.env import ENV
from vt_front.message import (
    AbortMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    ErrorReply,
    TokenizeMsg,
    UserReply,
)
from vt_front.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger

from .args import FrontendConfig

logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE = None


def get_global_state() -> FrontendManager:
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


class BackendError(RuntimeError):
    status_code = 502


class BackendTimeoutError(BackendError):
    status_code = 504


class ClientDisconnected(BackendError):
    status_code = 499


def _unwrap_msg(msg: BaseFrontendMsg) -> List[BaseFrontendMsg]:
    if isinstance(msg, BatchFrontendMsg):
        return [reply for item in msg.data for reply in _unwrap_msg(item)]
    return [msg]


class SamplingRequest(BaseModel):
    max_tokens: int = Field(default=16, gt=0)
    temperature: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    top_k: int = -1
    top_p: float = Field(default=1.0, gt=0, le=1)
    ignore_eos: bool = False

    @model_validator(mode="after")
    def validate_sampling(self):
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError("'top_k' must be -1 or a positive integer")
        return self

    def sampling_params(self):
        return SamplingParams(
            **{name: getattr(self, name) for name in SamplingParams.__dataclass_fields__}
        )


class GenerateRequest(SamplingRequest):
    prompt: str
    max_tokens: int = Field(gt=0)
    temperature: float = Field(default=0.0, ge=0, allow_inf_nan=False)


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAICompletionRequest(SamplingRequest):
    """Unified request model for OpenAI-style completions and chat-completions."""

    model: str

    prompt: str | None = None
    messages: List[Message] | None = None

    n: Literal[1] = 1
    stream: bool = False
    stop: List[str] = Field(default_factory=list)
    presence_penalty: Literal[0.0] = 0.0
    frequency_penalty: Literal[0.0] = 0.0

    @model_validator(mode="after")
    def validate_supported_request(self):
        if (self.prompt is not None) == (self.messages is not None):
            raise ValueError("Provide exactly one of 'messages' or 'prompt'")
        if self.messages == []:
            raise ValueError("'messages' must not be empty")
        if self.stop:
            raise ValueError("Stop sequences are not supported")
        return self


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "vt-front"
    root: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


@dataclass
class FrontendManager:
    config: FrontendConfig
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    uid_counter: int = 0
    initialized: bool = False
    ack_map: Dict[int, List[UserReply | ErrorReply]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)

    deadlines: Dict[int, float | None] = field(default_factory=dict)
    _listen_task: asyncio.Task | None = field(default=None, init=False)
    _listener_error: str | None = field(default=None, init=False)
    _closed: bool = field(default=False, init=False)
    _cleanup_tasks: set = field(default_factory=set, init=False)

    def new_user(self) -> int:
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        timeout = self.config.request_timeout
        self.deadlines[uid] = time.monotonic() + timeout if timeout else None
        return uid

    def _remaining(self, uid):
        deadline = self.deadlines.get(uid)
        return max(0, deadline - time.monotonic()) if deadline is not None else None

    async def listen(self):
        try:
            while True:
                try:
                    msg = await self.recv_tokenizer.get()
                    replies = _unwrap_msg(msg)
                except (ValueError, KeyError, TypeError, AttributeError, AssertionError):
                    logger.exception("Dropping malformed frontend message")
                    continue
                for reply in replies:
                    if not isinstance(reply, (UserReply, ErrorReply)):
                        logger.warning(
                            "Ignoring unknown frontend message: %s", type(reply).__name__
                        )
                        continue
                    if (
                        not isinstance(reply.uid, int)
                        or (
                            isinstance(reply, UserReply)
                            and (
                                not isinstance(reply.incremental_output, str)
                                or not isinstance(reply.finished, bool)
                            )
                        )
                        or (isinstance(reply, ErrorReply) and not isinstance(reply.message, str))
                    ):
                        logger.warning("Ignoring malformed reply fields")
                        continue
                    if reply.uid in self.ack_map:
                        self.ack_map[reply.uid].append(reply)
                        self.event_map[reply.uid].set()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Frontend listener failed")
        finally:
            self._listener_error = "Frontend reply connection closed"
            for uid, pending in self.ack_map.items():
                pending.append(ErrorReply(uid, self._listener_error))
                self.event_map[uid].set()

    def _create_listener_once(self):
        if self._closed or self._listener_error:
            raise BackendError(self._listener_error or "Frontend is shutting down")
        if not self.initialized:
            self._listen_task = asyncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        self._create_listener_once()
        try:
            await asyncio.wait_for(self.send_tokenizer.put(msg), self._remaining(msg.uid))
        except asyncio.TimeoutError as exc:
            raise BackendTimeoutError("Request timed out while sending to tokenizer") from exc
        except zmq.ZMQError as exc:
            raise BackendError("Tokenizer connection failed") from exc

    async def dispatch(self, msg: TokenizeMsg):
        try:
            await self.send_one(msg)
        except BackendError as exc:
            await self._abort_safely(msg.uid)
            raise HTTPException(exc.status_code, str(exc)) from exc
        except BaseException:
            await self._abort_safely(msg.uid)
            raise

    async def wait_for_ack(self, uid: int, request: Request | None = None):
        event = self.event_map[uid]
        finished = False
        try:
            while True:
                if request is not None and await request.is_disconnected():
                    raise ClientDisconnected("Client disconnected")
                remaining = self._remaining(uid)
                if remaining is not None and remaining <= 0:
                    raise BackendTimeoutError("Inference request timed out")
                # Poll disconnects even while the backend is silent. The deadline
                # is absolute, so receiving a token never resets the timeout.
                interval = min(0.25, remaining) if remaining is not None else 0.25
                try:
                    await asyncio.wait_for(event.wait(), interval)
                except asyncio.TimeoutError:
                    continue
                event.clear()
                pending = self.ack_map[uid]
                self.ack_map[uid] = []
                for ack in pending:
                    if isinstance(ack, ErrorReply):
                        raise BackendError(ack.message)
                    finished = ack.finished
                    yield ack
                    if finished:
                        return
        finally:
            if not finished:
                await self._abort_safely(uid)
            self.ack_map.pop(uid, None)
            self.event_map.pop(uid, None)
            self.deadlines.pop(uid, None)

    async def stream_generate(self, uid: int, request: Request | None = None):
        async with aclosing(self.wait_for_ack(uid, request)) as replies:
            async for ack in replies:
                yield (
                    "data: "
                    + ack.incremental_output.replace("\r\n", "\n")
                    .replace("\r", "\n")
                    .replace("\n", "\ndata: ")
                    + "\n\n"
                ).encode()

        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_chat_completions(self, uid: int, model: str, request: Request | None = None):
        created = int(time.time())
        first_chunk = True

        async with aclosing(self.wait_for_ack(uid, request)) as replies:
            async for ack in replies:
                delta = {}

                if first_chunk:
                    delta["role"] = "assistant"
                    first_chunk = False

                if ack.incremental_output:
                    delta["content"] = ack.incremental_output

                chunk = {
                    "id": f"chatcmpl-{uid}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "delta": delta,
                            "index": 0,
                            "finish_reason": None,
                        }
                    ],
                }

                yield f"data: {json.dumps(chunk)}\n\n".encode()
        # Send the terminal chunk.
        end_chunk = {
            "id": f"chatcmpl-{uid}",
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "delta": {},
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
        }

        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_with_cancellation(self, generator, request: Request, uid: int):
        completed = False
        try:
            async with aclosing(generator):
                async for chunk in generator:
                    if request is not None and await request.is_disconnected():
                        logger.info("Client disconnected for user %s", uid)
                        return
                    yield chunk
                completed = True
        except ClientDisconnected:
            return
        except BackendError as exc:
            error = {
                "error": {"message": str(exc), "type": "server_error", "code": exc.status_code}
            }
            yield f"data: {json.dumps(error)}\n\n".encode()
            yield b"data: [DONE]\n\n"
        finally:
            if not completed:
                await self._abort_safely(uid)

    async def _abort_safely(self, uid):
        task = asyncio.create_task(self.abort_user(uid))
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)
        await asyncio.shield(task)

    async def abort_user(self, uid: int):
        if uid not in self.ack_map:
            return
        self.ack_map.pop(uid, None)
        self.event_map.pop(uid, None)
        self.deadlines.pop(uid, None)
        logger.info("Aborting request for user %s", uid)
        try:
            # Cleanup must remain bounded when the tokenizer/backend is gone.
            await asyncio.wait_for(self.send_tokenizer.put(AbortMsg(uid=uid)), timeout=1)
        except (asyncio.TimeoutError, zmq.ZMQError):
            logger.warning("Could not deliver abort for user %s", uid)

    async def aclose(self):
        tasks = list(self._cleanup_tasks)
        if self._listen_task is not None:
            tasks.append(self._listen_task)
        self.shutdown()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        if self._listen_task is not None:
            self._listen_task.cancel()
        for task in list(self._cleanup_tasks):
            task.cancel()
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # shutdown code here
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        await _GLOBAL_STATE.aclose()


app = FastAPI(title="VT-Front API Server", version="0.1.0", lifespan=lifespan)


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    logger.debug("Received generate request %s", req)
    state = get_global_state()
    uid = state.new_user()
    await state.dispatch(
        TokenizeMsg(
            uid=uid,
            text=req.prompt,
            sampling_params=req.sampling_params(),
        )
    )

    return StreamingResponse(
        state.stream_with_cancellation(state.stream_generate(uid, request), request, uid),
        media_type="text/event-stream",
    )


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    state = get_global_state()
    if req.messages:
        prompt = [msg.model_dump() for msg in req.messages]
    else:
        assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"
        prompt = req.prompt

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.dispatch(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=req.sampling_params(),
        )
    )

    if req.stream:
        return StreamingResponse(
            state.stream_with_cancellation(
                state.stream_chat_completions(uid, req.model, request), request, uid
            ),
            media_type="text/event-stream",
        )

    # Non-streaming: collect all chunks and return a single JSON response
    full_content = ""

    try:
        async with aclosing(state.wait_for_ack(uid, request)) as replies:
            async for ack in replies:
                full_content += ack.incremental_output
    except BackendError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc

    return {
        "id": f"chatcmpl-{uid}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.get("/v1/models")
async def available_models():
    state = get_global_state()
    return ModelList(data=[ModelCard(id=state.config.model_path, root=state.config.model_path)])


async def shell_completion(req: OpenAICompletionRequest):
    state = get_global_state()
    assert req.messages is not None, "Shell completion only supports chat-completions"
    prompt = [msg.model_dump() for msg in req.messages]

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.dispatch(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=req.sampling_params(),
        )
    )

    async def text_chunks():
        # The shell consumes text directly. HTTP SSE error events must not be
        # mistaken for assistant content or added to the conversation history.
        async with aclosing(state.wait_for_ack(uid)) as replies:
            async for ack in replies:
                yield ack.incremental_output

    return StreamingResponse(text_chunks(), media_type="text/plain")


async def shell():
    commands = ["/exit", "/reset"]
    completer = WordCompleter(commands)
    session = PromptSession("$ ", completer=completer)

    try:
        history: List[Tuple[str, str]] = []
        while True:
            cmd = (await session.prompt_async()).strip()
            if cmd == "":
                continue
            if cmd.startswith("/"):
                if cmd == "/exit":
                    return
                if cmd == "/reset":
                    history = []
                    continue
                raise ValueError(f"Unknown command: {cmd}")
            history_messages: List[Message] = []
            for user_msg, assistant_msg in history:
                history_messages.append(Message(role="user", content=user_msg))
                history_messages.append(Message(role="assistant", content=assistant_msg))
            # send to server
            req = OpenAICompletionRequest(
                model="",
                messages=history_messages + [Message(role="user", content=cmd)],
                max_tokens=ENV.SHELL_MAX_TOKENS.value,
                top_k=ENV.SHELL_TOP_K.value,
                top_p=ENV.SHELL_TOP_P.value,
                temperature=ENV.SHELL_TEMPERATURE.value,
                stream=True,
            )
            cur_msg = ""
            try:
                async with aclosing((await shell_completion(req)).body_iterator) as response:
                    async for text in response:
                        cur_msg += text
                        print(text, end="", flush=True)
            except (BackendError, HTTPException) as exc:
                detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
                print(f"\nRequest failed: {detail}", flush=True)
                continue
            print("", flush=True)
            history.append((cmd, cur_msg))
    except EOFError:
        # user pressed Ctrl-D
        pass
    finally:
        print("Exiting shell...")


def run_api_server(
    config: FrontendConfig, start_backend: Callable[[], None], run_shell: bool
) -> None:
    """
    Run the frontend API server (FastAPI + uvicorn) and wire it to the tokenizer process via ZMQ.

    Args:
        config: Server configuration (host/port, ZMQ IPC addresses, etc).
        start_backend: Callback that launches the tokenizer/detokenizer worker processes.
        run_shell: If True, run an interactive terminal shell instead of starting uvicorn.
    """

    global _GLOBAL_STATE

    host = config.server_host
    port = config.server_port

    assert _GLOBAL_STATE is None, "Global state is already initialized"
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )

    try:
        start_backend()
        logger.info(f"API server is ready to serve on {host}:{port}")
        if not run_shell:
            uvicorn.run(app, host=host, port=port)
        else:
            asyncio.run(shell())
    finally:
        _GLOBAL_STATE.shutdown()
        _GLOBAL_STATE = None
