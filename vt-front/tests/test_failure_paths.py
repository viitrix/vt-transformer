import asyncio
import logging
import queue
from dataclasses import replace

import pytest
import zmq
from fastapi import HTTPException
from fastapi.testclient import TestClient
from test_regressions import FakeQueue, FakeTokenizer, manager

from vt_front.core import SamplingParams
from vt_front.message import (
    AbortBackendMsg,
    AbortMsg,
    BaseFrontendMsg,
    BatchFrontendMsg,
    DetokenizeMsg,
    ErrorReply,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from vt_front.server import api_server
from vt_front.server.api_server import (
    BackendError,
    BackendTimeoutError,
    ClientDisconnected,
    OpenAICompletionRequest,
)
from vt_front.server.args import FrontendConfig, parse_args
from vt_front.server.launch import WorkerGroup, wait_for_workers
from vt_front.tokenizer.detokenize import DetokenizeManager
from vt_front.tokenizer.server import process_message
from vt_front.tokenizer.tokenize import TokenizeManager


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("failure", ["error", "timeout"])
def test_http_failures_are_terminal(monkeypatch, stream, failure):
    state = manager()
    state.config = replace(state.config, request_timeout=0.02)

    async def reply(msg):
        if failure == "error":
            state.ack_map[msg.uid].append(ErrorReply(msg.uid, "backend failed"))
            state.event_map[msg.uid].set()

    monkeypatch.setattr(state, "send_one", reply)
    monkeypatch.setattr(state, "shutdown", lambda: None)
    monkeypatch.setattr(api_server, "_GLOBAL_STATE", state)
    with TestClient(api_server.app) as client:
        response = client.post(
            "/v1/chat/completions", json={"model": "test", "prompt": "x", "stream": stream}
        )
    expected = 502 if failure == "error" else 504
    if stream:
        assert response.status_code == 200
        assert '"error"' in response.text and f'"code": {expected}' in response.text
        assert response.text.endswith("data: [DONE]\n\n")
        assert '"finish_reason": "stop"' not in response.text
    else:
        assert response.status_code == expected
    assert not state.ack_map and not state.event_map and not state.deadlines
    assert state.send_tokenizer.sent == [AbortMsg(0)]


def test_listener_survives_poison_and_error_reply():
    class Incoming:
        def __init__(self):
            self.items = asyncio.Queue()

        async def get(self):
            value = await self.items.get()
            if isinstance(value, Exception):
                raise value
            return value

    async def run():
        state = manager()
        state.initialized = False
        state.recv_tokenizer = Incoming()
        uid = state.new_user()
        state._create_listener_once()
        for item in [
            KeyError("bad type"),
            BaseFrontendMsg(),
            UserReply([], "bad", True),
            BatchFrontendMsg([ErrorReply(uid, "bad prompt")]),
        ]:
            await state.recv_tokenizer.items.put(item)
        with pytest.raises(BackendError, match="bad prompt"):
            await asyncio.wait_for(anext(state.wait_for_ack(uid)), 1)
        assert not state._listen_task.done()
        uid = state.new_user()
        await state.recv_tokenizer.items.put(UserReply(uid, "ok", True))
        assert [ack.incremental_output async for ack in state.wait_for_ack(uid)] == ["ok"]
        state._listen_task.cancel()
        await asyncio.gather(state._listen_task, return_exceptions=True)

    asyncio.run(run())


def test_listener_connection_failure_wakes_requests_and_rejects_new_ones():
    class Broken:
        async def get(self):
            raise zmq.ZMQError(zmq.ETERM)

    async def run():
        state = manager()
        state.initialized = False
        state.recv_tokenizer = Broken()
        uid = state.new_user()
        state._create_listener_once()
        with pytest.raises(BackendError, match="connection closed"):
            await asyncio.wait_for(anext(state.wait_for_ack(uid)), 1)
        uid = state.new_user()
        with pytest.raises(HTTPException) as caught:
            await state.dispatch(TokenizeMsg(uid, "x", SamplingParams()))
        assert caught.value.status_code == 502
        assert not state.ack_map

    asyncio.run(run())


def test_send_timeout_is_bounded_and_cleans_state():
    class Blocked(FakeQueue):
        async def put(self, msg):
            if isinstance(msg, TokenizeMsg):
                await asyncio.Event().wait()
            else:
                await super().put(msg)

    async def run():
        state = manager()
        state.config = replace(state.config, request_timeout=0.01)
        state.send_tokenizer = Blocked()
        uid = state.new_user()
        with pytest.raises(HTTPException) as caught:
            await state.dispatch(TokenizeMsg(uid, "x", SamplingParams()))
        assert caught.value.status_code == 504
        assert not state.ack_map and not state.deadlines
        assert state.send_tokenizer.sent == [AbortMsg(uid)]

    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
def test_disconnect_without_any_backend_reply(stream):
    class Request:
        disconnected = False

        async def is_disconnected(self):
            return self.disconnected

    async def run():
        state = manager()
        state.config = replace(state.config, request_timeout=0)
        request = Request()
        uid = state.new_user()

        async def consume():
            if stream:
                output = state.stream_with_cancellation(
                    state.stream_generate(uid, request), request, uid
                )
                return [chunk async for chunk in output]
            return [ack async for ack in state.wait_for_ack(uid, request)]

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.01)
        request.disconnected = True
        if stream:
            assert await asyncio.wait_for(task, 0.6) == []
        else:
            with pytest.raises(ClientDisconnected):
                await asyncio.wait_for(task, 0.6)
        assert state.send_tokenizer.sent == [AbortMsg(uid)]
        assert not state.ack_map and not state.deadlines

    asyncio.run(run())


def test_timeout_is_total_not_per_token():
    async def run():
        state = manager()
        state.config = replace(state.config, request_timeout=0.04)
        uid = state.new_user()

        async def feed():
            while uid in state.ack_map:
                state.ack_map[uid].append(UserReply(uid, "x", False))
                state.event_map[uid].set()
                await asyncio.sleep(0.005)

        task = asyncio.create_task(feed())
        try:
            with pytest.raises(BackendTimeoutError):
                async for _ in state.wait_for_ack(uid):
                    pass
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert not state.ack_map

    asyncio.run(run())


def test_shell_early_close_executes_abort(monkeypatch):
    async def run():
        state = manager()

        async def reply(msg):
            state.ack_map[msg.uid].append(UserReply(msg.uid, "partial", False))
            state.event_map[msg.uid].set()

        monkeypatch.setattr(api_server, "_GLOBAL_STATE", state)
        monkeypatch.setattr(state, "send_one", reply)
        response = await api_server.shell_completion(
            OpenAICompletionRequest(model="test", messages=[{"role": "user", "content": "x"}])
        )
        await anext(response.body_iterator)
        await response.body_iterator.aclose()
        assert state.send_tokenizer.sent == [AbortMsg(0)]
        assert not state.ack_map

    asyncio.run(run())


def test_generate_passes_sampling_parameters(monkeypatch):
    state = manager()
    sent = []

    async def reply(msg):
        sent.append(msg)
        state.ack_map[msg.uid].append(UserReply(msg.uid, "ok", True))
        state.event_map[msg.uid].set()

    monkeypatch.setattr(api_server, "_GLOBAL_STATE", state)
    monkeypatch.setattr(state, "send_one", reply)
    monkeypatch.setattr(state, "shutdown", lambda: None)
    with TestClient(api_server.app) as client:
        for extra in [{}, {"temperature": 0.7, "top_k": 20, "top_p": 0.9}]:
            assert (
                client.post("/generate", json={"prompt": "x", "max_tokens": 8, **extra}).status_code
                == 200
            )
        assert (
            client.post("/generate", json={"prompt": "x", "max_tokens": 8, "top_k": 0}).status_code
            == 422
        )
    assert sent[0].sampling_params.temperature == 0
    assert sent[1].sampling_params == SamplingParams(
        temperature=0.7, top_k=20, top_p=0.9, max_tokens=8
    )


class SyncQueue:
    def __init__(self):
        self.sent = []

    def put(self, msg):
        self.sent.append(msg)


def test_worker_isolates_bad_prompt_and_clears_aborted_state():
    class Tokenizer(FakeTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            raise ValueError("no template")

    tokenizer = Tokenizer()
    encode, decode = TokenizeManager(tokenizer), DetokenizeManager(tokenizer)
    backend, frontend = SyncQueue(), SyncQueue()
    logger = logging.getLogger("test")
    for msg in [
        TokenizeMsg(0, [], SamplingParams()),
        TokenizeMsg(1, "valid", SamplingParams()),
        DetokenizeMsg(2, 1, False),
        AbortMsg(2),
    ]:
        process_message(msg, encode, decode, backend, frontend, logger)
    assert isinstance(frontend.sent[0], ErrorReply)
    assert isinstance(backend.sent[0], AbortBackendMsg)
    assert isinstance(backend.sent[1], UserMsg)
    assert not decode.decode_map
    # A dedicated detokenizer must not forward an abort back in a loop.
    count = len(backend.sent)
    process_message(AbortMsg(2), encode, decode, backend, frontend, logger, forward_aborts=False)
    assert len(backend.sent) == count


@pytest.mark.parametrize(
    "eos, tokens, expected", [(None, [1, 2], "Hello world"), ([0, 3], [1, 3], "Hello")]
)
def test_eos_normalization(eos, tokens, expected):
    tokenizer = FakeTokenizer()
    tokenizer.eos_token_id = eos
    decode = DetokenizeManager(tokenizer)
    assert (
        "".join(
            decode.detokenize(
                [DetokenizeMsg(0, token, i == len(tokens) - 1) for i, token in enumerate(tokens)]
            )
        )
        == expected
    )
    assert not decode.decode_map


def test_idle_cleanup_without_more_tokens():
    decode = DetokenizeManager(FakeTokenizer(), max_idle_seconds=10)
    decode.detokenize([DetokenizeMsg(0, 1, False), DetokenizeMsg(1, 1, False)])
    decode.decode_map[0].updated_at -= 11
    assert decode.expire_idle() == [0]
    assert set(decode.decode_map) == {1}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_tokenizer": -1},
        {"request_timeout": -1},
        {"startup_timeout": 0},
        {"request_timeout": float("nan")},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        FrontendConfig("test", **kwargs)


def test_local_path_validation_and_remote_repo_id(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(["--model-path", str(tmp_path / "missing")])
    config, _ = parse_args(["--model-path", "organization/repository", "--request-timeout", "0"])
    assert config.model_path == "organization/repository" and config.request_timeout == 0


def test_worker_startup_deadline_and_early_death():
    class Worker:
        name = "test-worker"
        exitcode = None

    with pytest.raises(RuntimeError, match="within"):
        wait_for_workers([Worker()], queue.Queue(), 0.01)
    Worker.exitcode = 1
    with pytest.raises(RuntimeError, match="exit 1"):
        wait_for_workers([Worker()], queue.Queue(), 10)


def test_worker_shutdown_reaps_only_owned_processes():
    class Process:
        def __init__(self):
            self.calls = []
            self.alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.calls.append("terminate")

        def join(self, timeout):
            self.calls.append("join")

        def kill(self):
            self.calls.append("kill")
            self.alive = False

        def close(self):
            self.calls.append("close")

    process = Process()
    group = WorkerGroup(FrontendConfig("test", _unique_suffix=".test-shutdown"), [process])
    group.shutdown()
    group.shutdown()
    assert process.calls == ["terminate", "join", "kill", "join", "close"]
    assert not group.processes


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_asgi_disconnect_releases_request(monkeypatch, stream, spec_version):
    import json

    async def run():
        state = manager()
        state.config = replace(state.config, request_timeout=0)
        monkeypatch.setattr(api_server, "_GLOBAL_STATE", state)
        disconnected = asyncio.Event()
        body = json.dumps({"model": "test", "prompt": "silent", "stream": stream}).encode()
        body_sent = False

        async def receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": spec_version},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1234),
            "server": ("127.0.0.1", 80),
        }
        task = asyncio.create_task(api_server.app(scope, receive, send))
        # Wait until the request has actually been sent before disconnecting.
        for _ in range(100):
            if state.send_tokenizer.sent:
                break
            await asyncio.sleep(0.001)
        assert isinstance(state.send_tokenizer.sent[0], TokenizeMsg)
        disconnected.set()
        await asyncio.wait_for(task, 1)
        if state._cleanup_tasks:
            await asyncio.gather(*state._cleanup_tasks)
        assert not state.ack_map and not state.event_map and not state.deadlines
        assert state.send_tokenizer.sent[-1] == AbortMsg(0)

    asyncio.run(run())
