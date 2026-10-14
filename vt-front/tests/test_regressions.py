import asyncio
import json

import pytest
import torch
from fastapi.testclient import TestClient

from vt_front.core import SamplingParams
from vt_front.message import AbortMsg, DetokenizeMsg, TokenizeMsg, UserReply
from vt_front.server.api_server import FrontendManager, app
from vt_front.server.args import FrontendConfig
from vt_front.tokenizer.detokenize import DetokenizeManager
from vt_front.tokenizer.tokenize import TokenizeManager


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return "<bos>hello"

    def encode(self, prompt, *, return_tensors, add_special_tokens):
        ids = [1, 2] if prompt.startswith("<bos>") else [2]
        if add_special_tokens:
            ids.insert(0, 1)
        return torch.tensor([ids])

    def batch_decode(self, batches):
        vocab = {1: "Hello", 2: " world", 3: "\n你好"}
        return ["".join(vocab[token] for token in batch) for batch in batches]


class FakeQueue:
    def __init__(self):
        self.sent = []

    async def put(self, msg):
        self.sent.append(msg)


def manager():
    return FrontendManager(
        FrontendConfig(model_path="test"), FakeQueue(), FakeQueue(), initialized=True
    )


def test_chat_template_does_not_duplicate_special_tokens():
    tokenizer = TokenizeManager(FakeTokenizer())
    chat = TokenizeMsg(0, [{"role": "user", "content": "hello"}], SamplingParams())
    raw = TokenizeMsg(1, "hello", SamplingParams())
    assert [ids.tolist() for ids in tokenizer.tokenize([chat, raw])] == [[1, 2], [1, 2]]


@pytest.mark.parametrize("tokens", [[1, 2, 0], [1, 2, 3]])
def test_batched_tokens_match_sequential_decoding(tokens):
    messages = [DetokenizeMsg(7, token, i == len(tokens) - 1) for i, token in enumerate(tokens)]
    sequential = DetokenizeManager(FakeTokenizer())
    expected = [sequential.detokenize([msg])[0] for msg in messages]
    batched = DetokenizeManager(FakeTokenizer())
    assert batched.detokenize(messages) == expected
    assert not batched.decode_map


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"messages": []},
        {"prompt": "x", "messages": [{"role": "user", "content": "x"}]},
        {"prompt": "x", "max_tokens": 0},
        {"prompt": "x", "temperature": -1},
        {"prompt": "x", "top_p": 0},
        {"prompt": "x", "top_p": 1.1},
        {"prompt": "x", "top_k": 0},
        {"prompt": "x", "n": 2},
        {"prompt": "x", "stop": ["END"]},
        {"prompt": "x", "presence_penalty": 1},
        {"prompt": "x", "frequency_penalty": 1},
    ],
)
def test_invalid_requests_rejected_before_dispatch(payload):
    with TestClient(app) as client:
        assert (
            client.post("/v1/chat/completions", json={"model": "test", **payload}).status_code
            == 422
        )


def test_generate_rejects_nonpositive_limit():
    with TestClient(app) as client:
        assert client.post("/generate", json={"prompt": "x", "max_tokens": -1}).status_code == 422


def test_stream_formats_and_cleanup():
    async def run():
        state = manager()
        uid = state.new_user()
        state.ack_map[uid] = [UserReply(uid, "hello\nworld\n", True)]
        state.event_map[uid].set()
        frames = [chunk async for chunk in state.stream_generate(uid)]
        assert frames == [b"data: hello\ndata: world\ndata: \n\n", b"data: [DONE]\n\n"]
        assert not state.ack_map and not state.event_map
        uid = state.new_user()
        state.ack_map[uid] = [UserReply(uid, "hello", True)]
        state.event_map[uid].set()
        frames = [chunk async for chunk in state.stream_chat_completions(uid, "test")]
        chunks = [json.loads(frame[6:]) for frame in frames[:-1]]
        assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
        assert all(
            chunk["model"] == "test" and isinstance(chunk["created"], int) for chunk in chunks
        )
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": "hello"}
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert not state.ack_map and not state.event_map

    asyncio.run(run())


def test_closing_stream_cleans_up_and_aborts_backend():
    class ConnectedRequest:
        async def is_disconnected(self):
            return False

    async def run():
        state = manager()
        uid = state.new_user()
        state.ack_map[uid] = [UserReply(uid, "hello", False)]
        state.event_map[uid].set()
        stream = state.stream_with_cancellation(state.stream_generate(uid), ConnectedRequest(), uid)
        await anext(stream)
        await stream.aclose()
        assert not state.ack_map and not state.event_map
        assert state.send_tokenizer.sent == [AbortMsg(uid)]

    asyncio.run(run())


def test_cancelling_while_waiting_aborts_backend():
    async def run():
        state = manager()
        uid = state.new_user()
        stream = state.stream_with_cancellation(state.stream_generate(uid), None, uid)
        task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not state.ack_map and not state.event_map
        assert state.send_tokenizer.sent == [AbortMsg(uid)]

    asyncio.run(run())


def test_successful_stream_does_not_abort():
    class ConnectedRequest:
        async def is_disconnected(self):
            return False

    async def run():
        state = manager()
        uid = state.new_user()
        state.ack_map[uid] = [UserReply(uid, "done", True)]
        state.event_map[uid].set()
        stream = state.stream_with_cancellation(state.stream_generate(uid), ConnectedRequest(), uid)
        assert len([frame async for frame in stream]) == 2
        assert state.send_tokenizer.sent == []
        assert not state.ack_map and not state.event_map

    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
def test_chat_http_round_trip(monkeypatch, stream):
    from vt_front.server import api_server

    state = manager()

    async def reply(msg):
        state.ack_map[msg.uid].append(UserReply(msg.uid, "你好\nworld", True))
        state.event_map[msg.uid].set()

    monkeypatch.setattr(state, "send_one", reply)
    monkeypatch.setattr(state, "shutdown", lambda: None)
    monkeypatch.setattr(api_server, "_GLOBAL_STATE", state)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": stream,
            },
        )
    assert response.status_code == 200
    if stream:
        events = response.text.strip().split("\n\n")
        assert events[-1] == "data: [DONE]"
        chunk = json.loads(events[0][6:])
        assert chunk["choices"][0]["delta"]["content"] == "你好\nworld"
    else:
        assert response.json()["choices"][0]["message"]["content"] == "你好\nworld"
    assert not state.ack_map and not state.event_map
