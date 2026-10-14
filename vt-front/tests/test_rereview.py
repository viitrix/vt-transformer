import asyncio
import logging
from types import SimpleNamespace

import pytest
import zmq
from fastapi import HTTPException
from test_failure_paths import SyncQueue
from test_regressions import FakeTokenizer, manager

from vt_front.core import SamplingParams
from vt_front.message import AbortMsg, DetokenizeMsg, ErrorReply, TokenizeMsg, UserReply
from vt_front.server import api_server
from vt_front.server.args import FrontendConfig
from vt_front.server.launch import WorkerGroup
from vt_front.tokenizer import server
from vt_front.tokenizer.detokenize import DetokenizeManager
from vt_front.tokenizer.tokenize import TokenizeManager


class CongestedQueue(SyncQueue):
    def __init__(self, failures=2):
        super().__init__()
        self.failures = failures
        self.attempts = 0

    def put(self, msg):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise zmq.Again()
        super().put(msg)


@pytest.fixture
def clock(monkeypatch):
    current = [0.0]

    def sleep(seconds):
        current[0] += seconds

    monkeypatch.setattr(server, "time", SimpleNamespace(monotonic=lambda: current[0], sleep=sleep))
    return current


@pytest.mark.parametrize("kind", ["tokenize", "detokenize", "abort", "error"])
def test_worker_sends_recover_without_duplicates(clock, kind):
    class Tokenizer(FakeTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            raise ValueError("bad template")

    backend, frontend = CongestedQueue(), CongestedQueue()
    tokenizer = Tokenizer()
    messages = {
        "tokenize": TokenizeMsg(1, "text", SamplingParams()),
        "detokenize": DetokenizeMsg(1, 1, True),
        "abort": AbortMsg(1),
        "error": TokenizeMsg(1, [], SamplingParams()),
    }
    server.process_message(
        messages[kind],
        TokenizeManager(tokenizer),
        DetokenizeManager(tokenizer),
        backend,
        frontend,
        logging.getLogger("test"),
    )
    assert len(backend.sent) == (kind != "detokenize")
    assert len(frontend.sent) == (kind in ("detokenize", "error"))
    for queue in (backend, frontend):
        if queue.sent:
            assert queue.attempts == 3
    # The same worker remains able to process the next request.
    server.process_message(
        TokenizeMsg(2, "next", SamplingParams()),
        TokenizeManager(tokenizer),
        DetokenizeManager(tokenizer),
        backend,
        frontend,
        logging.getLogger("test"),
    )
    assert backend.sent[-1].uid == 2


def test_send_retry_budget_and_fatal_errors(clock):
    queue = CongestedQueue(failures=100)
    with pytest.raises(zmq.Again):
        server.send_with_retry(queue, AbortMsg(0), logging.getLogger("test"), retry_timeout=0.2)
    assert clock[0] == pytest.approx(0.2)
    assert queue.attempts == 3 and not queue.sent

    class Closed:
        def put(self, msg):
            raise zmq.ZMQError(zmq.ETERM)

    with pytest.raises(zmq.ZMQError) as caught:
        server.send_with_retry(Closed(), AbortMsg(0), logging.getLogger("test"))
    assert caught.value.errno == zmq.ETERM
    assert clock[0] == pytest.approx(0.2)


@pytest.mark.parametrize("failure", ["dispatch", "stream"])
def test_shell_continues_and_does_not_save_failed_turn(monkeypatch, capsys, failure):
    commands = iter(["bad", "good", "followup", "/exit"])
    requests = []
    state = manager()

    class Session:
        async def prompt_async(self):
            return next(commands)

    async def reply(msg):
        requests.append(msg)
        if msg.uid == 0:
            if failure == "dispatch":
                raise HTTPException(502, "dispatch failed")
            state.ack_map[msg.uid].extend(
                [UserReply(msg.uid, "partial", False), ErrorReply(msg.uid, "stream failed")]
            )
        else:
            # A literal SSE sentinel is ordinary text in the shell.
            state.ack_map[msg.uid].append(UserReply(msg.uid, "[DONE]", True))
        state.event_map[msg.uid].set()

    monkeypatch.setattr(api_server, "PromptSession", lambda *args, **kwargs: Session())
    monkeypatch.setattr(api_server, "_GLOBAL_STATE", state)
    monkeypatch.setattr(state, "send_one", reply)
    asyncio.run(api_server.shell())
    assert len(requests) == 3
    assert requests[1].text == [{"role": "user", "content": "good"}]
    assert requests[2].text == [
        {"role": "user", "content": "good"},
        {"role": "assistant", "content": "[DONE]"},
        {"role": "user", "content": "followup"},
    ]
    assert "Request failed:" in capsys.readouterr().out
    assert not state.ack_map and not state.event_map and not state.deadlines


def test_shutdown_continues_after_close_failure_and_retains_handle():
    class Process:
        def __init__(self, stubborn=False):
            self.stubborn = stubborn
            self.alive = True
            self.closed = False

        def is_alive(self):
            return self.alive

        def terminate(self):
            if not self.stubborn:
                self.alive = False

        def join(self, timeout):
            pass

        def kill(self):
            pass

        def close(self):
            if self.alive:
                raise ValueError("still running")
            self.closed = True

    stubborn, normal = Process(True), Process()
    group = WorkerGroup(FrontendConfig("test", _unique_suffix=".rereview-test"), [stubborn, normal])
    group.shutdown()
    assert normal.closed and group.processes == [stubborn]
    stubborn.alive = False
    group.shutdown()
    assert stubborn.closed and not group.processes
