"""Offline integration smoke test: HTTP handlers + real ZMQ + spawned workers.

Run `python tests/e2e_smoke.py` from vt-front. A tiny tokenizer is generated
locally; no GPU, model download, fixed HTTP port, or cached model is required.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import tempfile
import threading
import time
from pathlib import Path

import zmq
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from vt_front.message import (
    AbortBackendMsg,
    AbortMsg,
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    DetokenizeMsg,
    ErrorReply,
    UserMsg,
)
from vt_front.server import api_server
from vt_front.server.api_server import FrontendManager
from vt_front.server.args import FrontendConfig
from vt_front.server.launch import WorkerGroup
from vt_front.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, ZmqPullQueue, ZmqPushQueue


def make_tokenizer(path):
    core = Tokenizer(
        models.WordLevel(
            {"<eos>": 0, "Hello": 1, "world": 2, "<unk>": 3, "error": 4, "silent": 5},
            unk_token="<unk>",
        )
    )
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, eos_token="<eos>", unk_token="<unk>")
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    tokenizer.save_pretrained(path)


def mock_backend(config, ready):
    recv = ZmqPullQueue(config.zmq_backend_addr, create=True, decoder=BaseBackendMsg.decoder)
    send = ZmqPushQueue(
        config.zmq_detokenizer_addr,
        create=config.backend_create_detokenizer_link,
        encoder=BaseTokenizerMsg.encoder,
    )
    frontend = ZmqPushQueue(config.zmq_frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    ready.put("ready")
    try:
        while True:
            msg = recv.get()
            if isinstance(msg, AbortBackendMsg):
                if not config.share_tokenizer:
                    send.put(AbortMsg(msg.uid))
                continue
            if not isinstance(msg, UserMsg):
                continue
            if 4 in msg.input_ids.tolist():
                frontend.put(ErrorReply(msg.uid, "mock backend failure"))
            elif 5 in msg.input_ids.tolist():
                send.put(DetokenizeMsg(msg.uid, 1, False))
            else:
                for token in [1, 2, 0]:
                    send.put(DetokenizeMsg(msg.uid, token, token == 0))
    finally:
        recv.stop()
        send.stop()
        frontend.stop()


def run_mode(model_path, num_tokenizer):
    config = FrontendConfig(
        model_path,
        num_tokenizer=num_tokenizer,
        startup_timeout=20,
        request_timeout=0.5,
        _unique_suffix=f".smoke-{os.getpid()}-{num_tokenizer}",
    )
    ctx = mp.get_context("spawn")
    ready = ctx.Queue()
    backend = ctx.Process(target=mock_backend, args=(config, ready))
    workers = WorkerGroup(config)
    state = FrontendManager(
        config,
        ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            config.frontend_create_tokenizer_link,
            BaseTokenizerMsg.encoder,
        ),
        ZmqAsyncPullQueue(config.zmq_frontend_addr, True, BaseFrontendMsg.decoder),
    )
    backend.start()
    backend_pid = backend.pid
    worker_pids = []
    try:
        ready.get(timeout=20)
        workers.start()
        worker_pids = [process.pid for process in workers.processes]
        api_server._GLOBAL_STATE = state
        with TestClient(api_server.app) as client:
            payload = {"model": "test", "messages": [{"role": "user", "content": "Hello"}]}
            response = client.post("/v1/chat/completions", json=payload)
            assert response.status_code == 200, response.text
            assert response.json()["choices"][0]["message"]["content"] == "Hello world"
            response = client.post("/v1/chat/completions", json={**payload, "stream": True})
            chunks = [json.loads(event[6:]) for event in response.text.strip().split("\n\n")[:-1]]
            assert (
                "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
                == "Hello world"
            )
            assert response.text.endswith("data: [DONE]\n\n")
            response = client.post("/generate", json={"prompt": "Hello", "max_tokens": 4})
            assert response.status_code == 200 and response.text.endswith("data: [DONE]\n\n")
            response = client.post(
                "/v1/chat/completions", json={"model": "test", "prompt": "error"}
            )
            assert response.status_code == 502, response.text
            response = client.post("/generate", json={"prompt": "error", "max_tokens": 4})
            assert '"code": 502' in response.text and response.text.endswith("data: [DONE]\n\n")
            response = client.post(
                "/v1/chat/completions", json={"model": "test", "prompt": "silent"}
            )
            assert response.status_code == 504, response.text
            # A failed/aborted request must not prevent subsequent success.
            assert client.post("/v1/chat/completions", json=payload).status_code == 200
            backend.kill()
            backend.join(timeout=5)
            response = client.post("/v1/chat/completions", json=payload)
            assert response.status_code == 504, response.text
            assert not state.ack_map and not state.event_map and not state.deadlines
        print(f"E2E passed: num_tokenizer={num_tokenizer}", flush=True)
    finally:
        state.shutdown()
        api_server._GLOBAL_STATE = None
        workers.shutdown()
        if backend.is_alive():
            backend.terminate()
        backend.join(timeout=5)
        if backend.is_alive():
            backend.kill()
            backend.join(timeout=5)
        backend.close()
        ready.close()
        ready.join_thread()
        assert not {p.pid for p in mp.active_children()} & {backend_pid, *worker_pids}
        # The backend owns these endpoints; its test harness cleans them up.
        for address in (config.zmq_backend_addr, config.zmq_detokenizer_addr):
            Path(address.removeprefix("ipc://")).unlink(missing_ok=True)
        assert not Path(config.zmq_frontend_addr.removeprefix("ipc://")).exists()
        assert not Path(config.zmq_tokenizer_addr.removeprefix("ipc://")).exists()


def check_startup_timeout(model_path):
    config = FrontendConfig(
        model_path, startup_timeout=0.001, _unique_suffix=f".smoke-timeout-{os.getpid()}"
    )
    before = {process.pid for process in mp.active_children()}
    workers = WorkerGroup(config)
    try:
        workers.start()
    except RuntimeError as exc:
        assert "within" in str(exc)
    else:
        raise AssertionError("Expected worker startup timeout")
    finally:
        workers.shutdown()
    assert {process.pid for process in mp.active_children()} == before
    print("E2E passed: startup timeout reaps workers", flush=True)


def check_send_recovery():
    from vt_front.tokenizer.server import send_with_retry

    address = f"ipc:///tmp/vt-front-retry-{os.getpid()}"
    sender = ZmqPushQueue(address, create=True, encoder=BaseBackendMsg.encoder)
    sender.socket.setsockopt(zmq.SNDTIMEO, 50)
    received, failures = [], []

    def late_receiver():
        time.sleep(0.3)  # Force several send timeouts before a peer is available.
        receiver = ZmqPullQueue(address, create=False, decoder=BaseBackendMsg.decoder)
        receiver.socket.setsockopt(zmq.RCVTIMEO, 5000)
        try:
            received.extend([receiver.get(), receiver.get()])
        except Exception as exc:
            failures.append(exc)
        finally:
            receiver.stop()

    thread = threading.Thread(target=late_receiver)
    thread.start()
    try:
        for uid in [1, 2]:
            send_with_retry(
                sender, AbortBackendMsg(uid), logging.getLogger("retry-test"), retry_timeout=3
            )
        thread.join(timeout=6)
        assert not thread.is_alive() and not failures
        assert received == [AbortBackendMsg(1), AbortBackendMsg(2)]
        print("E2E passed: send recovers after peer arrival without duplicate messages", flush=True)
    finally:
        sender.stop()
        thread.join(timeout=6)
        Path(address.removeprefix("ipc://")).unlink(missing_ok=True)


def main():
    check_send_recovery()
    with tempfile.TemporaryDirectory(prefix="vt-front-smoke-") as path:
        make_tokenizer(path)
        check_startup_timeout(path)
        for count in [0, 1]:
            run_mode(path, count)
    print("E2E_TEST_PASSED", flush=True)


if __name__ == "__main__":
    main()
