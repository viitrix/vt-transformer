"""End-to-end smoke test for the standalone vt_front package.

Starts:
  - a mock inference backend (speaks the ZMQ message protocol)
  - the real vt_front frontend (tokenizer/detokenizer workers + FastAPI server)
then issues HTTP requests against /v1/chat/completions (non-streaming and
streaming) and /generate, and verifies the outputs.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import threading
import time
import urllib.request

import torch
from transformers import AutoTokenizer

MODEL_PATH = (
    __import__("os").path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots"
    )
)
import glob

MODEL_PATH = glob.glob(MODEL_PATH + "/*")[0]

from vt_front.message import (
    BaseBackendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    UserMsg,
)
from vt_front.server.api_server import run_api_server
from vt_front.server.args import parse_args
from vt_front.utils import ZmqPullQueue, ZmqPushQueue

REPLY_TEXT = "Hello from the mock vt-front backend! 你好，来自 vt-front。"
PORT = 8931


def mock_backend(cfg, tokenizer):
    reply_ids = tokenizer.encode(REPLY_TEXT, add_special_tokens=False)
    recv = ZmqPullQueue(cfg.zmq_backend_addr, create=True, decoder=BaseBackendMsg.decoder)
    send = ZmqPushQueue(cfg.zmq_detokenizer_addr, create=False, encoder=BatchTokenizerMsg.encoder)
    print("[mock-backend] ready", flush=True)
    while True:
        msg = recv.get()
        if not isinstance(msg, UserMsg):
            continue
        print(f"[mock-backend] got UserMsg uid={msg.uid} input_len={len(msg.input_ids)}", flush=True)
        for i, tid in enumerate(reply_ids):
            last = i == len(reply_ids) - 1
            send.put(DetokenizeMsg(uid=msg.uid, next_token=int(tid), finished=last))


def start_frontend(cfg):
    def start_subprocess():
        from vt_front.tokenizer import tokenize_worker

        mp.set_start_method("spawn", force=True)
        ack_queue: mp.Queue[str] = mp.Queue()
        mp.Process(
            target=tokenize_worker,
            kwargs=dict(
                tokenizer_path=cfg.model_path,
                addr=cfg.zmq_detokenizer_addr,
                backend_addr=cfg.zmq_backend_addr,
                frontend_addr=cfg.zmq_frontend_addr,
                local_bs=1,
                create=cfg.tokenizer_create_addr,
                tokenizer_id=cfg.num_tokenizer,
                ack_queue=ack_queue,
            ),
            daemon=False,
            name="vt-front-detokenizer-0",
        ).start()
        for _ in range(cfg.num_tokenizer + 1):
            print("[frontend]", ack_queue.get(), flush=True)

    t = threading.Thread(target=lambda: run_api_server(cfg, start_subprocess, run_shell=False), daemon=True)
    t.start()
    return t


def http_post(path, payload, stream=False):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    cfg, _ = parse_args(["--model-path", MODEL_PATH, "--port", str(PORT)])
    print(f"[test] config: {cfg}", flush=True)

    backend_thread = threading.Thread(target=mock_backend, args=(cfg, tokenizer), daemon=True)
    backend_thread.start()

    start_frontend(cfg)

    # wait for HTTP server
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1", timeout=2)
            break
        except Exception:
            time.sleep(0.5)
    else:
        raise RuntimeError("frontend HTTP server did not start")
    print("[test] HTTP server is up", flush=True)

    # 1. non-streaming chat completions
    r = http_post(
        "/v1/chat/completions",
        {"model": "test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 32},
    )
    content = r["choices"][0]["message"]["content"]
    assert content == REPLY_TEXT, f"unexpected content: {content!r}"
    print("[test] /v1/chat/completions OK:", content[:50], flush=True)

    # 2. streaming chat completions
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 32,
                "stream": True,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode()
    assert "data: [DONE]" in body
    streamed = "".join(
        json.loads(line[6:])["choices"][0]["delta"].get("content", "")
        for line in body.splitlines()
        if line.startswith("data: {")
    )
    assert streamed == REPLY_TEXT, f"unexpected stream: {streamed!r}"
    print("[test] /v1/chat/completions stream OK", flush=True)

    # 3. /generate raw endpoint
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/generate",
        data=json.dumps({"prompt": "hi", "max_tokens": 32}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode()
    assert "data: [DONE]" in body and REPLY_TEXT.split()[0] in body
    print("[test] /generate OK", flush=True)

    # 4. /v1/models
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=10) as resp:
        models = json.loads(resp.read())
    assert models["data"][0]["id"] == MODEL_PATH
    print("[test] /v1/models OK", flush=True)

    print("E2E_TEST_PASSED", flush=True)


if __name__ == "__main__":
    main()
