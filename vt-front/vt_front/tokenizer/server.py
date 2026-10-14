from __future__ import annotations

import multiprocessing as mp
import time
from typing import List

import torch
import zmq

from vt_front.message import (
    AbortBackendMsg,
    AbortMsg,
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    ErrorReply,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from vt_front.utils import ZmqPullQueue, ZmqPushQueue, init_logger, load_tokenizer


def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    return [msg]


def send_with_retry(queue, msg, logger, retry_timeout=30.0):
    """Retry only transient backpressure; socket sends themselves are bounded.

    A failed send was not enqueued, so retrying preserves message order without
    duplicating it. Persistent failure still exits the worker after this budget
    (plus at most one SNDTIMEO interval).
    """
    deadline = time.monotonic() + retry_timeout
    warned = False
    while True:
        try:
            queue.put(msg)
            if warned:
                logger.info("Worker send recovered for user %s", msg.uid)
            return
        except zmq.Again:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.error("Worker send retry deadline exceeded for user %s", msg.uid)
                raise
            if not warned:
                logger.warning(
                    "Worker send blocked for user %s; retrying for up to %ss",
                    msg.uid,
                    retry_timeout,
                )
                warned = True
            time.sleep(min(0.1, remaining))


def process_message(
    msg,
    tokenize_manager,
    detokenize_manager,
    send_backend,
    send_frontend,
    logger,
    forward_aborts=True,
):
    """Isolate model/tokenizer errors; transport failures remain fatal."""
    try:
        if isinstance(msg, TokenizeMsg):
            tensor = tokenize_manager.tokenize([msg])[0]
            output = UserMsg(msg.uid, tensor, msg.sampling_params)
        elif isinstance(msg, DetokenizeMsg):
            text = detokenize_manager.detokenize([msg])[0]
            output = UserReply(msg.uid, text, msg.finished)
        elif isinstance(msg, AbortMsg):
            detokenize_manager.abort(msg.uid)
            if not forward_aborts:
                return
            output = AbortBackendMsg(msg.uid)
        else:
            logger.warning("Ignoring unknown tokenizer message: %s", type(msg).__name__)
            return
    except Exception as exc:
        logger.exception("Tokenizer failed for user %s", msg.uid)
        detokenize_manager.abort(msg.uid)
        send_with_retry(send_frontend, ErrorReply(msg.uid, f"Tokenizer failed: {exc}"), logger)
        send_with_retry(send_backend, AbortBackendMsg(msg.uid), logger)
        return
    if isinstance(output, UserReply):
        send_with_retry(send_frontend, output, logger)
    else:
        send_with_retry(send_backend, output, logger)


@torch.inference_mode()
def tokenize_worker(
    *,
    tokenizer_path: str,
    addr: str,
    create: bool,
    backend_addr: str,
    frontend_addr: str,
    local_bs: int,
    forward_aborts: bool = True,
    tokenizer_id: int = -1,
    model_source: str = "huggingface",
    ack_queue: mp.Queue[str] | None = None,
) -> None:
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    send_backend.socket.setsockopt(zmq.SNDTIMEO, 1000)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    send_frontend.socket.setsockopt(zmq.SNDTIMEO, 1000)
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    assert local_bs > 0

    if model_source == "modelscope":
        import os

        if not os.path.isdir(tokenizer_path):
            from modelscope import snapshot_download

            tokenizer_path = snapshot_download(
                tokenizer_path,
                ignore_patterns=["*.bin", "*.safetensors", "*.pt", "*.ckpt"],
            )

    tokenizer = load_tokenizer(tokenizer_path)
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")

    from .detokenize import DetokenizeManager
    from .tokenize import TokenizeManager

    tokenize_manager = TokenizeManager(tokenizer)
    detokenize_manager = DetokenizeManager(tokenizer)

    if ack_queue is not None:
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")

    next_sweep = 0.0
    try:
        while True:
            # Poll so abandoned decode state is reclaimed even with no traffic.
            now = time.monotonic()
            if now >= next_sweep:
                detokenize_manager.expire_idle()
                next_sweep = now + 1
            if not recv_listener.poll(timeout=1000):
                continue
            try:
                pending_msg = _unwrap_msg(recv_listener.get())
            except (ValueError, KeyError, TypeError, AttributeError, AssertionError):
                logger.exception("Dropping malformed tokenizer message")
                continue
            for msg in pending_msg:
                process_message(
                    msg,
                    tokenize_manager,
                    detokenize_manager,
                    send_backend,
                    send_frontend,
                    logger,
                    forward_aborts,
                )
    except KeyboardInterrupt:
        pass
    finally:
        send_backend.stop()
        send_frontend.stop()
        recv_listener.stop()
