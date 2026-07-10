from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from typing import TYPE_CHECKING

from sglfront.utils import init_logger

if TYPE_CHECKING:
    from .args import FrontendArgs


def launch_frontend(run_shell: bool = False) -> None:
    """Start the standalone frontend: HTTP API server + tokenizer/detokenizer worker(s).

    The inference backend is NOT launched here - it must be started externally
    and connected to the same four ZMQ links. See docs/protocol.md.
    """
    from .api_server import run_api_server
    from .args import parse_args

    config, run_shell = parse_args(sys.argv[1:], run_shell)
    logger = init_logger(__name__, "initializer")
    logger.info(f"Frontend args:\n{config}")

    def start_workers() -> None:
        mp.set_start_method("spawn", force=True)

        ack_queue: mp.Queue[str] = mp.Queue()

        # Detokenizer (always 1). In shared mode (num_tokenizer=0) it also
        # handles tokenize requests via the same socket; in separate mode
        # it only handles DetokenizeMsg.
        mp.Process(
            target=_detokenizer_entrypoint,
            kwargs={"config": config, "ack_queue": ack_queue},
            daemon=False,
            name="sglfront-detokenizer-0",
        ).start()

        # Additional dedicated tokenizer workers if requested.
        for i in range(config.num_tokenizer):
            mp.Process(
                target=_tokenizer_entrypoint,
                kwargs={"config": config, "tokenizer_id": i, "ack_queue": ack_queue},
                daemon=False,
                name=f"sglfront-tokenizer-{i}",
            ).start()

        # Wait for all workers to report ready:
        #   1 detokenizer + num_tokenizer separate tokenizers (or 0 if shared)
        expected_acks = 1 + config.num_tokenizer
        for _ in range(expected_acks):
            logger.info(ack_queue.get())

    run_api_server(config, start_workers, run_shell=run_shell)


def _detokenizer_entrypoint(*, config: FrontendArgs, ack_queue: mp.Queue[str]) -> None:
    """The single detokenizer worker.

    Always listens on zmq_detokenizer_addr. Binds iff shared mode
    (otherwise the external backend binds and we connect).
    """
    from sglfront.tokenizer import tokenize_worker

    if config.silent_output:
        logging.disable(logging.INFO)

    tokenize_worker(
        tokenizer_path=config.tokenizer_path,
        addr=config.zmq_detokenizer_addr,
        create=config.tokenizer_create_addr,
        backend_addr=config.zmq_backend_addr,
        frontend_addr=config.zmq_frontend_addr,
        local_bs=1,
        tokenizer_id=0,
        ack_queue=ack_queue,
    )


def _tokenizer_entrypoint(
    *, config: FrontendArgs, tokenizer_id: int, ack_queue: mp.Queue[str]
) -> None:
    """A dedicated tokenizer worker (only used when --num-tokenizer > 0).

    Listens on zmq_tokenizer_addr. Connects (does NOT bind) because the
    API server is the binder for this link and load-balances across all
    tokenizer workers via PUSH round-robin.
    """
    from sglfront.tokenizer import tokenize_worker

    if config.silent_output:
        logging.disable(logging.INFO)

    tokenize_worker(
        tokenizer_path=config.tokenizer_path,
        addr=config.resolved_tokenizer_addr,
        create=config.tokenizer_create_addr,  # False in separate mode -> connect
        backend_addr=config.zmq_backend_addr,
        frontend_addr=config.zmq_frontend_addr,
        local_bs=1,
        tokenizer_id=tokenizer_id,
        ack_queue=ack_queue,
    )


if __name__ == "__main__":
    launch_frontend()
