from __future__ import annotations

import multiprocessing as mp
import os
import queue
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vt_front.utils import init_logger

if TYPE_CHECKING:
    from .args import FrontendConfig

logger = init_logger(__name__, "initializer")


def wait_for_workers(workers, ack_queue, timeout):
    """One deadline for the whole group; fail promptly when a worker exits."""
    deadline = time.monotonic() + timeout
    remaining_acks = len(workers)
    while remaining_acks:
        failed = [p for p in workers if p.exitcode is not None]
        if failed:
            raise RuntimeError(
                f"Tokenizer worker exited during startup: {failed[0].name} (exit {failed[0].exitcode})"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"Tokenizer workers failed to start within {timeout}s")
        try:
            logger.info(ack_queue.get(timeout=min(0.1, remaining)))
            remaining_acks -= 1
        except queue.Empty:
            continue


@dataclass
class WorkerGroup:
    config: FrontendConfig
    processes: list = field(default_factory=list)
    ack_queue: object = None

    def start(self):
        from vt_front.tokenizer import tokenize_worker

        ctx = mp.get_context("spawn")
        self.ack_queue = ctx.Queue()
        count = self.config.num_tokenizer
        try:
            for i in range(count + 1):
                detokenizer = i == count
                process = ctx.Process(
                    target=tokenize_worker,
                    kwargs={
                        "tokenizer_path": self.config.model_path,
                        "addr": self.config.zmq_detokenizer_addr
                        if detokenizer
                        else self.config.zmq_tokenizer_addr,
                        "backend_addr": self.config.zmq_backend_addr,
                        "frontend_addr": self.config.zmq_frontend_addr,
                        "local_bs": 1,
                        "forward_aborts": not detokenizer or self.config.share_tokenizer,
                        "create": self.config.tokenizer_create_addr,
                        "tokenizer_id": i,
                        "model_source": self.config.model_source,
                        "ack_queue": self.ack_queue,
                    },
                    name=f"vt-front-{'detokenizer' if detokenizer else 'tokenizer'}-{i}",
                )
                process.start()
                self.processes.append(process)
            wait_for_workers(self.processes, self.ack_queue, self.config.startup_timeout)
        except BaseException:
            self.shutdown()
            raise

    def shutdown(self):
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        deadline = time.monotonic() + 5
        unclosed = []
        for process in self.processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
            try:
                process.close()
            except ValueError:
                logger.warning(
                    "Worker is not yet reaped; retaining handle for another cleanup attempt"
                )
                unclosed.append(process)
        self.processes = unclosed
        if self.ack_queue is not None:
            self.ack_queue.close()
            self.ack_queue.join_thread()
            self.ack_queue = None
        if unclosed:
            return
        # Only remove the IPC endpoints this frontend/its workers bind.
        for address in (self.config.zmq_frontend_addr, self.config.zmq_tokenizer_addr):
            if address.startswith("ipc://"):
                try:
                    os.unlink(address.removeprefix("ipc://"))
                except FileNotFoundError:
                    pass


def launch_frontend(run_shell: bool = False) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    config, run_shell = parse_args(sys.argv[1:], run_shell)
    workers = WorkerGroup(config)
    try:
        run_api_server(config, workers.start, run_shell=run_shell)
    finally:
        workers.shutdown()


def main() -> None:
    launch_frontend()


if __name__ == "__main__":
    main()
