from __future__ import annotations

import multiprocessing as mp
import sys
from typing import TYPE_CHECKING

from vt_front.utils import init_logger

if TYPE_CHECKING:
    from .args import FrontendConfig


def launch_frontend(run_shell: bool = False) -> None:
    """Launch the standalone HTTP frontend.

    This spawns the tokenizer/detokenizer worker processes and starts the
    FastAPI server. The GPU inference backend (scheduler) is NOT launched
    here; it must be running separately and connected over the ZMQ links
    (see FrontendConfig for the addresses).
    """
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    logger = init_logger(__name__, "initializer")

    def start_subprocess() -> None:
        from vt_front.tokenizer import tokenize_worker

        mp.set_start_method("spawn", force=True)

        # a multiprocessing queue to receive ack from subprocesses
        # so that we can guarantee all subprocesses are ready
        ack_queue: mp.Queue[str] = mp.Queue()

        num_tokenizers = server_args.num_tokenizer
        # DeTokenizer, only 1
        mp.Process(
            target=tokenize_worker,
            kwargs={
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_detokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers,
                "model_source": server_args.model_source,
                "ack_queue": ack_queue,
            },
            daemon=False,
            name="vt-front-detokenizer-0",
        ).start()
        for i in range(num_tokenizers):
            mp.Process(
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_tokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                    "model_source": server_args.model_source,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name=f"vt-front-tokenizer-{i}",
            ).start()

        # Wait for acknowledgments from all worker processes:
        # - num_tokenizers tokenizers
        # - 1 detokenizer
        for _ in range(num_tokenizers + 1):
            logger.info(ack_queue.get())

    run_api_server(server_args, start_subprocess, run_shell=run_shell)


def main() -> None:
    launch_frontend()


if __name__ == "__main__":
    main()
