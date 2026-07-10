from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class FrontendArgs:
    """Configuration for the standalone sglfront process.

    sglfront is a pure frontend: it runs the HTTP API server plus the
    tokenizer/detokenizer worker(s). The inference backend (scheduler +
    GPU engine) is started externally and connected to via the four ZMQ
    links below. The bind/connect semantics of each link match the
    in-tree python/minisgl scheduler so the two can be mixed freely;
    see docs/protocol.md for the full spec.
    """

    # HTTP
    server_host: str = "127.0.0.1"
    server_port: int = 1919

    # Tokenizer
    tokenizer_path: str = ""
    num_tokenizer: int = 0
    silent_output: bool = False

    # ZMQ links - each is fully configurable so the external backend
    # can live on another machine (use tcp://) or on the same machine
    # (default ipc://). Defaults match the in-tree python/minisgl
    # scheduler.
    zmq_frontend_addr: str = "ipc:///tmp/minisgl_3"
    zmq_tokenizer_addr: str = ""  # empty -> falls back to detokenizer_addr (shared mode)
    zmq_backend_addr: str = "ipc:///tmp/minisgl_0"
    zmq_detokenizer_addr: str = "ipc:///tmp/minisgl_1"

    # Reported to clients via /v1/models; informational only.
    model_name: str = "sglfront"

    @property
    def share_tokenizer(self) -> bool:
        return self.num_tokenizer == 0

    @property
    def resolved_tokenizer_addr(self) -> str:
        """The actual address workers and the API server use for the
        API-server -> tokenizer PUSH/PULL link.

        In shared mode (num_tokenizer=0), this collapses onto
        zmq_detokenizer_addr so the single detokenizer worker can serve
        both directions. In separate mode it is the user-provided
        zmq_tokenizer_addr (must differ from detokenizer_addr).
        """
        if self.zmq_tokenizer_addr:
            if not self.share_tokenizer:
                assert self.zmq_tokenizer_addr != self.zmq_detokenizer_addr, (
                    "zmq_tokenizer_addr must differ from zmq_detokenizer_addr "
                    "when num_tokenizer > 0"
                )
            return self.zmq_tokenizer_addr
        return self.zmq_detokenizer_addr

    @property
    def tokenizer_create_addr(self) -> bool:
        """Whether a tokenizer/detokenizer worker binds (True) or connects (False)
        the address it listens on.

        - Shared mode: the single detokenizer worker binds zmq_detokenizer_addr.
        - Separate mode: the detokenizer connects zmq_detokenizer_addr (backend
          binds it); the separate tokenizer workers also *connect*
          zmq_tokenizer_addr because the API server is the binder for that link.
        """
        return self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        """Whether the API server binds (True) or connects (False) the
        API-server -> tokenizer PUSH socket.

        - Shared mode: connect (the detokenizer worker binds).
        - Separate mode: bind (the dedicated tokenizer workers connect).
        """
        return not self.share_tokenizer


def parse_args(args: List[str], run_shell: bool = False) -> Tuple[FrontendArgs, bool]:
    """Parse command line arguments for the standalone frontend."""
    parser = argparse.ArgumentParser(
        description="SGLFront - standalone frontend for Mini-SGLang MQ protocol"
    )

    parser.add_argument(
        "--tokenizer-path",
        "--model-path",
        "--model",
        type=str,
        required=True,
        help=(
            "Path to the HuggingFace tokenizer (a local dir or repo id). "
            "Used purely for tokenization/detokenization; no model weights "
            "are loaded by sglfront."
        ),
    )

    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=FrontendArgs.server_host,
        help="The host address for the HTTP server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=FrontendArgs.server_port,
        help="The port number for the HTTP server.",
    )

    parser.add_argument(
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=FrontendArgs.num_tokenizer,
        help=(
            "Number of separate tokenizer processes. 0 means the tokenizer "
            "is shared with the detokenizer (single worker). When >0, the "
            "bind/connect side of zmq_detokenizer_addr flips: the frontend "
            "connects and the backend must bind."
        ),
    )

    parser.add_argument(
        "--zmq-frontend-addr",
        type=str,
        default=FrontendArgs.zmq_frontend_addr,
        help=(
            "ZMQ address for UserReply messages (detokenizer -> API server). "
            "The API server binds (PULL). Use tcp:// for cross-machine deploys."
        ),
    )

    parser.add_argument(
        "--zmq-tokenizer-addr",
        type=str,
        default=FrontendArgs.zmq_tokenizer_addr,
        help=(
            "ZMQ address for TokenizeMsg/AbortMsg (API server -> tokenizer). "
            "Empty (default) falls back to --zmq-detokenizer-addr in shared mode. "
            "Bind/connect side depends on --num-tokenizer; see docs/protocol.md."
        ),
    )

    parser.add_argument(
        "--zmq-backend-addr",
        type=str,
        default=FrontendArgs.zmq_backend_addr,
        help=(
            "ZMQ address for UserMsg/AbortBackendMsg (tokenizer -> backend scheduler). "
            "The tokenizer connects (PUSH); the backend binds (PULL). "
            "Use tcp:// for cross-machine deploys."
        ),
    )

    parser.add_argument(
        "--zmq-detokenizer-addr",
        type=str,
        default=FrontendArgs.zmq_detokenizer_addr,
        help=(
            "ZMQ address for DetokenizeMsg (backend -> detokenizer). "
            "Bind/connect side depends on --num-tokenizer: shared mode -> frontend binds, "
            "separate mode -> backend binds. Use tcp:// for cross-machine deploys."
        ),
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=FrontendArgs.model_name,
        help="Name reported by /v1/models and stored in chat completion responses.",
    )

    parser.add_argument(
        "--shell-mode",
        action="store_true",
        help="Run the frontend in interactive terminal shell mode instead of starting uvicorn.",
    )

    kwargs = parser.parse_args(args).__dict__.copy()

    run_shell |= kwargs.pop("shell_mode")
    if run_shell:
        kwargs["silent_output"] = True

    if kwargs["tokenizer_path"].startswith("~"):
        kwargs["tokenizer_path"] = os.path.expanduser(kwargs["tokenizer_path"])

    result = FrontendArgs(**kwargs)
    return result, run_shell
