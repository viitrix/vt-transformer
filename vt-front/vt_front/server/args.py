from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import List, Tuple


def _get_pid_suffix() -> str:
    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class FrontendConfig:
    """Configuration for the standalone HTTP frontend.

    The GPU inference backend (scheduler) is NOT launched by this frontend; it
    must be running separately and connected over the ZMQ links below.
    """

    model_path: str  # only used to load the tokenizer
    model_source: str = "huggingface"  # "huggingface" or "modelscope"
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    num_tokenizer: int = 0  # 0 means the tokenizer is shared with the detokenizer

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def share_tokenizer(self) -> bool:
        return self.num_tokenizer == 0

    @property
    def zmq_backend_addr(self) -> str:
        # tokenizer/detokenizer worker -> backend scheduler
        return "ipc:///tmp/vt_front_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        # backend scheduler -> detokenizer worker
        return "ipc:///tmp/vt_front_1" + self._unique_suffix

    @property
    def zmq_frontend_addr(self) -> str:
        # detokenizer worker -> HTTP frontend
        return "ipc:///tmp/vt_front_3" + self._unique_suffix

    @property
    def zmq_tokenizer_addr(self) -> str:
        # HTTP frontend -> tokenizer worker (only used when not shared)
        if self.share_tokenizer:
            return self.zmq_detokenizer_addr
        result = "ipc:///tmp/vt_front_4" + self._unique_suffix
        assert result != self.zmq_detokenizer_addr
        return result

    @property
    def tokenizer_create_addr(self) -> bool:
        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        return not self.share_tokenizer


def parse_args(args: List[str], run_shell: bool = False) -> Tuple[FrontendConfig, bool]:
    """
    Parse command line arguments and return a FrontendConfig.

    Args:
        args: Command line arguments (e.g., sys.argv[1:])
        run_shell: Whether shell mode was requested by the caller.

    Returns:
        FrontendConfig instance and the resolved run_shell flag.
    """
    parser = argparse.ArgumentParser(description="VT-Front Server Arguments")

    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        required=True,
        help="The path of the model tokenizer. This can be a local folder or a Hugging Face repo ID.",
    )

    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=FrontendConfig.server_host,
        help="The host address for the server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=FrontendConfig.server_port,
        help="The port number for the server to listen on.",
    )

    parser.add_argument(
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=FrontendConfig.num_tokenizer,
        help="The number of tokenizer processes to launch. 0 means the tokenizer is shared with the detokenizer.",
    )

    parser.add_argument(
        "--model-source",
        type=str,
        default="huggingface",
        choices=["huggingface", "modelscope"],
        help="The source to download the tokenizer from. Either 'huggingface' or 'modelscope'.",
    )

    parser.add_argument(
        "--shell-mode",
        action="store_true",
        help="Run the server in shell mode.",
    )

    # Parse arguments
    kwargs = parser.parse_args(args).__dict__.copy()

    # resolve some arguments
    run_shell |= kwargs.pop("shell_mode")

    model_source = kwargs.pop("model_source")
    model_path = kwargs["model_path"]
    if model_path.startswith("~"):
        model_path = os.path.expanduser(model_path)
        kwargs["model_path"] = model_path

    kwargs["model_source"] = model_source

    result = FrontendConfig(**kwargs)
    return result, run_shell
