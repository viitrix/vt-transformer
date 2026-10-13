"""kv4 -- a pure-PyTorch reference implementation of DeepSeek-V4-Flash.

DeepSeek's released inference code (``inference/model.py`` + ``inference/kernel.py``)
targets CUDA through tilelang kernels and loads the whole 160 GB checkpoint at
once. This package keeps ``model.py`` as a near-verbatim copy and replaces only
the kernels and the loading strategy, so that a single prefill forward pass --
the "reference activations" that vt-lang verifies ``.vt`` nodes against -- can
be produced on a machine with far less memory than the model.

Modules:
    weights.py -- streaming shard reader + fp8/fp4 dequantisation
    kernels.py -- plain-torch equivalents of the six tilelang kernels
    model.py   -- verbatim port of inference/model.py (4 auditable changes)
    loader.py  -- meta-device shell, per-layer filling, buffer resetting
"""

from .kernels import (
    act_quant,
    fp4_act_quant,
    fp4_gemm,
    fp8_gemm,
    hadamard_matrix,
    hc_split_sinkhorn,
    rotate_activation,
    sparse_attn,
)
from .loader import build_shell, fill, free, load_prefix, prepare_tensor, reset_buffers
from .weights import Checkpoint, dequantize_fp4, dequantize_fp8, e8m0_to_float32

__all__ = [
    "Checkpoint",
    "act_quant",
    "build_shell",
    "dequantize_fp4",
    "dequantize_fp8",
    "e8m0_to_float32",
    "fill",
    "fp4_act_quant",
    "fp4_gemm",
    "fp8_gemm",
    "free",
    "hadamard_matrix",
    "hc_split_sinkhorn",
    "load_prefix",
    "prepare_tensor",
    "reset_buffers",
    "rotate_activation",
    "sparse_attn",
]
