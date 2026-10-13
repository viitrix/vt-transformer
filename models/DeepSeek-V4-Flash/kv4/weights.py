"""Streaming reader and dequantizer for the released DeepSeek-V4-Flash checkpoint.

The released shards are already laid out for DeepSeek's own
``inference/model.py`` (keys such as ``layers.8.attn.wq_a.weight``), and each
shard holds exactly one decoder layer (~3.5 GB), so a full forward pass can be
run by streaming one layer at a time on a machine that could never hold the
160 GB checkpoint.

Quantisation formats in the checkpoint:

* routed MoE experts -- FP4 packed two-per-byte (``I8``), one ``F8_E8M0``
  scale per 32 values along K;
* most other linears -- ``F8_E4M3`` with one ``F8_E8M0`` scale per 128x128
  block;
* norms / compressors / indexer / embedding / router -- plain BF16;
* ``hc_*`` / ``ape`` / ``attn_sink`` / ``gate.bias`` -- plain FP32.

E8M0 is exponent-only: ``value = 2 ** (bits - 127)``.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import torch
from safetensors import safe_open

# E2M1 (FP4) code -> value, same table as inference/convert.py::FP4_TABLE.
FP4_TABLE = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)

FP8_BLOCK = 128
FP4_BLOCK = 32


def read_header(path: Path) -> dict:
    """Tensor name -> {dtype, shape, offsets} for one shard, without reading data."""
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header


def e8m0_to_float32(scale: torch.Tensor) -> torch.Tensor:
    """Decode an F8_E8M0 (exponent-only) tensor into float32."""
    exponent = scale.view(torch.uint8).to(torch.int32) - 127
    return torch.pow(2.0, exponent.to(torch.float32))


def dequantize_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """F8_E4M3 [N, K] with 128x128 F8_E8M0 blocks -> float32 (exact)."""
    n, k = weight.shape
    expanded = e8m0_to_float32(scale).repeat_interleave(FP8_BLOCK, 0).repeat_interleave(FP8_BLOCK, 1)
    return weight.float() * expanded[:n, :k]


def _as_uint8(packed: torch.Tensor) -> torch.Tensor:
    """Normalise the three ways a packed FP4 tensor can arrive."""
    if packed.dtype in (torch.int8, torch.uint8):
        return packed.view(torch.uint8)
    if hasattr(torch, "float4_e2m1fn_x2") and packed.dtype == torch.float4_e2m1fn_x2:
        return packed.view(torch.uint8)
    raise TypeError(f"not a packed FP4 tensor: dtype={packed.dtype}")


def dequantize_fp4(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Packed FP4 [N, K/2] with one F8_E8M0 scale per 32 values -> float32.

    Unpacking follows inference/convert.py: the low nibble holds element 2i and
    the high nibble element 2i+1 of the logical [N, K] matrix.
    """
    packed = _as_uint8(weight)
    low = (packed & 0x0F).to(torch.long)
    high = ((packed >> 4) & 0x0F).to(torch.long)
    values = torch.stack([FP4_TABLE[low], FP4_TABLE[high]], dim=-1).flatten(-2)
    expanded = e8m0_to_float32(scale).repeat_interleave(FP4_BLOCK, dim=1)
    return values * expanded[:, : values.size(1)]


class Checkpoint:
    """Random access to the sharded checkpoint; headers only until ``get``."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        if not self.root.is_dir():
            raise FileNotFoundError(f"checkpoint directory not found: {self.root}")
        self.shards: list[Path] = sorted(self.root.glob("*.safetensors"))
        if not self.shards:
            raise FileNotFoundError(f"no *.safetensors shards under {self.root}")
        self.index: dict[str, Path] = {}
        for shard in self.shards:
            for key in read_header(shard):
                self.index[key] = shard
        self._handles: dict[Path, object] = {}

    # -- lookup ------------------------------------------------------------
    def __contains__(self, key: str) -> bool:
        return key in self.index

    def __len__(self) -> int:
        return len(self.index)

    def keys(self) -> list[str]:
        """Every tensor name in the checkpoint, sorted."""
        return sorted(self.index)

    def keys_with_prefix(self, prefix: str) -> list[str]:
        """Every tensor whose key is ``prefix`` or starts with ``prefix.``."""
        head = prefix + "."
        return sorted(k for k in self.index if k == prefix or k.startswith(head))

    def has_scale(self, weight_key: str) -> bool:
        return weight_key.endswith(".weight") and weight_key[: -len("weight")] + "scale" in self.index

    def layer_indices(self) -> list[int]:
        return sorted({int(k.split(".")[1]) for k in self.index if k.startswith("layers.")})

    def _handle(self, shard: Path):
        handle = self._handles.get(shard)
        if handle is None:
            handle = safe_open(shard, framework="pt")
            self._handles[shard] = handle
        return handle

    def get(self, key: str) -> torch.Tensor:
        """Load one tensor. safetensors returns an mmap-backed view, so the OS
        can evict the pages again instead of pinning the whole checkpoint."""
        shard = self.index.get(key)
        if shard is None:
            raise KeyError(f"tensor not found in checkpoint: {key}")
        return self._handle(shard).get_tensor(key)

    def dequantized(self, weight_key: str) -> torch.Tensor:
        """``<x>.weight`` in float32, whatever format it is stored in."""
        weight = self.get(weight_key)
        if weight.dtype == torch.float8_e4m3fn:
            return dequantize_fp8(weight, self.get(weight_key[: -len("weight")] + "scale"))
        if weight.dtype in (torch.int8, torch.uint8):
            return dequantize_fp4(weight, self.get(weight_key[: -len("weight")] + "scale"))
        return weight.float()

    def close(self) -> None:
        for handle in self._handles.values():
            close = getattr(handle, "close", None)
            if close is not None:
                close()
        self._handles.clear()

    def describe(self) -> str:
        return (
            f"Checkpoint({self.root.name}: {len(self.shards)} shards, "
            f"{len(self.index)} tensors, layers {self.layer_indices()[:1]}..{self.layer_indices()[-1:]})"
        )
