#!/usr/bin/env python3
"""Self check for the delivered DeepSeek-V4-Flash reference dump.

Verifies the `.safetensors` file itself, not the port that produced it
(`dev_check.py` does that). Anyone who downloads the dump can run it:

  1. **file integrity** -- key count, dtype histogram, payload size vs file size;
  2. **structure** -- every layer must carry exactly the key set its type
     implies (the spec from the README), experts must come in complete groups
     of four, and the MTP passes must be present;
  3. **values** -- no NaN / Inf anywhere, integer tensors kept an integer dtype;
  4. **end-to-end anchor** -- `embed.out` must equal a fresh embedding lookup on
     the released weights (needs ``--ckpt``; skipped without it).

Usage:
    vt-lang/.venv/bin/python models/DeepSeek-V4-Flash/check_dump.py \
        --dump models/DeepSeek-V4-Flash/deepseek-v4-flash-vt.safetensors \
        --ckpt ~/workspace/infra/dsv4-flash
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import struct
import sys
from pathlib import Path

import torch

# Keys every decoder layer has, independently of its attention type.
COMMON_KEYS = frozenset({
    "out",
    "attn_norm.out",
    "ffn_norm.out",
    "attn.out",
    "attn.wq_a.out",
    "attn.q_norm.out",
    "attn.wq_b.out",
    "attn.wkv.out",
    "attn.kv_norm.out",
    "attn.wo_b.in",          # the wo_a projection result (see the README)
    "attn.wo_b.out",
    "attn.sparse_attn.in.q",
    "attn.sparse_attn.in.kv",
    "attn.sparse_attn.in.topk_idxs",
    "attn.sparse_attn.out",
    "attn.rope.rot.0.out",
    "attn.rope.rot.1.out",
    "attn.rope.o_inv.out",
    "hc_attn.pre", "hc_attn.post", "hc_attn.comb",
    "hc_ffn.pre", "hc_ffn.post", "hc_ffn.comb",
    "ffn.out",
    "ffn.gate.out.0", "ffn.gate.out.1",
    "ffn.shared_experts.out",
    "ffn.shared_experts.w1.out",
    "ffn.shared_experts.w2.out",
    "ffn.shared_experts.w3.out",
})

def extra_keys(ratio: int, seq_len: int) -> frozenset[str]:
    """Capture keys depend on both layer type and input length."""
    if ratio == 0:
        return frozenset()
    keys = {"attn.compressor.wkv.out", "attn.compressor.wgate.out"}
    if ratio == 4:
        keys.update({
            "attn.indexer.out", "attn.indexer.wq_b.out", "attn.indexer.weights_proj.out",
            "attn.indexer.compressor.wkv.out", "attn.indexer.compressor.wgate.out",
            "attn.rope.rot.2.out",
        })
        if seq_len >= ratio:
            keys.update({
                "attn.indexer.compressor.out", "attn.indexer.compressor.norm.out",
                "attn.rope.rot.3.out", "attn.rope.rot.4.out",
            })
    elif ratio != 128:
        raise ValueError(f"unsupported compress ratio: {ratio}")
    if seq_len >= ratio:
        keys.update({"attn.compressor.out", "attn.compressor.norm.out"})
        if ratio != 4:
            keys.add("attn.rope.rot.2.out")
    return frozenset(keys)


# The MTP block is a decoder layer (ratio 0) plus its own projections and head.
MTP_EXTRA = frozenset({
    "e_proj.out", "h_proj.out", "enorm.out", "hnorm.out", "norm.out",
    "embed.out", "head.out", "logits",
})

TOP_LEVEL_KEYS = frozenset({"input_ids", "embed.out", "norm.out", "head.out"})
EXPERT_KEY = re.compile(r"ffn\.experts\.(\d+)\.(out|w1\.out|w2\.out|w3\.out)$")
EXPERT_PARTS = frozenset({"out", "w1.out", "w2.out", "w3.out"})
SIZE_BY_DTYPE = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "U8": 1, "I8": 1, "F64": 8}

PROBLEMS: list[str] = []


def fail(message: str) -> None:
    PROBLEMS.append(message)
    print(f"  ✗ {message}")


def read_header(path: Path) -> dict:
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(length)), 8 + length


def check_file(dump: Path) -> dict:
    """Header level: key count, dtypes, payload size."""
    header, offset = read_header(dump)
    header.pop("__metadata__", None)
    payload = sum(
        int(torch.tensor(entry["shape"]).prod()) * SIZE_BY_DTYPE.get(entry["dtype"], 4)
        if entry["shape"] else SIZE_BY_DTYPE.get(entry["dtype"], 4)
        for entry in header.values()
    )
    on_disk = dump.stat().st_size - offset
    dtypes = collections.Counter(entry["dtype"] for entry in header.values())
    print(f"  file        : {dump}")
    print(f"  tensors     : {len(header)}")
    print(f"  dtypes      : {dict(dtypes)}")
    print(f"  payload     : {payload / 2**20:.1f} MiB (header says) vs {on_disk / 2**20:.1f} MiB (on disk)")
    if payload != on_disk:
        fail(f"payload size mismatch: {payload} != {on_disk}")

    integers = [k for k, v in header.items() if v["dtype"] in ("I64", "I32")]
    print(f"  integer keys: {len(integers)} (input_ids / top-k indices / router indices)")
    if not any(k == "input_ids" for k in integers):
        fail("input_ids is not an integer tensor")
    return header


def check_structure(header: dict, compress_ratios: list[int], n_layers: int = 43,
                    seq_len: int = 6, n_experts: int = 256) -> None:
    """Require every main-stack layer, all base keys, and complete expert groups."""
    layers: dict[int, set[str]] = collections.defaultdict(set)
    mtp: dict[str, set[str]] = collections.defaultdict(set)
    top_level = set()
    for key in header:
        match = re.fullmatch(r"layers\.(\d+)\.(.+)", key)
        if match:
            layers[int(match.group(1))].add(match.group(2))
            continue
        match = re.fullmatch(r"mtp\.0\.(prefill|decode)\.(.+)", key)
        if match:
            mtp[match.group(1)].add(match.group(2))
            continue
        top_level.add(key)

    expected_layers = set(range(n_layers))
    if expected_layers - set(layers):
        fail(f"missing whole layers: {sorted(expected_layers - set(layers))}")
    if set(layers) - expected_layers:
        fail(f"unexpected layer indices: {sorted(set(layers) - expected_layers)}")
    print(f"  layers      : {len(layers)} / {n_layers}")
    if top_level != TOP_LEVEL_KEYS:
        fail(f"top-level keys are {sorted(top_level)}, expected {sorted(TOP_LEVEL_KEYS)}")

    def check_group(label: str, present: set[str], expected: set[str]) -> None:
        experts = {key for key in present if EXPERT_KEY.fullmatch(key)}
        other = present - experts
        if other - expected:
            fail(f"{label}: unexpected keys {sorted(other - expected)[:4]}")
        if expected - other:
            fail(f"{label}: missing keys {sorted(expected - other)[:4]}")
        groups: dict[int, set[str]] = collections.defaultdict(set)
        for key in experts:
            match = EXPERT_KEY.fullmatch(key)
            groups[int(match.group(1))].add(match.group(2))
        if not groups:
            fail(f"{label}: all routed expert outputs are missing")
        for expert_id, parts in groups.items():
            if not 0 <= expert_id < n_experts:
                fail(f"{label}: expert id out of range: {expert_id}")
            if parts != EXPERT_PARTS:
                fail(f"{label}: expert {expert_id} has incomplete outputs {sorted(parts)}")
        print(f"    {label:<18s} keys={len(present):<4d} base={len(other):<3d} experts={len(groups)}")

    for index in sorted(set(layers) & expected_layers):
        if index >= len(compress_ratios):
            fail(f"missing compress ratio for layer {index}")
            continue
        expected = COMMON_KEYS | extra_keys(compress_ratios[index], seq_len)
        check_group(f"layers.{index}", layers[index], expected)
    for tag in ("prefill", "decode"):
        if tag not in mtp:
            fail(f"mtp.0.{tag}.* is missing")
        else:
            check_group(f"mtp.0.{tag}", mtp[tag], COMMON_KEYS | MTP_EXTRA)


def check_routed_experts(dump: Path, header: dict, n_experts: int = 256) -> None:
    """The recorded router decides exactly which experts must have outputs."""
    from safetensors import safe_open

    checked = 0
    with safe_open(dump, framework="pt") as handle:
        for key in header:
            if not key.endswith(".ffn.gate.out.1"):
                continue
            routing = handle.get_tensor(key)
            if routing.dtype not in (torch.int32, torch.int64):
                fail(f"{key}: router indices must be integer")
                continue
            if routing.numel() == 0:
                fail(f"{key}: router indices are empty")
                continue
            expected = set(int(value) for value in routing.unique().tolist())
            if any(value < 0 or value >= n_experts for value in expected):
                fail(f"{key}: router expert indices are out of range")
                continue
            prefix = key[:-len("ffn.gate.out.1")]
            observed = set()
            for name in header:
                if name.startswith(prefix):
                    match = EXPERT_KEY.fullmatch(name[len(prefix):])
                    if match:
                        observed.add(int(match.group(1)))
            if expected - observed:
                fail(f"{prefix} missing routed experts: {sorted(expected - observed)}")
            if observed - expected:
                fail(f"{prefix} unexpected expert outputs: {sorted(observed - expected)}")
            for expert_id in expected:
                for part in EXPERT_PARTS:
                    name = f"{prefix}ffn.experts.{expert_id}.{part}"
                    if name not in header:
                        fail(f"missing routed expert output: {name}")
            checked += 1
    print(f"  routing     : checked {checked} router outputs against expert groups")


def check_values(dump: Path, header: dict, seq_len: int = 6) -> None:
    """Check finite values, nonempty tensors and integer index dtypes."""
    from safetensors import safe_open

    floats = non_finite = empty = 0
    with safe_open(dump, framework="pt") as handle:
        for key in header:
            tensor = handle.get_tensor(key)
            if tensor.numel() == 0:
                empty += 1
                fail(f"{key} is unexpectedly empty")
            integer_key = (key == "input_ids" or key.endswith(".ffn.gate.out.1")
                           or key.endswith(".sparse_attn.in.topk_idxs")
                           or key.endswith(".attn.indexer.out"))
            if integer_key and tensor.dtype not in (torch.int32, torch.int64):
                fail(f"{key} lost its integer dtype")
            if tensor.is_floating_point():
                floats += 1
                if not torch.isfinite(tensor).all():
                    non_finite += 1
                    fail(f"{key} contains NaN or Inf")
    print(f"  checked     : {floats} floats, {empty} empty, {non_finite} non-finite")


def check_anchor(dump: Path, ckpt: Path) -> None:
    """embed.out must equal a fresh embedding lookup on the released weights."""
    from safetensors import safe_open

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from kv4.weights import Checkpoint

    checkpoint = Checkpoint(ckpt)
    weight = checkpoint.get("embed.weight")
    with safe_open(dump, framework="pt") as handle:
        input_ids = handle.get_tensor("input_ids")
        recorded = handle.get_tensor("embed.out").float()
    expected = torch.nn.functional.embedding(input_ids, weight).to(torch.bfloat16)
    gap = (recorded.float() - expected.float()).abs().max().item()
    print(f"  embed.out   : recomputed from {tuple(weight.shape)} {weight.dtype} weights, "
          f"max|diff|={gap:.3e}")
    if gap != 0.0:
        fail(f"embed.out does not match a fresh embedding lookup (max|diff|={gap:.3e})")


def main() -> int:
    parser = argparse.ArgumentParser(description="self check for the V4-Flash reference dump")
    parser.add_argument("--dump", default=str(Path(__file__).resolve().parent / "deepseek-v4-flash-vt.safetensors"))
    parser.add_argument("--ckpt", default=None, help="checkpoint directory, for the embed.out anchor")
    args = parser.parse_args()

    dump = Path(args.dump).expanduser()
    if not dump.is_file():
        raise SystemExit(f"dump not found: {dump}")

    PROBLEMS.clear()
    torch.set_num_threads(4)
    print("[1] file integrity")
    header = check_file(dump)

    print("\n[2] structure")
    config = Path(args.ckpt).expanduser() / "inference" / "config.json" if args.ckpt else None
    if config and config.is_file():
        settings = json.loads(config.read_text())
        compress_ratios = settings["compress_ratios"]
        n_layers = settings["n_layers"]
        n_experts = settings["n_routed_experts"]
    else:
        compress_ratios = [0, 0] + [4 if i % 2 == 0 else 128 for i in range(41)] + [0]
        n_layers, n_experts = 43, 256
        print("  (config not found, using the released configuration)")
    shape = header.get("input_ids", {}).get("shape", [])
    if len(shape) != 2 or min(shape) <= 0:
        fail(f"input_ids must have positive [batch, sequence] shape, got {shape}")
        return 1
    seq_len = shape[1]
    check_structure(header, compress_ratios, n_layers, seq_len, n_experts)

    print("\n[3] values and routed expert coverage")
    check_values(dump, header, seq_len)
    check_routed_experts(dump, header, n_experts)

    if args.ckpt:
        print("\n[4] end-to-end anchor")
        check_anchor(dump, Path(args.ckpt).expanduser())
    else:
        print("\n[4] end-to-end anchor skipped (no --ckpt)")

    print("\n" + "=" * 68)
    if PROBLEMS:
        print(f"{len(PROBLEMS)} problem(s) found:")
        for problem in PROBLEMS[:20]:
            print(f"  - {problem}")
        return 1
    print("all file checks passed; numerical model correctness is checked separately by vt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
