#!/usr/bin/env python3
"""dump.py -- DeepSeek-V4-Flash reference inference dump (plain PyTorch).

Exported per-node activations (batch dim kept, batch == 1):
    input_ids                                        -> [batch, seq_len]
    embed.out                                        -> [batch, seq_len, hidden]
    layers.{N}.attn_norm.out
    layers.{N}.attn.wq_a.out
    layers.{N}.attn.q_norm.out
    layers.{N}.attn.wq_b.out
    layers.{N}.attn.wkv.out
    layers.{N}.attn.kv_norm.out
    layers.{N}.attn.compressor.wkv.out                 (compressed layers)
    layers.{N}.attn.compressor.wgate.out
    layers.{N}.attn.compressor.norm.out                (when a full block exists)
    layers.{N}.attn.compressor.out
    layers.{N}.attn.indexer.wq_b.out                   (ratio-4 layers)
    layers.{N}.attn.indexer.weights_proj.out
    layers.{N}.attn.indexer.compressor.wkv.out
    layers.{N}.attn.indexer.compressor.wgate.out
    layers.{N}.attn.indexer.compressor.norm.out         (when a full block exists)
    layers.{N}.attn.indexer.compressor.out
    layers.{N}.attn.indexer.out                        -> compressed KV indices
    layers.{N}.attn.sparse_attn.in.q
    layers.{N}.attn.sparse_attn.in.kv
    layers.{N}.attn.sparse_attn.in.topk_idxs
    layers.{N}.attn.sparse_attn.out
    layers.{N}.attn.rope.rot.{i}.out                   (forward rotations)
    layers.{N}.attn.rope.o_inv.out                     (inverse output rotation)
    layers.{N}.attn.wo_b.in                            (grouped wo_a projection)
    layers.{N}.attn.wo_b.out
    layers.{N}.attn.out
    layers.{N}.hc_attn.pre / .post / .comb
    layers.{N}.ffn_norm.out
    layers.{N}.ffn.gate.out.0                          -> routing weights
    layers.{N}.ffn.gate.out.1                          -> expert indices
    layers.{N}.ffn.experts.{E}.w1.out                   (only routed experts)
    layers.{N}.ffn.experts.{E}.w3.out
    layers.{N}.ffn.experts.{E}.w2.out
    layers.{N}.ffn.experts.{E}.out
    layers.{N}.ffn.shared_experts.w1.out
    layers.{N}.ffn.shared_experts.w3.out
    layers.{N}.ffn.shared_experts.w2.out
    layers.{N}.ffn.shared_experts.out
    layers.{N}.ffn.out
    layers.{N}.hc_ffn.pre / .post / .comb
    layers.{N}.out                                    -> [batch, seq_len, 4, hidden]
    norm.out                                         -> [batch, seq_len, hidden]
    head.out                                         -> [batch, vocab], last position

MTP records the same attention, expert and hyper-connection outputs under
mtp.0.{prefill,decode}.*, plus:
    mtp.0.{prefill,decode}.embed.out
    mtp.0.{prefill,decode}.enorm.out
    mtp.0.{prefill,decode}.hnorm.out
    mtp.0.{prefill,decode}.e_proj.out
    mtp.0.{prefill,decode}.h_proj.out
    mtp.0.{prefill,decode}.norm.out
    mtp.0.{prefill,decode}.head.out
    mtp.0.{prefill,decode}.out
    mtp.0.{prefill,decode}.logits                       -> [batch, vocab]

Runs one CPU prefill over the 43 text layers, followed by the MTP prefill and
one MTP decode step. The checkpoint is loaded one layer at a time. Floating
activations are saved as float32; token ids and indices keep their integer
dtype. All results are written to deepseek-v4-flash-vt.safetensors beside this
script, for model.vt to compare against.

NOTE on environments:
    Dump and VT verification use the same CPU stack and thread count (default 4).

Usage:
    vt-lang/.venv/bin/python models/DeepSeek-V4-Flash/dump.py
    vt-lang/.venv/bin/python models/DeepSeek-V4-Flash/dump.py --list
"""

from __future__ import annotations

import argparse
import collections
import re
import resource
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kv4.capture import Recorder, layer_range, run_reference  # noqa: E402
from kv4.loader import build_shell  # noqa: E402
from kv4.weights import Checkpoint  # noqa: E402

# The prompt the Qwen reference dump used; the reference only has to be a fixed
# input, its text does not matter for verification.
DEFAULT_PROMPT = "你好，简单介绍一下你自己。"
DEFAULT_OUT = Path(__file__).resolve().parent / "deepseek-v4-flash-vt.safetensors"
# Layer types: 0 is plain sliding window, 2 compresses with an indexer, 3
# compresses without one -- representatives of the three attention layer types.
DEFAULT_LIST_LAYERS = "0,2,3"


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


def build_input_ids(ckpt_root: Path, prompt: str | None, input_ids: str | None, log) -> torch.Tensor:
    """Token ids from ``--input-ids`` or from the checkpoint's tokenizer."""
    if input_ids:
        ids = [int(piece) for piece in input_ids.replace(",", " ").split()]
        log(f"  input ids from --input-ids: {len(ids)} tokens")
        return torch.tensor([ids], dtype=torch.long)

    tokenizer_path = ckpt_root / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise SystemExit(f"tokenizer not found: {tokenizer_path} (use --input-ids instead)")
    try:
        from tokenizers import Tokenizer
    except ImportError:
        raise SystemExit(
            "tokenising needs the 'tokenizers' package:\n"
            "  uv pip install --python vt-lang/.venv/bin/python tokenizers\n"
            "or pass --input-ids to skip the tokenizer entirely"
        )
    prompt = prompt if prompt is not None else DEFAULT_PROMPT
    ids = Tokenizer.from_file(str(tokenizer_path)).encode(prompt).ids
    log(f"  prompt {prompt!r} -> {len(ids)} tokens: {ids}")
    return torch.tensor([ids], dtype=torch.long)


def print_listing(recorder: Recorder) -> None:
    """Print the recorded keys with layer/expert indices collapsed into templates."""
    collapse = re.compile(r"layers\.\d+\.|experts\.\d+\.")
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for key in sorted(recorder.tensors):
        template = collapse.sub(
            lambda match: "layers.N." if match.group(0).startswith("layers") else "experts.E.", key
        )
        groups[template].append(key)

    print(f"\n会记录 {len(recorder.tensors)} 个张量，合计 {recorder.nbytes() / 2**20:.1f} MiB")
    print(f"\n  {'键名模板':60s} {'个数':>5s}  {'形状':24s} {'单个':>10s}")
    print("  " + "-" * 104)
    for template, keys in groups.items():
        sample = recorder.tensors[keys[0]]
        size = sample.numel() * sample.element_size()
        print(f"  {template:60s} {len(keys):5d}  {str(tuple(sample.shape)):24s} {size / 1024:8.1f} KiB")

    per_layer = collections.Counter()
    for key in recorder.tensors:
        match = re.match(r"layers\.(\d+)\.", key)
        if match:
            per_layer[int(match.group(1))] += 1
    if per_layer:
        print("\n  每层张量数: " + ", ".join(f"layer {i}: {n}" for i, n in sorted(per_layer.items())))


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepSeek-V4-Flash reference activation dump")
    parser.add_argument("--ckpt", default="~/workspace/infra/dsv4-flash", help="checkpoint directory")
    parser.add_argument("--config", default=None, help="defaults to <ckpt>/inference/config.json")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--prompt", default=None, help=f"default: {DEFAULT_PROMPT!r}")
    parser.add_argument("--input-ids", default=None, help="explicit token ids, e.g. \"0,1,2\"")
    parser.add_argument("--layers", default="all", help="all | 0 | 0,2,3 | 0-5")
    parser.add_argument("--mtp", dest="mtp", action="store_true", default=True)
    parser.add_argument("--no-mtp", dest="mtp", action="store_false")
    parser.add_argument("--list", dest="list_only", action="store_true",
                        help="dry run: record, print every key, write nothing")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    ckpt_root = Path(args.ckpt).expanduser()
    config = Path(args.config) if args.config else ckpt_root / "inference" / "config.json"
    started = time.perf_counter()

    print(f"torch {torch.__version__}  threads={torch.get_num_threads()}  device=cpu  seed={args.seed}")
    print(f"checkpoint : {ckpt_root}")
    print(f"config     : {config}")

    ckpt = Checkpoint(ckpt_root)
    print(f"{ckpt.describe()}")
    args_model, model = build_shell(config)
    print(f"shell built on meta: peak RSS {peak_rss_gib():.1f} GiB")

    input_ids = build_input_ids(ckpt_root, args.prompt, args.input_ids, print)
    layers = layer_range(DEFAULT_LIST_LAYERS if args.list_only and args.layers == "all" else args.layers,
                         ckpt.layer_indices())
    if args.list_only and args.layers == "all":
        print(f"  --list defaults to the three layer types: {layers}")
    print(f"layers     : {layers if len(layers) < 6 else f'{layers[0]}..{layers[-1]} ({len(layers)})'}"
          f"   mtp={args.mtp}")

    recorder = Recorder()
    logits = run_reference(model, args_model, ckpt, input_ids, recorder, layers,
                           include_mtp=args.mtp)

    elapsed = time.perf_counter() - started
    print(f"\nrecorded {recorder.summary()} in {elapsed:.1f} s, peak RSS {peak_rss_gib():.1f} GiB")
    top = torch.topk(torch.softmax(logits[0].float(), dim=-1), 5)
    print("next-token sanity check (greedy top-5): "
          + ", ".join(f"{int(i)}:{float(p):.4f}" for p, i in zip(top.values, top.indices)))

    if args.list_only:
        print_listing(recorder)
        print("\n(--list: nothing written)")
        return 0

    out = Path(args.out)
    non_contiguous = [key for key, tensor in recorder.tensors.items() if not tensor.is_contiguous()]
    if non_contiguous:
        raise SystemExit(f"internal error: non-contiguous tensors {non_contiguous[:3]}")
    from safetensors.torch import save_file

    save_file(recorder.tensors, str(out))
    print(f"\nwrote {out} ({out.stat().st_size / 2**20:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
