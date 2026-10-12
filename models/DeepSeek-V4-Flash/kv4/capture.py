"""Recording a reference forward pass for DeepSeek-V4-Flash.

Two capture mechanisms, because not every quantity of interest sits on a module
boundary (this is the capture spec agreed for this dump):

1. **forward hooks** on *every* module of a decoder layer record its return
   value under ``<module path>.out`` -- the naming convention of the Qwen
   reference dumps. Tuple returns (the router's weights and indices) are
   expanded to ``.out.0`` / ``.out.1``. Only the experts that are actually
   routed to ever run, so only their outputs appear.
2. the three fused helpers that have no module of their own are **wrapped for
   the duration of one layer**:
   ``hc_split_sinkhorn`` (``pre``/``post``/``comb``), ``sparse_attn`` (its
   inputs and output) and ``apply_rotary_emb`` (the rotated slice -- it mutates
   in place, so the value is read after the call). model.py imported those names
   into its own namespace, so patching them there changes only the call sites:
   model.py itself is not touched, and the originals are restored in a
   ``finally`` block.

``act_quant`` / ``fp4_act_quant`` are deliberately not wrapped: they run inside
every Linear and their results are already visible at the surrounding module
boundaries.

Two details matter for the record to stay correct:

* every tensor is cloned to CPU immediately -- ``apply_rotary_emb`` and the
  in-place activation quantisation mutate their inputs after the call;
* integer tensors keep their dtype (token ids and top-k indices must not be
  rounded through float32).
"""

from __future__ import annotations

import contextlib
import itertools

import torch

from . import model as model_module
from .loader import fill, free, load_prefix
from .model import Transformer
from .weights import Checkpoint

# ``apply_rotary_emb`` is called several times per layer and the number of calls
# depends on the layer type (plain sliding-window, compressed, or compressed
# with an indexer -- the compressor and the indexer rotate their own KV/q), so
# the forward calls are labelled by index and the inverse call by name:
#
#   ratio 0    : rot.0 = q        rot.1 = kv                              o_inv
#   ratio 128  : rot.0 = q        rot.1 = kv    rot.2 = compressor kv     o_inv
#   ratio 4    : rot.0 = q        rot.1 = kv    rot.2 = indexer q
#                rot.3 = indexer compressor kv  rot.4 = compressor kv     o_inv
_ROPE_LABELS = "rot"


class Recorder:
    """Collects ``key -> tensor`` while the model runs."""

    def __init__(self, enabled: bool = True):
        self.tensors: dict[str, torch.Tensor] = {}
        self.enabled = enabled

    def add(self, key: str, value) -> None:
        if not self.enabled or value is None:
            return
        if torch.is_tensor(value):
            tensor = value.detach().to("cpu")
            if tensor.is_floating_point():
                tensor = tensor.float()
            # clone: in-place ops mutate the buffer after the hook fires;
            # contiguous: safetensors requires it
            self.tensors[key] = tensor.clone().contiguous()
        elif isinstance(value, (tuple, list)):
            for position, item in enumerate(value):
                self.add(f"{key}.{position}", item)
        # plain ints/floats are not activations: skipped

    def __len__(self) -> int:
        return len(self.tensors)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors.values())

    def summary(self) -> str:
        return f"{len(self.tensors)} tensors, {self.nbytes() / 2**20:.1f} MiB"


def _hook(recorder: Recorder, key: str):
    def hook(module, inputs, output):
        recorder.add(f"{key}.out", output)

    return hook


def _input_hook(recorder: Recorder, key: str):
    def hook(module, args):
        if args:
            recorder.add(f"{key}.in", args[0])

    return hook


# Attention.forward builds the grouped low-rank output projection inline --
# ``wo_a = self.wo_a.weight.view(...); o = torch.einsum("bsgd,grd->bsgr", o, wo_a)``
# -- and never calls the ``wo_a`` module, so its result is invisible to a module
# output hook. Its only appearance is as the input of ``wo_b`` (flattened), so
# that input is recorded as ``<path>.wo_b.in``.
_INPUT_HOOK_SUFFIXES = ("wo_b",)


def record_module_tree(recorder: Recorder, module: torch.nn.Module, prefix: str) -> list:
    """Hook ``module`` and all of its submodules; returns the hook handles."""
    handles = [module.register_forward_hook(_hook(recorder, prefix))]
    for name, submodule in module.named_modules():
        if not name:
            continue
        handles.append(submodule.register_forward_hook(_hook(recorder, f"{prefix}.{name}")))
        if name.rsplit(".", 1)[-1] in _INPUT_HOOK_SUFFIXES:
            handles.append(
                submodule.register_forward_pre_hook(_input_hook(recorder, f"{prefix}.{name}"))
            )
    return handles


@contextlib.contextmanager
def record_fused(recorder: Recorder, prefix: str):
    """Wrap the three fused helpers of kv4.model for the duration of one layer."""
    originals = {
        name: getattr(model_module, name)
        for name in ("hc_split_sinkhorn", "sparse_attn", "apply_rotary_emb")
    }
    sinkhorn_calls = itertools.count()
    rope_calls = itertools.count()

    def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
        index = next(sinkhorn_calls)
        label = ("hc_attn", "hc_ffn")[index] if index < 2 else f"hc_{index}"
        pre, post, comb = originals["hc_split_sinkhorn"](
            mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps
        )
        recorder.add(f"{prefix}.{label}.pre", pre)
        recorder.add(f"{prefix}.{label}.post", post)
        recorder.add(f"{prefix}.{label}.comb", comb)
        return pre, post, comb

    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, *args, **kwargs):
        key = f"{prefix}.attn.sparse_attn"
        recorder.add(f"{key}.in.q", q)
        recorder.add(f"{key}.in.kv", kv)
        recorder.add(f"{key}.in.topk_idxs", topk_idxs)
        out = originals["sparse_attn"](q, kv, attn_sink, topk_idxs, softmax_scale, *args, **kwargs)
        recorder.add(f"{key}.out", out)
        return out

    def apply_rotary_emb(x, freqs_cis, inverse=False):
        index = next(rope_calls)
        label = "o_inv" if inverse else f"{_ROPE_LABELS}.{index}"
        out = originals["apply_rotary_emb"](x, freqs_cis, inverse)
        recorder.add(f"{prefix}.attn.rope.{label}.out", out)  # in place: read after the call
        return out

    model_module.hc_split_sinkhorn = hc_split_sinkhorn
    model_module.sparse_attn = sparse_attn
    model_module.apply_rotary_emb = apply_rotary_emb
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(model_module, name, original)


def layer_range(spec: str, available: list[int]) -> list[int]:
    """Parse a ``--layers`` specification: "all", "0,2,3" or "0-5"."""
    if spec in ("all", "", None):
        return list(available)
    chosen: list[int] = []
    for piece in spec.split(","):
        piece = piece.strip()
        if "-" in piece:
            start, end = (int(part) for part in piece.split("-", 1))
            chosen.extend(range(start, end + 1))
        elif piece:
            chosen.append(int(piece))
    missing = [index for index in chosen if index not in available]
    if missing:
        raise SystemExit(f"layers not present in the checkpoint: {missing}")
    return chosen


def _run_layers(model, args, ckpt, layers, hidden, input_ids, recorder, log):
    for index in layers:
        loaded, unmatched = load_prefix(model, ckpt, args, f"layers.{index}")
        if unmatched:
            raise RuntimeError(f"layers.{index}: unmatched keys {unmatched[:3]}")
        handles = record_module_tree(recorder, model.layers[index], f"layers.{index}")
        try:
            with record_fused(recorder, f"layers.{index}"):
                hidden = model.layers[index](hidden, 0, input_ids)
        finally:
            for handle in handles:
                handle.remove()
        free(model, f"layers.{index}")
        log(f"  layers.{index}: {loaded} tensors loaded, {recorder.summary()}")
    return hidden


def _run_mtp(model, args, ckpt, hidden, input_ids, recorder, log):
    """MTP prefill plus one decode step, mirroring model.py's own smoke test.

    The two passes share the MTP block's KV cache on purpose: buffers are *not*
    reset in between, because the decode step must continue the state the
    prefill wrote. The decode step runs at ``start_pos = seq_len`` (the natural
    continuation); model.py's self test uses ``start_pos = 1`` there, but that
    reuses an already written cache slot -- it is a shape smoke test.
    """
    loaded, unmatched = load_prefix(model, ckpt, args, "mtp.0")
    if unmatched:
        raise RuntimeError(f"mtp.0: unmatched keys {unmatched[:3]}")
    block = model.mtp[0]
    seq_len = input_ids.size(1)

    for tag, (hidden_in, ids_in, start_pos) in (
        ("prefill", (hidden, input_ids, 0)),
        ("decode", (hidden[:, -1:], input_ids[:, -1:], seq_len)),
    ):
        prefix = f"mtp.0.{tag}"
        handles = record_module_tree(recorder, block, prefix)
        try:
            with record_fused(recorder, prefix):
                logits = block(hidden_in, start_pos, ids_in)
        finally:
            for handle in handles:
                handle.remove()
        recorder.add(f"{prefix}.logits", logits)
        log(f"  {prefix}: start_pos={start_pos}, {recorder.summary()}")

    free(model, "mtp.0")
    log(f"  mtp.0: {loaded} tensors loaded")


def run_reference(
    model: Transformer,
    args,
    ckpt: Checkpoint,
    input_ids: torch.Tensor,
    recorder: Recorder,
    layers: list[int],
    include_mtp: bool = True,
    log=print,
) -> torch.Tensor:
    """One prefill forward pass, streaming one layer at a time, recording as we go."""
    with torch.no_grad():
        recorder.add("input_ids", input_ids)

        fill(model, ckpt, "embed")
        hidden = model.embed(input_ids)
        recorder.add("embed.out", hidden)
        hidden = hidden.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)

        hidden = _run_layers(model, args, ckpt, layers, hidden, input_ids, recorder, log)

        for key in ("hc_head_fn", "hc_head_base", "hc_head_scale", "norm", "head"):
            fill(model, ckpt, key)
        # Transformer.forward is `self.head(h, hc_head_fn, hc_head_scale, hc_head_base, self.norm)`,
        # i.e. the final hyper-connection reduction is ParallelHead.hc_head; the
        # steps are spelled out here so that norm.out can be recorded. `hidden`
        # itself stays 4D (the hc copies) because the MTP block consumes that.
        reduced = model.head.hc_head(
            hidden, model.hc_head_fn, model.hc_head_scale, model.hc_head_base
        )
        normed = model.norm(reduced)
        recorder.add("norm.out", normed)
        # ParallelHead.get_logits projects the last position only -- that is what
        # the official forward computes, so that is what the dump records.
        logits = model.head.get_logits(normed)
        recorder.add("head.out", logits)

        if include_mtp:
            _run_mtp(model, args, ckpt, hidden, input_ids, recorder, log)

        return logits
