"""Model shell construction, checkpoint filling and buffer resetting.

The whole point of this module is memory: the shell is built on the ``meta``
device (shapes without storage), so a 284B-parameter Transformer costs
nothing until a single layer is filled from its shard. After that layer has
run, ``free_layer`` puts its storage back to ``meta`` and the memory is gone.

Filling works by name, with no mapping table: the checkpoint's key namespace
(``layers.0.attn.wq_a.weight``) is exactly the module tree of kv4/model.py, so
``model.get_parameter(key)`` resolves every tensor.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .model import ModelArgs, Transformer, precompute_freqs_cis
from .weights import Checkpoint, dequantize_fp8

# Dtypes that mean "stored quantised" rather than "stored as the model uses it".
_PACKED_FP4 = tuple(
    dtype for dtype in (torch.int8, getattr(torch, "float4_e2m1fn_x2", None)) if dtype is not None
)


def build_shell(config_path: str | Path) -> tuple[ModelArgs, Transformer]:
    """Build the model on the meta device (no storage allocated).

    The torch default dtype is bfloat16 while building, exactly as
    inference/generate.py does it: that is what decides the dtype of the
    parameters that Linear does not create explicitly (embedding, router).
    """
    args = ModelArgs(**json.loads(Path(config_path).read_text()))
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            model = Transformer(args)
    finally:
        torch.set_default_dtype(previous)
    return args, model


def prepare_tensor(key: str, ckpt: Checkpoint, want_dtype: torch.dtype) -> torch.Tensor:
    """Turn a stored checkpoint tensor into what the parameter expects.

    Three cases matter (everything else is a plain dtype cast):

    * routed experts are stored as packed FP4 (dtype ``I8``); model.py declares
      them as ``torch.float4_e2m1fn_x2``, so the bytes are reinterpreted, not
      converted;
    * ``attn.wo_a`` is FP8 in the checkpoint but declared bfloat16 in model.py
      (see the NOTE in Attention.forward), so it is dequantised -- exactly what
      inference/convert.py does for this one tensor;
    * norms / lm_head / compressor weights are stored bf16 but used in fp32.
    """
    tensor = ckpt.get(key)
    if tensor.dtype in _PACKED_FP4 and ckpt.has_scale(key):
        return tensor.view(torch.float4_e2m1fn_x2)
    if tensor.dtype == torch.float8_e4m3fn and want_dtype == torch.bfloat16:
        return dequantize_fp8(tensor, ckpt.get(key[: -len("weight")] + "scale")).to(torch.bfloat16)
    if tensor.dtype != want_dtype:
        return tensor.to(want_dtype)
    return tensor


def _replace_parameter(model: torch.nn.Module, key: str, tensor: torch.Tensor) -> None:
    """Replace the nn.Parameter at dotted path ``key`` with ``tensor``.

    Assigning ``parameter.data = tensor`` refuses a dtype change ("variable and
    tensor have incompatible tensor type"), and the checkpoint legitimately
    changes dtype: packed FP4 arrives as int8, norms arrive as bf16 for fp32
    parameters, tid2eid arrives as int64 for int32. Replacing the Parameter
    object keeps the module tree intact and skips that check.
    """
    parts = key.split(".")
    holder = model.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else model
    name = parts[-1]
    existing = holder._parameters.get(name)
    requires_grad = existing.requires_grad if existing is not None else False
    holder.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=requires_grad))


def is_consumed(key: str) -> bool:
    """Checkpoint tensors that intentionally have no parameter of their own.

    ``attn.wo_a`` is the one tensor inference/convert.py dequantises -- model.py
    declares it bfloat16 (see the NOTE in Attention.forward) -- so its E8M0
    scale is folded into the weight by ``prepare_tensor`` and never appears in
    the module tree. 44 such scales exist (43 layers + the MTP block).
    """
    return key.endswith(".wo_a.scale")


def _relink_linear_scales(module: torch.nn.Module) -> int:
    """Re-establish ``Linear.weight.scale`` after parameters have been replaced.

    model.py attaches the quantisation scale to the weight Parameter itself
    (``self.weight.scale = self.scale = nn.Parameter(...)``) *in addition to*
    registering it as a module parameter, and ``linear()`` reads
    ``weight.scale``. Replacing the Parameter objects breaks that alias, so it
    is restored here. ``weight.scale`` is the only such implicit attribute in
    the port (model.py reads it on lines 143/146 and sets it on 165/170).
    """
    relinked = 0
    for submodule in module.modules():
        parameters = getattr(submodule, "_parameters", None)
        if not parameters:
            continue
        weight, scale = parameters.get("weight"), parameters.get("scale")
        if weight is not None and scale is not None and getattr(weight, "scale", None) is not scale:
            weight.scale = scale
            relinked += 1
    return relinked


def fill(model: torch.nn.Module, ckpt: Checkpoint, prefix: str) -> tuple[int, list[str]]:
    """Copy every ``prefix.*`` checkpoint tensor into the matching parameter.

    Returns (number of tensors filled, keys that matched no parameter).
    """
    filled, unmatched = 0, []
    for key in ckpt.keys_with_prefix(prefix):
        if is_consumed(key):
            continue
        try:
            parameter = model.get_parameter(key)
        except AttributeError:
            parameter = None
        if parameter is None:
            unmatched.append(key)
            continue
        _replace_parameter(model, key, prepare_tensor(key, ckpt, parameter.dtype))
        filled += 1
    # Re-link Linear.weight.scale, but only when the prefix actually names a
    # module: bare top-level parameters (hc_head_fn/base/scale) do not.
    try:
        holder = model.get_submodule(prefix)
    except AttributeError:
        holder = None
    if holder is not None:
        _relink_linear_scales(holder)
    return filled, unmatched


def free(model: torch.nn.Module, prefix: str) -> int:
    """Return every ``prefix.*`` parameter to the meta device, freeing memory."""
    freed = 0
    for key in _parameter_keys(model, prefix):
        parameter = model.get_parameter(key)
        if parameter is None or parameter.device.type == "meta":
            continue
        _replace_parameter(
            model, key, torch.empty(parameter.shape, dtype=parameter.dtype, device="meta")
        )
        freed += 1
    return freed


def _parameter_keys(model: torch.nn.Module, prefix: str) -> list[str]:
    head = prefix + "."
    return [n for n, _ in model.named_parameters() if n == prefix or n.startswith(head)]


def reset_buffers(model: torch.nn.Module, args: ModelArgs, prefix: str | None = None) -> None:
    """Re-create the buffers that model.py computes at construction time.

    The meta build leaves uninitialised storage behind for every buffer, and
    three of them carry meaning that a forward pass depends on:

    * ``attn.kv_cache`` / ``indexer.kv_cache`` -- zeros;
    * ``compressor.kv_state`` -- zeros, ``compressor.score_state`` -- -inf
      ("nothing compressed yet");
    * ``attn.freqs_cis`` -- the YaRN rotary tables, which depend on whether the
      layer compresses KV or is a plain sliding-window layer (model.py
      Attention.__init__).

    ``compressor.kv_cache`` / ``indexer.compressor.kv_cache`` and the
    ``freqs_cis`` attributes of the compressors are plain attributes set to
    None in model.py and filled in lazily during the first forward pass.
    """
    layers = [(name, module) for name, module in model.named_modules()]
    for name, module in layers:
        if prefix is not None and not (name == prefix or name.startswith(prefix + ".")):
            continue
        if name.endswith(".compressor") or name.endswith(".indexer.compressor"):
            if hasattr(module, "kv_state"):
                module.kv_state = _fresh(module.kv_state, 0.0)
            if hasattr(module, "score_state"):
                module.score_state = _fresh(module.score_state, float("-inf"))
            continue
        attention = getattr(module, "attn", None)
        if attention is None or not hasattr(attention, "freqs_cis"):
            continue
        attention.kv_cache = _fresh(attention.kv_cache, 0.0)
        indexer = getattr(attention, "indexer", None)
        if indexer is not None:
            indexer.kv_cache = _fresh(indexer.kv_cache, 0.0)
        attention.freqs_cis = _rotary_tables(attention, args)


def _fresh(meta_tensor: torch.Tensor, fill: float) -> torch.Tensor:
    """A real CPU tensor shaped and typed like its meta counterpart."""
    return torch.empty(meta_tensor.shape, dtype=meta_tensor.dtype, device="cpu").fill_(fill)


def _rotary_tables(attention: torch.nn.Module, args: ModelArgs) -> torch.Tensor:
    """Recompute Attention.freqs_cis (model.py, Attention.__init__ lines 475-482)."""
    if attention.compress_ratio:
        original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
    else:
        original_seq_len, rope_theta = 0, args.rope_theta
    precompute_freqs_cis.cache_clear()          # the meta build cached meta tensors
    with torch.device("cpu"):
        return precompute_freqs_cis(
            attention.rope_head_dim,
            args.max_seq_len,
            original_seq_len,
            rope_theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        ).clone()


def load_prefix(model: torch.nn.Module, ckpt: Checkpoint, args: ModelArgs, prefix: str) -> tuple[int, list[str]]:
    """Fill one layer (or the head/MTP block) and reset its buffers."""
    filled, unmatched = fill(model, ckpt, prefix)
    reset_buffers(model, args, prefix)
    return filled, unmatched
