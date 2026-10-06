#!/usr/bin/env python3
"""dump.py -- Qwen3.5-0.8B reference inference dump (HF Transformers).

Runs the model with stock HuggingFace Transformers and saves the tensors
needed for verification into a single safetensors file
(qwen3.5-0.8b-vt.safetensors) that model.vt compares against.

Exported per-node activations (batch dim kept, batch == 1):
    input_ids                                        -> [batch, seq_len]
    model.language_model.embed_tokens.out            -> [batch, seq_len, hidden]
    model.language_model.layers.{N}.input_layernorm.out
    model.language_model.layers.{N}.linear_attn.in_proj_qkv.out   (linear layers)
    model.language_model.layers.{N}.linear_attn.in_proj_a.out
    model.language_model.layers.{N}.linear_attn.in_proj_b.out
    model.language_model.layers.{N}.linear_attn.in_proj_z.out
    model.language_model.layers.{N}.linear_attn.out_proj.out
    model.language_model.layers.{N}.self_attn.q_proj.out          (full layers)
    model.language_model.layers.{N}.self_attn.k_proj.out
    model.language_model.layers.{N}.self_attn.v_proj.out
    model.language_model.layers.{N}.self_attn.o_proj.out
    model.language_model.layers.{N}.mlp.gate_proj.out
    model.language_model.layers.{N}.mlp.up_proj.out
    model.language_model.layers.{N}.mlp.down_proj.out
    model.language_model.layers.{N}.post_attention_layernorm.out
    model.language_model.layers.{N}.out
    model.language_model.norm.out
    lm_head.out                                      -> [batch, seq_len, vocab]

The attention implementation is forced to "eager" and the FLA kernels are
not used (fla is not installed), so the reference matches the plain torch
formulas that model.vt re-implements.

NOTE on environments:
    The vt-lang venv (CPU torch) is enough: only one prefill forward pass
    plus a short generate is executed. A CUDA venv also works via --device.

Usage:
    vt-lang/.venv/bin/python models/Qwen3.5-0.8B/dump.py
    vt-lang/.venv/bin/python models/Qwen3.5-0.8B/dump.py \
        --prompt "Hello, introduce yourself." --max-new-tokens 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "/home/teaonly/workspace/Qwen3.5-0.8B"
DEFAULT_OUT = Path(__file__).resolve().parent / "qwen3.5-0.8b-vt.safetensors"

# Central export table: every intermediate tensor is collected here and
# written to the output file in one shot.
_exported: dict[str, torch.Tensor] = {}

# True while the capture forward pass runs: hooks record only then, so the
# generate() loop (many cached decode steps) does not overwrite the dump.
_capturing = False


def _save(out_path: Path) -> None:
    """Write all tensors in _exported to one safetensors file (overwrites).

    Floating-point tensors are cast to float32 for easy comparison on the
    vt side; integer tensors (e.g. input_ids) keep their dtype.
    """
    tensors = {}
    for k, v in _exported.items():
        v = v.detach().to("cpu").contiguous()
        if v.is_floating_point():
            v = v.to(torch.float32)
        tensors[k] = v
    save_file(tensors, str(out_path))
    print(f"saved {len(tensors)} tensors -> {out_path}")


def _hook(name: str):
    """Forward hook that records a submodule output under `name`."""

    def fn(module, args, output):
        if _capturing:
            _exported[name] = output

    return fn


def install_hooks(model) -> None:
    """Register hooks for every node model.vt verifies."""
    lm = model.model  # Qwen3_5TextModel (CausalLM top level is the text model)

    model.get_input_embeddings().register_forward_hook(
        _hook("model.language_model.embed_tokens.out")
    )
    if model.lm_head is not None and model.lm_head.weight is not None:
        model.lm_head.register_forward_hook(_hook("lm_head.out"))
    lm.norm.register_forward_hook(_hook("model.language_model.norm.out"))

    for n, layer in enumerate(lm.layers):
        p = f"model.language_model.layers.{n}"
        layer.input_layernorm.register_forward_hook(_hook(f"{p}.input_layernorm.out"))
        layer.post_attention_layernorm.register_forward_hook(
            _hook(f"{p}.post_attention_layernorm.out")
        )
        layer.register_forward_hook(_hook(f"{p}.out"))
        layer.mlp.gate_proj.register_forward_hook(_hook(f"{p}.mlp.gate_proj.out"))
        layer.mlp.up_proj.register_forward_hook(_hook(f"{p}.mlp.up_proj.out"))
        layer.mlp.down_proj.register_forward_hook(_hook(f"{p}.mlp.down_proj.out"))
        if hasattr(layer, "linear_attn"):
            for sub in ("in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj"):
                getattr(layer.linear_attn, sub).register_forward_hook(
                    _hook(f"{p}.linear_attn.{sub}.out")
                )
        if hasattr(layer, "self_attn"):
            for sub in ("q_proj", "k_proj", "v_proj", "o_proj"):
                getattr(layer.self_attn, sub).register_forward_hook(
                    _hook(f"{p}.self_attn.{sub}.out")
                )


@torch.no_grad()
def capture(model, input_ids: torch.Tensor) -> None:
    """One plain forward pass (no cache) recording every hooked tensor."""
    global _capturing
    _exported["input_ids"] = input_ids.cpu()
    _capturing = True
    try:
        model(input_ids=input_ids, use_cache=False)
    finally:
        _capturing = False


def load(model_path: str, device: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load tokenizer and model (bf16; returned model is in eval mode on the device)."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )
    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate(tokenizer, model, prompt: str, max_new_tokens: int, verbose: bool = True):
    """One generation pass (verification values come from capture(), not here)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    new_ids = output[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    if verbose:
        print(f"prompt> {prompt}")
        print(f"generated ({new_ids.shape[0]} tokens):")
        print(text)
        print(f"new token ids: {new_ids.tolist()}")
    return text, new_ids.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5-0.8B HF reference inference dump")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model directory")
    parser.add_argument("--device", default="auto", help="auto / cuda / cpu")
    parser.add_argument(
        "--prompt", default="你好，简单介绍一下你自己。",
        help="prompt used for inference",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT),
        help="output safetensors file for intermediate tensors",
    )
    parser.add_argument(
        "--no-generate", action="store_true",
        help="skip the generation pass (dump intermediates only)",
    )
    args = parser.parse_args()

    tokenizer, model = load(args.model, args.device)
    print(f"model loaded: {args.model} (device={args.device}, dtype=bfloat16, attn=eager)")
    install_hooks(model)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    capture(model, inputs["input_ids"])
    print(f"captured {len(_exported)} tensors from one prefill forward")

    if not args.no_generate:
        generate(tokenizer, model, args.prompt, args.max_new_tokens)

    _save(Path(args.out))


if __name__ == "__main__":
    main()
