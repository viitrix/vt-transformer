#!/usr/bin/env python3
"""dump.py -- Qwen3.5-0.8B reference inference dump (HF Transformers).

Runs the model with stock HuggingFace Transformers and saves the tensors
needed for verification (currently: input_ids and the embed_tokens output;
later: per-layer intermediate results) into a single safetensors file
(qwen3.5-0.8b-vt.safetensors) that model.vt compares against.

All exported tensors keep the batch dimension, even when batch == 1:
    input_ids                         -> [batch, seq_len]
    model.language_model.embed_tokens.out -> [batch, seq_len, hidden]

NOTE on environments:
    Prefer running this script in a CUDA-enabled venv (e.g. the vt-reference
    venv) -- generation on CPU is slow for this model. The vt-lang venv ships
    a CPU-only torch build; the script still runs there via --device cpu /
    --device auto, just slower.

Usage:
    <cuda-venv>/bin/python models/Qwen3.5-0.8B/dump.py
    <cuda-venv>/bin/python models/Qwen3.5-0.8B/dump.py \
        --prompt "Hello, introduce yourself." --max-new-tokens 64
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
    print(f"saved {len(tensors)} tensors -> {out_path}: {sorted(tensors)}")


def load(model_path: str, device: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load tokenizer and model (bf16; returned model is in eval mode on the device)."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate(tokenizer, model, prompt: str, max_new_tokens: int, verbose: bool = True):
    """One generation pass; returns (full text, new token id list)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Keep the batch dimension: [batch, seq_len]
    _exported["input_ids"] = inputs["input_ids"]
    # Dump the embedding output, also with batch kept: [batch, seq_len, hidden]
    embed_tokens = model.get_input_embeddings()
    _exported["model.language_model.embed_tokens.out"] = embed_tokens(inputs["input_ids"])
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
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT),
        help="output safetensors file for intermediate tensors",
    )
    args = parser.parse_args()

    tokenizer, model = load(args.model, args.device)
    print(f"model loaded: {args.model} (device={args.device}, dtype=bfloat16)")
    generate(tokenizer, model, args.prompt, args.max_new_tokens)
    _save(Path(args.out))


if __name__ == "__main__":
    main()
