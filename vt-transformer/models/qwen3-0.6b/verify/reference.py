"""
Eager-mode reference forward for Qwen3-0.6B.

把 /home/teaonly/workspace/qwen3-0.6b 用 FP16 + eager attention 跑一遍 forward，
对每一层的 Linear / LayerNorm 输出挂 forward hook，落盘成 .npy，
用于和 vt-transformer 那边的 C++/CUDA 实现逐位对比。

用法（在 verify 目录下）：
    uv run reference.py
    uv run reference.py --prompt "Hello world" --out-dir ./dumps
    uv run reference.py --dtype bf16 --device cpu

落盘约定：
    <out_dir>/input_ids.npy                                   # int64, [seq_len]
    <out_dir>/model.embed_tokens.npy                          # [seq_len, 1024]
    <out_dir>/model.layers.{i}.input_layernorm.npy            # [seq_len, 1024]
    <out_dir>/model.layers.{i}.self_attn.q_proj.npy           # [seq_len, 2048]
    <out_dir>/model.layers.{i}.self_attn.k_proj.npy           # [seq_len, 1024]
    <out_dir>/model.layers.{i}.self_attn.v_proj.npy           # [seq_len, 1024]
    <out_dir>/model.layers.{i}.self_attn.o_proj.npy           # [seq_len, 1024]
    <out_dir>/model.layers.{i}.post_attention_layernorm.npy   # [seq_len, 1024]
    <out_dir>/model.layers.{i}.mlp.gate_proj.npy              # [seq_len, 3072]
    <out_dir>/model.layers.{i}.mlp.up_proj.npy                # [seq_len, 3072]
    <out_dir>/model.layers.{i}.mlp.down_proj.npy              # [seq_len, 1024]
    <out_dir>/model.layers.{i}.npy                            # block 输出 [seq_len, 1024]
    <out_dir>/model.norm.npy                                  # [seq_len, 1024]
    <out_dir>/lm_head.npy                                     # [seq_len, 151936]

所有 .npy 统一存为 float32 + cpu，方便 numpy 端做 abs/rel 误差对比。
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "/home/teaonly/workspace/qwen3-0.6b"
DEFAULT_PROMPT = "Hello"

DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

# 每层要挂 hook 的子模块相对路径；和 C++ 端 Weight 命名一一对应。
PER_LAYER_HOOKS = [
    "input_layernorm",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "post_attention_layernorm",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def resolve_submodule(layer, dotted: str):
    """从 layer 取出 'self_attn.q_proj' 这种点分路径对应的子模块。"""
    mod = layer
    for part in dotted.split("."):
        mod = getattr(mod, part)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF model dir (default: %(default)s)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPE_MAP))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dtype = DTYPE_MAP[args.dtype]

    print(f"[ref] model   : {args.model}")
    print(f"[ref] dtype   : {args.dtype}  device: {args.device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="eager",   # 关键：避免 sdpa / flash 融合隐藏中间值
    ).to(args.device).eval()

    # ---- 与 sglfront shell 模式对齐（sglfront/tokenizer/tokenize.py 的处理路径）----
    #
    #   messages = [{"role":"user", "content": text}]
    #   prompt   = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    #   ids      = tokenizer.encode(prompt)
    #
    # 关键点 1：shell 模式构造的 messages **不含 system**（见 sglfront/server/api_server.py
    #   的 shell() —— 第一轮就是 history + [Message(role="user", content=cmd)]）。
    #
    # 关键点 2：Qwen3 的 chat template 在 messages[0].role != "system" 时**不**注入默认
    #   system prompt（区别于 Qwen2 的 "You are a helpful assistant."）。所以 shell 模式
    #   第一轮的实际 token 前缀就是：
    #
    #       <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
    #
    #   verify 端必须复刻这条路径，否则第一层 embed 出来就和后端跑的对不上。
    messages = [{"role": "user", "content": args.prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(prompt_text, return_tensors="pt")
    ids = enc.input_ids.to(args.device)
    print(f"[ref] prompt  : {args.prompt!r}")
    print(f"[ref] chat    : {prompt_text!r}")
    print(f"[ref] ids     : shape={tuple(ids.shape)} value={ids.tolist()}")

    # ---- 注册 hook ----
    captures = {}   # name -> torch.Tensor（保留原 device/dtype，落盘时再转）

    def mk(name):
        def hook(_module, _inp, out):
            if isinstance(out, tuple):
                out = out[0]
            captures[name] = out.detach()
        return hook

    handles = [
        model.model.embed_tokens.register_forward_hook(mk("model.embed_tokens")),
        model.model.norm.register_forward_hook(mk("model.norm")),
        model.lm_head.register_forward_hook(mk("lm_head")),
    ]
    for i, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{i}."
        # 整个 block 的输出（残差流）也抓一份
        handles.append(layer.register_forward_hook(mk(f"model.layers.{i}")))
        for sub in PER_LAYER_HOOKS:
            mod = resolve_submodule(layer, sub)
            handles.append(mod.register_forward_hook(mk(prefix + sub)))

    # ---- forward ----
    with torch.no_grad():
        out = model(ids)

    # ---- 落盘：统一 fp32 + cpu + numpy ----
    for name, t in captures.items():
        print(f"====== {name}")
        print(t)

    # ---- sanity 输出 ----
    logits = out.logits[0, -1]
    next_id = int(torch.argmax(logits).item())
    topk = torch.topk(logits, k=5)
    print(f"[ref] logits.shape : {tuple(out.logits.shape)}")
    print(f"[ref] next token   : {next_id}  ({tokenizer.decode([next_id])!r})")
    print(f"[ref] top-5        : "
          + ", ".join(f"{int(i)}:{float(v):.3f}"
                      for v, i in zip(topk.values.tolist(), topk.indices.tolist())))
    print(f"[ref] dumped {len(captures) + 1} tensors.")

    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
