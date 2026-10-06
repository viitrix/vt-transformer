# VT code for Qwen3.8-27B

Qwen3.8-27B (multimodal Qwen3_5ForConditionalGeneration checkpoint,
bf16 weights, ~55 GiB). Only the text stack is dumped and verified;
the vision tower and the MTP layer are not covered.

HuggingFace weights: `https://huggingface.co/Qwen/Qwen3.8-27B`
VT-Transfomer verified tensors: `https://huggingface.co/vt-transformer/Qwen3.8-27B-vt`


## Layout

`model.vt` describes the full text stack as verification nodes:

- `embedding` -- token embedding (untied from lm_head)
- `rms_norm` / `final_norm` -- zero-centered RMSNorm (per-layer via the
  `layer`/`module` templates, plus the final norm)
- `linear_attn` -- Qwen3.5 gated delta net with GVA (48 value heads vs
  16 key heads, q/k repeat-interleaved 3x inside the kernel):
  in_proj_qkv/a/b/z, decay g = -exp(A_log) * softplus(a + dt_bias),
  depthwise causal conv1d + silu, the chunked gated delta rule
  (plain-torch reference kernel), gated RMSNorm, out_proj
- `self_attn` -- gated GQA attention (every 4th layer): q/k/v/o projections,
  per-head q/k RMSNorm, partial RoPE (rotary dim 64 of head dim 256,
  theta 1e7), causal softmax, sigmoid output gate
- `mlp` -- SwiGLU (gate/up/down, intermediate 17408)
- `decoder_layer` -- residual wiring of one layer
- `lm_head` -- logits (separate weight, vocab 248320)
- `llm` -- the whole stack

## Environment

The vt-lang venv needs a CUDA torch to dump/verify on the GPU (the model
is 27B dense; 64 GiB VRAM is enough, CPU needs ~60 GiB RAM):

    cd vt-lang
    uv pip install --python .venv/bin/python 'torch==2.14.1+cu130' \
        --index-url https://download.pytorch.org/whl/cu130

Dump and verification MUST run on the same device: bf16 matmuls only
match the reference bitwise on the device they were computed on.

## Usage

Regenerate the reference dump (one GPU prefill forward, eager attention):

    HF_HUB_OFFLINE=1 vt-lang/.venv/bin/python models/Qwen3.8-27B/dump.py --device cuda

Verify every node against it (same device):

    vt-lang/.venv/bin/vt models/Qwen3.8-27B/model.vt llm input_ids
