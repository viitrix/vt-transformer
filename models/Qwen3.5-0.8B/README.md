VT code for Qwen3.5-0.8B

HuggingFace model:
https://huggingface.co/Qwen/Qwen3.5-0.8B

vt-transformer and verify model:
https://github.com/viitrix/vt-transformer
https://huggingface.co/vt-transformer/Qwen3.5-0.8B-vt

## Layout

`model.vt` describes the full text stack as verification nodes:

- `embedding` -- token embedding
- `rms_norm` / `final_norm` -- zero-centered RMSNorm (per-layer via the
  `layer`/`module` templates, plus the final norm)
- `linear_attn` -- Qwen3.5 gated delta net (per linear-attention layer):
  in_proj_qkv/a/b/z, depthwise causal conv1d + silu, the chunked gated
  delta rule (fused kernel, mirrored from the reference implementation),
  gated RMSNorm, out_proj
- `self_attn` -- gated GQA attention (every 4th layer): q/k/v/o projections,
  per-head q/k RMSNorm, partial RoPE (rotary dim 64 of head dim 256,
  theta 1e7), causal softmax, sigmoid output gate
- `mlp` -- SwiGLU (gate/up/down)
- `decoder_layer` -- residual wiring of one layer
- `lm_head` -- logits (tied to the embedding weight)
- `llm` -- the whole stack

## Usage

Regenerate the reference dump (one CPU prefill forward, eager attention):

    vt-lang/.venv/bin/python models/Qwen3.5-0.8B/dump.py --device cpu

Verify every node against it:

    vt-lang/.venv/bin/vt models/Qwen3.5-0.8B/model.vt llm input_ids
