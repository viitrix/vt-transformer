# `model.vt` for DeepSeek-V4-Flash — how it is sliced

How the model is cut into `@vt.NodeChecker` nodes, which node verifies which key
of `deepseek-v4-flash-vt.safetensors`, and what has to live in a helper. This is
the design note behind [model.vt](model.vt); read it together with
[README.md](README.md) (what the dump contains) and `kv4/model.py` (the
reference formulas).

Status: the six-token CPU reference was regenerated after the attention
precision correction. The full model passes 1878 nested node checks using the
unchanged vt-lang runtime. This description requires at least four prefill
tokens and does not claim equivalence to official CUDA execution. See
[Results](#results).

## Ground rules

1. **The dump is a superset.** A node verifies only the keys it declares in
   `outputs`; recorded keys nobody declares are simply not verified. So the
   slicing follows "what reads well", it does not have to cover every key.
2. **Verification is per node.** `vt model.vt <node> <activation keys...>
   name=value...` loads activation keys as positional arguments, so every node
   should be able to take its inputs from the dump directly and run on its own.
3. **Two prefixes, not one.** `prefix` is the **activation** prefix
   (`layers.7.`, `mtp.0.prefill.`) and `wprefix` is the **weight** prefix
   (`layers.7.`, `mtp.0.`). They coincide for the text stack but not for MTP:
   the checkpoint stores `mtp.0.attn.wkv.weight` while the dump records
   `mtp.0.prefill.attn.wkv.out`. One `prefix` for both was the single largest
   source of confusion while writing the file.
4. **Prefill for the text stack, both passes for MTP.** The dump was produced
   with `start_pos = 0`; the decode branches of the main-stack `Attention` /
   `Compressor` / `Indexer` (their KV-cache state machines) are out of scope.
   The MTP decode step *is* covered: its window cache and index table are
   recorded keys, so `mtp_decode` receives them as arguments (see below).

## The constraint that shapes everything: quantisation

V4 quantises **activations** as well as weights, so a plain `torch.linear` is
not the right primitive: `kernels.act_quant` snaps the activations to the E4M3
grid with a power-of-two scale per 128-element block (up to ~6% relative error),
and the weights are FP8 (128x128 blocks) or FP4 (32-element blocks) with E8M0
scales. Reproducing the dump therefore requires, for every linear:

```python
x = act_quant(x, block=128)                    # quantise-dequantise, fp32
w = dequant_weight(w_q, w_s, fp4=...)          # fp32
h = torch.linear(x, w).to(torch.bfloat16)      # fp32 accumulate, bf16 out
```

The allowlist cannot express this (`log2`, `ceil`, `torch.float8_e4m3fn`,
`torch.float4_e2m1fn_x2` are all outside it), so `act_quant`, `fp4_act_quant`
and `dequant_weight` are **helpers**, exactly like `topk`. What stays in the
node body is the sequence of operations and the dtype rounding, which is what
the static checker and the verification can still see.

Note the asymmetry the reference has: the **gemm activation side stays fp32**
(the quantise-dequantise result, not a bf16 rounding of it), while the
**in-place** quantisations (`kv[..., :-rope]`, the compressor's pooled block,
the indexer's `q`) do round back to bf16. `act_quant` returns fp32 and
`act_quant_bf16` wraps it for the in-place call sites.

## Helpers (unchecked, at the top of the file)

| helper | why |
| --- | --- |
| `act_quant` / `act_quant_bf16`, `fp4_act_quant` / `fp4_act_quant_bf16` | activation quantise-dequantise (fp32 for gemms, bf16 for in-place sites) |
| `dequant_weight(w, s, fp4=False)`, `e8m0_to_float` | FP8 / FP4 / E8M0 weight dequantisation |
| `qlinear(x, w, s, fp4=False)`, `rms_norm_formula` | the three steps combined, and the shared RMSNorm |
| `topk = torch.topk` | the indexer's selection; not in the allowlist |
| `scatter_add(dst, i, v)` | the MoE dispatch; not in the allowlist |
| `softplus = torch.nn.functional.softplus` | routing score. Do **not** rewrite it as `log(1+exp(x))` (see below) |
| `rope_tables`, `apply_rope`, `rope_tail`, `layer_rope_tables` | trig and complex arithmetic are outside the allowlist (Qwen precedent) |
| `hadamard`, `rotate_activation` | replaces the CUDA `fast_hadamard_transform` |
| `window_index`, `compress_index`, `expert_batches`, `overlap_transform`, `layer_prefix` | index-table construction, expert grouping, string building |
| `grouped_projection`, `index_scores`, `attention_scores`, `attention_values` | the reference's four `einsum`s, kept verbatim (see below) |

Inside node bodies (checked): the Sinkhorn iterations, `relu` as `maximum(x, 0)`,
`unflatten` as `reshape`, all softmaxes, the MoE loop, the residual wiring.

## Keys: which node declares what

Keys below are relative to `prefix`; weights are bound from `wprefix`.

| node | `outputs` |
| --- | --- |
| `embedding(ids)` | `embed.out` (top level only) |
| `rms_norm(x, prefix, wprefix, module)` | `{module}.out` |
| `hc_split(x, prefix, wprefix, which)` | `hc_{which}.pre`, `.post`, `.comb`, `None` |
| `hc_merge(y, pre, post, comb, residual)` | — (none) |
| `attn_q(x, prefix, wprefix, layer, start_pos=0)` | `attn.wq_a.out`, `attn.q_norm.out`, `attn.wq_b.out`, `attn.rope.rot.0.out`, `attn.sparse_attn.in.q` |
| `attn_kv(x, prefix, wprefix, layer, start_pos=0)` | `attn.wkv.out`, `attn.kv_norm.out`, `attn.rope.rot.1.out`, `None` |
| `compressor_split(x, prefix, path)` | `{path}.wkv.out`, `{path}.wgate.out` |
| `compressor_pool(wkv, wgate, prefix, path, rot_out, ratio, rotate)` | `{path}.out`, `{path}.norm.out`, `{rot_out}` |
| `indexer(x, qr, prefix, offset)` | `attn.indexer.wq_b.out`, `.weights_proj.out`, `attn.indexer.out` |
| `sparse_attention(q, kv, idxs, prefix, wprefix)` | `attn.sparse_attn.out` |
| `attn_out(x, prefix, wprefix, layer, start_pos=0)` | `attn.rope.o_inv.out`, `attn.wo_b.in`, `attn.wo_b.out`, `attn.out` (the last two are the same tensor) |
| `attention(h, prefix, wprefix, layer)` | `attn.sparse_attn.in.kv`, `attn.sparse_attn.in.topk_idxs` |
| `router_hash(x, ids, prefix, wprefix)` / `router_score(x, prefix, wprefix)` | `ffn.gate.out.0`, `ffn.gate.out.1` |
| `expert(x, weight, prefix, wprefix, expert)` | `ffn.experts.{expert}.w1/w3/w2.out`, `ffn.experts.{expert}.out` |
| `shared_expert(x, prefix, wprefix)` | `ffn.shared_experts.{w1,w3,w2,out}.out` |
| `moe(x, ids, prefix, wprefix, layer)` | `ffn.out` |
| `decoder_layer(x, ids, prefix, wprefix, layer)` | `{prefix}out` |
| `final_norm(x)`, `lm_head(x)`, `final_head(x)` | `norm.out`, `head.out` (the last also does the final hyper-connection reduction, which has no key) |
| `mtp_fuse(hidden, ids, prefix)` | `{prefix}embed.out`, `.enorm.out`, `.hnorm.out`, `.e_proj.out`, `.h_proj.out`, `None` |
| `mtp_head(x, prefix)` | `{prefix}norm.out`, `.head.out`, `.out`, `.logits` |
| `attention_decode(cache_kv, cache_idxs, h, prefix, wprefix, layer, start_pos)` | — (takes the recorded cache and index table as arguments) |
| `mtp_run(hidden, ids, prefix, layer)`, `mtp_decode(last_hidden, ids, cache_kv, cache_idxs, prefix, start_pos)`, `llm(ids)` | — (pure orchestration) |

`head.out` is `[1, 129280]`: `ParallelHead.get_logits` projects the **last
position only**, which is what the dump recorded.

## How it was built and verified

Written and verified leaf-first, so that a mistake shows up in one small node
instead of at the end of a 70-second run:

| step | nodes | verified standalone with |
| --- | --- | --- |
| 1 | `embedding`, `rms_norm`, `final_norm`, `lm_head` | `input_ids`, `layers.0.attn.wq_a.out` |
| 2 | `hc_split`, `hc_merge` | a layer input (nested only) |
| 3 | `attn_q`, `attn_kv`, `attn_out` | `layers.N.attn_norm.out`, `...sparse_attn.out` |
| 4 | `sparse_attention` | the three `layers.N.attn.sparse_attn.in.*` |
| 5 | `compressor_split`, `compressor_pool`, `indexer` | `attn_norm.out`, `q_norm.out` |
| 6 | `router_hash`, `router_score`, `expert`, `shared_expert`, `moe` | `ffn_norm.out` |
| 7 | `attention`, `decoder_layer` | `layers.N-1.out` |
| 8 | `mtp_fuse`, `mtp_head`, `mtp_run` | nested from `llm` |
| 8b | `attention_decode`, `mtp_decode` | `layers.42.out`, `input_ids`, the recorded decode cache and index table |
| 9 | `llm` | `input_ids` |

```console
$ V=vt-lang/.venv/bin/vt; F=models/DeepSeek-V4-Flash/model.vt
$ $V $F attn_q layers.0.attn_norm.out prefix=layers.0. wprefix=layers.0. layer=0
$ $V $F sparse_attention layers.0.attn.sparse_attn.in.q \
      layers.0.attn.sparse_attn.in.kv layers.0.attn.sparse_attn.in.topk_idxs prefix=layers.0.
$ $V $F llm input_ids
```

## Results: retained six-token baseline

| run | result |
| --- | --- |
| `llm` | **1878 node verifications, 0 errors** |
| `mtp_decode` (decode step) | 23 node verifications, 0 errors; 61/63 keys compared, 2 cache inputs supplied |

All 43 main-stack layer outputs pass the original vt-lang comparison against
the regenerated CPU dump (atol=rtol=1e-4). The indexer reports an explicit error
for inputs shorter than four tokens, so no empty outputs are passed to the
original checker. The historical implementation notes below explain earlier
numerical and indexing issues.

## Bit-level traps (found the hard way)

The per-node tolerance is 1e-4, so small differences hide inside a passing node
and only surface later, when a quantisation boundary flips. The full-model run
first failed at layer 27 on one expert's `w2.out` with `max|diff| = 1.221e-04`;
the fixes below took the 43-layer drift to exactly zero.

1. **Keep the reference's contractions verbatim.** `F.softplus` uses `log1p`, so
   rewriting it as `log(1+exp(x))` differs in the last bits. Likewise the four
   `einsum`s (`einsum("bsgd,grd->bsgr")`, `("bshd,btd->bsht")`,
   `("bmhd,bmtd->bmht")`, `("bmht,bmtd->bmhd")`) must stay `einsum`: torch's
   bf16 CPU kernel accumulates in its own order, and an fp32 `matmul` rewrite
   changes the result by one bf16 ulp.
2. **`torch.gather` numbers the rows of the index tensor, not your tokens.**
   `weights[idx, top]` (advanced indexing) cannot become
   `gather(weights, 1, slots)` — with a single-row index that silently reads
   row 0. The allowed form is
   `gather(index_select(weights, 0, tokens), 1, slots)`.
3. **Nodes return tuples.** `attention` returns `(kv, idxs, out)`; assigning it
   to one name and using that as a tensor fails at the *next* call, not at the
   return.
4. **A node may not bind a weight name an enclosing node already bound** —
   `compressor_pool` had to rename its `norm_w` to `comp_norm_w` because its
   nested `rms_norm` binds `norm_w` too.
5. **The CLI parses `name=value` as int/float only.** `rotate=False` arrives as
   the string `"False"`; the flag is an int (`rotate=0` / `rotate=1`).
6. **`torch.linear` / `torch.silu` / `torch.gelu` do not exist**; alias
   `linear = torch.nn.functional.linear` at the top (the allowlist names are
   resolved through `std_operators.TORCH_ALIASES`). `torch.embedding` *does*
   exist and takes `(weight, indices)`.
7. **The MTP block's `out` is the logits, not a hidden state** — its forward
   ends with the head, so `decoder_layer` must not be reused with the MTP
   prefix (`mtp_run` does the block composition itself and declares nothing).

## Known wrinkles

* **`rot_index` differs per layer type**: the compressor's own rotation is
  `rope.rot.2` on ratio-128 layers, `rope.rot.4` on ratio-4 layers and
  `rope.rot.3` for the indexer's compressor, so the caller passes it as the
  `rot_out` fragment. On ratio-128 layers at 6 tokens it is simply absent.
* **The compressor is split in two nodes** because the reference runs
  `wkv`/`wgate` before deciding whether to compress: `compressor_split` always
  runs on a compressed layer (its keys exist even when nothing is pooled), while
  `compressor_pool` is only called when `seqlen >= ratio`.
* **Router needs two nodes**: layers 0-2 route through `ffn.gate.tid2eid` and
  have no `bias`; the other 40 score with `ffn.gate.weight` + `bias`. The MTP
  layer (index 43) is a scoring layer.
* **`expert` cannot be run standalone** (its input is the per-expert subset of
  the tokens, which the dump does not record); it is verified through the nested
  call from `moe` — once per activated expert, 1045 times for the six-token
  baseline. Different prompts activate different expert unions.
* **The compressors are prefill-only**; `Compressor.forward`'s `start_pos > 0`
  state machine is out of scope.
* ratio-128 layers produce no compressed block at 6 tokens, so on those layers
  `compressor.out` / `.norm.out` / `rope.rot.2` are absent from the dump — the
  node must branch on `seqlen >= ratio` exactly like the reference does.

## Multiple fixed inputs, one model description

`prompts.json` fixes the three inputs of lengths 6, 32 and 129, including text,
token IDs, separate reference filenames and the MTP decode position. All use
the same `model.vt`, checkpoint, CPU stack and four threads. The original
six-token reference is retained; the additional files are
`deepseek-v4-flash-vt-prompt32.safetensors` and
`deepseek-v4-flash-vt-prompt129.safetensors`.

The shared functions obtain sequence length from tensor shapes. At 32 tokens,
ratio-4 pooling handles eight complete blocks. At 129 tokens it handles 32
complete blocks and leaves one trailing token; ratio-128 pooling now executes
in all 20 corresponding layers, and the sliding-window index table excludes
position zero from the final row. `check_dump.py` derives required output keys
from the sequence length and requires complete outputs for the actually routed
experts. There are no separate model descriptions for these lengths.

The same node bindings are compared against each case's own reference by
changing `VT_ACTIVATION_FILE`. MTP decode uses `start_pos=6`, `32` or `129`,
with that case's recorded cache and index table. This validates derived
outputs, not cache construction/update. None of the cases exercises top-512
truncation from a larger candidate set. See the README and manifest for the
per-case sizes and verification evidence.

| input tokens | full `llm` checks | MTP decode checks | result |
| --- | --- | --- | --- |
| 6 | 1878 | 23 | previous verified baseline retained |
| 32 | 3496 | 23 | passed |
| 129 | 5922 | 23 | passed |

The added cases pass without changing `model.vt`, the reference computation,
vt-lang or the tolerance. Routed expert-node invocations are 1045, 2663 and
5049 respectively; the 129-token input also invokes ratio-128 pooling and
its nested norm in all 20 corresponding layers. These are invocation counts
within three fixed inputs, not counts of separate test inputs.
