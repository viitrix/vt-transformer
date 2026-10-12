# VT code for DeepSeek-V4-Flash

DeepSeek-V4-Flash is a 284B-parameter MoE (13B activated per token, 43 decoder
layers plus one MTP layer) released in FP4 + FP8 mixed precision as 46
safetensors shards, ~149 GiB in total.

This directory holds **three reference-activation dumps** for it: one prefill
forward pass over the whole text stack plus the MTP block, every node boundary
recorded for each fixed input. All three use the same **single `model.vt`**
and the same checkpoint. Select the reference with `VT_ACTIVATION_FILE`;
vt-lang compares the nodes at `atol=rtol=1e-4`, the same contract the Qwen
models in this repository use. The original six-token file remains the default.

`model.vt` describes the main-stack prefill and the existing MTP decode-cache
case. The attached six-token CPU reference was regenerated after correcting
sparse-attention denominator precision. The full six-token run passes 1878 nested
node checks. This is CPU reference verification, not a claim of end-to-end
equivalence to the official CUDA implementation.

Unlike the Qwen models, this one **cannot** be dumped with `transformers`, for
three independent reasons:

1. the released checkpoint is laid out for DeepSeek's own `inference/model.py`
   (`layers.0.attn.wkv.weight`, `mtp.0.*`, ...), not for a HF model class —
   none of the 69187 tensors carries a HF module name, so `from_pretrained`
   would silently initialise every weight at random;
2. transformers has no MTP module at all and discards the 1575 `mtp.*` tensors
   (`_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]`), and covering MTP
   is exactly the point of this dump;
3. `from_pretrained` materialises the whole state dict, and on a machine without
   CUDA the FP8/FP4 weights are dequantised to bf16 (568 GB) on the way in.

So the dump is produced by a plain-PyTorch port of DeepSeek's own inference
code, streaming one shard (one layer, ~3.5 GB) at a time.

## Layout

| Path | What it is |
| --- | --- |
| `kv4/model.py` | verbatim copy of the released `inference/model.py` with four auditable changes (see its header) |
| `kv4/kernels.py` | plain-torch equivalents of the six tilelang kernels: `act_quant`, `fp4_act_quant`, `fp8_gemm`, `fp4_gemm`, `sparse_attn`, `hc_split_sinkhorn`, plus `rotate_activation` (the CUDA-only Hadamard transform) |
| `kv4/weights.py` | shard indexing, mmap-backed reads, FP8/FP4/E8M0 dequantisation |
| `kv4/loader.py` | meta-device model shell, filling one layer by name, buffer reset, release |
| `kv4/capture.py` | the capture mechanism (module hooks + fused-helper wrappers) and the streaming driver |
| `dump.py` | CLI: tokenise, run, record, save |
| `dev_check.py` | self check of the port: structure, dequantisation, kernels, one real layer |
| `check_dump.py` | self check of the produced file: integrity, per-layer key spec, values, anchors |
| `model.vt` | the vt-lang description of the whole model: 27 nodes, verified |
| `VT_PLAN.md` | how the model was sliced into nodes, and what to watch out for |
| `prompts.json` | fixed prompt text, token IDs, lengths, output filenames and verification evidence |
| `deepseek-v4-flash-vt.safetensors` | original six-token reference: 5953 tensors, 256.5 MiB payload |
| `deepseek-v4-flash-vt-prompt32.safetensors` | independent 32-token reference |
| `deepseek-v4-flash-vt-prompt129.safetensors` | independent 129-token reference |

## Fixed prompts

The inputs are recorded in
[prompts.json](/Users/jiamin/workspace/vt-transformer/models/DeepSeek-V4-Flash/prompts.json).
Token IDs were checked with the checkpoint's `tokenizer.json`, using the same
encoding as `dump.py`, without a chat template or padding. The prompts are
ordinary questions; verification checks intermediate numerical results.

| case | tokens | subject | tensors | file size (MiB, including header) |
| --- | --- | --- | --- | --- |
| `baseline6` | 6 | self introduction | 5953 | 257.0 |
| `prompt32` | 32 | Python list and tuple | 12425 | 1349.1 |
| `prompt129` | 129 | what to bring for a park walk | 22029 | 5424.4 |

Each file uses the same activation key names. Values and many shapes differ
between inputs, so the references are kept separate. No model or weight copy
is created for a prompt. `dump.py`, `kv4/`, `check_dump.py`, `model.vt` and
vt-lang are shared. The new references were generated sequentially with CPU,
four threads, seed zero, all 43 main layers and the existing MTP passes.

| coverage | 6 tokens | 32 tokens | 129 tokens |
| --- | --- | --- | --- |
| complete ratio-4 blocks | 1, with 2 trailing tokens | 8, no trailing tokens | 32, with 1 trailing token |
| complete ratio-128 blocks | 0 | 0 | 1, with 1 trailing token |
| ratio-128 pooling/norm/rotation | not executed | not executed | executed in all 20 ratio-128 layers |
| exceeds sliding window 128 | no | no | yes |
| MTP decode `start_pos` | 6 | 32 | 129 |

The 129-token plain-layer index table was also checked to contain positions
1 through 128 in its last row. The number of recorded experts changes with
the token IDs and routing, so neither tensor count nor node-check count is
fixed across prompts. None of these inputs has more than 512 compressed
candidates: the top-512 truncation branch remains untested. Main-stack decode
and cache construction/update are still outside the VT verification scope;
MTP decode receives its recorded cache and index table as inputs.

### Verified results

| case | file check | full `llm` node checks | MTP decode node checks | dump time (s) | peak dump RSS (GiB) | full VT time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| `baseline6` | passed | 1878, passed | 23, passed | 61.9 | 15.8 | previous run |
| `prompt32` | passed | 3496, passed | 23, passed | 130.0 | 13.2 | 186.66 |
| `prompt129` | passed | 5922, passed | 23, passed | 238.3 | 13.0 | 430.12 |

The two additional cases were generated and checked in this run. The baseline
results are retained from its previous verified run; the original reference
SHA-256 is unchanged. All computation files, including the single `model.vt`,
are byte-for-byte unchanged, as are the original tracked project files.
No per-length model code or tolerance changes were required. All three MTP
decode references have 63 keys: 61 compared outputs and two supplied cache
inputs. Counts and file hashes are recorded in `prompts.json`. Timings are
observations on this machine, not performance guarantees or GPU measurements.

## Environment

```console
$ cd vt-lang && uv sync                     # Python 3.10 + torch 2.14.1
$ uv pip install --python vt-lang/.venv/bin/python tokenizers   # only for tokenising
```

The dump runs **on the CPU with mixed precision**, deliberately. Most matrix
work accumulates in FP32, hidden states use BF16 at the model's boundaries,
and Recorder saves floating activations as FP32. Saving BF16 values as FP32
does not recover the precision lost during computation.

* this implementation uses CPU quantisation conversions. On the tested M5
  with PyTorch 2.14.1, MPS supports ordinary floating-point matmuls, but the
  required FP8/FP4 conversion operations fail; GPU acceleration needs a
  separately adapted implementation;
* the current implementation defines a reproducible CPU reference; its timing
  is not a production GPU benchmark.

`dump.py` pins `torch.set_num_threads(4)`. Dump and verification must run on the
**same machine, same device, same thread count** — see "known numerical
deviations".

## The six-token baseline dump

```console
$ vt-lang/.venv/bin/python models/DeepSeek-V4-Flash/dump.py \
      --out models/DeepSeek-V4-Flash/deepseek-v4-flash-vt.safetensors
```

| | |
| --- | --- |
| input | `你好，简单介绍一下你自己。` → 6 tokens `[30594, 303, 6573, 70979, 36005, 320]` |
| device | CPU, mixed precision, `torch.set_num_threads(4)`, `seed=0` |
| recorded | **5953 tensors, 256.5 MiB**, in 61.9 s (peak RSS 15.8 GiB) |
| top-level keys | `input_ids`, `embed.out`, `norm.out`, `head.out` |
| per layer | `layers.0` … `layers.42`, 93–181 tensors each |
| MTP | `mtp.0.prefill.*` (start_pos=0) and `mtp.0.decode.*` (start_pos=6) |

### What is recorded

The rule is the one the Qwen reference dumps use: **the output of every module,
named by its module path plus `.out`**. On top of that, four quantities that
have no module of their own are recorded as well:

| key | what it is |
| --- | --- |
| `layers.N.hc_attn.pre/.post/.comb`, `layers.N.hc_ffn.*` | the hyper-connection split (Sinkhorn coefficients; `hc_pre`/`hc_post` are methods, not modules) |
| `layers.N.attn.sparse_attn.in.{q,kv,topk_idxs}` and `.out` | the sparse-attention operator's inputs and result |
| `layers.N.attn.rope.rot.<i>.out`, `layers.N.attn.rope.o_inv.out` | what `apply_rotary_emb` produced (in place). The number of forward rotations depends on the layer type; see below |
| `layers.N.attn.wo_b.in` | the grouped low-rank output projection: `Attention.forward` uses `self.wo_a.weight` directly in an `einsum` and never calls the `wo_a` module, so its result only exists as the (flattened) input of `wo_b` |

Only the experts that are actually routed to are recorded (4 keys each:
`experts.E.out` + `w1/w2/w3.out`), and the router's tuple return is split into
`ffn.gate.out.0` (weights) and `ffn.gate.out.1` (expert indices, integer).

### Three layer types

`compress_ratios = [0, 0, 4, 128, 4, 128, ...]`, and the recorded key set differs
per type. The following base-key counts describe the 6- and 32-token inputs:

| layer type | layers | base keys | extra keys |
| --- | --- | --- | --- |
| plain sliding window | 0, 1 | 31 | — |
| compressed, ratio 128 | odd layers 3…41 (20) | 33 | `attn.compressor.wkv/wgate.out` |
| compressed, ratio 4 | even layers 2…42 (21) | 45 | compressor (4), indexer (7), three extra rope rotations |

At 129 tokens, ratio-128 layers have 36 base keys: the three extra keys are
`attn.compressor.out`, `attn.compressor.norm.out` and `attn.rope.rot.2.out`.

## Known numerical deviations from the official CUDA implementation

This port defines the reference the `.vt` nodes are verified against, so the
deviations from the tilelang implementation are documented rather than hidden.
This is a CPU reference, not a measured end-to-end CUDA-equivalence claim.
FP32 denominator accumulation and the BF16 numerator probability copy are
kept separate, as in the official sparse-attention kernel. Remaining
cross-backend numerical differences include:

* **accumulation order** — the CUDA kernels tile and reduce in their own order,
  this code accumulates in BLAS order;
* **`sparse_attn`** — the kernel walks the top-k positions in blocks of 64 with
  an online softmax; this port computes the same softmax but chunks over query
  positions. The probability cast to bf16 before the `P @ V` product is
  reproduced;
* **`rotate_activation`** — the original calls the CUDA extension
  `fast_hadamard_transform`; this port multiplies by an explicit Sylvester
  Walsh-Hadamard matrix and scales by `d ** -0.5`;
* **transcendental functions** — `exp`, `sigmoid`, `softplus` differ between
  CUDA and CPU implementations.

Because quantisation steps act as decision boundaries, a difference of 1e-7 in
the input can occasionally become a whole quantisation step in the output, which
is why dump and verification must share the same stack. Deriving a reference
from the CUDA implementation would require running it on a machine with enough
memory for the 149 GiB checkpoint.

## Known coverage limits of the 6-token input

The input is deliberately short (the Qwen dumps use 6 tokens too), which means
the length-dependent machinery of V4 is only exercised in its degenerate form:

* **ratios 128 layers never compress**: `seqlen >= ratio` is false, so
  `Compressor.forward` returns `None` before its norm and rotation. Those 20
  layers have **no** `attn.compressor.out` / `.norm.out` to verify against;
* **ratio 4 layers compress a single block** (the first 4 tokens);
* the **indexer** has one candidate block, so it can only select `top-1` — its
  selection logic is not exercised;
* `head.out` is `[1, 129280]`: `ParallelHead.get_logits` projects the **last
  position only**, which is what the official forward computes;
* the MTP decode pass runs at `start_pos = 6` (the natural continuation of the
  prefill) rather than `1` as in `model.py`'s smoke test, which reuses an
  already written cache slot. `model.vt` covers it as a separate run: the window
  cache and its index table are themselves recorded (they are
  `attn.sparse_attn.in.*`), so `mtp_decode` takes them as arguments instead of
  rebuilding them. For the original six-token case, 61 of the 63 decode keys
are compared as outputs; the two cache/index keys are supplied as inputs
and their construction or update is not verified. Only the main-stack
  decode branches, whose cache state is *not* part of the capture, stay out of
  scope.

A longer input can be dumped with `--input-ids` when those paths need coverage;
the key set also changes when compression starts or stops. The checker
derives the expected keys from the input length and the layer configuration.

## Reproducing / checking

```console
# does the port work? (structure, dequantisation, kernels, one real layer)
$ vt-lang/.venv/bin/python models/DeepSeek-V4-Flash/dev_check.py --ckpt <ckpt>

# what would be recorded? (three representative layers, prints the key list, writes nothing)
$ vt-lang/.venv/bin/python models/DeepSeek-V4-Flash/dump.py --list

# is the delivered file complete and well formed?
$ vt-lang/.venv/bin/python models/DeepSeek-V4-Flash/check_dump.py \
      --dump models/DeepSeek-V4-Flash/deepseek-v4-flash-vt.safetensors --ckpt <ckpt>
```

`dev_check.py` verifies the port itself, `dump.py --list` prints the capture
spec, and `check_dump.py` verifies the delivered dump: header/payload size, the
per-layer key spec of the three layer types (including that experts come in
complete groups of four), no NaN/Inf, integer dtypes preserved, and an
end-to-end anchor — `embed.out` must equal a fresh embedding lookup on the
released weights. Its last run reported 0 problems (5857 float tensors, none
empty, none non-finite, anchor diff 0.000e+00).

## Verification

One node at a time — `vt` loads the activation keys as inputs and compares the
node's outputs with the same keys in the dump:

```console
$ V=vt-lang/.venv/bin/vt; F=models/DeepSeek-V4-Flash/model.vt

# a whole layer, one per layer type
$ $V $F decoder_layer layers.0.out input_ids prefix=layers.1. wprefix=layers.1. layer=1
$ $V $F decoder_layer layers.1.out input_ids prefix=layers.2. wprefix=layers.2. layer=2
$ $V $F decoder_layer layers.2.out input_ids prefix=layers.3. wprefix=layers.3. layer=3

# the whole model: 43 layers + the prefill MTP block + the head
$ $V $F llm input_ids

# the MTP decode step: one token, attending over the recorded window cache
$ $V $F mtp_decode layers.42.out input_ids \
      mtp.0.decode.attn.sparse_attn.in.kv \
      mtp.0.decode.attn.sparse_attn.in.topk_idxs \
      prefix=mtp.0.decode. start_pos=6
```

The following results describe the retained six-token baseline, CPU and 4 threads.
The original vt-lang implementation is unchanged. This VT description requires
at least four prefill tokens; shorter inputs report that limit explicitly.

| run | result |
| --- | --- |
| `llm` (43 layers + MTP prefill + head) | **1878 node verifications, 0 errors** |
| `mtp_decode` (the MTP decode step) | 23 node verifications, 0 errors; 61/63 keys compared, 2 supplied cache inputs |

Every main-stack layer output is checked against the regenerated reference
at the vt-lang tolerance of atol=rtol=1e-4. The file checker additionally
requires all 43 layers and complete outputs for every routed expert.

`llm` itself prints no `verify OK` line: it is pure orchestration and declares
no outputs, so comparison takes place in its nested nodes. Their invocation
count depends on the input and the routed experts.

### Verify all three inputs with the same model.vt

Run from `/Users/jiamin/workspace/vt-transformer`. This reads existing
references and runs the file checker, full `llm` and existing MTP decode for
each case. It does not regenerate the reference files.

```bash
vt-lang/.venv/bin/python - <<'PY'
from pathlib import Path
import json
import os
import subprocess

root = Path.cwd()
base = root / "models/DeepSeek-V4-Flash"
ckpt = Path("/Users/jiamin/workspace/infra/dsv4-flash")
python = str(root / "vt-lang/.venv/bin/python")
vt = str(root / "vt-lang/.venv/bin/vt")
model = str(base / "model.vt")
manifest = json.loads((base / "prompts.json").read_text())
for case in manifest["cases"]:
    reference = str(base / case["dump_file"])
    print("Checking", case["id"], case["seq_len"], "tokens", flush=True)
    subprocess.run([python, str(base / "check_dump.py"), "--ckpt", str(ckpt),
                    "--dump", reference], check=True)
    env = {**os.environ, "VT_WEIGHT_FILE": str(ckpt / "model.safetensors.index.json"),
           "VT_ACTIVATION_FILE": reference, "VT_NUM_THREADS": "4"}
    subprocess.run([vt, model, "llm", "input_ids"], env=env, check=True)
    subprocess.run([vt, model, "mtp_decode", "layers.42.out", "input_ids",
                    "mtp.0.decode.attn.sparse_attn.in.kv",
                    "mtp.0.decode.attn.sparse_attn.in.topk_idxs",
                    "prefix=mtp.0.decode.",
                    f"start_pos={case['mtp_decode_start_pos']}"], env=env, check=True)
PY
```

### Regenerate the two additional references

Only use this when regeneration is intended: it overwrites the two new files,
leaves the six-token baseline alone, and uses the frozen IDs from the manifest.
Afterwards run the verification block above.

```bash
vt-lang/.venv/bin/python - <<'PY'
from pathlib import Path
import json
import subprocess

root = Path.cwd()
base = root / "models/DeepSeek-V4-Flash"
manifest = json.loads((base / "prompts.json").read_text())
for case in manifest["cases"]:
    if case["id"] == "baseline6":
        continue
    subprocess.run([str(root / "vt-lang/.venv/bin/python"), str(base / "dump.py"),
                    "--ckpt", "/Users/jiamin/workspace/infra/dsv4-flash",
                    "--threads", "4", "--seed", "0", "--input-ids",
                    ",".join(map(str, case["input_ids"])),
                    "--out", str(base / case["dump_file"])], check=True)
PY
```

## Provenance

`kv4/model.py` is a copy of `inference/model.py` from
[deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
(MIT License, Copyright DeepSeek-AI); the four changes are listed in its header.
`kv4/kernels.py` reimplements `inference/kernel.py` from the same release.
