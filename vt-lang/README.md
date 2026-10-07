# vt-lang

A DSL for describing and verifying LLM computation.

It serves two audiences at once:

- **Human developers** — understand a model's computation architecture in a simple, readable language.
- **AI agents** — generate inference code against an executable description, where every computation can be verified.

## Project context

vt-lang is one component of [VT-Transformer](../README.md), an LLM inference
engine built on AI-generated kernels. The workflow: dump a model's weights and
reference activations, let an AI agent write `.vt` nodes describing each part
of the computation, and use vt-lang's operator checks and output verification
to confirm every generated node is correct before it becomes inference code.
See the top-level README for the full picture.

## How it works

A model is described in a `.vt` file: standard Python with one restriction — every
computation node is a function decorated with `@vt.NodeChecker`. Each node is a
short, readable sequence of primitive tensor ops, statically checked against a
lean operator subset (`std_operators`), so it can be mapped 1:1 onto backend
kernels. High-level composites (`torch.einsum`, `torch.nn.*`, `jit`, `compile`,
...) are denied by default.

Nodes declare three bindings:

- **`weights`** — `{local_name: weight_key}`: the safetensors keys loaded and
  bound as globals for the duration of one call, then freed.
- **`outputs`** — `[activation_key, ...]` by output position: after each call,
  outputs are compared against reference activations (`atol=rtol=1e-4`); a
  mismatch raises. `None` skips a position.
- **`templates`** — `[name, ...]`: placeholders such as `{layer}` expanded in
  weight/output keys per call from keyword arguments, so one node covers every
  transformer layer.

## Example

```python
# qwen3_mha.vt
import torch
import vt_lang as vt

@vt.NodeChecker(
    weights={
        "q_proj_w": "model.layers.{layer}.self_attn.q_proj.weight",
        "k_proj_w": "model.layers.{layer}.self_attn.k_proj.weight",
        "v_proj_w": "model.layers.{layer}.self_attn.v_proj.weight",
    },
    outputs=["model.layers.{layer}.self_attn.qkv"],
    templates=["layer"],
)
def qkv_proj(hidden_states, layer):
    q = torch.linear(hidden_states, q_proj_w)
    k = torch.linear(hidden_states, k_proj_w)
    v = torch.linear(hidden_states, v_proj_w)
    return torch.cat([q, k, v], dim=-1)
```

Run and verify it against dumped activations:

```python
import vt_lang as vt

vt.register_safetensors(
    "weights/model.safetensors.index.json",   # weights (single file or sharded index)
    "activations/forward.safetensors",        # reference activations for verification
)

nodes = vt.load_vt_file("qwen3_mha.vt")
hidden = vt.load_activation("model.layers.0.input_layernorm")
out = nodes["qkv_proj"](hidden, layer=0)      # prints the call stack; verifies outputs
```

Or from the command line:

```console
$ vt qwen3_mha.vt                      # list nodes and their weight bindings
$ vt qwen3_mha.vt qkv_proj activation_key layer=0
```

Positional arguments are activation keys loaded as inputs; `name=value`
arguments are passed to the node as keywords (template values such as
`layer=3` are converted to int/float when possible).

## Install

Requires Python ≥ 3.10, torch ≥ 2.9, safetensors, numpy.

```console
$ uv sync          # or: pip install -e .
```

## API

| Name | Purpose |
| --- | --- |
| `vt.register_safetensors(weight_path, activate_path=None)` | Register the weight source and activation source (`.safetensors` file or sharded-index `.json`). |
| `vt.load_vt_file(path)` | Execute a `.vt` file and return `{name: NodeChecker}`. |
| `vt.load_weight(key)` / `vt.load_activation(key)` | Load one tensor by key from the registered sources. |
| `@vt.NodeChecker(weights, outputs, templates)` | Declare a computation node. |

## The operator subset

Node functions may only use the operators allowlisted in
`std_operators.py`:

- **elementwise**: `add`, `mul`, `sub`, `div`, `pow`, `abs`, `clamp`, `exp`,
  `log`, `sqrt`, `rsqrt`, `reciprocal`, `neg`, `sigmoid`, `tanh`, `gelu`, `silu`
- **linear**: `linear` (the one composite allowed by design), `matmul`, `bmm`,
  `embedding`
- **norm**: `layer_norm`, `rms_norm`, `softmax`
- **reduce**: `sum`, `mean`, `max`, `min`, `argmax`, `cumsum`
- **shape / layout**: `cat`, `stack`, `reshape`, `view`, `permute`,
  `transpose`, `squeeze`, `unsqueeze`, `gather`, `index_select`, `expand`,
  `contiguous`
- **compare / select**: `where`, `maximum`, `minimum`
- **create**: `zeros`, `ones`, `full`, `arange`, `zeros_like`, `ones_like`

Plus a small set of tensor methods (`to`, `float`, `masked_fill`, ...) and
attributes (`shape`, `dtype`, `T`, ...), dtype constants, and a short list of
allowed builtins (`len`, `range`, `min`, ...). Anything else — especially
`einsum`, `nn.*`, `jit`, `compile`, `no_grad` — is rejected at decoration time
with a pointed error message.
