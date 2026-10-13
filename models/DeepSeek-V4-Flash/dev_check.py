#!/usr/bin/env python3
"""Stage-1 self check for the DeepSeek-V4-Flash reference port (kv4).

Four checks, in increasing order of "how much of the pipeline they prove":

  1. structure -- the port's module tree matches the released checkpoint's key
     namespace, in both directions (this is the fidelity argument);
  2. weights   -- fp8 / fp4 / e8m0 dequantisation produces sane tensors;
  3. kernels   -- the six plain-torch kernels agree with independent reference
     implementations written a different way;
  4. layer 0   -- one real decoder layer runs end to end on real weights.

Usage:
    vt-lang/.venv/bin/python models/DeepSeek-V4-Flash/dev_check.py \
        [--ckpt ~/workspace/infra/dsv4-flash] [--seq-len 16] [--skip-layer-run]
"""

from __future__ import annotations

import argparse
import math
import resource
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kv4 import kernels  # noqa: E402
from kv4.loader import build_shell, is_consumed, load_prefix  # noqa: E402
from kv4.weights import Checkpoint  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def run_check(name: str, function) -> bool:
    print(f"\n--- {name} " + "-" * max(0, 60 - len(name)))
    try:
        ok, detail = function()
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {detail}")
    return ok


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


# ---------------------------------------------------------------------------
# 1. structure
# ---------------------------------------------------------------------------
def check_structure(model, ckpt: Checkpoint):
    keys = ckpt.keys()
    layer_ids = set(ckpt.layer_indices())
    top_tokens = {key.split(".")[0] for key in ckpt.index}

    def resolves(key: str) -> bool:
        try:
            return model.get_parameter(key) is not None
        except AttributeError:
            return False

    def scope_available(name: str) -> bool:
        """Is the part of the model this parameter belongs to downloaded yet?"""
        if name.startswith("layers."):
            return int(name.split(".")[1]) in layer_ids
        return name.split(".")[0] in top_tokens

    consumed = [k for k in keys if is_consumed(k)]
    unmatched = [k for k in keys if not is_consumed(k) and not resolves(k)]
    shell_names = [name for name, _ in model.named_parameters()]
    missing = [name for name in shell_names if name not in ckpt.index and scope_available(name)]
    pending = [name for name in shell_names if name not in ckpt.index and not scope_available(name)]

    print(f"  checkpoint tensors        : {len(keys)}")
    print(f"  shell parameters          : {len(shell_names)}")
    print(f"  keys folded into a weight : {len(consumed)} (wo_a.scale, expected: no parameter)")
    print(f"  shell params not yet in ckpt (download pending): {len(pending)}")
    for label, items in (("keys with no parameter", unmatched), ("parameters with no key", missing)):
        if items:
            print(f"  {label}: {len(items)} e.g. {items[:5]}")
    return (
        not unmatched and not missing,
        f"{len(keys)} keys <-> {len(shell_names)} parameters "
        f"(+{len(consumed)} folded); {len(unmatched)} unmatched, "
        f"{len(missing)} unfilled, {len(pending)} pending download",
    )


# ---------------------------------------------------------------------------
# 2. weights
# ---------------------------------------------------------------------------
def check_weights(ckpt: Checkpoint):
    problems = []
    for prefix in (
        "layers.0.attn.wq_a",
        "layers.0.attn.wq_b",
        "layers.0.attn.wkv",
        "layers.0.attn.wo_a",
        "layers.0.ffn.shared_experts.w1",
    ):
        weight = ckpt.dequantized(f"{prefix}.weight")
        std = weight.std().item()
        print(f"  {prefix:36s} {str(tuple(weight.shape)):14s} std={std:.5f} max={weight.abs().max():.4f}")
        if not 0.005 < std < 0.1 or torch.isnan(weight).any():
            problems.append(prefix)
    for matrix in ("w1", "w2", "w3"):
        weight = ckpt.dequantized(f"layers.0.ffn.experts.0.{matrix}.weight")
        distinct = torch.unique(weight).numel()
        print(f"  layers.0.ffn.experts.0.{matrix:14s} {str(tuple(weight.shape)):14s} "
              f"std={weight.std():.5f} distinct values={distinct}")
        if torch.isnan(weight).any() or weight.abs().max() > 1.0:
            problems.append(matrix)
    return not problems, f"{len(problems)} suspicious tensors"


# ---------------------------------------------------------------------------
# 3. kernels
# ---------------------------------------------------------------------------
def naive_act_quant(x: torch.Tensor, block: int, round_scale: bool = True):
    """Explicit per-block reference for kernels.act_quant (python loops)."""
    n = x.size(-1)
    value = x.float()
    quantized = torch.empty_like(value)
    scales = []
    for start in range(0, n, block):
        piece = value[..., start : start + block]
        amax = piece.abs().amax(dim=-1).clamp(min=1e-4)
        step = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0))) if round_scale else amax / 448.0
        quantized[..., start : start + block] = piece / step.unsqueeze(-1)
        scales.append(step)
    return quantized.clamp(-448.0, 448.0), torch.stack(scales, dim=-1)


def naive_sinkhorn(mixes, hc_scale, hc_base, hc=4, iters=20, eps=1e-6):
    """Scalar python reference for kernels.hc_split_sinkhorn."""
    pres, posts, combs = [], [], []
    for row in mixes.tolist():
        pre = [1.0 / (1.0 + math.exp(-(row[j] * hc_scale[0].item() + hc_base[j].item()))) + eps for j in range(hc)]
        post = [2.0 / (1.0 + math.exp(-(row[hc + j] * hc_scale[1].item() + hc_base[hc + j].item()))) for j in range(hc)]
        comb = [[row[2 * hc + j * hc + k] * hc_scale[2].item() + hc_base[2 * hc + j * hc + k].item()
                 for k in range(hc)] for j in range(hc)]
        for j in range(hc):                                    # softmax over k
            top = max(comb[j])
            total = sum(math.exp(v - top) for v in comb[j])
            for k in range(hc):
                comb[j][k] = math.exp(comb[j][k] - top) / total + eps
        for k in range(hc):                                    # divide by column sums
            total = sum(comb[j][k] for j in range(hc))
            for j in range(hc):
                comb[j][k] /= total + eps
        for _ in range(iters - 1):
            for j in range(hc):                                # divide by row sums
                total = sum(comb[j])
                for k in range(hc):
                    comb[j][k] /= total + eps
            for k in range(hc):                                # divide by column sums
                total = sum(comb[j][k] for j in range(hc))
                for j in range(hc):
                    comb[j][k] /= total + eps
        pres.append(pre)
        posts.append(post)
        combs.append(comb)
    return (torch.tensor(pres), torch.tensor(posts), torch.tensor(combs))


def brute_sparse_attn(q, kv, sink, idxs, scale):
    """Direct softmax over the gathered positions, no online softmax, no chunking."""
    b, m, h, d = q.shape
    out = torch.zeros(b, m, h, d, dtype=torch.bfloat16)
    for bi in range(b):
        for mi in range(m):
            index = idxs[bi, mi]
            keep = index >= 0
            if not bool(keep.any()):
                continue
            k = kv[bi, index[keep]].float()
            scores = (q[bi, mi].float() @ k.t()) * scale
            top = scores.amax(-1, keepdim=True)
            probs_fp32 = torch.exp(scores - top)
            denom = probs_fp32.sum(-1, keepdim=True) + torch.exp(sink.view(h, 1) - top)
            probs_bf16 = probs_fp32.to(torch.bfloat16).float()
            out[bi, mi] = ((probs_bf16 @ k) / denom).to(torch.bfloat16)
    return out


def check_kernels(ckpt: Checkpoint):
    torch.manual_seed(0)
    failures = []

    # -- act_quant: scales must match an explicit loop, values must sit on the E4M3 grid
    x = torch.randn(4, 256, dtype=torch.bfloat16) * 3.0
    quantized, scales = kernels.act_quant(x, 64, "ue8m0", torch.float32)
    naive_unscaled, naive_scales = naive_act_quant(x, 64)
    clamped = naive_unscaled.clamp(-448.0, 448.0)
    scale_gap = (scales - naive_scales).abs().max().item()
    powers_of_two = bool((scales.log2() % 1 == 0).all())
    on_grid = torch.equal(kernels._to_e4m3(quantized), quantized)
    # E4M3 has 3 mantissa bits, so half an ulp is 2**-4 of the value; below 2**-6
    # the grid is fixed at 2**-9, hence the absolute floor.
    allowed = torch.maximum(clamped.abs() * 2.0 ** -4, torch.full_like(clamped, 2.0 ** -9))
    worst = ((quantized - clamped).abs() / allowed).max().item()
    print(f"  act_quant   scale match={scale_gap:.2e}  powers of two={powers_of_two}  "
          f"on e4m3 grid={on_grid}  worst error/bound={worst:.3f}")
    if scale_gap != 0.0 or not powers_of_two or not on_grid or worst > 1.0:
        failures.append("act_quant")

    # -- fp4: the result must sit on the E2M1 grid, and rounding may not move a
    #    value further than half a grid step. That step is *not* uniform: the
    #    widest gap in E2M1 is 4.0 -> 6.0, so 5.47 legitimately rounds up to 6.0.
    x4 = torch.randn(2, 64, dtype=torch.bfloat16) * 2.0
    before = x4.clone().float()
    amax = before.reshape(2, 2, 32).abs().amax(-1).clamp(min=6.0 * 2.0 ** -126)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 6.0)))
    kernels.fp4_act_quant(x4, 32, inplace=True)
    after = x4.float()
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    scaled = (after.reshape(2, 2, 32) / scale.unsqueeze(-1)).abs()
    on_grid = bool((scaled.unsqueeze(-1) - grid).abs().min(dim=-1).values.lt(1e-6).all())
    moved = (after - before).abs().max().item()
    bound = scale.max().item() * 1.0
    print(f"  fp4_act_quant  on E2M1 grid={on_grid}  max|after-before|={moved:.3f} "
          f"(half-step bound {bound:.3f})  changed={bool((x4 != before.to(torch.bfloat16)).any())}")
    if not on_grid or moved > bound + 1e-6:
        failures.append("fp4_act_quant")

    # -- fp8_gemm / fp4_gemm: a straightforward dequant + matmul, rounded to bf16
    # (the kernels round their output to bf16, so the reference is rounded too;
    #  both sides then use the same BLAS call and must agree exactly)
    weight_fp8 = ckpt.get("layers.0.attn.wq_a.weight")
    scale_fp8 = ckpt.get("layers.0.attn.wq_a.scale")
    act = torch.randn(4, 4096, dtype=torch.bfloat16)
    quantized_act, act_scale = kernels.act_quant(act, 128, "ue8m0", torch.float32)
    activations = quantized_act.float() * act_scale.float().repeat_interleave(128, dim=-1)

    got = kernels.fp8_gemm(quantized_act, act_scale, weight_fp8, scale_fp8)
    want = (activations @ ckpt.dequantized("layers.0.attn.wq_a.weight").t()).to(torch.bfloat16)
    gap = (got.float() - want.float()).abs().max().item()
    print(f"  fp8_gemm    shape={tuple(got.shape)} dtype={got.dtype} max|diff| vs reference={gap:.3e}")
    if got.shape != (4, 1024) or gap > 1e-6:
        failures.append("fp8_gemm")

    expert = ckpt.get("layers.0.ffn.experts.0.w1.weight")
    expert_scale = ckpt.get("layers.0.ffn.experts.0.w1.scale")
    got4 = kernels.fp4_gemm(quantized_act, act_scale, expert, expert_scale)
    want4 = (activations @ ckpt.dequantized("layers.0.ffn.experts.0.w1.weight").t()).to(torch.bfloat16)
    gap4 = (got4.float() - want4.float()).abs().max().item()
    print(f"  fp4_gemm    shape={tuple(got4.shape)} dtype={got4.dtype} max|diff| vs reference={gap4:.3e}")
    if got4.shape != (4, 2048) or gap4 > 1e-2:
        failures.append("fp4_gemm")

    # -- hc_split_sinkhorn against the scalar reference
    mixes = torch.randn(2, 24) * 0.5
    hc_scale = torch.tensor([0.7, 0.3, 1.1])
    hc_base = torch.randn(24) * 0.1
    pre, post, comb = kernels.hc_split_sinkhorn(mixes, hc_scale, hc_base)
    n_pre, n_post, n_comb = naive_sinkhorn(mixes, hc_scale, hc_base)
    gaps = ((pre - n_pre).abs().max().item(), (post - n_post).abs().max().item(),
            (comb - n_comb).abs().max().item())
    row_err = (comb.sum(-1) - 1.0).abs().max().item()
    print(f"  sinkhorn    shapes={tuple(pre.shape)},{tuple(post.shape)},{tuple(comb.shape)} "
          f"gaps={gaps[0]:.1e}/{gaps[1]:.1e}/{gaps[2]:.1e} rowsum-1={row_err:.1e}")
    if max(gaps) > 1e-5 or torch.isnan(comb).any():
        failures.append("hc_split_sinkhorn")

    # -- sparse_attn against a direct softmax
    b, m, h, d, n, topk = 1, 5, 3, 8, 10, 6
    q = torch.randn(b, m, h, d, dtype=torch.bfloat16)
    kv = torch.randn(b, n, d, dtype=torch.bfloat16)
    sink = torch.randn(h)
    idxs = torch.randint(-1, n, (b, m, topk))
    idxs[:, 0, :] = torch.arange(topk) % n                      # make sure row 0 has valid entries
    idxs[idxs >= n] = -1
    got_s = kernels.sparse_attn(q, kv, sink, idxs, d ** -0.5)
    want_s = brute_sparse_attn(q, kv, sink, idxs, d ** -0.5)
    gap_s = (got_s.float() - want_s.float()).abs().max().item()
    print(f"  sparse_attn shape={tuple(got_s.shape)} dtype={got_s.dtype} max|diff|={gap_s:.3e}")
    if got_s.shape != (b, m, h, d) or gap_s > 1e-2:
        failures.append("sparse_attn")

    # -- rotate_activation: orthonormal, and its own inverse up to a sign
    small = torch.randn(4, 16, dtype=torch.bfloat16)
    rotated = kernels.rotate_activation(small)
    twice = kernels.rotate_activation(rotated)
    norm_ratio = (rotated.float().norm(dim=-1) / small.float().norm(dim=-1)).mean().item()
    identity_gap = (twice.float() - small.float()).abs().max().item()
    expected = torch.tensor([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], dtype=torch.float32)
    matrix_ok = bool((kernels.hadamard_matrix(4) == expected).all())
    print(f"  hadamard    H(4) as expected={matrix_ok}  norm ratio={norm_ratio:.4f}  "
          f"R(R(x))-x={identity_gap:.3e}")
    if not matrix_ok or abs(norm_ratio - 1) > 0.02 or identity_gap > 0.05:
        failures.append("rotate_activation")

    return not failures, f"failing kernels: {failures or 'none'}"


# ---------------------------------------------------------------------------
# 4. one real layer
# ---------------------------------------------------------------------------
def check_layer_run(model, ckpt: Checkpoint, args, seq_len: int, layer: int = 0):
    started = time.perf_counter()
    filled, unmatched = load_prefix(model, ckpt, args, f"layers.{layer}")
    load_seconds = time.perf_counter() - started
    if unmatched:
        print(f"  unmatched keys: {unmatched[:5]}")
        return False, f"{len(unmatched)} keys did not match a parameter"

    hidden = (torch.randn(1, seq_len, args.hc_mult, args.dim) * 0.02).to(torch.bfloat16)
    input_ids = torch.randint(0, args.vocab_size, (1, seq_len))
    started = time.perf_counter()
    with torch.no_grad():
        out = model.layers[layer](hidden, 0, input_ids)
    forward_seconds = time.perf_counter() - started

    print(f"  weights    : {filled} tensors in {load_seconds:.1f} s")
    print(f"  forward    : {tuple(hidden.shape)} -> {tuple(out.shape)} in {forward_seconds:.1f} s")
    print(f"  output     : dtype={out.dtype} mean={out.float().mean():+.5f} std={out.float().std():.5f} "
          f"nan={int(torch.isnan(out.float()).sum())}")
    print(f"  peak RSS   : {peak_rss_gib():.1f} GiB")
    ok = out.shape == hidden.shape and not torch.isnan(out.float()).any()
    return ok, f"layer {layer} ran, {forward_seconds:.1f} s forward, peak {peak_rss_gib():.1f} GiB"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="~/workspace/infra/dsv4-flash")
    parser.add_argument("--config", default=None, help="defaults to <ckpt>/inference/config.json")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--layer", type=int, default=0, help="layer to run in check 4")
    parser.add_argument("--skip-layer-run", action="store_true")
    args_in = parser.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(0)

    root = Path(args_in.ckpt).expanduser()
    config = Path(args_in.config) if args_in.config else root / "inference" / "config.json"

    print(f"torch {torch.__version__}  threads={torch.get_num_threads()}  "
          f"mps={torch.backends.mps.is_available()}")
    print(f"checkpoint : {root}")
    print(f"config     : {config}")

    ckpt = Checkpoint(root)
    print(f"{ckpt.describe()}")
    print(f"layers present locally: {ckpt.layer_indices()}")

    meta_before = peak_rss_gib()
    args, model = build_shell(config)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"\nshell built on meta: {parameters / 1e9:.1f}B parameter entries "
          f"(FP4 experts are stored packed, so this reads below the 284B headline), "
          f"peak RSS {meta_before:.1f} -> {peak_rss_gib():.1f} GiB")

    run_check("1. structure: port vs checkpoint keys", lambda: check_structure(model, ckpt))
    run_check("2. weights: dequantisation", lambda: check_weights(ckpt))
    run_check("3. kernels: agree with independent references", lambda: check_kernels(ckpt))
    if args_in.skip_layer_run:
        print("\n--- 4. layer run skipped (--skip-layer-run)")
    else:
        run_check(
            f"4. run layer {args_in.layer} on real weights",
            lambda: check_layer_run(model, ckpt, args, args_in.seq_len, args_in.layer),
        )

    print("\n" + "=" * 68)
    for name, ok, detail in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    failed = [name for name, ok, _ in RESULTS if not ok]
    print("=" * 68)
    if failed:
        print(f"{len(failed)} check(s) failed -- paste this output back for the next round.")
        return 1
    print("all checks passed -- ready for the streaming dump (stage 2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
