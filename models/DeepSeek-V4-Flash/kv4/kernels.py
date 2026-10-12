"""Plain-torch equivalents of the six tilelang kernels in inference/kernel.py.

The signatures and the arithmetic follow the CUDA originals one-for-one, so
that kv4/model.py can stay a near-verbatim copy of DeepSeek's inference code.
What changes is only the execution strategy:

* everything runs in float32 on the CPU (torch's CPU bf16 matmul is emulated
  and about 2000x slower than its fp32 path; MPS has neither fp8 nor fp4);
* tilelang is replaced by plain tensor ops;
* ``fast_hadamard_transform`` (a CUDA extension) is replaced by an explicit
  Walsh-Hadamard matrix multiply.

Known deviation: the CUDA kernels accumulate in a tile-dependent order, this
code accumulates in BLAS order. Both are fp32; results differ only in the last
bits (see the README, section "known numerical deviations").
"""

from __future__ import annotations

import torch

from .weights import dequantize_fp4, dequantize_fp8, e8m0_to_float32

FP8_MAX = 448.0
FP4_MAX = 6.0

# E2M1 (FP4) representable magnitudes and the midpoints between them.
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
# Round-to-nearest-even at an exact midpoint picks the code with an even
# mantissa (codes 0, 2, 4, 6); bucketize alone would always round up.
E2M1_TIE_CODE = torch.tensor([0, 2, 2, 4, 4, 6, 6], dtype=torch.long)


def _native_fp8_cast_available() -> bool:
    try:
        torch.zeros(1).to(torch.float8_e4m3fn)
        return True
    except Exception:  # pragma: no cover - depends on the torch build
        return False


_HAS_NATIVE_FP8 = _native_fp8_cast_available()


def _round_to_e4m3(value: torch.Tensor) -> torch.Tensor:
    """Round fp32 to the E4M3 grid (round-half-to-even), returned as fp32.

    Used only when the running torch cannot cast to float8_e4m3fn natively.
    ``value`` must already be clamped to [-448, 448].
    """
    magnitude = value.abs()
    exponent = torch.floor(torch.log2(magnitude.clamp(min=2.0 ** -9)))
    step = torch.pow(2.0, exponent - 3)
    step = torch.where(magnitude < 2.0 ** -6, torch.full_like(step, 2.0 ** -9), step)
    rounded = torch.round(magnitude / step) * step
    return torch.sign(value) * rounded.clamp(max=FP8_MAX)


def _to_e4m3(value: torch.Tensor) -> torch.Tensor:
    """Values of ``value`` snapped to the E4M3 grid, as float32."""
    if _HAS_NATIVE_FP8:
        return value.to(torch.float8_e4m3fn).float()
    return _round_to_e4m3(value)


def _to_e2m1(magnitude: torch.Tensor) -> torch.Tensor:
    """Magnitudes snapped to the E2M1 (FP4) grid, round-half-to-even."""
    code = torch.bucketize(magnitude, E2M1_MIDPOINTS)
    lower = (code - 1).clamp(min=0)
    on_midpoint = (code > 0) & (magnitude == E2M1_MIDPOINTS[lower])
    code = torch.where(on_midpoint, E2M1_TIE_CODE[lower], code)
    return E2M1_VALUES[code]


def _round_pow2(amax: torch.Tensor, max_value: float) -> torch.Tensor:
    """The kernels' fast_round_scale: 2 ** ceil(log2(amax / max_value))."""
    return torch.pow(2.0, torch.ceil(torch.log2(amax / max_value)))


def act_quant(x, block_size: int = 128, scale_fmt=None, scale_dtype=torch.float32, inplace: bool = False):
    """Block-wise FP8 quantization (inference/kernel.py::act_quant).

    inplace=True  -> x is overwritten with quantize-then-dequantize values and
                     returned (this is what inference/model.py uses for KV).
    inplace=False -> returns (values snapped to the E4M3 grid as float32,
                     per-block scale). The values are what fp8_gemm consumes.
    """
    n = x.size(-1)
    assert n % block_size == 0, f"last dim {n} not divisible by block {block_size}"
    blocks = x.reshape(*x.shape[:-1], n // block_size, block_size).float()
    amax = blocks.abs().amax(dim=-1).clamp(min=1e-4)
    if scale_fmt is not None:                      # "ue8m0": power-of-two scales
        scale = _round_pow2(amax, FP8_MAX)
    else:
        scale = amax / FP8_MAX
    scaled = (blocks / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    quantized = _to_e4m3(scaled).reshape(x.shape)
    if inplace:
        x.copy_((quantized * scale.repeat_interleave(block_size, dim=-1)).to(x.dtype))
        return x
    return quantized, scale


def fp4_act_quant(x, block_size: int = 32, inplace: bool = False):
    """Block-wise FP4 quantization (inference/kernel.py::fp4_act_quant).

    Always used with inplace=True in inference/model.py (the indexer path):
    x is overwritten with quantize-then-dequantize values and returned.
    """
    n = x.size(-1)
    assert n % block_size == 0, f"last dim {n} not divisible by block {block_size}"
    blocks = x.reshape(*x.shape[:-1], n // block_size, block_size).float()
    amax = blocks.abs().amax(dim=-1).clamp(min=FP4_MAX * 2.0 ** -126)
    scale = _round_pow2(amax, FP4_MAX)
    scaled = (blocks / scale.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX)
    quantized = (torch.sign(scaled) * _to_e2m1(scaled.abs())).reshape(x.shape)
    dequantized = quantized * scale.repeat_interleave(block_size, dim=-1)
    if inplace:
        x.copy_(dequantized.to(x.dtype))
        return x
    return quantized, scale


def _expand_act_scale(scale: torch.Tensor, block: int, size: int) -> torch.Tensor:
    """[.., K/block] -> [.., K]: repeat_interleave along the last (K) dimension.

    Activation scales keep every leading dimension (``linear()`` can be called
    with x of shape [batch, seq, K]), so only the last axis is expanded.
    """
    expanded = e8m0_to_float32(scale) if scale.dtype == torch.float8_e8m0fnu else scale.float()
    return expanded.repeat_interleave(block, dim=-1)[..., :size]


def fp8_gemm(x, x_scale, weight, weight_scale, scale_dtype=torch.float32):
    """FP8 activations x FP8 weights, fp32 accumulate, bf16 out.

    Both sides use 128-element blocks along K
    (inference/kernel.py::fp8_gemm_kernel).
    """
    activations = x.float() * _expand_act_scale(x_scale, 128, x.size(-1))
    w = dequantize_fp8(weight, weight_scale) if weight.dtype == torch.float8_e4m3fn else weight.float()
    return (activations @ w.t()).to(torch.bfloat16)


def fp4_gemm(x, x_scale, weight, weight_scale, scale_dtype=torch.float32):
    """FP8 activations x FP4 weights, fp32 accumulate, bf16 out.

    The two sides use different block sizes (inference/kernel.py::fp4_gemm_kernel):
    activations are quantised per 128 elements along K, weights are FP4 with one
    E8M0 scale per 32 elements.
    """
    activations = x.float() * _expand_act_scale(x_scale, 128, x.size(-1))
    w = dequantize_fp4(weight, weight_scale)
    return (activations @ w.t()).to(torch.bfloat16)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
    """Hyper-connection split (inference/kernel.py::hc_split_sinkhorn).

    mixes: [..., (2 + hc) * hc] fp32 -> pre [..., hc], post [..., hc],
    comb [..., hc, hc] (a Sinkhorn-normalised doubly stochastic matrix).
    """
    hc = hc_mult
    lead = mixes.shape[:-1]
    flat = mixes.reshape(-1, (2 + hc) * hc).float()

    pre = torch.sigmoid(flat[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * torch.sigmoid(flat[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (flat[:, 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).view(-1, hc, hc)

    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return (
        pre.view(*lead, hc),
        post.view(*lead, hc),
        comb.view(*lead, hc, hc),
    )


_HADAMARD_CACHE: dict[int, torch.Tensor] = {}


def hadamard_matrix(size: int) -> torch.Tensor:
    """Sylvester Walsh-Hadamard matrix: square, entries +-1, H @ H.T = size * I."""
    if size not in _HADAMARD_CACHE:
        matrix = torch.ones(1, 1)
        while matrix.shape[0] < size:
            matrix = torch.cat(
                [torch.cat([matrix, matrix], dim=1), torch.cat([matrix, -matrix], dim=1)], dim=0
            )
        assert matrix.shape[0] == size, f"Hadamard size must be a power of two, got {size}"
        _HADAMARD_CACHE[size] = matrix
    return _HADAMARD_CACHE[size]


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Randomised-Hadamard rotation used before the indexer's FP4 quantisation.

    The CUDA original calls fast_hadamard_transform.hadamard_transform(
    x, scale=x.size(-1) ** -0.5); this is the same orthonormal transform done
    as an explicit matrix product.
    """
    assert x.dtype == torch.bfloat16, "rotate_activation expects bf16"
    size = x.size(-1)
    matrix = hadamard_matrix(size).to(x.device)
    return ((x.float() @ matrix) * (size ** -0.5)).to(torch.bfloat16)


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale: float, block: int = 64):
    """Sparse attention with a learnable sink (inference/kernel.py::sparse_attn).

    q: [b, m, h, d] bf16, kv: [b, n, d] bf16, attn_sink: [h] fp32,
    topk_idxs: [b, m, topk] int (negative entries are "no position").
    Returns [b, m, h, d] bf16.

    The CUDA kernel walks the top-k positions in blocks of 64 with an online
    softmax; this implementation is equivalent (softmax over the valid
    positions, plus exp(sink - max) in the denominator) and chunks over query
    positions so that a 2k-token prefill does not materialise the gathered KV.
    """
    b, m, h, d = q.shape
    topk = topk_idxs.size(-1)
    sink = attn_sink.float().view(1, 1, h)
    out = torch.empty(b, m, h, d, dtype=torch.bfloat16)

    for start in range(0, m, block):
        stop = min(start + block, m)
        idx = topk_idxs[:, start:stop].long()
        valid = idx >= 0
        flat = idx.clamp(min=0).reshape(b, -1, 1).expand(b, -1, kv.size(-1))
        gathered = kv.gather(1, flat).view(b, stop - start, topk, kv.size(-1)).float()

        scores = torch.einsum("bmhd,bmtd->bmht", q[:, start:stop].float(), gathered) * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))

        row_max = scores.amax(dim=-1, keepdim=True)
        empty = ~torch.isfinite(row_max)                     # a row with no valid position
        row_max = torch.where(empty, torch.zeros_like(row_max), row_max)

        # Official kernel: sum_exp is reduced from FP32 probabilities.
        # Only the P @ V numerator uses the BF16 probability copy.
        probs_fp32 = torch.exp(scores - row_max)
        denom = probs_fp32.sum(dim=-1) + torch.exp(sink - row_max.squeeze(-1))
        probs_bf16 = probs_fp32.to(torch.bfloat16).float()

        chunk = torch.einsum("bmht,bmtd->bmhd", probs_bf16, gathered) / denom.unsqueeze(-1)
        out[:, start:stop] = torch.where(empty, torch.zeros_like(chunk), chunk).to(torch.bfloat16)

    return out
