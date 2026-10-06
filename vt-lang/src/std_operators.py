"""Lean standard Torch operator subset for @vt.NodeChecker functions.

A node must read like a short sequence of primitive tensor ops so it can be
mapped 1:1 onto backend kernels. Only the operators listed here may appear
inside a checker function; everything else — especially high-level composites
like torch.einsum, torch.nn modules, jit/compile — is denied by default.

Lists are plain arrays of operator names relative to ``torch`` (or to a
tensor, for the method allowlists). TORCH_ALIASES resolves names that live
elsewhere (e.g. "linear" is really torch.nn.functional.linear).
"""

# Canonical VT operator name -> real dotted attribute path under torch.
TORCH_ALIASES = {
    "gelu": "nn.functional.gelu",
    "linear": "nn.functional.linear",
    "rms_norm": "nn.functional.rms_norm",
    "silu": "nn.functional.silu",
}

# ---------------------------------------------------------------------------
# torch.<name>(...): pointwise arithmetic / activations
# ---------------------------------------------------------------------------
ELEMENTWISE_OPS = [
    "abs",
    "add",
    "clamp",
    "div",
    "exp",
    "gelu",
    "log",
    "mul",
    "neg",
    "pow",
    "reciprocal",
    "rsqrt",
    "sigmoid",
    "silu",
    "sqrt",
    "sub",
    "tanh",
]

# ---------------------------------------------------------------------------
# torch.<name>(...): matrix / embedding primitives
# ---------------------------------------------------------------------------
LINEAR_OPS = [
    "bmm",
    "embedding",
    "linear",        # y = x @ W^T + b, the one composite we allow by design
    "matmul",
]

# ---------------------------------------------------------------------------
# torch.<name>(...): normalization primitives (single fused kernels)
# ---------------------------------------------------------------------------
NORM_OPS = [
    "layer_norm",
    "rms_norm",
    "softmax",
]

# ---------------------------------------------------------------------------
# torch.<name>(...): reductions
# ---------------------------------------------------------------------------
REDUCE_OPS = [
    "argmax",
    "cumsum",
    "max",
    "mean",
    "min",
    "sum",
]

# ---------------------------------------------------------------------------
# torch.<name>(...): shape / layout (no numerics)
# ---------------------------------------------------------------------------
SHAPE_OPS = [
    "cat",
    "gather",
    "index_select",
    "permute",
    "reshape",
    "stack",
    "squeeze",
    "transpose",
    "unsqueeze",
]

# ---------------------------------------------------------------------------
# torch.<name>(...): comparison / selection
# ---------------------------------------------------------------------------
COMPARE_OPS = [
    "maximum",
    "minimum",
    "where",
]

# ---------------------------------------------------------------------------
# torch.<name>(...): tensor creation
# ---------------------------------------------------------------------------
CREATE_OPS = [
    "arange",
    "full",
    "ones",
    "ones_like",
    "zeros",
    "zeros_like",
]

# ---------------------------------------------------------------------------
# Method-only ops: no torch.<name> exists; usable as t.<name>(...).
# ---------------------------------------------------------------------------
METHOD_ONLY_OPS = [
    "contiguous",
    "expand",
    "view",
]

# ---------------------------------------------------------------------------
# Extra tensor methods (t.<name>(...)) beyond the op groups above.
# ---------------------------------------------------------------------------
TENSOR_METHOD_OPS = [
    "float",
    "half",
    "to",
    "masked_fill",
    "clone",
    "detach",
    "item",
    "size",
    "numel",
]

# ---------------------------------------------------------------------------
# Tensor attributes (t.<name>, no call), e.g. x.shape, x.dtype.
# ---------------------------------------------------------------------------
TENSOR_ATTR_OPS = [
    "T",
    "device",
    "dim",
    "dtype",
    "ndim",
    "numel",
    "shape",
    "size",
]

# ---------------------------------------------------------------------------
# dtype constants (values, not called): torch.float, torch.bfloat16, ...
# ---------------------------------------------------------------------------
DTYPE_CONSTANTS = [
    "bfloat16",
    "bool",
    "double",
    "float",
    "float16",
    "half",
    "int",
    "int32",
    "int64",
    "long",
]

# ---------------------------------------------------------------------------
# Explicitly denied high-level composites. Anything not in the allowlists is
# denied by default; this list documents the tempting ones so error messages
# can call them out. "nn" / "nn.functional" are denied as user-written paths;
# the four TORCH_ALIASES entries are the only sanctioned exceptions.
# ---------------------------------------------------------------------------
DENIED_OPS = [
    "einsum",           # arbitrary contraction: hides many primitive ops
    "tensordot",        # ditto
    "inner",            # ditto
    "cross",            # ditto
    "nn",               # torch.nn.* modules are not node-level primitives
    "nn.functional",    # use the torch.* primitive instead (e.g. softmax)
    "jit",
    "compile",
    "vmap",
    "autograd",
    "linalg",
    "fft",
    "special",
    "optim",
    "no_grad",
    "inference_mode",
]

_FUNCTION_GROUPS = {
    "elementwise": ELEMENTWISE_OPS,
    "linear": LINEAR_OPS,
    "norm": NORM_OPS,
    "reduce": REDUCE_OPS,
    "shape": SHAPE_OPS,
    "compare": COMPARE_OPS,
    "create": CREATE_OPS,
}

STD_TORCH_OPS = frozenset(
    name for group in _FUNCTION_GROUPS.values() for name in group
)
"""Allowed ``torch.<name>`` functions; TORCH_ALIASES maps a name to its real
attribute path when the two differ."""

STD_TENSOR_METHODS = frozenset(
    TENSOR_METHOD_OPS
    + METHOD_ONLY_OPS
    + ELEMENTWISE_OPS
    + NORM_OPS
    + REDUCE_OPS
    + SHAPE_OPS
    + COMPARE_OPS
)
"""Allowed ``t.<name>(...)`` tensor methods."""

STD_TENSOR_ATTRS = frozenset(TENSOR_ATTR_OPS)
"""Allowed ``t.<name>`` tensor attributes (no call)."""

STD_DTYPE_CONSTANTS = frozenset(DTYPE_CONSTANTS)
"""Allowed dtype constant names, usable as values (not called)."""

STD_DENIED_OPS = frozenset(DENIED_OPS)
"""Documented high-level composites; denied with a pointed error message."""
