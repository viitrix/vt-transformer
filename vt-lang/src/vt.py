"""Core DSL: the @vt.NodeChecker decorator and .vt file loading."""

import ast
import inspect
import json
import textwrap
from pathlib import Path
from typing import TypeAlias

import torch

from safetensors import safe_open

from .std_operators import (
    STD_DTYPE_CONSTANTS,
    STD_TENSOR_ATTRS,
    STD_TENSOR_METHODS,
    STD_TORCH_OPS,
)

class VtError(Exception):
    """User-facing error raised while loading or running a .vt file."""

def load_vt_file(path):
    """Execute a .vt file (standard decorated Python) and return {name: NodeChecker}."""
    file = Path(path)
    if not file.is_file():
        raise VtError(f"file not found: {file}")
    try:
        code = compile(file.read_text(), str(file), "exec")
    except SyntaxError as exc:
        raise VtError(f"syntax error in {file} (line {exc.lineno}): {exc.msg}") from exc
    namespace = {"__name__": file.stem, "__file__": str(file.resolve())}
    exec(code, namespace)
    return {
        name: obj
        for name, obj in sorted(namespace.items())
        if isinstance(obj, NodeChecker)
    }

class TensorStore:
    """Tensors from one .safetensors file or a sharded index (.json)."""

    def __init__(self, path):
        path = Path(path)
        if not path.is_file():
            raise VtError(f"file not found: {path}")
        self.path = path
        if path.suffix == ".safetensors":
            try:
                with safe_open(path, framework="pt") as f:
                    self.keys = set(f.keys())
            except Exception as exc:
                raise VtError(f"cannot open safetensors file: {path} ({exc})")
            self.shards = {path: self.keys}
            self._by_key = {key: path for key in self.keys}
        elif path.suffix == ".json":
            try:
                index = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                raise VtError(f"cannot parse index json: {path} ({exc})")
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise VtError(f"not a safetensors index (missing weight_map): {path}")
            self.shards = {}
            self._by_key = {}
            for key, shard in weight_map.items():
                # Shards always live in the same directory as the index json,
                # so ignore any directory prefix carried in the weight_map entry.
                shard_path = (path.parent / Path(shard).name).resolve()
                if not shard_path.is_file():
                    raise VtError(f"shard not found: {shard_path}")
                self.shards.setdefault(shard_path, set()).add(key)
                self._by_key[key] = shard_path
            self.keys = set(weight_map)
        else:
            raise VtError(f"unsupported file type (.safetensors or .json): {path}")

    def load(self, key):
        """Load one tensor by key from the shard that holds it."""
        shard = self._by_key.get(key)
        if shard is None:
            raise VtError(
                f"tensor not found: {key} ({self.path.name} holds {len(self.keys)} tensors)"
            )
        with safe_open(shard, framework="pt") as f:
            return f.get_tensor(key)

    def __repr__(self):
        return f"TensorStore({self.path.name}: {len(self.keys)} tensors)"


_weight_store = None
_activate_store = None


def register_safetensors(weight_path, activate_path=None):
    """Register the weight source and the activation source used for verification.

    Both accept a .safetensors file or a sharded-index .json; a bad file raises.
    activate_path may be omitted when no verification data is used yet.
    """
    global _weight_store, _activate_store
    _weight_store = TensorStore(weight_path)
    _activate_store = TensorStore(activate_path) if activate_path else None


def load_weight(key: str):
    """Load one tensor by key from the registered weight source; raises VtError on failure."""
    if _weight_store is None:
        raise VtError(
            "no weight source registered; "
            "call vt.register_safetensors(...) before loading weights"
        )
    return _weight_store.load(key)


def load_activation(key: str):
    """Load one tensor by key from the registered activation source; raises VtError on failure."""
    if _activate_store is None:
        raise VtError(
            "no activation source registered; "
            "call vt.register_safetensors(..., activate_path=...) before loading activations"
        )
    return _activate_store.load(key)


WeightConfig: TypeAlias = dict[str, str]
"""Weight bindings for a node: {local_name: weight_key}, e.g.
{"embed_tokens": "model.language_model.embed_tokens.weight"}."""

OutputConfig: TypeAlias = list[str]
"""Output verification for a node: [activation_key, ...] by output position.
A single (non-tuple) return uses position 0; a None entry skips that position."""


def _check_weight_config(config) -> WeightConfig:
    """Validate and copy a WeightConfig ({name: weight_key}); None means no weights."""
    if config is None:
        return {}
    if not isinstance(config, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in config.items()
    ):
        raise VtError(f"WeightConfig expects {{name: weight_key}}, got: {config!r}")
    return dict(config)


def _check_output_config(config) -> OutputConfig:
    """Validate and copy an OutputConfig ([activation_key, ...]) at runtime."""
    if not isinstance(config, (list, tuple)) or not all(
        entry is None or isinstance(entry, str) for entry in config
    ):
        raise VtError(f"OutputConfig expects [activation_key, ...], got: {config!r}")
    return list(config)


_ALLOWED_BUILTINS = frozenset({
    "abs", "bool", "enumerate", "float", "int", "len", "list", "max", "min",
    "print", "range", "reversed", "round", "sorted", "sum", "tuple", "zip",
})


def _dotted_name(node):
    """Return 'a.b.c' for an Attribute/Name chain, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _check_std_operators(func):
    """Statically check that func only uses the std_operators Torch subset.

    Parses the function source with ast and reports, as a list of strings:
      - torch.<x> calls/attributes not in STD_TORCH_OPS / STD_DTYPE_CONSTANTS,
      - t.<x>() method calls not in STD_TENSOR_METHODS,
      - t.<x> attribute loads not in STD_TENSOR_ATTRS.
    Plain calls of local names (helpers, parameters) are not restricted.
    """
    try:
        source = textwrap.dedent(inspect.getsource(func))
        # inspect.getsource includes the @decorator lines; drop them so the
        # checker only sees the function body itself.
        source = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("@")
        )
        tree = ast.parse(source)
    except (OSError, TypeError, IndentationError, SyntaxError) as exc:
        raise VtError(
            f"cannot read source of node {func.__name__!r} for the "
            f"std-operators check ({exc})"
        ) from exc

    violations = []
    skip = set()  # ids of Call.func nodes handled by the call check

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                skip.add(id(func))
                dotted = _dotted_name(func) or ""
                if dotted.startswith("torch."):
                    name = dotted[len("torch."):]
                    if name not in STD_TORCH_OPS and name not in STD_DTYPE_CONSTANTS:
                        violations.append(f"torch.{name}(...) is not in std-operators")
                elif func.attr not in STD_TENSOR_METHODS:
                    violations.append(f".{func.attr}(...) is not a std tensor method")
            # plain Name calls (helpers, params, builtins) are unrestricted

        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            if id(node) in skip:
                continue
            dotted = _dotted_name(node) or ""
            if dotted.startswith("torch."):
                name = dotted[len("torch."):]
                if (name not in STD_TORCH_OPS
                        and name not in STD_DTYPE_CONSTANTS
                        and name not in STD_TENSOR_ATTRS):
                    violations.append(f"torch.{name} is not in std-operators")
            elif node.attr not in STD_TENSOR_ATTRS:
                violations.append(f".{node.attr} is not a std tensor attribute")

    return violations


class NodeChecker:
    """Mark a function as a computation node. Use as @vt.NodeChecker(weights, outputs)."""

    def __init__(self, config: WeightConfig | None = None, outputs: OutputConfig | None = None):
        self.weight_bindings = _check_weight_config(config)
        self.output_bindings = _check_output_config(outputs) if outputs else {}
        self.func = None

    def __call__(self, *args, **kwargs):
        if self.func is None:
            # Decoration: the checker was applied to the function below @vt.NodeChecker.
            self.func = args[0]
            violations = _check_std_operators(self.func)
            if violations:
                raise VtError(
                    f"node {self.func.__name__!r} uses operators outside "
                    f"std-operators: {'; '.join(violations)}"
                )
            for name in self.weight_bindings:
                if name in self.func.__globals__:
                    raise VtError(
                        f"weight name {name!r} collides with an existing module name"
                    )
            return self
        return self._invoke(args, kwargs)

    def _invoke(self, args, kwargs):
        """Call the node with weight variables bound, then unbind them to free memory."""
        if self.weight_bindings and _weight_store is None:
            raise VtError(
                "no weight source registered; "
                "call vt.register_safetensors(...) before running nodes"
            )
        globals_ = self.func.__globals__
        bound = []
        try:
            for name, key in self.weight_bindings.items():
                if name in globals_:
                    raise VtError(f"weight name {name!r} is already bound in this module")
                globals_[name] = load_weight(key)
                bound.append(name)
            result = self.func(*args, **kwargs)
        finally:
            # Drop the global bindings so the loaded tensors are freed after the call.
            for name in bound:
                del globals_[name]
        if self.output_bindings:
            self._verify_outputs(result)
        return result

    def _verify_outputs(self, result):
        """Verify node outputs against activations; mismatch raises VtError.

        Temporary activation tensors are freed (del) once checked, whether or
        not every output matches.
        """
        if _activate_store is None:
            raise VtError(
                "no activation source registered; "
                "call vt.register_safetensors(..., activate_path=...) to verify outputs"
            )
        outputs = result if isinstance(result, tuple) else (result,)
        for position, key in enumerate(self.output_bindings):
            if key is None:
                continue
            if position >= len(outputs):
                raise VtError(
                    f"cannot verify output {position}: node returned "
                    f"{len(outputs)} value(s) ({type(result).__name__})"
                )
            actual = outputs[position]
            expected = load_activation(key)  # temporary; deleted below
            try:
                if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape:
                    raise VtError(
                        f"output {position} mismatch vs activation {key!r}: "
                        f"got shape {tuple(actual.shape) if isinstance(actual, torch.Tensor) else type(actual).__name__}, "
                        f"expected {tuple(expected.shape)}"
                    )
                diff = (actual.float() - expected.float()).abs().max().item()
                if not torch.allclose(actual.float(), expected.float(), atol=1e-4, rtol=1e-4):
                    raise VtError(
                        f"output {position} mismatch vs activation {key!r}: "
                        f"max abs diff {diff:.3e} (atol=1e-4, rtol=1e-4)"
                    )
            finally:
                del expected  # free the temporary activation tensor

