"""Core DSL: the @vt.NodeChecker decorator and .vt file loading."""

import ast
import builtins
import inspect
import json
import string
import textwrap
from pathlib import Path
from types import ModuleType
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

TemplateConfig: TypeAlias = list[str]
"""Template parameters for a node: [name, ...], e.g. ["layer"].

Each name is a placeholder that may appear in weight keys and output
activation keys as {name} (e.g. "model.layers.{layer}.self_attn.q_proj.weight")
and receives its value per call as a keyword argument of the same name.
Every template name must also be a parameter of the node function: the value
is forwarded to it so nested node calls can reuse it."""


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


def _check_template_config(config) -> TemplateConfig:
    """Validate and copy a TemplateConfig ([name, ...]); empty/None means no templates."""
    if config is None:
        return []
    if not isinstance(config, (list, tuple)) or not all(
        isinstance(name, str) and name.isidentifier() for name in config
    ):
        raise VtError(
            f"TemplateConfig expects a list of parameter names, got: {config!r}"
        )
    names = list(config)
    if len(set(names)) != len(names):
        raise VtError(f"TemplateConfig has duplicate names: {names!r}")
    return names


def _key_placeholders(key: str) -> set[str]:
    """Placeholder names used in a template key, e.g. '{layer}' -> {'layer'}."""
    return {
        field
        for _, field, _, _ in string.Formatter().parse(key)
        if field
    }


_ALLOWED_BUILTINS = frozenset({
    "abs", "bool", "enumerate", "float", "int", "len", "list", "max", "min",
    "print", "range", "reversed", "round", "sorted", "sum", "tuple", "zip",
})
"""Builtins a node function may use; any other builtin name is a violation.

Checked statically by _check_std_operators. Names bound by the .vt file or
by the function itself (shadowing a builtin) count as local and are exempt."""

_PYTHON_BUILTINS = frozenset(dir(builtins))
"""All names Python's builtins module exposes, including dunders."""


def _disallowed_builtin(name: str, exempt: frozenset[str] | set[str]) -> str | None:
    """Violation message if name refers to a builtin outside _ALLOWED_BUILTINS.

    exempt holds every name the .vt file or the function itself binds
    (module globals, parameters, assigned locals); those shadow the builtin
    and are treated as local, not restricted.
    """
    if name in _ALLOWED_BUILTINS or name not in _PYTHON_BUILTINS or name in exempt:
        return None
    return f"{name} is not an allowed builtin (add it to _ALLOWED_BUILTINS if needed)"


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


_call_stack: list[str] = []
"""Names of NodeChecker functions currently executing; drives call-stack output."""


def _stack_prefix() -> str:
    """Indentation matching the current node-call depth."""
    return "  " * len(_call_stack)


def _describe_value(value) -> str:
    """Short human-readable description of one call argument."""
    if isinstance(value, torch.Tensor):
        return f"Tensor{tuple(value.shape)}"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}[{len(value)}]"
    return f"{type(value).__name__}({value!r})" if value is not None else "None"


def _describe_args(args, kwargs) -> str:
    """Render a call's arguments for the call-stack line."""
    parts = [_describe_value(a) for a in args]
    parts += [f"{k}={_describe_value(v)}" for k, v in kwargs.items()]
    return ", ".join(parts)


def _check_std_operators(func):
    """Statically check that func only uses the std_operators Torch subset.

    Parses the function source with ast and reports, as a list of strings:
      - torch.<x> calls/attributes not in STD_TORCH_OPS / STD_DTYPE_CONSTANTS,
      - t.<x>() method calls not in STD_TENSOR_METHODS,
      - t.<x> attribute loads not in STD_TENSOR_ATTRS,
      - builtin names (called or referenced) not in _ALLOWED_BUILTINS.
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

    # Names of non-torch modules imported in the .vt file (e.g. vt_lang).
    # Calls/attributes reached through them, like vt.load_weight(...), are
    # runtime helpers, not tensor ops, so they are exempt; torch and its
    # submodules stay fully checked.
    module_names = {
        name
        for name, value in func.__globals__.items()
        if isinstance(value, ModuleType)
        and getattr(value, "__name__", "") != "torch"
        and not getattr(value, "__name__", "").startswith("torch.")
    }

    # Every name the file or the function binds (globals, parameters,
    # assigned locals, loop/comprehension targets). A builtin name in this
    # set is shadowed locally, so using it is not a builtin violation.
    local_names = set(func.__globals__) | set(inspect.signature(func).parameters)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            local_names.add(node.id)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            callee = node.func
            if isinstance(callee, ast.Attribute):
                skip.add(id(callee))
                dotted = _dotted_name(callee) or ""
                if dotted.partition(".")[0] in module_names:
                    continue
                if dotted.startswith("torch."):
                    name = dotted[len("torch."):]
                    if name not in STD_TORCH_OPS and name not in STD_DTYPE_CONSTANTS:
                        violations.append(f"torch.{name}(...) is not in std-operators")
                elif callee.attr not in STD_TENSOR_METHODS:
                    violations.append(f".{callee.attr}(...) is not a std tensor method")
            elif isinstance(callee, ast.Name):
                skip.add(id(callee))
                violation = _disallowed_builtin(callee.id, local_names)
                if violation:
                    violations.append(violation)
            # plain Name calls of local names (helpers, params) are unrestricted

        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if id(node) in skip:
                continue
            violation = _disallowed_builtin(node.id, local_names)
            if violation:
                violations.append(violation)

        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            if id(node) in skip:
                continue
            dotted = _dotted_name(node) or ""
            if dotted.partition(".")[0] in module_names:
                continue
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
    """Mark a function as a computation node.

    Use as @vt.NodeChecker(weights, outputs, templates); templates is a list
    of placeholder names (default []) expanded per call from keyword args.
    """

    def __init__(
        self,
        config: WeightConfig | None = None,
        outputs: OutputConfig | None = None,
        templates: TemplateConfig | None = None,
    ):
        self.weight_bindings = _check_weight_config(config)
        self.output_bindings = _check_output_config(outputs) if outputs else {}
        self.template_bindings = _check_template_config(templates)
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
            for name in [*self.weight_bindings, *self.template_bindings]:
                if name in self.func.__globals__:
                    kind = "weight name" if name in self.weight_bindings else "template name"
                    raise VtError(
                        f"{kind} {name!r} collides with an existing module name"
                    )
            params = set(inspect.signature(self.func).parameters)
            for name in self.template_bindings:
                if name not in params:
                    raise VtError(
                        f"template name {name!r} must be a parameter of the "
                        f"node function; template values are forwarded to it "
                        f"so nested node calls can reuse them"
                    )
            self._check_template_usage()
            return self
        return self._invoke(args, kwargs)

    def _check_template_usage(self):
        """Cross-check template names against the placeholders used in keys.

        Every placeholder in a weight/output key must be declared in
        templates, and every declared template must be used by at least one
        key — otherwise the declaration is dead and almost certainly a typo.
        """
        used = set()
        for key in self.weight_bindings.values():
            used |= _key_placeholders(key)
        for key in self.output_bindings:
            if key:
                used |= _key_placeholders(key)
        declared = set(self.template_bindings)
        unknown = used - declared
        if unknown:
            raise VtError(
                f"node {self.func.__name__!r} uses template placeholders "
                f"{sorted(unknown)} not declared in templates "
                f"{self.template_bindings!r}"
            )
        unused = declared - used
        if unused:
            raise VtError(
                f"node {self.func.__name__!r} declares templates "
                f"{sorted(unused)} that no weight/output key uses"
            )

    def _template_values(self, kwargs) -> dict:
        """Read template values from call kwargs (left in place for the function)."""
        values = {}
        for name in self.template_bindings:
            if name not in kwargs:
                raise VtError(
                    f"node {self.func.__name__!r} requires template argument "
                    f"{name!r} (declared in templates {self.template_bindings!r})"
                )
            values[name] = kwargs[name]
        return values

    def _invoke(self, args, kwargs):
        """Call the node with weight variables bound, then unbind them to free memory.

        Template arguments (declared in templates) are read from the call's
        keyword arguments and expanded into the weight/output keys; they stay
        in kwargs and are forwarded to the function, so its body can use the
        layer index and pass it on to nested node calls.
        """
        if self.weight_bindings and _weight_store is None:
            raise VtError(
                "no weight source registered; "
                "call vt.register_safetensors(...) before running nodes"
            )
        name = self.func.__name__
        template_values = self._template_values(kwargs)
        print(f"{_stack_prefix()}call -> {name}({_describe_args(args, kwargs)})")
        _call_stack.append(name)
        globals_ = self.func.__globals__
        bound = []
        try:
            for weight_name, key in self.weight_bindings.items():
                if weight_name in globals_:
                    raise VtError(
                        f"weight name {weight_name!r} is already bound in this module"
                    )
                globals_[weight_name] = load_weight(key.format(**template_values))
                bound.append(weight_name)
            result = self.func(*args, **kwargs)
            if self.output_bindings:
                self._verify_outputs(result, template_values)
                print(f"{_stack_prefix()}verify OK <- {name}")
        except Exception as exc:
            print(f"{_stack_prefix()}error <- {name}: {type(exc).__name__}: {exc}")
            raise
        finally:
            # Drop the global bindings so the loaded tensors are freed after the call.
            for weight_name in bound:
                del globals_[weight_name]
            _call_stack.pop()
        return result

    def _verify_outputs(self, result, template_values):
        """Verify node outputs against activations; mismatch raises VtError.

        Output activation keys are template-expanded with the call's template
        values before loading. Temporary activation tensors are freed (del)
        once checked, whether or not every output matches.
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
            key = key.format(**template_values)
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

