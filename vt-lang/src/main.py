"""Command-line entry: vt <file> [node] [input ...] [name=value ...].

Each input is an activation key; the tensor is loaded via vt.load_activation(key)
and passed to the node as a positional argument. A name=value argument is
passed to the node as a keyword argument instead (for template parameters
such as layer=3); the value is converted to int, then float, else kept str.
"""

import sys

from . import vt


def _parse_value(text: str):
    """Convert 'name=value' text to a Python value: int, then float, else str."""
    for convert in (int, float):
        try:
            return convert(text)
        except ValueError:
            pass
    return text


def _split_arguments(args):
    """Split CLI args into positional activation keys and keyword arguments."""
    keys, kwargs = [], {}
    for arg in args:
        name, sep, value = arg.partition("=")
        if sep and name.isidentifier():
            kwargs[name] = _parse_value(value)
        else:
            keys.append(arg)
    return keys, kwargs


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if not args:
        sys.exit("usage: vt <file> [node] [input ...] [name=value ...]")

    try:
        nodes = vt.load_vt_file(args[0])
    except vt.VtError as exc:
        sys.exit(f"vt: {exc}")

    if len(args) == 1:
        for name, fn in nodes.items():
            print(f"{name}  weights={fn.weight_bindings}")
        return

    if args[1] not in nodes:
        sys.exit(f"vt: node not found: {args[1]} ({', '.join(nodes) or 'no nodes'})")

    try:
        input_keys, template_kwargs = _split_arguments(args[2:])
        inputs = [vt.load_activation(key) for key in input_keys]
        node = nodes[args[1]]
        result = node(*inputs, **template_kwargs)
    except vt.VtError as exc:
        sys.exit(f"vt: {exc}")
    except Exception as exc:  # node crashed: the call stack above shows where
        sys.exit(f"vt: {type(exc).__name__}: {exc}")
    print("Output is:")
    print(result)
