"""Command-line entry: vt <file> [node] [input ...].

Each input is an activation key; the tensor is loaded via vt.load_activation(key)
and passed to the node as a positional argument.
"""

import sys

from . import vt


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if not args:
        sys.exit("usage: vt <file> [node] [input ...]")

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
        inputs = [vt.load_activation(key) for key in args[2:]]
        node = nodes[args[1]]
        result = node(*inputs)
    except vt.VtError as exc:
        sys.exit(f"vt: {exc}")
    except Exception as exc:  # node crashed: the call stack above shows where
        sys.exit(f"vt: {type(exc).__name__}: {exc}")
    print("Output is:")
    print(result)
