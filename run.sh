#!/usr/bin/env bash
# Project launcher. Subcommands:
#   ./run.sh front [args...]   launch sglfront HTTP frontend
#   ./run.sh help              show this help
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLFRONT_DIR="${ROOT_DIR}/sglfront"
SGLFRONT_VENV="${SGLFRONT_DIR}/.venv"

usage() {
    cat <<'EOF'
Usage: ./run.sh <command> [args...]

Commands:
  front [args...]    Launch sglfront frontend (python -m sglfront)
                     e.g. ./run.sh front --tokenizer-path Qwen/Qwen3-0.6B --port 1919

  help               Show this help
EOF
}

ensure_venv() {
    if [[ ! -d "${SGLFRONT_VENV}" ]]; then
        echo ">> sglfront venv not found at ${SGLFRONT_VENV}"
        echo ">> creating one with uv (python 3.12) ..."
        (cd "${SGLFRONT_DIR}" && uv venv --python=3.12 && uv pip install -e .)
    fi
}

run_front() {
    ensure_venv
    # shellcheck disable=SC1091
    source "${SGLFRONT_VENV}/bin/activate"
    cd "${SGLFRONT_DIR}"
    exec python -m sglfront "$@"
}

main() {
    local cmd="${1:-help}"
    shift || true
    case "${cmd}" in
        front)  run_front "$@" ;;
        help|-h|--help) usage ;;
        *)
            echo "Unknown command: ${cmd}" >&2
            usage
            exit 1
            ;;
    esac
}

main "$@"
