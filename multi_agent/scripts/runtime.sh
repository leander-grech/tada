#!/bin/sh
# Portable project runtime. Thread policy precedes every Python/Ray import.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
# Workers must reuse this verified environment, not auto-create uv environments.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
# Deterministic child-local ordering policy, before the first Python process.
export PYTHONHASHSEED=0
# Keep Ray's socket paths short; operators may select another temporary directory.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/tada-ray}"
mkdir -p "$RAY_TMPDIR"
runtime=runs/runtime
command=${1:-}
[ "$#" -eq 0 ] || shift
if [ "$command" = setup ]; then
    { [ "$#" -eq 2 ] || [ "$#" -eq 4 ]; } && [ "$1" = --backend ] || { echo 'setup requires --backend cpu|cuda' >&2; exit 2; }
    backend=$2
    native=
    if [ "$#" -eq 4 ]; then
        [ "$3" = --native-wheel ] || { echo "Expected --native-wheel ZIP|WHL" >&2; exit 2; }
        native=$4
    fi
    case "$backend" in cpu|cuda) ;; *) echo 'backend must be cpu or cuda' >&2; exit 2;; esac
    uv python install 3.12
    mkdir -p "$runtime"
    set -- --python 3.12 --extra "$backend" --extra dev --no-install-package flight-simulator
    if [ -f uv.lock ]; then
        set -- --locked "$@"
    fi
    uv sync "$@"
    if [ -n "$native" ]; then
        .venv/bin/python -m scripts.verify_runtime --install-native --native-wheel "$native"
    else
        .venv/bin/python -m scripts.verify_runtime --install-native
    fi
    printf '%s\n' "$backend" > "$runtime/backend"
    exec .venv/bin/python -m scripts.verify_runtime --device cpu --expect-backend "$backend"
fi
[ -f "$runtime/backend" ] || { echo 'Run setup --backend cpu|cuda first' >&2; exit 2; }
# No implicit uv sync: preserve the selected backend and installed Ray.
export UV_NO_SYNC=1
.venv/bin/python -m scripts.verify_runtime --identity-only
slot=runs/run-slot
case "$command" in
    runtime-check) exec uv run --no-sync python -m scripts.verify_runtime "$@" ;;
    test)
        if [ -n "${TADA_RUN_DIR:-}" ]; then
            exec uv run --no-sync test "$@"
        fi
        exec uv run --no-sync python -m training.process --root "$slot" exec -- uv run --no-sync test "$@" ;;
    status|stop) exec uv run --no-sync python -m training.process --root "$slot" "$command" "$@" ;;
    check) uv run --no-sync ruff format --check . && uv run --no-sync ruff check . && exec uv run --no-sync pyright ;;
    run) exec uv run --no-sync python -m training.process --root "$slot" launch -- uv run --no-sync "$@" ;;
    *) echo 'Expected setup, test, check, runtime-check, run, status, or stop' >&2; exit 2 ;;
esac
