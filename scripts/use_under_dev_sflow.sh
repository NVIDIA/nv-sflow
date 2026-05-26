#!/bin/bash

resolve_under_dev_sflow_ref() {
    local repo_dir="$1"
    local ref

    ref=$(git -C "$repo_dir" rev-parse --abbrev-ref HEAD 2>/dev/null || true)
    if [ -z "$ref" ]; then
        return 1
    fi
    if [ "$ref" = "HEAD" ]; then
        git -C "$repo_dir" rev-parse HEAD 2>/dev/null
    else
        printf '%s\n' "$ref"
    fi
}

assert_under_dev_sflow_editable_install() {
    local repo_dir="$1"

    "$SFLOW_TEST_PYTHON" - "$repo_dir" <<'PY'
import json
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from urllib.parse import unquote, urlparse

repo_dir = Path(sys.argv[1]).resolve()

try:
    dist = importlib_metadata.distribution("sflow")
except importlib_metadata.PackageNotFoundError:
    print(
        "ERROR: sflow is not installed in the selected Python environment.",
        file=sys.stderr,
    )
    print(
        f"       Install this checkout editable first: {sys.executable} -m pip install -e {repo_dir}",
        file=sys.stderr,
    )
    sys.exit(1)

direct_url_text = dist.read_text("direct_url.json")
try:
    direct_url = json.loads(direct_url_text or "{}")
except json.JSONDecodeError:
    direct_url = {}

parsed = urlparse(direct_url.get("url") or "")
install_path = Path(unquote(parsed.path)).resolve() if parsed.scheme == "file" else None
is_editable = bool(direct_url.get("dir_info", {}).get("editable"))

if not is_editable or install_path != repo_dir:
    print(
        "ERROR: full sample tests must run against this checkout installed editable.",
        file=sys.stderr,
    )
    print(f"       Selected Python : {sys.executable}", file=sys.stderr)
    print(f"       Expected repo   : {repo_dir}", file=sys.stderr)
    print(f"       Installed path  : {install_path or 'not a local editable install'}", file=sys.stderr)
    print(
        f"       Fix with        : {sys.executable} -m pip install -e {repo_dir}",
        file=sys.stderr,
    )
    sys.exit(1)
PY
}

setup_under_dev_sflow() {
    local repo_dir="$1"

    if [ -z "$repo_dir" ] || [ ! -d "$repo_dir/src/sflow" ]; then
        echo "ERROR: invalid sflow repo directory: $repo_dir" >&2
        return 1
    fi
    repo_dir="$(cd "$repo_dir" && pwd)"

    if [ -z "${SFLOW_TEST_PYTHON:-}" ]; then
        if [ -x "$repo_dir/.venv/bin/python" ]; then
            SFLOW_TEST_PYTHON="$repo_dir/.venv/bin/python"
        else
            SFLOW_TEST_PYTHON="$(command -v python3 || true)"
        fi
    fi
    if [ -z "$SFLOW_TEST_PYTHON" ] || { [ ! -x "$SFLOW_TEST_PYTHON" ] && ! command -v "$SFLOW_TEST_PYTHON" >/dev/null 2>&1; }; then
        echo "ERROR: unable to find a Python interpreter for under-dev sflow" >&2
        return 1
    fi

    assert_under_dev_sflow_editable_install "$repo_dir"

    if [ -z "${SFLOW_UNDER_DEV_REF:-}" ]; then
        SFLOW_UNDER_DEV_REF="$(resolve_under_dev_sflow_ref "$repo_dir" || true)"
    fi
    if [ -z "$SFLOW_UNDER_DEV_REF" ]; then
        echo "ERROR: unable to resolve the under-dev sflow git ref from $repo_dir" >&2
        return 1
    fi

    export SFLOW_TEST_PYTHON
    export SFLOW_UNDER_DEV_REPO="$repo_dir"
    export SFLOW_UNDER_DEV_SRC="$repo_dir/src"
    export SFLOW_UNDER_DEV_REF
    export PYTHONPATH="$SFLOW_UNDER_DEV_SRC${PYTHONPATH:+:$PYTHONPATH}"

    SFLOW_WRAPPER_DIR="$(mktemp -d)"
    export SFLOW_WRAPPER_DIR
    cat > "$SFLOW_WRAPPER_DIR/sflow" <<'EOF'
#!/bin/bash

has_sflow_version_arg() {
    for arg in "$@"; do
        if [ "$arg" = "--sflow-version" ]; then
            return 0
        fi
    done
    return 1
}

if [ "${1:-}" = "batch" ] && ! has_sflow_version_arg "$@"; then
    set -- "$@" --sflow-version "$SFLOW_UNDER_DEV_REF"
fi

export PYTHONPATH="$SFLOW_UNDER_DEV_SRC${PYTHONPATH:+:$PYTHONPATH}"
exec "$SFLOW_TEST_PYTHON" -m sflow "$@"
EOF
    chmod +x "$SFLOW_WRAPPER_DIR/sflow"
    export PATH="$SFLOW_WRAPPER_DIR:$PATH"

    echo "Using under-dev sflow from $SFLOW_UNDER_DEV_REPO (ref: $SFLOW_UNDER_DEV_REF)"
    if ! git -C "$repo_dir" diff --quiet 2>/dev/null || ! git -C "$repo_dir" diff --cached --quiet 2>/dev/null; then
        echo "WARNING: local sflow checkout has uncommitted changes; submitted Slurm jobs can only install ref '$SFLOW_UNDER_DEV_REF'." >&2
    fi
    local sflow_runtime_info
    sflow_runtime_info="$(sflow --version)"
    printf '%s\n' "$sflow_runtime_info"
    if ! printf '%s\n' "$sflow_runtime_info" | grep -q "source  : local editable dev"; then
        echo "ERROR: sflow runtime source is not local editable dev." >&2
        echo "       full_sample_tests.sh is intended to validate local editable sflow changes." >&2
        return 1
    fi
}

cleanup_under_dev_sflow() {
    if [ -n "${SFLOW_WRAPPER_DIR:-}" ]; then
        rm -rf "$SFLOW_WRAPPER_DIR"
    fi
}
