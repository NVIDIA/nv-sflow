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

assert_sflow_installed_from() {
    # The suite exists to validate THIS checkout, so prove the sflow on PATH came
    # from it. Not by grepping `sflow --version` for "source : local editable
    # dev" -- that made the runtime-info format load-bearing and rejected install
    # shapes that were the right code (a wheel built from the checkout, a
    # direct-URL install). direct_url.json records the path pip installed FROM
    # whatever the shape, so compare that and do not require "editable".
    #
    # Worth keeping now that the CI workspace and its venv are reused across
    # runs: without it a stale install silently passes as this branch.
    local repo_dir="$1"

    # PYTHONPATH deliberately cleared: setup_under_dev_sflow puts $repo_dir/src on
    # it, and importlib.metadata would then discover src/sflow.egg-info -- which
    # has no direct_url.json -- ahead of the real dist-info in site-packages. The
    # question here is what is INSTALLED, not what is importable, so the source
    # tree must not be on the path. (Otherwise this rejects every checkout that
    # ever ran a plain `pip install -e .` and left an egg-info behind.)
    PYTHONPATH= "$SFLOW_TEST_PYTHON" - "$repo_dir" <<'SFLOW_INSTALL_CHECK'
import json, sys
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse

repo_dir = Path(sys.argv[1]).resolve()
# Either name: nv-sflow (PyPI, internal builds) or sflow (GitHub checkouts).
for name in ("nv-sflow", "sflow"):
    try:
        dist = metadata.distribution(name)
        break
    except metadata.PackageNotFoundError:
        dist = None
if dist is None:
    sys.exit(f"ERROR: sflow is not installed in {sys.executable}.")

url = json.loads(dist.read_text("direct_url.json") or "{}").get("url") or ""
parsed = urlparse(url)
src = Path(unquote(parsed.path)).resolve() if parsed.scheme == "file" else None
if src != repo_dir:
    sys.exit(
        "ERROR: the sflow on PATH was not installed from this checkout.\n"
        f"       Selected Python : {sys.executable}\n"
        f"       Expected repo   : {repo_dir}\n"
        f"       Installed from  : {src or url or 'a package index'}\n"
        f"       Fix with        : {sys.executable} -m pip install -e {repo_dir}"
    )
SFLOW_INSTALL_CHECK
}

assert_sflow_is_published_version() {
    # Release-gate counterpart of assert_sflow_installed_from: the suite must run
    # the wheel users get from the index -- that exact version, and no
    # direct_url.json, which every checkout/path/VCS install records. Without it a
    # stale editable install would pass the gate as the release.
    PYTHONPATH= "$SFLOW_TEST_PYTHON" - "$1" <<'SFLOW_PUBLISHED_CHECK'
import sys
from importlib import metadata

want = sys.argv[1]
try:
    dist = metadata.distribution("nv-sflow")
except metadata.PackageNotFoundError:
    sys.exit(f"ERROR: nv-sflow is not installed in {sys.executable}.")
origin = dist.read_text("direct_url.json")
if dist.version != want or origin:
    sys.exit(
        f"ERROR: expected the published nv-sflow {want} from the index, "
        f"found {dist.version} installed from {origin or 'an index'}."
    )
SFLOW_PUBLISHED_CHECK
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
    # SFLOW_E2E_INSTALL_VERSION (the release gate): test the installed published
    # wheel, so src/ must not shadow it.
    if [ -z "${SFLOW_E2E_INSTALL_VERSION:-}" ]; then
        export PYTHONPATH="$SFLOW_UNDER_DEV_SRC${PYTHONPATH:+:$PYTHONPATH}"
    fi

    SFLOW_WRAPPER_DIR="$(mktemp -d)"
    export SFLOW_WRAPPER_DIR
    cat > "$SFLOW_WRAPPER_DIR/sflow" <<'EOF'
#!/bin/bash

has_sflow_install_override() {
    # The caller already chose how the per-job venv installs sflow. Don't inject
    # our default on top: --sflow-source-path and --sflow-version are mutually
    # exclusive, so injecting either over an existing choice would be rejected.
    for arg in "$@"; do
        case "$arg" in
            --sflow-source-path|--sflow-version) return 0 ;;
        esac
    done
    return 1
}

# Under-dev runs install the local checkout editable into each job's fresh
# per-job venv via --sflow-source-path, so submitted Slurm jobs run the exact
# working tree (no git push / remotely reachable ref required).
# The release gate (SFLOW_E2E_INSTALL_VERSION) tests the published wheel
# instead: jobs install that same version from the index, as a user's would.
if [ "${1:-}" = "batch" ] && ! has_sflow_install_override "$@"; then
    if [ -n "${SFLOW_E2E_INSTALL_VERSION:-}" ]; then
        set -- "$@" --sflow-index-url "$SFLOW_E2E_INDEX_URL" --sflow-version "${SFLOW_E2E_INSTALL_VERSION#[vV]}"
    else
        set -- "$@" --sflow-source-path "$SFLOW_UNDER_DEV_REPO"
    fi
fi

if [ -z "${SFLOW_E2E_INSTALL_VERSION:-}" ]; then
    export PYTHONPATH="$SFLOW_UNDER_DEV_SRC${PYTHONPATH:+:$PYTHONPATH}"
fi
exec "$SFLOW_TEST_PYTHON" -m sflow "$@"
EOF
    chmod +x "$SFLOW_WRAPPER_DIR/sflow"
    export PATH="$SFLOW_WRAPPER_DIR:$PATH"

    if [ -n "${SFLOW_E2E_INSTALL_VERSION:-}" ]; then
        echo "Release gate: testing the published nv-sflow ${SFLOW_E2E_INSTALL_VERSION#[vV]} from $SFLOW_E2E_INDEX_URL, not this checkout"
        sflow --version
        assert_sflow_is_published_version "${SFLOW_E2E_INSTALL_VERSION#[vV]}"
        return
    fi
    echo "Using under-dev sflow from $SFLOW_UNDER_DEV_REPO (ref: $SFLOW_UNDER_DEV_REF)"
    echo "Submitted Slurm jobs install this checkout editable (--sflow-source-path), so uncommitted working-tree changes are included."
    sflow --version
    assert_sflow_installed_from "$repo_dir"
}

cleanup_under_dev_sflow() {
    if [ -n "${SFLOW_WRAPPER_DIR:-}" ]; then
        rm -rf "$SFLOW_WRAPPER_DIR"
    fi
}
