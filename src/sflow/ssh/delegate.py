# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run an ordinary sflow CLI command on an SSH host.

The delegate deliberately knows nothing about Slurm, Docker, Kubernetes, or the
local backend.  It stages the controller inputs, starts the same CLI remotely,
and copies the controller-side logs back.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import logging
import os
import platform
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import typer
import yaml

from sflow.utils.distribution import DIST_NAME, installed_distribution
from sflow.core.artifact_registry import resolve_file_like_uri_to_path
from sflow.ssh.invocation import Invocation, argv_to_invocation
from sflow.ssh.receipt import initial_receipt, write_receipt
from sflow.ssh.transport import DEFAULT_TRANSPORT, Transport
from sflow.ssh.progress import configure_logging as configure_ssh_logging, log

_SSH_OPTIONS_WITH_VALUE = {
    "-B",
    "-b",
    "-c",
    "-D",
    "-E",
    "-e",
    "-F",
    "-I",
    "-i",
    "-J",
    "-L",
    "-l",
    "-m",
    "-O",
    "-o",
    "-p",
    "-R",
    "-S",
    "-W",
    "-w",
}
_INCOMPATIBLE_SSH_OPTIONS = {
    "-D",
    "-f",
    "-L",
    "-N",
    "-n",
    "-O",
    "-R",
    "-W",
    "-t",
}
_DELEGATE_OPTIONS = {
    "--ssh",
    "--ssh-follow",
    "--ssh-fetch",
    "--ssh-remote-root",
    "--ssh-tty",
}
_IGNORED_INPUT_DIRS = {".git", ".sflow", ".venv", "__pycache__", "sflow_output"}
# The remote session file `sflow batch --submit` records job ids in, and
# ssh/remote_helper.py reads to cancel them. The helper runs standalone and
# cannot import this; tests/unit/test_cli_ssh_delegate.py asserts they match.
SUBMITTED_JOBS_FILE = "submitted-jobs"
_SSH_FAILURE_HINT_SHOWN = False


def _workflow_tag(input_files: list[Path]) -> str | None:
    """Derive a short recipe label (``dynamo``, ``trtllm``) from the first input file.

    Lets ``ssh-<tag>-<stamp>-<hex>`` runs be told apart by recipe when several
    invocations land in the same output directory.
    """
    if not input_files:
        return None
    stem = input_files[0].stem
    tag = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return tag or None


def _invocation_id(input_files: list[Path] | None = None) -> str:
    """Name one SSH execution identically on both hosts.

    The UTC stamp sorts chronologically for a human browsing the local output
    dir; the random suffix keeps concurrent same-second invocations apart
    (PIDs and wall clocks alone collide across clients). The optional recipe
    tag lets runs from different workflows be told apart at a glance.
    """
    tag = _workflow_tag(input_files or [])
    prefix = f"ssh-{tag}-" if tag else "ssh-"
    return f"{prefix}{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{secrets.token_hex(4)}"


def ssh_argv(connection: str) -> list[str]:
    """Normalize ``host -p 22`` and conventional ``-p 22 host`` alike."""
    try:
        tokens = shlex.split(connection)
    except ValueError as exc:
        raise typer.BadParameter(f"invalid --ssh value: {exc}") from exc
    options: list[str] = []
    destination: str | None = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("-"):
            options.append(token)
            option = token[:2]
            if option == "-t":
                raise typer.BadParameter(
                    "SSH option -t would break the payload upload and output "
                    "fetch, which need byte-clean channels; use --ssh-tty "
                    "always to request a PTY for the remote run itself"
                )
            if option in _INCOMPATIBLE_SSH_OPTIONS:
                raise typer.BadParameter(
                    f"SSH option {option} is incompatible with delegated command/file transfer"
                )
            if option in _SSH_OPTIONS_WITH_VALUE and token == option:
                index += 1
                if index >= len(tokens):
                    raise typer.BadParameter(f"SSH option {option} requires a value")
                options.append(tokens[index])
                if option == "-o" and tokens[index].split("=", 1)[0].lower() in {
                    "remotecommand",
                    "requesttty",
                    "sessiontype",
                    "stdinnull",
                }:
                    raise typer.BadParameter(
                        f"SSH option -o {tokens[index]} is incompatible with delegation"
                    )
            elif option == "-o" and token[2:].split("=", 1)[0].lower() in {
                "remotecommand",
                "requesttty",
                "sessiontype",
                "stdinnull",
            }:
                raise typer.BadParameter(
                    f"SSH option {token} is incompatible with delegation"
                )
        elif destination is None:
            destination = token
        else:
            raise typer.BadParameter(
                "--ssh accepts connection options only, not a remote command"
            )
        index += 1
    if destination is None:
        raise typer.BadParameter("--ssh must include a host, for example user@login")
    return ["ssh", *options, destination]


def _tty_argv(ssh: list[str], mode: str, command: str) -> list[str]:
    """Return the SSH argv for the execution channel, with an optional PTY.

    Only this channel may carry a PTY: the payload upload writes a tarball on
    stdin and the fetch reads one from stdout, so a PTY there corrupts them.
    ``compose`` also writes YAML on stdout, so ``auto`` leaves it alone and only
    an explicit ``always`` opts it in.
    """
    if mode == "always":
        allocate = True
    elif mode == "never" or command == "compose":
        allocate = False
    else:
        allocate = _is_a_tty(sys.stdin) and _is_a_tty(sys.stdout)
    # -tt, not -t: sflow's own stdin is usually not a terminal (pipes, CI), and
    # plain -t silently declines the PTY in that case.
    return [*ssh[:-1], "-tt", ssh[-1]] if allocate else ssh


def _is_a_tty(stream: object) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError, OSError):
        return False


_STAGES = 4
_STAGED_FACTS = 5


def _stage(number: int, description: str) -> None:
    """Announce one delegation stage; stdout stays reserved for the command."""
    log(logging.INFO, "[ssh %s/%s] %s...", number, _STAGES, description)


def _fact(label: str, value: str) -> None:
    log(logging.INFO, "    %s: %s", label, value)


def _staged_report(
    staged: dict[Path, Path], workspace: Path, remote_work: str
) -> tuple[list[tuple[str, str]], int]:
    """Map a bounded sample of staged inputs to the remote paths they became.

    ``staged`` values are already relative to the staging root, which is what
    the remote work dir unpacks to.
    """
    pairs = [
        (
            str(
                local.relative_to(workspace)
                if local.is_relative_to(workspace)
                else local
            ),
            f"{remote_work}/{relative}",
        )
        for local, relative in staged.items()
    ]
    return pairs[:_STAGED_FACTS], max(0, len(pairs) - _STAGED_FACTS)


def _tty_mode(mode: str, raw_args: list[str]) -> str:
    """Let ``--tui`` win over an explicit ``never``; it cannot render on a pipe."""
    if "--tui" in raw_args and mode == "never":
        log(
            logging.WARNING,
            "--tui needs a remote PTY to render; ignoring --ssh-tty never.",
        )
        return "always"
    return mode


def _cancel_remote(
    ssh: list[str],
    helper: str,
    session: str,
    *,
    transport: Transport = DEFAULT_TRANSPORT,
) -> bool:
    """Stop the remote controller, reporting whether it is confirmed gone.

    Killing the local SSH client only closes the channel: without this the
    remote controller (and anything it allocated) keeps running.
    """
    log(logging.INFO, "Cancelling the remote run...")
    try:
        result = transport.run_to_stderr(
            ssh,
            f"exec python3 {shlex.quote(helper)} cancel {shlex.quote(session)}",
        )
    except KeyboardInterrupt:
        # A second Ctrl-C means the user is done waiting; the remote state is
        # then genuinely unknown.
        log(logging.WARNING, "Stopped waiting for the remote cancel to confirm.")
        return False
    except OSError as exc:
        log(logging.ERROR, "Could not reach the host to cancel: %s", exc)
        return False
    if result.returncode == 255:
        ssh_failure_hint(ssh)
    return result.returncode == 0


def _resolve_follow(follow: str, *, live_stream: bool) -> str:
    """Pick the display mode so exactly one writer owns the local terminal.

    Over a PTY the remote sflow sees a terminal, keeps task logs streaming
    instead of offloading them to files, and may render a live TUI -- a second
    channel printing log tails or heartbeats on top of that interleaves
    mid-render (garbled, overlapping output). Without a PTY the remote offloads
    task logs to files, so the poller is the only way to see them while the
    run is attached.

    An explicit ``--ssh-follow logs``/``status`` cannot be honored once a PTY
    is already live-streaming everything: the tail/heartbeat would add no
    information (the PTY already shows it) while corrupting the terminal, so
    it is downgraded to ``none`` with a warning rather than honored verbatim.
    """
    if follow == "auto":
        return "none" if live_stream else "logs"
    if live_stream and follow in {"logs", "status"}:
        log(
            logging.WARNING,
            f"A PTY is already streaming the run live; ignoring "
            f"--ssh-follow {follow} to avoid two writers on the terminal.",
        )
        return "none"
    return follow


def ssh_failure_hint(ssh: list[str]) -> None:
    global _SSH_FAILURE_HINT_SHOWN
    if _SSH_FAILURE_HINT_SHOWN:
        return
    _SSH_FAILURE_HINT_SHOWN = True
    options: list[str] = []
    index = 1
    while index < len(ssh) - 1:
        token = ssh[index]
        if token == "-M":
            index += 1
            continue
        if token == "-S":
            index += 2
            continue
        if token.startswith("-S"):
            index += 1
            continue
        if token == "-o" and index + 1 < len(ssh) - 1:
            value = ssh[index + 1]
            if value.split("=", 1)[0].lower() in {
                "batchmode",
                "controlmaster",
                "controlpath",
                "controlpersist",
            }:
                index += 2
                continue
            options.extend((token, value))
            index += 2
            continue
        if token.startswith("-o") and token[2:].split("=", 1)[0].lower() in {
            "batchmode",
            "controlmaster",
            "controlpath",
            "controlpersist",
        }:
            index += 1
            continue
        options.append(token)
        index += 1
    destination = ssh[-1]
    master = shlex.join(
        [
            "ssh",
            *options,
            "-MNf",
            "-o",
            "BatchMode=no",
            "-o",
            "ControlMaster=yes",
            "-o",
            "ControlPersist=24h",
            "-o",
            "ControlPath=%d/.ssh/sflow-%C",
            destination,
        ]
    )
    reuse = shlex.join(
        [
            *options,
            "-o",
            "BatchMode=yes",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPath=%d/.ssh/sflow-%C",
            destination,
        ]
    )
    log(
        logging.ERROR,
        "SSH could not establish a connection (exit 255). If this host requires "
        "MFA, run this command and complete MFA:",
    )
    # Shell commands must survive copy/paste even on a narrow terminal.
    typer.echo(
        f"  {master}\n"
        "Then rerun sflow using:\n"
        f"  --ssh {shlex.quote(reuse)}",
        err=True,
    )
    log(
        logging.ERROR,
        "Exit 255 can also mean a network, DNS, host-key, or SSH configuration "
        "failure. Documentation: "
        "https://nvidia.github.io/nv-sflow/docs/user/cli#run-through-ssh",
    )


def predispatch(argv: list[str] | None = None) -> None:
    """Delegate raw argv before Typer resolves remote paths on the local host."""
    invocation = argv_to_invocation(list(argv) if argv is not None else sys.argv[1:])
    if invocation is not None:
        delegate(invocation)


def _without_options(args: list[str], names: set[str]) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        name = token.split("=", 1)[0]
        if name in names:
            if "=" not in token:
                index += 1
            index += 1
            continue
        result.append(token)
        index += 1
    return result


def _resolve_remote_environment(
    ssh: list[str], value: str | None, *, transport: Transport = DEFAULT_TRANSPORT
) -> tuple[str, str]:
    script = (
        "import json,os,platform,sys; "
        "sys.version_info >= (3,10) or sys.exit('sflow SSH requires Python 3.10+'); "
        "value = sys.argv[1] if len(sys.argv) > 1 else "
        "os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')), 'sflow/ssh'); "
        "print(json.dumps({'root': os.path.abspath(os.path.expanduser(value)), "
        "'tag': '-'.join((sys.implementation.cache_tag or 'python', "
        "platform.system(), platform.machine()))}))"
    )
    command = f"python3 -c {shlex.quote(script)}"
    if value is not None:
        command += " " + shlex.quote(value)
    result = transport.capture(ssh, command)
    try:
        resolved = json.loads(result.stdout)
        root = resolved["root"]
        runtime_tag = resolved["tag"]
    except (json.JSONDecodeError, KeyError, TypeError):
        root = runtime_tag = ""
    if (
        result.returncode != 0
        or not isinstance(root, str)
        or not root.startswith("/")
        or not isinstance(runtime_tag, str)
        or not runtime_tag
    ):
        if result.returncode == 255:
            ssh_failure_hint(ssh)
        raise typer.BadParameter(
            result.stderr.strip() or "could not resolve --ssh-remote-root"
        )
    return root, runtime_tag


def _resolve_remote_root(
    ssh: list[str], value: str | None, *, transport: Transport = DEFAULT_TRANSPORT
) -> str:
    return _resolve_remote_environment(ssh, value, transport=transport)[0]


def _external_relative(path: Path, workspace: Path) -> Path:
    try:
        return path.relative_to(workspace)
    except ValueError:
        digest = hashlib.sha256(str(path.parent).encode()).hexdigest()[:10]
        return Path(".sflow-inputs") / digest / path.name


def _copy_input(path: Path, workspace: Path, work_stage: Path) -> Path:
    if path.is_symlink():
        raise typer.BadParameter(f"refusing to upload symlink input: {path}")
    path = path.resolve()
    if path.is_dir() and path.name in _IGNORED_INPUT_DIRS:
        raise typer.BadParameter(f"refusing to upload ignored directory: {path}")
    relative = _external_relative(path, workspace)
    destination = work_stage / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        symlink = None
        for root, dirs, files in os.walk(path):
            dirs[:] = [name for name in dirs if name not in _IGNORED_INPUT_DIRS]
            symlink = next(
                (
                    Path(root) / name
                    for name in [*dirs, *files]
                    if (Path(root) / name).is_symlink()
                ),
                None,
            )
            if symlink is not None:
                break
        if symlink is not None:
            raise typer.BadParameter(
                f"refusing to follow symlink inside uploaded directory: {symlink}"
            )
        shutil.copytree(
            path,
            destination,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(*_IGNORED_INPUT_DIRS),
        )
    elif path.is_file():
        shutil.copy2(path, destination)
    else:
        raise FileNotFoundError(path)
    return relative


def _directory_configs(path: Path) -> list[Path]:
    """Every YAML under an uploaded input directory.

    Not ``cli/batch._scan_sflow_yamls``: that one only looks one level deep and
    keeps files with a top-level ``workflow`` key, because it selects recipes to
    *run*. This selects files to *upload*, so nested directories and includes
    that are only fragments of a recipe have to come along too.
    """
    configs: list[Path] = []
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(name for name in dirs if name not in _IGNORED_INPUT_DIRS)
        configs.extend(
            Path(root) / name
            for name in sorted(files)
            if Path(name).suffix.lower() in {".yaml", ".yml"}
        )
    return configs


def _csv_configs(path: Path) -> list[Path]:
    """The configs a bulk CSV names, read exactly the way ``sflow batch`` reads it.

    A CSV this cannot parse is not reported here: it is the remote run's job to
    reject it, with the message it would have given locally.
    """
    from sflow.cli.batch import read_bulk_csv, resolve_row_files

    try:
        _, rows = read_bulk_csv(path)
    except (OSError, csv.Error, ValueError):
        return []
    return [config for row in rows for config in resolve_row_files(row, path.parent, [])]


def _artifact_uris(config: Path, artifact_overrides: list[str] | None = None) -> list[str]:
    """Every non-inline ``file``/``fs`` artifact URI a recipe file references.

    Read from the raw YAML rather than a loaded config: staging has to work for
    recipes this host cannot fully resolve (remote-only paths, backends it does
    not have), which is exactly what the loader refuses to do. ``-a``/``--artifact``
    overrides are still applied first, via the same merge ``ConfigLoader`` uses,
    so a CLI override replaces the placeholder URI instead of getting ignored.
    """
    try:
        document = yaml.safe_load(config.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []
    if artifact_overrides:
        from sflow.config.loader import ConfigLoader

        try:
            ConfigLoader()._apply_artifact_overrides(document, artifact_overrides)
        except ValueError:
            pass  # surfaced properly by _validate_recipe_locally
    found: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            uri = value.get("uri")
            if isinstance(uri, str) and value.get("content") is None:
                found.append(uri)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(document)
    return found


def _raw_uri_path(uri: str) -> str | None:
    """The unresolved path a ``file``/``fs`` URI names, if it names one at all.

    ``${{ ... }}`` references are not paths yet -- they are expanded during the
    run, on the host that owns them.
    """
    parsed = urlparse(uri)
    if parsed.scheme.lower() not in {"file", "fs"}:
        return None
    raw = unquote((parsed.netloc or "") + (parsed.path or ""))
    return raw if raw and "${{" not in raw else None


def _relative_uri_path(uri: str, workspace: Path) -> Path | None:
    """Resolve a workspace-relative artifact URI; absolute ones are remote paths."""
    raw = _raw_uri_path(uri)
    if raw is None or Path(raw).is_absolute():
        return None
    return resolve_file_like_uri_to_path(uri, workspace_dir=workspace).resolve()


def _local_artifacts(
    config: Path, workspace: Path, artifact_overrides: list[str] | None = None
) -> list[Path]:
    """Relative artifacts a recipe references, which have to be uploaded."""
    found: list[Path] = []
    for uri in _artifact_uris(config, artifact_overrides):
        path = _relative_uri_path(uri, workspace)
        if path is not None and path.exists():
            if not path.is_relative_to(workspace.resolve()):
                raise typer.BadParameter(
                    f"relative artifact {uri!r} resolves outside "
                    f"--workspace-dir: {path}"
                )
            found.append(path)
    return found


def _remote_artifact_paths(
    config: Path, artifact_overrides: list[str] | None = None
) -> list[str]:
    """Absolute file/fs artifact URIs a config references on the remote host.

    Relative artifacts get staged and checked locally by ``_local_artifacts``;
    absolute ones are assumed to already exist on the remote filesystem and
    were never checked anywhere, so a typo only surfaced deep into a remote
    ``sflow run``. Collected here so :func:`_check_remote_artifacts` can
    fail fast, right after the SSH connection is up.
    """
    raw_paths = (_raw_uri_path(uri) for uri in _artifact_uris(config, artifact_overrides))
    return [raw for raw in raw_paths if raw is not None and Path(raw).is_absolute()]


def _check_remote_artifacts(
    ssh: list[str], paths: list[str], *, transport: Transport = DEFAULT_TRANSPORT
) -> None:
    """Fail fast if an absolute artifact path is missing on the remote host."""
    if not paths:
        return
    script = (
        "import os,sys,json; "
        "print(json.dumps([p for p in sys.argv[1:] if not os.path.exists(p)]))"
    )
    result = transport.capture(
        ssh,
        f"python3 -c {shlex.quote(script)} " + " ".join(shlex.quote(p) for p in paths),
    )
    try:
        missing = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        missing = None
    if result.returncode != 0 or not isinstance(missing, list):
        log(
            logging.ERROR,
            f"✗ Could not verify remote artifact paths: "
            f"{result.stderr.strip() or 'unknown error'}",
        )
        raise typer.Exit(code=1)
    if missing:
        log(
            logging.ERROR,
            "✗ Artifact path(s) not found on remote host: " + ", ".join(missing)
            + ". Use --artifact NAME=URI to override the path, or "
            "--skip-artifact-check to skip this pre-check.",
        )
        raise typer.Exit(code=1)


def _source_pyproject() -> Path | None:
    """Locate this checkout's pyproject.toml, when running from source."""
    import sflow

    package = Path(sflow.__file__).resolve().parent
    return next(
        (
            parent / "pyproject.toml"
            for parent in package.parents
            if (parent / "pyproject.toml").is_file()
        ),
        None,
    )


def _requirements() -> list[str]:
    """The dependency list bootstrap installs; sourced from install metadata,
    or straight from pyproject.toml so an uninstalled checkout can't drift
    from what a real install would report.
    """
    dist = installed_distribution()
    if dist is not None:
        return sorted(dist.requires or [])
    pyproject = _source_pyproject()
    if pyproject is None:
        return []
    match = re.search(
        r"^dependencies\s*=\s*\[(.*?)^\]",
        pyproject.read_text(encoding="utf-8"),
        re.DOTALL | re.MULTILINE,
    )
    return sorted(re.findall(r'"([^"]+)"', match.group(1))) if match else []


def _version() -> str:
    dist = installed_distribution()
    return dist.version if dist is not None else "0.0+ssh"


def _runtime_fingerprint(
    requirements: list[str], runtime_tag: str | None = None
) -> str:
    pyproject = _source_pyproject()
    metadata_bytes = pyproject.read_bytes() if pyproject is not None else b""
    tag = runtime_tag or "-".join(
        (
            sys.implementation.cache_tag or "python",
            platform.system(),
            platform.machine(),
        )
    )
    return hashlib.sha256(
        metadata_bytes + b"\0" + tag.encode() + b"\0" + "\n".join(requirements).encode()
    ).hexdigest()[:16]


def _copy_source_project(destination: Path, requirements: list[str]) -> None:
    import sflow

    package = Path(sflow.__file__).resolve().parent
    pyproject = _source_pyproject()
    destination.mkdir(parents=True)
    if pyproject is not None:
        source_root = pyproject.parent
        shutil.copy2(pyproject, destination / "pyproject.toml")
        if (source_root / "README.md").is_file():
            shutil.copy2(source_root / "README.md", destination / "README.md")
    else:
        dependencies = ",\n  ".join(json.dumps(item) for item in requirements)
        (destination / "pyproject.toml").write_text(
            '[build-system]\nrequires = ["setuptools>=69", "wheel"]\n'
            'build-backend = "setuptools.build_meta"\n\n'
            f'[project]\nname = "{DIST_NAME}"\nversion = {json.dumps(_version())}\n'
            f"dependencies = [\n  {dependencies}\n]\n\n"
            "[project.optional-dependencies]\ndev = []\n\n"
            '[project.scripts]\nsflow = "sflow.cli:app"\n\n'
            '[tool.setuptools]\npackage-dir = {"" = "src"}\n\n'
            '[tool.setuptools.packages.find]\nwhere = ["src"]\n',
            encoding="utf-8",
        )
    pyproject = destination / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    marker = "[tool.setuptools_scm]\n"
    if marker in text and "fallback_version" not in text:
        text = text.replace(
            marker,
            marker + f"fallback_version = {json.dumps(_version())}\n",
            1,
        )
        pyproject.write_text(text, encoding="utf-8")
    shutil.copytree(
        package,
        destination / "src" / "sflow",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def _rewrite_staged_paths(
    args: list[str], staged: dict[Path, Path], workspace: Path, remote_work: str
) -> list[str]:
    rewritten: list[str] = []
    for token in args:
        prefix = ""
        value = token
        if token.startswith("-") and "=" in token:
            prefix, value = token.split("=", 1)
            prefix += "="
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = workspace / candidate
        relative = staged.get(candidate.resolve())
        rewritten.append(prefix + f"{remote_work}/{relative}" if relative else token)
    return rewritten


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not (root / member.name).resolve().is_relative_to(root):
                raise RuntimeError(
                    f"unsafe path returned by remote host: {member.name}"
                )
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(
                    f"unsafe file type returned by remote host: {member.name}"
                )
        archive.extractall(destination)


def _ensure_directory(path: Path) -> None:
    """Create a directory without ever following an existing symlink."""
    missing: list[Path] = []
    current = path
    while not current.exists() and not current.is_symlink():
        missing.append(current)
        current = current.parent
    if current.is_symlink():
        raise RuntimeError(f"refusing to write through destination symlink: {current}")
    if current.exists() and not current.is_dir():
        raise RuntimeError(f"fetch destination parent is not a directory: {current}")
    for directory in reversed(missing):
        directory.mkdir()


def _merge_tree(source: Path, destination: Path) -> None:
    """Merge fetched files with atomic per-file publication and no symlink escape."""
    _ensure_directory(destination)
    for source_path in sorted(source.rglob("*")):
        relative = source_path.relative_to(source)
        target = destination / relative
        if source_path.is_dir():
            _ensure_directory(target)
            continue
        if not source_path.is_file() or source_path.is_symlink():
            raise RuntimeError(f"unsafe fetched file: {relative}")
        _copy_file_atomic(source_path, target)


def _copy_file_atomic(source: Path, target: Path) -> None:
    _ensure_directory(target.parent)
    if target.is_symlink():
        raise RuntimeError(f"refusing to replace destination symlink: {target}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)


def snapshot(
    ssh: list[str],
    helper: str,
    remote_output: str,
    previous: str | None,
    unchanged: int,
    *,
    connection_hint: bool = False,
    transport: Transport = DEFAULT_TRANSPORT,
) -> tuple[str | None, int]:
    try:
        result = transport.capture(
            ssh,
            f"python3 {shlex.quote(helper)} snapshot {shlex.quote(remote_output)}",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result = subprocess.CompletedProcess(
            [], 1, "", f"remote log check failed: {exc}"
        )
    if connection_hint and result.returncode == 255:
        ssh_failure_hint(ssh)
    snapshot = (
        result.stdout.strip() or result.stderr.strip() or "(remote log check failed)"
    )
    unchanged = unchanged + 5 if snapshot == previous else 0
    suffix = f", unchanged for {unchanged}s" if unchanged else ""
    log(logging.INFO, "[ssh] remote log snapshot%s", suffix)
    # Remote output has its own severity/layout; never filter or rewrap it.
    typer.echo(snapshot, err=True)
    if connection_hint and result.returncode != 0:
        raise typer.Exit(code=result.returncode)
    return snapshot, unchanged


def fetch(
    ssh: list[str],
    helper: str,
    remote_output: str,
    mode: str,
    destination: Path,
    *,
    transport: Transport = DEFAULT_TRANSPORT,
) -> bool:
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive_file:
        command = (
            f"python3 {shlex.quote(helper)} pack {shlex.quote(remote_output)} "
            f"{shlex.quote(mode)}"
        )
        try:
            result = transport.run_to_file(ssh, command, archive_file)
        except OSError as exc:
            log(logging.WARNING, "Failed to fetch remote output: %s", exc)
            return False
        if result.returncode != 0:
            if result.returncode == 255:
                ssh_failure_hint(ssh)
            log(logging.WARNING, "Failed to fetch remote output.")
            return False
        archive_file.flush()
        try:
            with tempfile.TemporaryDirectory(prefix="sflow-fetch-") as temporary:
                staged = Path(temporary)
                _safe_extract(Path(archive_file.name), staged)
                _merge_tree(staged, destination)
        except (OSError, RuntimeError, tarfile.TarError) as exc:
            log(logging.WARNING, "Unsafe or incomplete remote output: %s", exc)
            return False
    return True


def _receipt_status(return_code: int) -> str:
    return {0: "completed", 130: "unknown", 255: "unknown"}.get(return_code, "failed")


def _receipt_outcome(command: str, return_code: int, args: list[str]) -> dict[str, str]:
    execution_status = _receipt_status(return_code)
    if command == "batch" and return_code == 0 and "--submit" in args:
        return {
            "status": "submitted",
            "execution_status": execution_status,
            "backend_status": "submitted",
        }
    return {"status": execution_status, "execution_status": execution_status}


def _validate_recipe_locally(
    config_files: list[Path],
    variable_overrides: list[str] | None,
    artifact_overrides: list[str] | None,
    missable_tasks: list[str] | None,
) -> None:
    """Reject a malformed recipe before paying for SSH setup.

    Runs the same ``ConfigLoader`` schema/expression validation ``sflow run``
    itself would hit first, so a bad recipe fails in milliseconds instead of
    only surfacing after the remote host is provisioned and ``--dry-run`` runs
    there. Skips anything that needs the actual target host (absolute
    artifact existence is checked separately by ``_check_remote_artifacts``
    once the SSH connection is up; backend resolution still surfaces
    remotely).
    """
    if not config_files:
        return
    from sflow.config.loader import ConfigLoader

    try:
        ConfigLoader().load_configs(
            config_files, variable_overrides, artifact_overrides, missable_tasks
        )
    except (ValueError, FileNotFoundError) as exc:
        log(logging.ERROR, "✗ Configuration error: %s", exc)
        raise typer.Exit(code=1) from None


def delegate(invocation: Invocation, transport: Transport | None = None) -> None:
    """Delegate one parsed CLI invocation, then exit with the remote status."""
    # Imported here, not at module scope: sflow.cli imports this module back.
    from sflow.cli.batch import SUBMITTED_JOBS_FILE_ENV
    from sflow.runtime_info import SSH_RUNTIME_ENV, SSH_SOURCE_ENV, install_info

    configure_ssh_logging(invocation.log_level)
    transport = transport or DEFAULT_TRANSPORT
    command = invocation.command
    connection = invocation.connection
    follow = invocation.follow
    fetch_mode = invocation.fetch
    remote_root = invocation.remote_root
    tty = invocation.tty
    workspace_dir = invocation.workspace_dir
    output_dir = invocation.output_dir
    input_files = list(invocation.input_files)
    config_files = invocation.config_files
    variable_overrides = invocation.variable_overrides
    artifact_overrides = invocation.artifact_overrides
    missable_tasks = invocation.missable_tasks
    bulk_input = invocation.bulk_input
    compose_output = invocation.compose_output
    compose_bulk = invocation.compose_bulk
    batch_sbatch_path = invocation.batch_sbatch_path
    batch_runtime_explicit = invocation.batch_runtime_explicit
    if follow not in {"auto", "logs", "status", "none"}:
        raise typer.BadParameter("--ssh-follow must be auto, logs, status, or none")
    if fetch_mode not in {"logs", "all", "none"}:
        raise typer.BadParameter("--ssh-fetch must be logs, all, or none")
    if tty not in {"auto", "always", "never"}:
        raise typer.BadParameter("--ssh-tty must be auto, always, or never")

    _validate_recipe_locally(
        config_files or [], variable_overrides, artifact_overrides, missable_tasks
    )

    ssh = ssh_argv(connection)
    workspace = (workspace_dir or Path.cwd()).resolve()
    run_id = _invocation_id(input_files)
    _stage(1, f"connecting to {ssh[-1]}")
    resolved_root, runtime_tag = _resolve_remote_environment(
        ssh, remote_root, transport=transport
    )
    remote_session = f"{resolved_root}/runs/{run_id}"
    remote_work = f"{remote_session}/work"
    remote_output = f"{remote_session}/output"
    remote_helper = f"{remote_session}/remote_helper.py"

    requirements = _requirements()
    fingerprint = _runtime_fingerprint(requirements, runtime_tag)
    remote_runtime = f"{resolved_root}/runtimes/{fingerprint}"
    # One folder per SSH execution, named like its remote session, so remote
    # artifacts never interleave with local runs under the same output dir.
    session_dir = (output_dir or workspace / "sflow_output").resolve() / run_id
    receipt_path = session_dir / "receipt.json"

    with tempfile.TemporaryDirectory(prefix="sflow-ssh-") as temporary:
        payload = Path(temporary) / "payload"
        work_stage = payload / "work"
        work_stage.mkdir(parents=True)
        shutil.copy2(
            Path(__file__).parent / "remote_helper.py",
            payload / "remote_helper.py",
        )
        _copy_source_project(payload / "source-project", requirements)

        if command in {"run", "batch"} and not input_files:
            input_files = [workspace / "sflow.yaml"]
        expanded_inputs: list[Path] = []
        for path in input_files:
            pattern = str(path if path.is_absolute() else workspace / path)
            if glob.has_magic(pattern):
                expanded_inputs.extend(Path(match) for match in glob.glob(pattern))
            else:
                expanded_inputs.append(path)
        input_files = expanded_inputs
        configs = [
            path.resolve()
            for path in input_files
            if path.suffix.lower() in {".yaml", ".yml"}
        ]
        for path in input_files:
            if path.is_dir():
                configs.extend(_directory_configs(path))
        if bulk_input is not None:
            configs.extend(_csv_configs(bulk_input.resolve()))
        inputs = list(input_files)
        inputs.extend(configs)
        for config in configs:
            inputs.extend(_local_artifacts(config, workspace, artifact_overrides))
        remote_artifact_paths = [
            path
            for config in configs
            for path in _remote_artifact_paths(config, artifact_overrides)
        ]
        if not invocation.skip_artifact_check:
            _check_remote_artifacts(ssh, remote_artifact_paths, transport=transport)
        for override in artifact_overrides or []:
            if "=" in override:
                artifact_path = _relative_uri_path(override.split("=", 1)[1], workspace)
                if artifact_path is not None and artifact_path.exists():
                    if not artifact_path.is_relative_to(workspace):
                        raise typer.BadParameter(
                            f"relative artifact override resolves outside "
                            f"--workspace-dir: {artifact_path}"
                        )
                    inputs.append(artifact_path)

        staged: dict[Path, Path] = {}
        for path in inputs:
            resolved = path.resolve()
            if resolved in staged:
                continue
            staged[resolved] = _copy_input(path, workspace, work_stage)

        archive_path = Path(temporary) / "payload.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            for child in payload.iterdir():
                archive.add(child, arcname=child.name)

        raw_args = _without_options(invocation.raw_args, _DELEGATE_OPTIONS)
        raw_args = _without_options(
            raw_args,
            {"--workspace-dir", "--output-dir", "--sbatch-output", "--sbatch-error"},
        )
        if command == "batch":
            raw_args = _without_options(raw_args, {"--sbatch-path", "-o"})
        raw_args = _rewrite_staged_paths(raw_args, staged, workspace, remote_work)

        fetch_destination: Path | None = None
        if command in {"run", "batch"}:
            raw_args.extend(["--workspace-dir", remote_work])
            raw_args.extend(["--output-dir", remote_output])
            fetch_destination = session_dir
        if command == "batch":
            raw_args.extend(
                [
                    "--sbatch-output",
                    f"{remote_output}/%j-sflow-submit.out",
                    "--sbatch-error",
                    f"{remote_output}/%j-sflow-submit.err",
                ]
            )
            if batch_sbatch_path is not None:
                raw_args.extend(
                    ["--sbatch-path", f"{remote_output}/{batch_sbatch_path.name}"]
                )
            if not batch_runtime_explicit:
                raw_args.extend(
                    ["--sflow-source-path", f"{remote_session}/source-project"]
                )
        elif command == "compose":
            raw_args = _without_options(raw_args, {"-o", "--output"})
            if compose_output is not None or compose_bulk:
                remote_compose = (
                    remote_output
                    if compose_bulk
                    else f"{remote_output}/{compose_output.name}"
                )
                raw_args.extend(["--output", remote_compose])
                # An explicit -o is a literal local path the caller asked for;
                # only the defaulted destination moves into the session folder.
                fetch_destination = (
                    compose_output.resolve()
                    if compose_output is not None
                    else session_dir
                )

        _stage(
            2,
            f"staging {len(staged)} input(s) and the sflow source "
            f"({archive_path.stat().st_size} bytes)",
        )
        _fact("remote session", remote_session)
        _fact("remote work dir", remote_work)
        _fact("remote output dir", remote_output)
        sampled, remaining = _staged_report(staged, workspace, remote_work)
        for local, remote in sampled:
            _fact("staged", f"{local} -> {remote}")
        if remaining:
            _fact("staged", f"... and {remaining} more")
        setup = (
            f"set -e; umask 077; mkdir -p {shlex.quote(remote_session)} "
            f"{shlex.quote(remote_output)} {shlex.quote(resolved_root + '/runtimes')}; "
            f"cat > {shlex.quote(remote_session + '/payload.tar.gz')}; "
            f"PYTHONWARNINGS=ignore::DeprecationWarning python3 -m tarfile -e "
            f"{shlex.quote(remote_session + '/payload.tar.gz')} "
            f"{shlex.quote(remote_session)}; "
            f"rm -f {shlex.quote(remote_session + '/payload.tar.gz')}; "
            f"exec python3 {shlex.quote(remote_helper)} bootstrap "
            f"{shlex.quote(remote_runtime)} "
            f"{shlex.quote(remote_session + '/source-project')} "
            f"{shlex.quote(_version())} {shlex.quote(invocation.log_level)}"
        )
        receipt = initial_receipt(
            command=command,
            connection=connection,
            session=run_id,
            session_dir=session_dir,
            remote_session=remote_session,
            local_output=fetch_destination,
            fetch=fetch_mode,
        )
        write_receipt(receipt_path, receipt)
        _fact("receipt", str(receipt_path))
        _stage(3, "preparing the remote runtime")
        _fact("remote runtime", remote_runtime)
        try:
            with archive_path.open("rb") as archive_stream:
                setup_result = transport.run_to_stderr(
                    ssh, setup, stdin=archive_stream
                )
        except KeyboardInterrupt:
            receipt.update(status="unknown", phase="bootstrap", return_code=130)
            write_receipt(receipt_path, receipt)
            raise typer.Exit(code=130) from None
        except OSError as exc:
            receipt.update(
                status="failed",
                execution_status="not_started",
                fetch_status="skipped",
                phase="bootstrap",
            )
            write_receipt(receipt_path, receipt)
            raise typer.BadParameter(f"could not start SSH bootstrap: {exc}") from exc
        if setup_result.returncode != 0:
            if setup_result.returncode == 255:
                ssh_failure_hint(ssh)
            receipt.update(
                status=_receipt_status(setup_result.returncode),
                execution_status="not_started",
                fetch_status="skipped",
                phase="bootstrap",
                return_code=setup_result.returncode,
            )
            write_receipt(receipt_path, receipt)
            raise typer.Exit(code=setup_result.returncode)

        receipt.update(status="running", execution_status="running", phase="execution")
        write_receipt(receipt_path, receipt)

    quoted_args = " ".join(shlex.quote(arg) for arg in [command, *raw_args])
    local_install_mode, _ = install_info()
    local_install_label = {
        "installed": "PyPI",
        "editable": "editable checkout",
        "source-tree": "source tree",
        "direct-url": "direct-url install",
    }.get(local_install_mode, local_install_mode)
    mirrored_source = f"mirrors local {local_install_label} {_version()}"
    remote_command = (
        f"umask 077; cd {shlex.quote(remote_work)}; "
        # exec keeps this PID, so the pid file addresses the controller itself
        # and Ctrl-C can signal its process group instead of only killing the
        # local SSH client.
        f"echo $$ > {shlex.quote(remote_session + '/controller.pid')}; "
        f"export PYTHONPATH={shlex.quote(remote_session + '/source-project/src')}; "
        f"export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SFLOW={shlex.quote(_version())}; "
        f"export {SSH_RUNTIME_ENV}={shlex.quote(remote_runtime)}; "
        f"export {SSH_SOURCE_ENV}={shlex.quote(mirrored_source)}; "
        f"export {SUBMITTED_JOBS_FILE_ENV}="
        f"{shlex.quote(remote_session + '/' + SUBMITTED_JOBS_FILE)}; "
        f"exec {shlex.quote(remote_runtime + '/bin/python')} -m sflow {quoted_args}"
    )
    execution_argv = _tty_argv(ssh, _tty_mode(tty, raw_args), command)
    follow = _resolve_follow(follow, live_stream="-tt" in execution_argv)

    _stage(4, f"running sflow {command} on {ssh[-1]}")
    # The translated argv, not the local one: paths were rewritten to their
    # staged remote locations, so this is what actually executes.
    _fact("remote command", f"sflow {quoted_args}")
    _fact("remote cwd", remote_work)
    _fact(
        "progress",
        {
            "none": "remote stream only",
            "logs": "remote stream plus a 5-second log tail",
            "status": "remote stream plus a heartbeat",
        }[follow],
    )

    try:
        process = transport.start(execution_argv, remote_command)
    except OSError as exc:
        receipt.update(
            status="failed",
            execution_status="not_started",
            fetch_status="skipped",
            phase="launch",
        )
        write_receipt(receipt_path, receipt)
        raise typer.BadParameter(f"could not start SSH: {exc}") from exc

    follower: subprocess.Popen | None = None
    if follow == "logs":
        try:
            follower = transport.start_to_stderr(
                ssh,
                f"exec python3 {shlex.quote(remote_helper)} follow {shlex.quote(remote_output)} "
                f"{shlex.quote(invocation.log_level)}",
            )
        except OSError as exc:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            receipt.update(
                status="unknown",
                execution_status="unknown",
                fetch_status="skipped_unknown",
                phase="launch",
            )
            write_receipt(receipt_path, receipt)
            log(logging.ERROR, "Could not start SSH log follower: %s", exc)
            raise typer.Exit(code=1) from exc

    previous: str | None = None
    unchanged = 0
    cancelled = False
    try:
        try:
            while True:
                try:
                    return_code = process.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    if (
                        follow == "logs"
                        and follower is not None
                        and follower.poll() is not None
                    ):
                        previous, unchanged = snapshot(
                            ssh,
                            remote_helper,
                            remote_output,
                            previous,
                            unchanged,
                            transport=transport,
                        )
                    elif follow == "status":
                        log(logging.INFO, "[ssh] remote process is still running")
        except KeyboardInterrupt:
            cancelled = _cancel_remote(
                ssh, remote_helper, remote_session, transport=transport
            )

            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            return_code = 130
            if cancelled:
                log(logging.INFO, "Remote run cancelled.")
            else:
                log(
                    logging.WARNING,
                    "Remote SSH session interrupted; remote state is unknown and "
                    "the receipt keeps its location.",
                )
    finally:
        if follower is not None and follower.poll() is None:
            follower.terminate()
            try:
                follower.wait(timeout=5)
            except subprocess.TimeoutExpired:
                follower.kill()
                follower.wait()

    receipt.update(
        **_receipt_outcome(command, return_code, raw_args),
        phase="postprocessing",
        return_code=return_code,
    )
    if cancelled:
        # Confirmed gone, so this is no longer the "unknown remote state" case.
        receipt.update(status="cancelled", execution_status="cancelled")
    write_receipt(receipt_path, receipt)

    if follow == "logs" and return_code not in {130, 255}:
        snapshot(
            ssh, remote_helper, remote_output, previous, unchanged, transport=transport
        )
    if return_code == 255:
        ssh_failure_hint(ssh)
        log(
            logging.ERROR,
            f"SSH transport failed; remote state is unknown. Inspect or reconnect with {receipt_path}.",
        )
    should_fetch = (
        fetch_mode != "none"
        and fetch_destination is not None
        and (return_code not in {130, 255} or cancelled)
    )
    if should_fetch:
        receipt.update(fetch_status="running", phase="fetch")
        write_receipt(receipt_path, receipt)
        try:
            if command == "compose" and compose_output is not None and not compose_bulk:
                fetched = fetch(
                    ssh,
                    remote_helper,
                    remote_output,
                    "all",
                    compose_output.parent,
                    transport=transport,
                )
            else:
                mode = (
                    "all"
                    if fetch_mode == "all" or command == "compose"
                    else ("batch" if command == "batch" else "logs")
                )
                fetched = fetch(
                    ssh,
                    remote_helper,
                    remote_output,
                    mode,
                    fetch_destination,
                    transport=transport,
                )
            if fetched and command == "batch" and batch_sbatch_path is not None:
                for local_path in (
                    batch_sbatch_path,
                    batch_sbatch_path.with_suffix(".yaml"),
                ):
                    fetched_file = fetch_destination / local_path.name
                    if fetched_file.is_file() and fetched_file != local_path:
                        try:
                            _copy_file_atomic(fetched_file, local_path)
                        except (OSError, RuntimeError) as exc:
                            log(
                                logging.WARNING,
                                "Failed to restore %s: %s", local_path, exc,
                            )
                            fetched = False
            if fetched:
                log(logging.INFO, "Fetched remote output to %s", fetch_destination)
        except KeyboardInterrupt:
            receipt.update(fetch_status="interrupted", phase="finished")
            write_receipt(receipt_path, receipt)
            raise typer.Exit(code=130) from None
        receipt.update(
            fetch_status="completed" if fetched else "failed", phase="finished"
        )
    else:
        receipt.update(
            fetch_status=(
                "skipped_unknown"
                if return_code in {130, 255} and not cancelled
                else "skipped"
            ),
            phase="finished",
        )
    write_receipt(receipt_path, receipt)
    raise typer.Exit(code=return_code)
