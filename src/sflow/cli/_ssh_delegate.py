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
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse

import typer
import yaml

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
_SSH_FAILURE_HINT_SHOWN = False


def _invocation_id() -> str:
    """Name one SSH execution identically on both hosts.

    The UTC stamp sorts chronologically for a human browsing the local output
    dir; the random suffix keeps concurrent same-second invocations apart
    (PIDs and wall clocks alone collide across clients).
    """
    return f"ssh-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{secrets.token_hex(4)}"


def _ssh_argv(connection: str) -> list[str]:
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
    typer.echo(f"[ssh {number}/{_STAGES}] {description}...", err=True)


def _fact(label: str, value: str) -> None:
    typer.echo(f"    {label}: {value}", err=True)


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
        typer.echo(
            "Warning: --tui needs a remote PTY to render; ignoring --ssh-tty never.",
            err=True,
        )
        return "always"
    return mode


def _cancel_remote(ssh: list[str], helper: str, session: str) -> bool:
    """Stop the remote controller, reporting whether it is confirmed gone.

    Killing the local SSH client only closes the channel: without this the
    remote controller (and anything it allocated) keeps running.
    """
    typer.echo("\nCancelling the remote run...", err=True)
    try:
        result = subprocess.run(
            [
                *ssh,
                f"exec python3 {shlex.quote(helper)} cancel {shlex.quote(session)}",
            ],
            stdin=subprocess.DEVNULL,
            stdout=sys.stderr,
            stderr=sys.stderr,
            check=False,
        )
    except KeyboardInterrupt:
        # A second Ctrl-C means the user is done waiting; the remote state is
        # then genuinely unknown.
        typer.echo("Stopped waiting for the remote cancel to confirm.", err=True)
        return False
    except OSError as exc:
        typer.echo(f"Could not reach the host to cancel: {exc}", err=True)
        return False
    if result.returncode == 255:
        _ssh_failure_hint(ssh)
    return result.returncode == 0


def _resolve_follow(follow: str, *, live_stream: bool) -> str:
    """Pick the display mode so exactly one writer owns the local terminal.

    Over a PTY the remote sflow sees a terminal, keeps task logs streaming
    instead of offloading them to files, and may render a live TUI -- a second
    channel printing log tails on top of that interleaves mid-render. Without a
    PTY the remote offloads task logs to files, so the poller is the only way to
    see them while the run is attached.
    """
    if follow != "auto":
        return follow
    return "none" if live_stream else "logs"


def _ssh_failure_hint(ssh: list[str]) -> None:
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
    typer.echo(
        "SSH could not establish a connection (exit 255). If this host requires "
        "MFA, run this command and complete MFA:\n"
        f"  {master}\n"
        "Then rerun sflow using:\n"
        f"  --ssh {shlex.quote(reuse)}\n"
        "Exit 255 can also mean a network, DNS, host-key, or SSH configuration "
        "failure. Documentation: "
        "https://nvidia.github.io/nv-sflow/docs/user/cli#run-through-ssh",
        err=True,
    )


def _raw_command_args(command: str) -> list[str]:
    argv = sys.argv[1:]
    try:
        return argv[argv.index(command) + 1 :]
    except ValueError:
        return argv[1:]


def _option_values(args: list[str], *names: str) -> list[str]:
    values: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        for name in names:
            if token == name and index + 1 < len(args):
                values.append(args[index + 1])
                index += 1
                break
            if token.startswith(name + "="):
                values.append(token.split("=", 1)[1])
                break
        index += 1
    return values


def predispatch() -> None:
    """Delegate raw argv before Typer resolves remote paths on the local host."""
    argv = sys.argv[1:]
    if not argv or argv[0] not in {"run", "batch", "compose"}:
        return
    command = argv[0]
    args = argv[1:]
    connections = _option_values(args, "--ssh")
    if not connections:
        return

    workspace_values = _option_values(args, "--workspace-dir")
    workspace = (
        Path(workspace_values[-1]).expanduser() if workspace_values else Path.cwd()
    )
    output_values = _option_values(args, "--output-dir")
    output_dir = Path(output_values[-1]).expanduser() if output_values else None
    bulk_values = _option_values(args, "--bulk-input", "-b")
    bulk_input = Path(bulk_values[-1]).expanduser() if bulk_values else None
    artifact_overrides = _option_values(args, "--artifact", "-a")

    input_values = _option_values(
        args, "--file", "-f", "--bulk-submit", "-B", "--sflow-source-path"
    )
    input_values.extend(bulk_values)
    for token in args:
        path = Path(token).expanduser()
        if path.suffix.lower() in {".yaml", ".yml"} and path.exists():
            input_values.append(token)
    input_files = list(
        dict.fromkeys(Path(value).expanduser() for value in input_values)
    )

    compose_values = (
        _option_values(args, "--output", "-o") if command == "compose" else []
    )
    sbatch_values = (
        _option_values(args, "--sbatch-path", "-o") if command == "batch" else []
    )
    follow_values = _option_values(args, "--ssh-follow")
    fetch_values = _option_values(args, "--ssh-fetch")
    root_values = _option_values(args, "--ssh-remote-root")
    tty_values = _option_values(args, "--ssh-tty")

    delegate(
        command,
        connection=connections[-1],
        follow=follow_values[-1]
        if follow_values
        else ("none" if command == "compose" else "auto"),
        fetch=fetch_values[-1] if fetch_values else "logs",
        remote_root=root_values[-1] if root_values else None,
        tty=tty_values[-1] if tty_values else "auto",
        workspace_dir=workspace,
        output_dir=output_dir,
        input_files=input_files,
        artifact_overrides=artifact_overrides,
        bulk_input=bulk_input,
        compose_output=Path(compose_values[-1]).expanduser()
        if compose_values
        else None,
        compose_bulk=command == "compose" and bulk_input is not None,
        batch_sbatch_path=Path(sbatch_values[-1]).expanduser()
        if sbatch_values
        else None,
        batch_runtime_explicit=any(
            _option_values(
                args, "--sflow-source-path", "--sflow-version", "--sflow-index-url"
            )
        ),
    )


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


def _resolve_remote_environment(ssh: list[str], value: str | None) -> tuple[str, str]:
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
    result = subprocess.run(
        [*ssh, command],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
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
            _ssh_failure_hint(ssh)
        raise typer.BadParameter(
            result.stderr.strip() or "could not resolve --ssh-remote-root"
        )
    return root, runtime_tag


def _resolve_remote_root(ssh: list[str], value: str | None) -> str:
    return _resolve_remote_environment(ssh, value)[0]


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
    configs: list[Path] = []
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                for value in shlex.split(row.get("sflow_config_file", "")):
                    configs.append((path.parent / value).resolve())
    except (OSError, csv.Error, ValueError):
        pass
    return configs


def _local_artifacts(config: Path, workspace: Path) -> list[Path]:
    try:
        document = yaml.safe_load(config.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []
    found: list[Path] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            uri = value.get("uri")
            if isinstance(uri, str) and value.get("content") is None:
                path = _relative_uri_path(uri, workspace)
                if path is not None and path.exists():
                    if not path.is_relative_to(workspace.resolve()):
                        raise typer.BadParameter(
                            f"relative artifact {uri!r} resolves outside "
                            f"--workspace-dir: {path}"
                        )
                    found.append(path)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(document)
    return found


def _relative_uri_path(uri: str, workspace: Path) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme.lower() not in {"file", "fs"}:
        return None
    raw = unquote((parsed.netloc or "") + (parsed.path or ""))
    if not raw or "${{" in raw or Path(raw).is_absolute():
        return None
    return (workspace / raw).resolve()


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
    try:
        return sorted(metadata.requires("sflow") or [])
    except metadata.PackageNotFoundError:
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
    try:
        return metadata.version("sflow")
    except metadata.PackageNotFoundError:
        return "0.0+ssh"


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
            f'[project]\nname = "sflow"\nversion = {json.dumps(_version())}\n'
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


def _write_receipt(path: Path, receipt: dict[str, object]) -> None:
    """Atomically publish a mode-0600 recovery receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)


def _snapshot(
    ssh: list[str],
    helper: str,
    remote_output: str,
    previous: str | None,
    unchanged: int,
    *,
    connection_hint: bool = False,
) -> tuple[str | None, int]:
    try:
        result = subprocess.run(
            [
                *ssh,
                f"python3 {shlex.quote(helper)} snapshot {shlex.quote(remote_output)}",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result = subprocess.CompletedProcess(
            [], 1, "", f"remote log check failed: {exc}"
        )
    if connection_hint and result.returncode == 255:
        _ssh_failure_hint(ssh)
    snapshot = (
        result.stdout.strip() or result.stderr.strip() or "(remote log check failed)"
    )
    unchanged = unchanged + 5 if snapshot == previous else 0
    suffix = f", unchanged for {unchanged}s" if unchanged else ""
    typer.echo(f"\n[ssh {time.strftime('%H:%M:%S')}{suffix}]\n{snapshot}", err=True)
    if connection_hint and result.returncode != 0:
        raise typer.Exit(code=result.returncode)
    return snapshot, unchanged


def _fetch(
    ssh: list[str], helper: str, remote_output: str, mode: str, destination: Path
) -> bool:
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive_file:
        command = (
            f"python3 {shlex.quote(helper)} pack {shlex.quote(remote_output)} "
            f"{shlex.quote(mode)}"
        )
        try:
            result = subprocess.run(
                [*ssh, command],
                stdin=subprocess.DEVNULL,
                stdout=archive_file,
                check=False,
            )
        except OSError as exc:
            typer.echo(f"Warning: failed to fetch remote output: {exc}", err=True)
            return False
        if result.returncode != 0:
            if result.returncode == 255:
                _ssh_failure_hint(ssh)
            typer.echo("Warning: failed to fetch remote output.", err=True)
            return False
        archive_file.flush()
        try:
            with tempfile.TemporaryDirectory(prefix="sflow-fetch-") as temporary:
                staged = Path(temporary)
                _safe_extract(Path(archive_file.name), staged)
                _merge_tree(staged, destination)
        except (OSError, RuntimeError, tarfile.TarError) as exc:
            typer.echo(f"Warning: unsafe or incomplete remote output: {exc}", err=True)
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


def delegate(
    command: str,
    *,
    connection: str,
    follow: str,
    fetch: str,
    remote_root: str | None,
    tty: str = "auto",
    workspace_dir: Path | None,
    output_dir: Path | None,
    input_files: list[Path],
    artifact_overrides: list[str] | None = None,
    bulk_input: Path | None = None,
    compose_output: Path | None = None,
    compose_bulk: bool = False,
    batch_sbatch_path: Path | None = None,
    batch_runtime_explicit: bool = False,
) -> None:
    """Delegate one parsed CLI invocation, then exit with the remote status."""
    if follow not in {"auto", "logs", "status", "none"}:
        raise typer.BadParameter("--ssh-follow must be auto, logs, status, or none")
    if fetch not in {"logs", "all", "none"}:
        raise typer.BadParameter("--ssh-fetch must be logs, all, or none")
    if tty not in {"auto", "always", "never"}:
        raise typer.BadParameter("--ssh-tty must be auto, always, or never")

    ssh = _ssh_argv(connection)
    workspace = (workspace_dir or Path.cwd()).resolve()
    run_id = _invocation_id()
    _stage(1, f"connecting to {ssh[-1]}")
    resolved_root, runtime_tag = _resolve_remote_environment(ssh, remote_root)
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
            Path(__file__).with_name("remote.py"), payload / "remote_helper.py"
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
            inputs.extend(_local_artifacts(config, workspace))
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

        raw_args = _without_options(_raw_command_args(command), _DELEGATE_OPTIONS)
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
            f"{shlex.quote(_version())}"
        )
        receipt: dict[str, object] = {
            "command": command,
            "connection": connection,
            "session": run_id,
            "session_dir": str(session_dir),
            "remote_session": remote_session,
            "local_output": str(fetch_destination) if fetch_destination else None,
            "status": "bootstrapping",
            "execution_status": "pending",
            "fetch_status": "skipped" if fetch == "none" else "pending",
            "phase": "bootstrap",
        }
        _write_receipt(receipt_path, receipt)
        _fact("receipt", str(receipt_path))
        _stage(3, "preparing the remote runtime")
        _fact("remote runtime", remote_runtime)
        try:
            with archive_path.open("rb") as archive_stream:
                setup_result = subprocess.run(
                    [*ssh, setup],
                    stdin=archive_stream,
                    stdout=sys.stderr,
                    stderr=sys.stderr,
                    check=False,
                )
        except KeyboardInterrupt:
            receipt.update(status="unknown", phase="bootstrap", return_code=130)
            _write_receipt(receipt_path, receipt)
            raise typer.Exit(code=130) from None
        except OSError as exc:
            receipt.update(
                status="failed",
                execution_status="not_started",
                fetch_status="skipped",
                phase="bootstrap",
            )
            _write_receipt(receipt_path, receipt)
            raise typer.BadParameter(f"could not start SSH bootstrap: {exc}") from exc
        if setup_result.returncode != 0:
            if setup_result.returncode == 255:
                _ssh_failure_hint(ssh)
            receipt.update(
                status=_receipt_status(setup_result.returncode),
                execution_status="not_started",
                fetch_status="skipped",
                phase="bootstrap",
                return_code=setup_result.returncode,
            )
            _write_receipt(receipt_path, receipt)
            raise typer.Exit(code=setup_result.returncode)

        receipt.update(status="running", execution_status="running", phase="execution")
        _write_receipt(receipt_path, receipt)

    quoted_args = " ".join(shlex.quote(arg) for arg in [command, *raw_args])
    remote_command = (
        f"umask 077; cd {shlex.quote(remote_work)}; "
        # exec keeps this PID, so the pid file addresses the controller itself
        # and Ctrl-C can signal its process group instead of only killing the
        # local SSH client.
        f"echo $$ > {shlex.quote(remote_session + '/controller.pid')}; "
        f"export PYTHONPATH={shlex.quote(remote_session + '/source-project/src')}; "
        f"export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SFLOW={shlex.quote(_version())}; "
        f"export SFLOW_SUBMITTED_JOBS_FILE={shlex.quote(remote_session + '/submitted-jobs')}; "
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
        process = subprocess.Popen([*execution_argv, remote_command])
    except OSError as exc:
        receipt.update(
            status="failed",
            execution_status="not_started",
            fetch_status="skipped",
            phase="launch",
        )
        _write_receipt(receipt_path, receipt)
        raise typer.BadParameter(f"could not start SSH: {exc}") from exc

    follower: subprocess.Popen | None = None
    if follow == "logs":
        try:
            follower = subprocess.Popen(
                [
                    *ssh,
                    f"exec python3 {shlex.quote(remote_helper)} follow {shlex.quote(remote_output)}",
                ],
                stdin=subprocess.DEVNULL,
                stdout=sys.stderr,
                stderr=sys.stderr,
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
            _write_receipt(receipt_path, receipt)
            typer.echo(f"Could not start SSH log follower: {exc}", err=True)
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
                        previous, unchanged = _snapshot(
                            ssh, remote_helper, remote_output, previous, unchanged
                        )
                    elif follow == "status":
                        typer.echo(
                            f"[ssh {time.strftime('%H:%M:%S')}] remote process is still running",
                            err=True,
                        )
        except KeyboardInterrupt:
            cancelled = _cancel_remote(ssh, remote_helper, remote_session)

            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            return_code = 130
            if cancelled:
                typer.echo("Remote run cancelled.", err=True)
            else:
                typer.echo(
                    "Remote SSH session interrupted; remote state is unknown and "
                    "the receipt keeps its location.",
                    err=True,
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
    _write_receipt(receipt_path, receipt)

    if follow == "logs" and return_code not in {130, 255}:
        _snapshot(ssh, remote_helper, remote_output, previous, unchanged)
    if return_code == 255:
        _ssh_failure_hint(ssh)
        typer.echo(
            f"SSH transport failed; remote state is unknown. Inspect or reconnect with {receipt_path}.",
            err=True,
        )
    should_fetch = (
        fetch != "none"
        and fetch_destination is not None
        and (return_code not in {130, 255} or cancelled)
    )
    if should_fetch:
        receipt.update(fetch_status="running", phase="fetch")
        _write_receipt(receipt_path, receipt)
        try:
            if command == "compose" and compose_output is not None and not compose_bulk:
                fetched = _fetch(
                    ssh, remote_helper, remote_output, "all", compose_output.parent
                )
            else:
                mode = (
                    "all"
                    if fetch == "all" or command == "compose"
                    else ("batch" if command == "batch" else "logs")
                )
                fetched = _fetch(
                    ssh, remote_helper, remote_output, mode, fetch_destination
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
                            typer.echo(
                                f"Warning: failed to restore {local_path}: {exc}",
                                err=True,
                            )
                            fetched = False
            if fetched:
                typer.echo(f"Fetched remote output to {fetch_destination}", err=True)
        except KeyboardInterrupt:
            receipt.update(fetch_status="interrupted", phase="finished")
            _write_receipt(receipt_path, receipt)
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
    _write_receipt(receipt_path, receipt)
    raise typer.Exit(code=return_code)
