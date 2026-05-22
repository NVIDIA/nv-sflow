# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for reporting which sflow executable is running."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import textwrap
from importlib import metadata as importlib_metadata
from pathlib import Path
from urllib.parse import unquote, urlparse

import sflow
from sflow import __version__
from sflow.logging import get_logger

_logger = get_logger(__name__)
RUNTIME_INFO_MIN_BOX_WIDTH = 48
RUNTIME_INFO_MAX_BOX_WIDTH = 80
_RICH_LOG_COLUMNS_WIDTH = 48
_DEFAULT_CONSOLE_WIDTH = 120
_RUNTIME_INFO_LABEL_WIDTH = 7
_BRANCH_BUILD_PREFIXES = ("feature.", "feat.", "bugfix.", "hotfix.")


def _resolve_bin_path() -> str:
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0:
        argv_path = Path(argv0)
        if argv_path.exists():
            return str(argv_path.resolve())
    return shutil.which("sflow") or argv0 or "unknown"


def _path_from_file_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None
    return unquote(parsed.path)


def _install_info() -> tuple[str, str | None]:
    try:
        dist = importlib_metadata.distribution("sflow")
    except importlib_metadata.PackageNotFoundError:
        return "source-tree", None

    direct_url_text = dist.read_text("direct_url.json")
    if not direct_url_text:
        return "installed", None

    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError:
        return "installed", None

    repo = _path_from_file_url(direct_url.get("url"))
    if direct_url.get("dir_info", {}).get("editable"):
        return "editable", repo
    return "direct-url", repo


def _find_repo_root(path: Path) -> Path | None:
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _repo_from_imported_source(package_path: Path) -> Path | None:
    repo = _find_repo_root(package_path)
    if repo is None:
        return None
    try:
        package_path.relative_to(repo / "src")
    except ValueError:
        return None
    return repo


def _repo_for_git_info(
    install_mode: str, install_repo: str | None, package_path: Path
) -> Path | None:
    if install_repo:
        return Path(install_repo).resolve()
    if install_mode == "source-tree":
        return _find_repo_root(package_path)
    return None


def _version_local_segment(version: str) -> str:
    _, separator, local = version.partition("+")
    if not separator:
        return ""
    return local


def _source_label(install_mode: str, install_repo: str | None, version: str) -> str:
    if install_mode == "editable":
        return "local editable dev"
    if install_mode == "direct-url" and install_repo:
        return "local build"
    if install_mode == "source-tree":
        return "local source tree"

    local_version = _version_local_segment(version)
    if install_mode == "installed":
        if local_version.startswith("develop."):
            return "remote pypi develop"
        if local_version.startswith(_BRANCH_BUILD_PREFIXES):
            return "remote pypi branch"
        if ".dev" not in version:
            return "remote pypi release"
        return "remote pypi package"

    return "unknown"


def _git_info_for_repo(repo: Path | None) -> str:
    if repo is None:
        return ""
    try:
        branch = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode().strip()
        sha = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode().strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repo), "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                timeout=3,
            ).decode().strip()
        )
    except Exception:
        return ""
    suffix = ", dirty" if dirty else ""
    return f"({branch} {sha}{suffix})"


def format_runtime_info() -> str:
    package_path = Path(sflow.__file__).resolve().parent
    install_mode, install_repo = _install_info()
    imported_source_repo = _repo_from_imported_source(package_path)
    if install_mode == "installed" and install_repo is None and imported_source_repo:
        install_mode = "editable"
        install_repo = str(imported_source_repo)
    repo_path = _repo_for_git_info(install_mode, install_repo, package_path)
    git_info = _git_info_for_repo(repo_path)

    fields = [
        ("version", __version__),
        ("bin", _resolve_bin_path()),
        ("python", str(Path(sys.executable).resolve())),
        ("package", str(package_path)),
        ("install", install_mode),
        ("source", _source_label(install_mode, install_repo, __version__)),
    ]
    if install_repo:
        fields.append(("repo", install_repo))
    if git_info:
        fields.append(("git", git_info))

    box_width = _runtime_info_box_width()
    lines = [_box_title("sflow executable", box_width)]
    for label, value in fields:
        lines.extend(_box_field(label, value, box_width))
    lines.append(_box_bottom(box_width))
    return "\n".join(lines)


def log_runtime_info() -> None:
    _logger.info(format_runtime_info())


def _current_console_width() -> int:
    for handler in logging.getLogger("sflow").handlers:
        console = getattr(handler, "console", None)
        width = getattr(console, "width", None)
        if isinstance(width, int) and width > 0:
            return width
    return shutil.get_terminal_size((_DEFAULT_CONSOLE_WIDTH, 20)).columns


def _runtime_info_box_width() -> int:
    message_width = _current_console_width() - _RICH_LOG_COLUMNS_WIDTH
    return max(
        RUNTIME_INFO_MIN_BOX_WIDTH,
        min(RUNTIME_INFO_MAX_BOX_WIDTH, message_width),
    )


def _box_title(title: str, box_width: int) -> str:
    prefix = f"╭─ {title} "
    return prefix + "─" * (box_width - len(prefix) - 1) + "╮"


def _box_bottom(box_width: int) -> str:
    return "╰" + "─" * (box_width - 2) + "╯"


def _box_line(content: str, box_width: int) -> str:
    inner_width = box_width - 4
    return f"│ {content:<{inner_width}} │"


def _box_field(label: str, value: str, box_width: int) -> list[str]:
    inner_width = box_width - 4
    prefix = f"{label:<{_RUNTIME_INFO_LABEL_WIDTH}} : "
    continuation = " " * len(prefix)
    wrap_width = inner_width - len(prefix)
    wrapped = textwrap.wrap(
        str(value),
        width=wrap_width,
        break_long_words=True,
        break_on_hyphens=False,
    ) or [""]

    lines = [_box_line(prefix + wrapped[0], box_width)]
    lines.extend(_box_line(continuation + part, box_width) for part in wrapped[1:])
    return lines
