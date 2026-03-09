# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import shlex

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def prepend_envs(script: list[str], envs: dict[str, str]) -> list[str]:
    """Prepend environment variables to a script.

    Args:
        script: The script to prepend the environment variables to.
        envs: The environment variables to prepend.

    Returns:
        The script with the environment variables prepended.
    """
    exports: list[str] = []
    for key, value in envs.items():
        k = str(key)
        if not _ENV_KEY_RE.match(k):
            raise ValueError(f"Invalid environment variable name: {k!r}")
        exports.append(f"export {k}={shlex.quote(str(value))}")
    return exports + script


def ensure_line_buffered(script: list[str]) -> list[str]:
    """Ensure the script is line buffered.

    Args:
        script: The script to ensure is line buffered.

    Returns:
        The script with the line buffering ensured.
    """
    # Idempotency: don't re-prepend or re-wrap if we've already done it.
    marker = "# sflow: line-buffered"
    if script and script[0].strip() == marker:
        return script

    # Best-effort line buffering:
    # - Prefer `stdbuf -oL -eL` if available (common on Linux/coreutils).
    # - Also set `PYTHONUNBUFFERED=1` for Python subprocesses.
    prologue = [
        marker,
        "export PYTHONUNBUFFERED=1",
        "__sflow_has_stdbuf=0",
        "command -v stdbuf >/dev/null 2>&1 && __sflow_has_stdbuf=1",
        "__sflow_linebuf() {",
        '  if [ "${__sflow_has_stdbuf}" = "1" ]; then',
        '    stdbuf -oL -eL "$@"',
        "  else",
        '    "$@"',
        "  fi",
        "}",
        "",
    ]

    def _should_wrap(stripped: str) -> bool:
        if not stripped:
            return False
        if stripped.startswith("#"):
            return False

        # Don't wrap things that aren't "a command with args".
        for prefix in (
            "export ",
            "unset ",
            "alias ",
            "source ",
            ". ",
            "cd ",
            "set ",
            "ulimit ",
            "return",
            "exit",
        ):
            if stripped.startswith(prefix):
                return False

        # Avoid bash control structures / reserved words.
        for prefix in (
            "if ",
            "then",
            "elif ",
            "else",
            "fi",
            "for ",
            "while ",
            "until ",
            "do",
            "done",
            "case ",
            "esac",
            "select ",
            "function ",
            "{",
            "}",
        ):
            if stripped == prefix or stripped.startswith(prefix + " "):
                return False

        # Avoid complex shell syntax we don't want to rewrite.
        for token in ("|", ">", "<", "&&", "||", ";"):
            if token in stripped:
                return False

        # Avoid env assignments preceding a command, e.g. "FOO=1 python ...".
        first_space = stripped.find(" ")
        first_eq = stripped.find("=")
        if 0 <= first_eq < first_space:
            return False

        # Avoid things that would change semantics when run as "$@".
        first_word = stripped.split(maxsplit=1)[0]
        if first_word in {"time", "exec", "eval", "command", "builtin"}:
            return False

        # `sudo` won't see our shell function; leave those lines alone.
        if first_word == "sudo":
            return False

        # Don't double-wrap.
        if stripped.startswith("__sflow_linebuf "):
            return False

        return True

    rewritten: list[str] = []
    for line in script:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if _should_wrap(stripped):
            rewritten.append(f"{indent}__sflow_linebuf {stripped}")
        else:
            rewritten.append(line)

    return prologue + rewritten
