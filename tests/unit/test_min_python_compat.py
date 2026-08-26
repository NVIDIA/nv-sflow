# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``src/`` must not use stdlib APIs newer than the ``requires-python`` floor.

``async with asyncio.timeout(...)`` (3.11+) reached both repos while the package
declared ``requires-python = ">=3.10"``, where it is an ``AttributeError``. The 3.10
CI leg did catch it, and stayed red unnoticed -- so this is the same signal on the
dev box, where it is in front of the author instead of in a matrix tab.
"""

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

# ponytail: only the API that actually bit. Add a row when the next one does -- a
# speculative table of every 3.11 addition is upkeep for a bug nobody has had.
# (regex, first version with it, what to use instead)
_TOO_NEW = [
    (r"\basyncio\.timeout\b", (3, 11), "asyncio.wait_for(coro, timeout)"),
]


def _requires_python_floor() -> tuple[int, int]:
    """The floor the package advertises, so this follows pyproject instead of a pin."""
    text = (_REPO / "pyproject.toml").read_text()
    match = re.search(r'^requires-python\s*=\s*"[><=~^]*(\d+)\.(\d+)', text, re.MULTILINE)
    assert match, "could not parse requires-python from pyproject.toml"
    return int(match.group(1)), int(match.group(2))


def test_src_uses_no_api_newer_than_requires_python():
    floor = _requires_python_floor()
    offenders = []
    for path in sorted((_REPO / "src").rglob("*.py")):
        for number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            code = line.split("#", 1)[0]
            for pattern, added, replacement in _TOO_NEW:
                if added > floor and re.search(pattern, code):
                    offenders.append(
                        f"{path.relative_to(_REPO).as_posix()}:{number}: needs "
                        f"py{added[0]}.{added[1]}+ but requires-python is "
                        f">={floor[0]}.{floor[1]} -- use {replacement}"
                    )
    assert not offenders, "stdlib APIs newer than the supported floor:\n  " + "\n  ".join(
        offenders
    )
