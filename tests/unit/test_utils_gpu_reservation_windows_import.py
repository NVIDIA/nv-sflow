# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The GPU reservation registry must stay importable without POSIX ``fcntl``.

``sflow.cli.run`` imports ``WAIT_FOR_GPUS_ENV`` from it, and ``sflow.cli`` imports
every command module at startup -- so an unguarded ``import fcntl`` would make the
*entire* CLI fail on Windows with ModuleNotFoundError, not just GPU reservation.
Mirrors ``test_core_launcher_windows_import`` (the same guard for pty/termios).
"""

import builtins
import importlib
import sys


def _reimport_without_fcntl(monkeypatch, module_name: str):
    real_import = builtins.__import__

    def block_fcntl(name, *args, **kwargs):
        if name == "fcntl":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    original = sys.modules.pop("sflow.utils.gpu_reservation", None)
    package = importlib.import_module("sflow.core")
    original_package_attr = getattr(package, "gpu_reservation", None)
    monkeypatch.setattr(builtins, "__import__", block_fcntl)
    try:
        return importlib.import_module(module_name)
    finally:
        sys.modules.pop("sflow.utils.gpu_reservation", None)
        if original is not None:
            sys.modules["sflow.utils.gpu_reservation"] = original
            setattr(package, "gpu_reservation", original_package_attr)


def test_gpu_reservation_imports_without_fcntl(monkeypatch):
    gr = _reimport_without_fcntl(monkeypatch, "sflow.utils.gpu_reservation")

    assert gr.fcntl is None
    assert gr.flock_available() is False
    # The env-var constant the CLI needs is still available...
    assert gr.WAIT_FOR_GPUS_ENV == "SFLOW_WAIT_FOR_GPUS"
    # ...and reservation reports itself off, so callers fall back rather than
    # blowing up on the missing lock primitive.
    monkeypatch.delenv(gr.GPU_RESERVATION_ENV, raising=False)
    assert gr.reservation_enabled() is False
    # Release is best-effort and must stay silent rather than raise.
    gr.release_gpus("never-reserved")


def test_cli_imports_without_fcntl(monkeypatch):
    """The whole Typer app must still build -- this is what actually breaks users."""
    for name in [m for m in sys.modules if m.startswith("sflow.cli")]:
        sys.modules.pop(name, None)
    cli = _reimport_without_fcntl(monkeypatch, "sflow.cli")
    assert cli.app is not None
