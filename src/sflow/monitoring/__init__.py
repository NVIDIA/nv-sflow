# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bundled hardware-monitor assets for the ``monitor:`` feature.

Ships two standalone, dependency-light scripts:

* ``hardware_monitor.py`` -- the per-node collector launched as a sidecar during
  a run (standard library + ``nvidia-smi``).
* ``postprocess_monitor_timeline.py`` -- the post-run processor that turns raw
  samples into ``sflow_monitor.log`` plus detailed per-consumer reports.

Accessors mirror :mod:`sflow.samples` so callers can either read a script's
source (to materialize it as a ``file://`` artifact) or get its on-disk path.
"""

from importlib import resources
from pathlib import Path

HARDWARE_MONITOR_FILENAME = "hardware_monitor.py"
POSTPROCESS_FILENAME = "postprocess_monitor_timeline.py"


def get_monitoring_dir() -> Path:
    """Return the path to the bundled monitoring assets directory."""
    return Path(resources.files(__package__))


def hardware_monitor_path() -> Path:
    """Return the on-disk path to the collector script."""
    return get_monitoring_dir() / HARDWARE_MONITOR_FILENAME


def postprocess_path() -> Path:
    """Return the on-disk path to the post-processor script."""
    return get_monitoring_dir() / POSTPROCESS_FILENAME


def hardware_monitor_source() -> str:
    """Return the collector script source (for inline ``file://`` artifacts)."""
    return hardware_monitor_path().read_text(encoding="utf-8")


def postprocess_source() -> str:
    """Return the post-processor script source."""
    return postprocess_path().read_text(encoding="utf-8")
