# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One monitor command, many deliveries.

What to sample is backend-agnostic and built in one place. How it reaches a node
and how samples come back (``MonitorDelivery``) is the per-backend part. Adding a
backend must cost a delivery, never a second copy of the collector invocation.
"""

import shlex

from sflow.app.monitor_planner import _build_collector_script
from sflow.core.monitor import MonitorDelivery

_PLAN = dict(
    interval_ms=2000, scopes=["cpu", "gpu"], gpu_fields="index,power.draw", custom=[]
)


def _script(delivery, script_path="/x.py", **plan):
    return _build_collector_script(
        script_path=script_path, delivery=delivery, **{**_PLAN, **plan}
    )


def test_the_collector_args_are_identical_across_deliveries():
    """The whole point: one argument vector, whatever the transport."""
    args = "--interval-ms 2000 --scopes cpu,gpu --gpu-fields index,power.draw"
    assert args in "\n".join(_script(MonitorDelivery()))
    # Off-host adds the output-channel flag and changes nothing else.
    assert f"{args} --stdout" in "\n".join(_script(MonitorDelivery(offhost=True)))


def test_default_delivery_runs_the_materialized_script_in_place():
    script = _script(MonitorDelivery(), script_path="/run/out/hardware_monitor.py")
    body = "\n".join(script)
    assert "python3 /run/out/hardware_monitor.py --interval-ms 2000" in body
    # Nothing inlined, nothing streamed: the node can read the driver's path.
    assert "--stdout" not in body


def test_stdin_delivery_redirects_the_file_into_the_container():
    """A container cannot see the driver's filesystem, so the source has to
    travel with the command -- but it travels as BYTES ON STDIN, redirected by
    the driver's own shell, not pasted into the command text.

    Inlining it as a heredoc worked and was unreadable: all 378 lines of the
    collector landed in sflow.log once per node, which was 93% of the file on a
    two-node run.
    """
    script = _script(
        MonitorDelivery(
            launch_prefix=("docker", "--host", "ssh://u@h", "run", "--rm", "-i", "img"),
            offhost=True,
        ),
        script_path="/run/out/hardware_monitor.py",
    )
    body = "\n".join(script)
    assert "docker --host ssh://u@h run --rm -i img python3 - " in body
    assert "--stdout" in body
    # The redirect is the driver's; the remote side still never sees this path.
    assert body.rstrip().endswith("< /run/out/hardware_monitor.py")
    # No heredoc, and the whole thing stays on one line.
    assert "<<" not in body
    assert len(script) == 2, script


def test_the_logged_command_stays_short():
    """The regression this replaced: a command the length of the collector."""
    script = _script(
        MonitorDelivery(launch_prefix=("docker", "run", "-i", "img"), offhost=True),
        script_path="/run/out/hardware_monitor.py",
    )
    assert max(len(line) for line in script) < 200


def test_streaming_keeps_status_text_off_the_data_channel():
    """stdout carries samples when streaming, so the banner must go to stderr."""
    streamed = "\n".join(_script(MonitorDelivery(offhost=True)))
    in_place = "\n".join(_script(MonitorDelivery()))
    assert 'echo "Starting hardware monitor" >&2' in streamed
    assert 'echo "Starting hardware monitor"' in in_place
    assert ">&2" not in in_place


def test_custom_commands_survive_every_delivery():
    for delivery, expected in (
        (MonitorDelivery(), "my_probe.sh &"),
        # Streaming: stdout is the sample channel and the driver's demultiplexer
        # drops every line that is not `<scope>\t<row>`, so custom output left on
        # stdout would vanish. stderr is still a log channel.
        (MonitorDelivery(offhost=True), "my_probe.sh >&2 &"),
    ):
        body = "\n".join(
            _script(
                delivery, interval_ms=1000, scopes=[], gpu_fields=None,
                custom=["my_probe.sh"],
            )
        )
        assert expected in body
        # Only custom commands -> the step must stay alive until they finish.
        assert body.rstrip().endswith("wait")


def test_launch_prefix_tokens_cannot_break_out_of_the_shell():
    """`docker_host:`/`context:`/`image:` are arbitrary config strings (and may
    come from a `${{ }}` expression), interpolated into a bash script. Unquoted,
    a `;` in one of them became a second command."""
    script = _script(
        MonitorDelivery(
            launch_prefix=("docker", "--host", "h; touch /tmp/pwned", "img name"),
            offhost=True,
        )
    )
    # The authoritative check: the shell must parse the line back into exactly
    # the tokens we intended -- no extra command, no split argument.
    invocation = next(line for line in script if line.startswith("docker"))
    argv = shlex.split(invocation)
    assert argv[:4] == ["docker", "--host", "h; touch /tmp/pwned", "img name"]
    assert argv[4] == "python3"
    # The `;` never becomes a command separator.
    assert ";" not in argv


def test_ordinary_prefix_tokens_are_left_alone():
    """Quoting must not churn the common command into something unreadable."""
    body = "\n".join(
        _script(
            MonitorDelivery(
                launch_prefix=(
                    "docker", "--host", "ssh://svc@10.0.0.11", "run", "--rm",
                    "-i", "--network", "host", "nvcr.io/example/img:1",
                ),
                offhost=True,
            )
        )
    )
    assert (
        "docker --host ssh://svc@10.0.0.11 run --rm -i --network host "
        "nvcr.io/example/img:1 python3 - " in body
    )
