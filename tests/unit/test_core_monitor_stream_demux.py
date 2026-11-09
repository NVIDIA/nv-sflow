# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Demultiplexing a streaming collector's stdout back into raw sample files.

A collector on a node that shares no filesystem with the driver (a container on
a remote Docker host) streams ``<scope>\\t<row>`` instead of writing files. The
driver reassembles the same on-disk layout the post-processor already reads, so
nothing downstream needs to know which transport produced it.
"""

from sflow.core.monitor import StreamedSampleWriter


def test_rows_land_in_per_scope_files_named_for_the_plan_time_node(tmp_path):
    writer = StreamedSampleWriter(tmp_path, node_id="1", node_name="dgx-a")
    writer("cpu\t123.0,7.5,1,2,3")
    writer("memory\t123.0,64,32,32,50,0,0")
    writer("cpu\t124.0,8.5,1,2,3")
    writer.close()

    cpu = tmp_path / "cpu_monitor_node_1_dgx-a.log"
    memory = tmp_path / "memory_monitor_node_1_dgx-a.log"
    assert cpu.read_text().splitlines() == ["123.0,7.5,1,2,3", "124.0,8.5,1,2,3"]
    assert memory.read_text().splitlines() == ["123.0,64,32,32,50,0,0"]


def test_filename_matches_what_the_post_processor_parses(tmp_path):
    """The post-processor keys node identity off the filename, and matches the
    `hostname` field against plan-time node names -- so that field must BE the
    plan-time node name, not whatever the remote machine calls itself."""
    from sflow.monitoring.postprocess_monitor_timeline import MONITOR_FILE_RE

    writer = StreamedSampleWriter(tmp_path, node_id="0", node_name="node-0")
    writer("gpu\t#fields=index,power.draw")
    writer.close()

    path = tmp_path / "gpu_monitor_node_0_node-0.log"
    match = MONITOR_FILE_RE.search(path.name)
    assert match is not None, f"post-processor cannot parse {path.name}"
    assert match.group("resource_type") == "gpu"
    assert match.group("node_id") == "0"
    assert match.group("hostname") == "node-0"


def test_the_gpu_fields_header_is_preserved_verbatim(tmp_path):
    writer = StreamedSampleWriter(tmp_path, node_id="0", node_name="n0")
    writer("gpu\t#fields=index,power.draw")
    writer("gpu\t123.0,0,42")
    writer.close()

    assert (tmp_path / "gpu_monitor_node_0_n0.log").read_text().splitlines() == [
        "#fields=index,power.draw",
        "123.0,0,42",
    ]


def test_untagged_and_unknown_lines_are_ignored(tmp_path):
    """Anything that is not a known scope tag is noise, not a sample. Creating a
    file per stray line would hand the post-processor junk resource types."""
    writer = StreamedSampleWriter(tmp_path, node_id="0", node_name="n0")
    writer("Hardware monitor started successfully: ...")
    writer("")
    writer("bogus\t1,2,3")
    writer("cpu\t123.0,1")
    writer.close()

    assert [p.name for p in sorted(tmp_path.iterdir())] == [
        "cpu_monitor_node_0_n0.log"
    ]


def test_rows_are_flushed_as_they_arrive(tmp_path):
    """A run killed mid-flight must still leave the samples it already took."""
    writer = StreamedSampleWriter(tmp_path, node_id="0", node_name="n0")
    writer("cpu\t123.0,1")
    # Deliberately no close(): simulates the driver dying with the file open.
    assert (tmp_path / "cpu_monitor_node_0_n0.log").read_text() == "123.0,1\n"
    writer.close()


def test_a_tab_inside_the_payload_is_preserved(tmp_path):
    """Only the FIRST tab is the scope separator."""
    writer = StreamedSampleWriter(tmp_path, node_id="0", node_name="n0")
    writer("cpu\t123.0,a\tb")
    writer.close()
    assert (tmp_path / "cpu_monitor_node_0_n0.log").read_text() == "123.0,a\tb\n"


def test_stream_scopes_match_the_collector_that_produces_them():
    """A tag the demultiplexer does not recognise is DROPPED, not logged.

    ``MONITOR_STREAM_SCOPES`` is a deliberate literal copy of the collector's
    ``ALL_SCOPES`` -- hardware_monitor.py must stay importable without sflow, so
    the two cannot share a constant. Nothing but this test stops them drifting,
    and drift costs a whole scope's samples on every streaming node in silence.
    """
    from sflow.core.monitor import MONITOR_STREAM_SCOPES
    from sflow.monitoring.hardware_monitor import ALL_SCOPES

    assert MONITOR_STREAM_SCOPES == set(ALL_SCOPES)


def test_the_separator_matches_the_collector_that_writes_it(tmp_path):
    """The other half of the same two-ended wire protocol.

    ``_TaggedStream`` writes ``<scope><SEPARATOR><row>`` and the demultiplexer
    splits on a hardcoded tab. Change the collector's SEPARATOR and every row
    stops matching a scope, so every sample is dropped in silence -- the same
    failure mode the scope guard above exists for, on the other field.
    """
    from sflow.core.monitor import StreamedSampleWriter
    from sflow.monitoring.hardware_monitor import _TaggedStream

    class _Buffer:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def write(self, text: str) -> None:
            self.lines.append(text)

        def flush(self) -> None:
            pass

    # Produce a row the way the collector really does, then feed it to the
    # demultiplexer the way the driver really does. No literal tab in between.
    buffer = _Buffer()
    _TaggedStream("cpu", buffer).write("123.0,7.5\n")
    emitted = "".join(buffer.lines).rstrip("\n")

    writer = StreamedSampleWriter(tmp_path, node_id="0", node_name="n0")
    writer(emitted)
    writer.close()

    assert (tmp_path / "cpu_monitor_node_0_n0.log").read_text() == "123.0,7.5\n"
