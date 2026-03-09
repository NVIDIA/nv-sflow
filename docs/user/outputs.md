---
title: Outputs & logs
sidebar_position: 9
---

`sflow` creates a consistent output directory layout and injects built-in env vars into every task.

## Output directory structure

Default output root is `./sflow_output/` (relative to `--workspace-dir`, default: current directory).

For a real run (non dry-run):

- `<output_dir>/<run_id>/sflow.log`: global sflow log
- `<output_dir>/<run_id>/<task>/<task>.log`: per-task log
- `<output_dir>/<run_id>/...`: anything your scripts write

Dry-run does not `mkdir` anything; it only prints planned output paths.

## Built-in env vars

These are always available inside task scripts:

- `SFLOW_WORKSPACE_DIR`: workspace root
- `SFLOW_OUTPUT_DIR`: output root (default: `<workspace>/sflow_output`)
- `SFLOW_WORKFLOW_OUTPUT_DIR`: per-run root (where `sflow.log` lives)
- `SFLOW_TASK_OUTPUT_DIR`: per-task dir (where `<task>.log` lives)

Example pattern:

```yaml
workflow:
  name: wf
  tasks:
    - name: write_files
      script:
        - echo "hello" > ${SFLOW_WORKFLOW_OUTPUT_DIR}/hello.txt
        - echo "task" > ${SFLOW_TASK_OUTPUT_DIR}/task.txt
```

## `task.outputs`: parse metrics from task logs (MVP)

In v0.1, `task.outputs` is supported as a **best-effort** “metrics extraction” mechanism:

- You declare one or more **parse-style patterns**
- After a task completes successfully, `sflow` scans the task log and extracts named fields
- The parsed outputs are written to `${SFLOW_TASK_OUTPUT_DIR}/outputs.json`

### Example: extract TTFT and throughput

```yaml
workflow:
  name: wf
  tasks:
    - name: benchmark
      script:
        - echo "TTFT: 42.5 ms"
        - echo "tok/s: 123.0"
      outputs:
        - pattern: "TTFT: {ttft:f} ms"
        - pattern: "tok/s: {tps:f}"
```

Result file:

- `${SFLOW_TASK_OUTPUT_DIR}/outputs.json`

It looks like:

```json
{
  "task": "benchmark",
  "specs": [
    { "pattern": "TTFT: {ttft:f} ms", "source": "stdout" },
    { "pattern": "tok/s: {tps:f}", "source": "stdout" }
  ],
  "outputs": {
    "ttft": 42.5,
    "tps": 123.0
  }
}
```

### Semantics (current MVP behavior)

- **Where it parses from**: the merged task log file (`${SFLOW_TASK_OUTPUT_DIR}/${task}.log`)
- **When it runs**: only after the task finishes with exit code 0
- **Multiple matches**: if the same key appears multiple times, you get a list; otherwise a scalar
- **Failure behavior**: missing log / parse errors return `{}` (best-effort; workflow does not fail)

## Common gotchas (worth knowing)

- **Parallel tasks writing the same file**: if two tasks run in parallel and both write to the same path under
  `${SFLOW_WORKFLOW_OUTPUT_DIR}` (e.g. `metrics.txt`), you'll have a race/overwrite. Prefer either:
  - write per-task files under `${SFLOW_TASK_OUTPUT_DIR}`, or
  - give each task a unique filename under `${SFLOW_WORKFLOW_OUTPUT_DIR}`.
