---
title: Outputs & logs
sidebar_position: 9
---

`sflow` creates a consistent output directory layout and injects built-in env vars into every task.

## Output directory structure

Default output root is `./sflow_output/` (relative to `--workspace-dir`, default: current directory).

For a real run (non dry-run):

- `<output_dir>/<run_id>/sflow.log`: global sflow log
- `<output_dir>/<run_id>/sflow_summary.log`: live execution summary, updated during the run and finalized when the workflow exits
- `<output_dir>/<run_id>/*_cmds.log`: command-only launch logs, grouped by command family such as `bash`, `slurm`, `docker`, `ssh`, or `python`
- `<output_dir>/<run_id>/<task>/<task>.log`: per-task log
- `<output_dir>/<run_id>/...`: anything your scripts write

Dry-run does not `mkdir` anything; it only prints planned output paths.

After a successful run, `sflow run` prints the output folder, summary path, and any command-log paths. When a run fails or is interrupted after the workflow output directory exists, the same paths are printed on the error path so you can jump straight to diagnostics.

## Execution summary

`sflow_summary.log` is a terminal-friendly status report for the whole run. It is useful for quick triage because it collects the most important details in one place:

- workflow status, start/end time, duration, output directory, and task counts
- executable/runtime details, including package version, binary path, Python path, install mode, repo path, and git branch/commit when available
- task duration timeline and task event timeline
- GPU and node usage charts when resource placement data exists
- command-log paths
- workflow DAG and dependency list
- failure hints with task name, attempts, reason, and task log path when a task fails or is cancelled

Example `sflow_summary.log`:

```text
Sflow Summary
=============
Workflow : quickstart_dag
Status   : COMPLETED
Started  : 2026-05-22T12:31:32+08:00
Ended    : 2026-05-22T12:31:41+08:00
Duration : 9.017s
Output   : /workspace/sflow_output/quickstart_dag-20260522-123132-1ba51e
Tasks    : 6
Summary  : /workspace/sflow_output/quickstart_dag-20260522-123132-1ba51e/sflow_summary.log
Counts   : COMPLETED=6

Runtime
-------
sflow executable:
  version : 0.2.2.dev7+g0858dce39.d20260522
  bin     : /workspace/.venv/bin/sflow
  python  : /workspace/.venv/bin/python
  package : /workspace/.venv/lib/python3.12/site-packages/sflow
  install : direct-url
  source  : https://github.com/NVIDIA/nv-sflow.git@develop

Task Duration Chart
-------------------
prepare_data         |###...........................| 1.002s COMPLETED
preprocess           |.......####...................| 1.002s COMPLETED
train                |..............####............| 1.001s COMPLETED
evaluate_on_dataset1 |.....................#####....| 1.004s COMPLETED
evaluate_on_dataset2 |.....................#####....| 1.003s COMPLETED
export_model         |............................##| 0.002s COMPLETED

Timeline
--------
Time      Elapsed   Task                  Event      Summary
--------  --------  --------------------  ---------  -------------------------------
12:31:33  +01.001s  prepare_data          SUBMITTED  attempt=1
12:31:34  +02.003s  prepare_data          COMPLETED  exit=0
12:31:37  +05.007s  train                 SUBMITTED  attempt=1
12:31:38  +06.008s  train                 COMPLETED  exit=0
12:31:41  +09.017s  export_model          COMPLETED  exit=0

Command Logs
------------
bash: /workspace/sflow_output/quickstart_dag-20260522-123132-1ba51e/bash_cmds.log

Dependencies
------------
START -> prepare_data
prepare_data -> preprocess
preprocess -> train
train -> evaluate_on_dataset1
train -> evaluate_on_dataset2
evaluate_on_dataset1, evaluate_on_dataset2 -> export_model
```

## Command logs

Command logs record launch commands without mixing in task stdout/stderr. They are grouped by command family and written only when matching commands are executed:

- `slurm_cmds.log` for `salloc`, `srun`, `scontrol`, `scancel`, and `sbatch`
- `bash_cmds.log` for `bash` / `sh`
- `docker_cmds.log` for Docker commands
- `ssh_cmds.log` for SSH commands
- `python_cmds.log` for Python commands
- `backend_cmds.log` for other backend commands

Each entry includes a timestamp, command family, task name when applicable, whether it used a shell, and the formatted command. Use these logs to reproduce launch commands or verify generated Slurm/container flags without scanning full task logs.

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
