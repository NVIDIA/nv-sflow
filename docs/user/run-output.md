---
title: Understanding run output
sidebar_position: 8.9
---

# Understanding run output

Every run writes everything it produces under one folder:

```text
<output_dir>/<run_id>/
```

`sflow run` prints that path when the run ends — on success **and** on failure, as
soon as the folder exists — together with the summary path and any command-log
paths, so you can jump straight to diagnostics. (A `--dry-run` creates nothing; it
only prints the paths it *would* use.)

This page is a map of what lives in there and which one answers your question. Each
row links to the page with the full detail.

## Start with the summary

`sflow_summary.log` is the single best entry point. It is written live during the
run and finalized when the workflow exits, and it collects in one place:

- workflow status, start/end time, duration, output directory and task counts
- runtime details — package version, binary and Python paths, install mode, repo
  path, and git branch/commit when available
- a task duration timeline and a task event timeline
- probe traces (the last attempt of every readiness/failure probe) when any task
  defines probes
- GPU and node usage charts when resource placement data exists
- failure hints — task name, attempts, reason, and the path to that task's log

See [Outputs & logs → Execution summary](./outputs.md#execution-summary) for an
annotated example.

## What do you want to know?

| Question | Look at | Details |
|---|---|---|
| Did the run succeed, and how long did each task take? | `sflow_summary.log` | [Outputs & logs](./outputs.md#execution-summary) |
| What did my script actually print? | `<task>/<task>.log` | [Outputs & logs](./outputs.md#output-directory-structure) |
| What did sflow itself do — scheduling, status transitions? | `sflow.log` | [Outputs & logs](./outputs.md#output-directory-structure) |
| What exact command was launched (srun / docker / kubectl)? | `*_cmds.log` | [Outputs & logs](./outputs.md#output-directory-structure) |
| What metrics did my benchmark produce? | `<task>/result.json`, `results.json` | [Results](./results.md) |
| Why did readiness never fire? | probe traces in `sflow_summary.log` | [Probes](./probes.md) |
| How busy were the GPUs / CPU / network? | `sflow_monitor.log`, `sflow_monitor/` | [Monitor](./monitor.md) |
| Which GPUs and nodes did each task get? | usage charts in `sflow_summary.log` | [Resources](./resources.md) |
| Which *physical* GPUs did this task really end up on? | `<task>/sflow_gpus.log`, plus the `GPU Assignment` section of `sflow_summary.log` | [Resources](./resources.md) |
| Was the cluster control plane slow or flaky, rather than my job? | `External Command Health` in `sflow_summary.log`, `command_trace.jsonl` | [Outputs & logs](./outputs.md#execution-summary) |
| sflow itself went unresponsive — what was it doing? | `loop_stalls.txt` | [Outputs & logs](./outputs.md#execution-summary) |
| How do I get all this off the cluster? | storage targets + `uploads:` | [Uploads](./uploads.md) |

## The output tree at a glance

```text
<output_dir>/<run_id>/
├── sflow_summary.log      # start here: status, timings, probe traces, failure hints
├── sflow.log              # orchestration + command/status lines (no task stdout)
├── *_cmds.log             # launch commands, grouped by family (bash/slurm/docker/ssh/python)
├── results.json           # workflow-level metric index      (only with `result:`)
├── command_trace.jsonl    # slow/failed external commands    (only when some call was notable)
├── loop_stalls.txt        # all-thread stacks on a driver stall (only when one happened)
├── sflow_monitor.log      # resource overview                (only with `monitor:`)
├── sflow_monitor/         # raw samples; per-task reports only when `report.enabled: true`   (only with `monitor:`)
└── <task>/
    ├── <task>.log         # full per-task stdout/stderr
    ├── result.json        # canonical per-task metrics       (only with `result:`)
    ├── sflow_gpus.log     # physical GPU placement record    (Slurm steps that pick their own devices)
    └── ...                # anything your scripts write
```

A task's stdout/stderr always goes to its own `<task>/<task>.log` and is
deliberately kept out of `sflow.log`, so the orchestration log stays readable. See
[Outputs & logs](./outputs.md#output-directory-structure) for the full contract,
including how Kubernetes log streaming is reconciled before anything parses it.

## Triaging a failure

1. **`sflow_summary.log`** — the failure hint names the task, the attempt count,
   the reason, and the path to that task's log.
2. **`<task>/<task>.log`** — the task's own stdout/stderr, the actual error.
3. **`*_cmds.log`** — confirm the launch command and its flags were what you
   expected (wrong mounts, wrong `--gpus`, missing env).
4. **probe traces** in the summary — if the task hung rather than crashed, these
   show the last readiness/failure probe attempt. See [Probes](./probes.md).
5. **`sflow_monitor/`** — if it was slow rather than broken, check whether the GPUs
   were actually busy. See [Monitor](./monitor.md).

Re-running with `--dry-run` is often the fastest way to confirm a *planning*
problem (node/GPU placement, resolved variables, mounts) without consuming
cluster resources.

## Machine-readable output

Two files are meant to be consumed by other tools rather than read by a human:

- `<task>/result.json` — the canonical per-task result, written after the task
  succeeds.
- `results.json` — the workflow-level index of every task's results.

Both appear only when a task declares `result:`. They are the stable contract for
downstream tasks and external tooling — see [Results](./results.md).

## Beyond the local folder

- [Monitor](./monitor.md) — hardware sampling and reports. Setting `monitor:` writes
  raw CSV samples *and* the per-task charts and summaries; reports are on by default
  (`report: { enabled: false }` opts out).
- [Uploads](./uploads.md) — declare storage targets and per-task `uploads:` to ship
  files to S3 as each task completes, so partial results survive a cancelled run.
