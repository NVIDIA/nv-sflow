---
title: Slurm notes
sidebar_position: 1
---

:::note
This page is **developer-facing background** on Slurm concepts and commands that matter when you build/debug
`sflow` on Slurm. It is not meant to be a complete Slurm tutorial.
:::

## Mental model (the 30-second version)

- **Job / Allocation**: a reservation of resources (nodes/CPUs/GPUs/memory/time). Created by `sbatch` (batch job) or `salloc` (interactive allocation).
- **Job step**: an execution inside an allocation. Created by `srun` (and also the implicit batch script step for `sbatch`).
- **Extern step**: a bookkeeping step Slurm creates automatically for tracking.
- **Interactive step**: the shell session created by `salloc` (unless you use `--no-shell`).

When you call `sacct` you will often see:

| Step type     | Example JobID               | Created by          | What it is                |
| ------------- | --------------------------- | ------------------- | ------------------------- |
| Main job      | `3792871`                   | `salloc` / `sbatch` | The allocation/job record |
| `extern`      | `3792871.extern`            | Slurm               | Tracking                  |
| `interactive` | `3792871.interactive`       | `salloc`            | Your interactive shell    |
| `batch`       | `3792871.batch`             | `sbatch`            | The batch script step     |
| numeric steps | `3792871.0`, `3792871.1`, … | `srun`              | Actual work               |

## Basic: run one command on allocated resources (`srun`)

```bash
srun --label --account=$(sacctmgr show user "$(whoami)" format=DefaultAccount -nP) hostname
```

Tips:

- `--label` prefixes output with task id (useful when multiple tasks run).
- If you’re inside an allocation already, `srun` will run within it.

## Interactive allocation (`salloc`) + steps (`srun`)

Allocate a node and drop into a shell on the allocated resources:

```bash
salloc \
  --time=00:10:00 \
  --nodes=1 \
  --account=$(sacctmgr show user "$(whoami)" format=DefaultAccount -nP) \
  --job-name=general_sflow.dev
```

In that shell, run steps:

```bash
srun hostname
```

Inspect job/step state:

```bash
sacct -P -j <jobid>
```

### What happens when time runs out?

- The allocation ends; your interactive step is cancelled; any running `srun` steps are killed.
- Expect `TIMEOUT` / `CANCELLED` state in `sacct`.

## Batch jobs (`sbatch`) are asynchronous by default

`sbatch` returns immediately with a job id. The batch script runs _later_ on the allocated compute node(s).

Minimal pattern:

```bash
sbatch --job-name=sflow --output=sflow-%j.out --wrap "cd $SLURM_SUBMIT_DIR && sflow run --file sflow.yaml"
```

Notes for devs:

- The batch script itself runs inside the allocation, so tools that “detect existing allocation” should work.
- Always ensure your output dir is on a **shared filesystem** if you want to watch logs while the job runs.

## Automation-friendly allocation: `salloc --no-shell` + `srun --jobid`

`salloc` without `--no-shell` drops you into a shell (bad for automation because the subprocess blocks).
For automation (and for `sflow`-like tooling), prefer:

```bash
salloc \
  --time=00:10:00 \
  --nodes=1 \
  --account=$(sacctmgr show user "$(whoami)" format=DefaultAccount -nP) \
  --job-name=general_sflow.dev \
  --no-shell
```

This keeps you on the login node while holding an allocation. To run steps inside that allocation you must
attach `--jobid=<jobid>`:

```bash
srun --jobid=<jobid> hostname
```

## Can multiple `srun` steps run in parallel?

**Sometimes.** Multiple steps _can_ run concurrently inside the same allocation, but only if:

- the allocation has enough free resources for the new step request, **and**
- your cluster/job settings allow steps to **share/overlap** resources (often gated by `srun --overlap`), **and**
- you are not forcing exclusivity (e.g. `--exclusive`) or otherwise consuming all resources in the first step.

Terminal A:

```bash
srun --jobid <jobid> --label --job-name general_sflow.demo.stepA bash -lc 'echo "A start $(date)"; sleep 60; echo "A end $(date)"'
```

Terminal B:

```bash
srun --jobid <jobid> --label --job-name general_sflow.demo.stepB bash -lc 'echo "B start $(date)"; sleep 60; echo "B end $(date)"'
```

If overlap is allowed and resources are available, `sacct -P -j <jobid>` will show separate numeric step ids (`<jobid>.0`, `<jobid>.1`, …) that run concurrently.

If overlap is _not_ allowed (or the earlier step effectively holds the resources), the second `srun` may block and you may see messages like "nodes are busy" / step creation retrying until resources are released.

## Logging: use `--output` and timestamping

Example with `--output=%J.log` (jobid-based filename) and line timestamps:

```bash
srun --jobid <jobid> \
  --label \
  --job-name general_sflow.demo.echo_time \
  --output=%J.log \
  bash -lc '(echo "Start time: $(date)"; sleep 10; echo "End time: $(date)") | awk "{print strftime(\"%Y-%m-%d %H:%M:%S\"), \$0; fflush()}"'
```

## Heterogeneous allocations (het jobs)

You can request different resource “groups” in one allocation:

```bash
salloc --nodes=2 --time=00:20:00 --account=$(sacctmgr show user "$(whoami)" format=DefaultAccount -nP) : --nodes=1 --time=00:20:00
```

This produces het groups (e.g. `3752946+0`, `3752946+1`).

## Debugging: `scontrol show node` / `scontrol show job`

Quick node inspection:

```bash
scontrol show node <node>
```

Quick job inspection:

```bash
scontrol show job <jobid>
```

:::tip
When you are debugging “works on login but fails on compute”, check `Arch=` in `scontrol show node` for the compute nodes.
Mixed-arch clusters are a common cause of Python `Exec format error` when reusing a venv created on a different architecture.
:::

## Why this matters for `sflow`

These are the Slurm behaviors `sflow` relies on:

- **Reuse existing allocation**: if `sflow` runs inside an allocation (interactive `salloc`, or inside an `sbatch` job), it should reuse it rather than creating a new one.
- **Non-interactive allocation**: for automation, `--no-shell` avoids blocking on an interactive shell.
- **Step naming**: giving `srun` a meaningful `--job-name` helps operators correlate Slurm steps to workflow tasks.
- **Parallel steps**: `sflow` may launch multiple tasks concurrently; Slurm supports parallel steps inside one allocation.
