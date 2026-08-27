# sflow v0.3.1 Release Notes

**Release date:** August 2026  
**Previous release:** [v0.3.0](https://github.com/NVIDIA/nv-sflow/releases/tag/v0.3.0) (July 2026)

---

## Highlights

v0.3.0 made a recipe portable across Local, Docker, Slurm, and Kubernetes. v0.3.1 is about trusting the run it produces: **knowing which physical GPU each task actually landed on**, and **not losing the run to a silent hang or a truncated log**.

Three things drive the release. **GPU placement** became explicit — you can pin device indices, sflow re-applies its plan *inside* the Slurm step where `slurmstepd` used to overwrite it, and concurrent runs on one box no longer collide on GPU 0. **The driver stopped hanging** — every `kubectl` call is bounded, console output is length-capped, an event-loop watchdog captures stalls that previously left no trace at all, and the post-run log re-fetch that silently truncated long-running pods is gone. **Monitoring became aimable** — reports are on by default, a `window:` clips them to the phase you actually measured, and GPU charts draw one line per device instead of an average.

| Area | v0.3.0 | v0.3.1 |
| --- | --- | --- |
| **GPU selection** | `gpus.count` only, index-agnostic | **+ `gpus.indices`** to pin device ids; `count` must divide the node count |
| **Slurm GPU placement** | `slurmstepd` overwrote `CUDA_VISIBLE_DEVICES` → every rank on GPU 0 | in-step re-select **by UUID** + a per-task `sflow_gpus.log` audit record |
| **Concurrent runs on one host** | each run packed from device 0, unaware of the others | machine-local **UUID reservation registry** (Docker) |
| **Kubernetes task logs** | post-run re-fetch, truncated to the last rotation window | exactly what `kubectl logs -f` delivered |
| **Kubernetes control plane** | one `get pod` per pod per tick; no call bounded | batched per context; **every call timeout-bounded** |
| **Monitoring** | reports opt-in, one averaged GPU line | reports **on by default**, `window:` markers, one line per device |
| **Merge-pod + intra-group deps** | rejected outright | **gated in-pod**, the run proceeds |
| **Upgrading sflow** | hand-written `uv pip install` incantation | **`sflow upgrade`** |

> **Read the [Breaking Changes](#breaking-changes) before upgrading.** Despite the patch version, this release changes several defaults that affect existing recipes — most notably monitor report output, GPU count validation, and resource release timing. The [Upgrade Guide](#upgrade-guide) has a five-minute checklist.

---

## New Features

### 1. Pin GPUs by device index — `resources.gpus.indices`

`resources.gpus` used to take only a `count`, which is index-agnostic: the planner packs the task into any contiguous idle run on one node. That is the right default, but it cannot express NUMA/NVLink affinity, reproduce a vendor benchmark's exact topology, or steer around a known-bad device.

```yaml
resources:
  gpus:
    indices: [2, 3]        # pin these device ids
```

Three modes, all on Local, Docker, and Slurm:

| You write | You get |
| --- | --- |
| `count: 4` | any contiguous idle run of 4 on a single node (never straddles a node boundary) |
| `indices: [2, 3]` | the first node where **both** are free; the node scan restarts at node 0, so later tasks backfill |
| `count: 8` + `indices: [0, 1]` | fan-out — `count` is the **total across nodes**, `indices` the per-node slice, so the task spans `8 / 2 = 4` nodes |

`CUDA_VISIBLE_DEVICES` preserves the order you wrote. Indices must be non-negative, unique, and non-empty — deliberately unlike `nodes.indices`, which allows `-1`.

**Kubernetes rejects `indices` at plan time.** The cluster's device plugin or DRA assigns the physical devices, so sflow will not pretend otherwise; use `count` to size the request.

Samples: `sflow sample self_contained/slurm/gpu_indices` (all three modes on a 4×4 board, with the expected placement map annotated) and `self_contained/slurm/gpu_placement_matrix` (a regression matrix that asserts placement by UUID, not by GPU count).

### 2. Slurm: the GPU plan is re-applied inside the step

On a GRES partition, `slurmstepd` overwrites the `CUDA_VISIBLE_DEVICES` sflow exports. Every step then saw the whole allocation and every rank picked device 0 — tasks planned onto different GPUs all piled onto the same one, and the run's numbers were quietly wrong rather than loudly broken.

sflow no longer trusts the inherited value. A prelude sourced inside each Slurm step probes the devices the step can really see, looks up the physical UUIDs the driver resolved this task's plan to, and re-exports the indices those same cards have *there*. Matching by UUID rather than by index is also what fixes pyxis/enroot containers, which renumber devices from 0.

On by default for any Slurm task with a `resources.gpus` slice; srun `gpus_per_task` opts out, since Slurm already carves per rank.

- **`<task>/sflow_gpus.log`** — a per-task audit record: planned indices and UUIDs, the inherited environment, the visible index→UUID map, and the final selection. Multi-node tasks write one per node.
- **`exit 97`** — a planned card not visible at all, a step holding fewer GPUs than planned, or a planned slot out of range now aborts the step instead of silently running on the wrong device. When *Slurm* chose the devices rather than sflow, this degrades to index arithmetic instead of failing.

See [Backends → GPU placement inside the step](https://nvidia.github.io/nv-sflow/docs/user/backends).

### 3. Concurrent runs on one host stop colliding (Docker)

Several `sflow run` processes on one machine each packed GPUs from device 0 independently — the in-process planner had no idea another run existed. Two runs on one workstation fought over the same cards.

A file-locked registry in machine-local temp now makes concurrent runs, and concurrent tasks within a run, claim **disjoint physical GPUs by nvidia-smi UUID**. It refuses to take a GPU a foreign workload is already on. The container is launched with `--gpus "device=<uuid,…>"` naming those exact cards.

**On by default for the Docker backend**, with no YAML opt-in. It is skipped — each case logging which one you hit — for remote `hosts:` pools, any `nodes > 1`, and hosts where `nvidia-smi` will not run. It requires POSIX `fcntl`, so it is inert on Windows.

| Environment variable | Default | Effect |
| --- | --- | --- |
| `SFLOW_GPU_RESERVATION` | `1` | `0` disables the registry entirely |
| `SFLOW_WAIT_FOR_GPUS` | unset | any value turns waiting on; unset restores fail-fast |
| `SFLOW_GPU_RESERVATION_DIR` | `$TMPDIR/sflow-gpu-reservations` | must stay machine-local — on NFS another host's records read as local |
| `SFLOW_GPU_BUSY_MEM_MIB` | `512` | how much foreign memory counts as "busy" |
| `SFLOW_GPU_IGNORE_FOREIGN` | unset | `1` when sflow owns the box |

Reservations are held per task, not per run, and released in a `finally` bounded at 10s so `Ctrl-C` can never hang on the lock. Records whose owning PID is gone are reaped on the next claim, guarded against PID recycling; another user's records are never reaped.

**New: `--wait-for-gpus <seconds>`** (Docker) — when too few GPUs are free at reserve time, wait instead of failing fast. `0` waits forever, `N` bounds the wait, omitting it fails fast. Also settable per-recipe as the backend field `wait_for_gpus`.

### 4. Kubernetes: the driver stops hanging, and logs stop lying

None of this needs a config change.

- **Task logs are now the streamed log.** The post-run one-shot `kubectl logs` re-fetch that replaced the streamed file is gone. Because the kubelet **rotates container logs**, that re-fetch returned only the last window — one hour of server output was persisted as its final ~11 seconds.
- **Every `kubectl` call is bounded** (30s poll, 300s delete) and retried on the next tick. A silently-dead TCP connection to the API server used to wedge the driver for 15–20 minutes with nothing logged.
- **Pod-status polling is batched** — pods in one context share a single `kubectl get pod a b c …` per tick. Status was ~90% of all kubectl traffic (862 of 957 calls in a measured 7-pod run), and at 19 concurrent recipes that traffic self-congested (mean `get pod` latency 0.2s → 1.3s). Terminal detection is at most ~1.5s staler.
- **The output collect is bounded and no longer intrudes into running pods.** It emits a heartbeat every 30s and, on timeout, names `collect_grace_seconds` and `collect_node_local_output` in the message. It no longer `kubectl exec`s into a live pod — that used to put 42 execs into a serving TRT-LLM pod.
- **`collect_node_local_output: false`** (backend, default `true`) turns the collect machinery off entirely: no in-pod `EXIT` trap, no driver-side copy. Task completion then depends only on pod status, probes, and the merge-pod marker, and outputs must reach you via a shared filesystem, `uploads:`, or a PVC.
- **The RBAC preflight no longer blocks on permissions sflow does not need.** `get nodes` and `get deviceclasses` are now *optional* — a denial warns and degrades node-level detection (set `gpus_per_node` explicitly) instead of failing the run. A namespace-scoped ServiceAccount on a shared multi-tenant cluster now works out of the box.
- **`--extra-kubectl-apply-args`** (repeatable) passes a flag to the `kubectl apply` **subcommand** (`--validate=false`, `--server-side`, `--force-conflicts`). kubectl takes global flags *before* the verb, so these cannot ride on `--extra-kubectl-args` — sflow now warns if it spots an apply-only flag there.

### 5. Merge-pod accepts dependencies between its own members

v0.3.0 refused to merge tasks that depended on each other: merged tasks run concurrently in one pod, so a benchmark depending on a co-located server could not use merged pods at all — exactly the case merge-pod exists for.

A **direct** member→member edge is now honored by gating rather than rejected. The dependent member waits on an in-pod gate; the driver opens it when the dependency reaches READY or COMPLETED. If the dependency failed, the gated member never starts and propagates its exit code. A member reachable only *transitively through a non-member* is still rejected.

Visible in `sflow_summary.log` as `gated_on=<deps>` on the SUBMITTED row and a new `UNGATED` timeline event. A gated member's duration is measured from gate-open, not submission — a 5-second client no longer reads as 40 seconds.

### 6. Monitoring you can aim

- **Reports are on by default.** Declaring `monitor:` now writes the report folders; `report: {enabled: false}` opts out. Worth doing on large fan-outs — a report folder is a per-view *copy* of the samples, so cost scales with `samples × views`.
- **`monitor.window`** clips a report to the phase you actually measured, using markers from the task's own log:

  ```yaml
  monitor:
    window:
      start: "Benchmark starting"
      end:   "re:Total throughput: [0-9.]+"
  ```

  Plain strings are case-sensitive literal substrings; prefix `re:` or `regex:` for a regex. `start` resolves first and `end` only from matches strictly after it. Task monitors only. The collector still runs for the whole task — only the report is clipped, so `sflow_monitor/lifecycle/` and `sflow_monitor/windowed/` sit side by side. If a marker never matches, sflow **warns and skips that report** rather than silently falling back to lifecycle timing, and writes `window_not_found.json` so you can see which pattern missed.
- **GPU charts draw one line per device**, each labelled with its own avg/max, instead of one averaged line that hid an idle card. Node-level scopes (cpu/memory/disk/network) stay averaged.
- **Multi-node reports split per node** — `timeline.<hostname>.svg` per node instead of one combined chart. CSVs stay combined.
- **Task-event markers were redesigned** — labels drawn in place with no legend to decode, near-simultaneous events merged into one labelled rule (`3 tasks submit +2 more`). Dotted = started, solid = ended.
- **Clock skew is corrected and warned about** — samples are shifted onto the driver's clock for reporting only when the estimated node offset excludes zero. Raw logs keep node timestamps.
- **Coverage fixes:** the workflow-level monitor now spans every monitorable backend instead of only the default one; a task on a backend no monitor covers no longer gets a bogus empty report folder; and a task that reserved no GPUs no longer picks up its node's GPUs in its report.

### 7. `sflow upgrade`

Reinstall sflow in place without hand-writing the `uv pip install` incantation:

```bash
sflow upgrade                                # latest main of the public repo
sflow upgrade --branch develop               # a specific ref
sflow upgrade --sflow-index-url <url>        # a private PyPI index
sflow upgrade --sflow-source-path ~/src/sflow  # editable, from a local checkout
sflow upgrade --dry-run                      # print the resolved command and stop
```

Prefers `uv`, falls back to `pip`. It refuses to upgrade over an editable/source-tree dev install unless you pass `--force`. `sflow update` is an alias.

Note the deliberate asymmetry: bare `sflow upgrade` installs **`main` of the public OSS repo**, whereas `sflow batch` installs whatever ref the *running* environment came from.

### 8. Run output that explains itself

- **`GPU Assignment`** in `sflow_summary.log` — per task, the physical GPUs it was planned onto next to the devices it actually saw, with a hint when the backend re-indexes inside the container. GPU/node charts now plot physical devices; previously every Docker task was drawn on GPU 0.
- **`Node Topology`** — the CPU/NUMA/GPU probe each backend captured at reservation time.
- **`External Command Health`** — call counts, failures, timeouts, and mean/max latency for `kubectl`/`srun`/`docker`, with a `healthy`/`DEGRADED` verdict. Written even when the run is cancelled or fails, which is when it matters. Backed by `command_trace.jsonl`, written lazily and only for *notable* calls (non-zero exit, or slower than 5s), so a healthy run leaves no file. A live warning fires when a control-plane call takes over 5s.
- **`loop_stalls.txt`** — if the driver's event loop stops being scheduled for 30s, sflow logs a warning and dumps every thread's Python stack here. Previously such a freeze produced no diagnostic output at all, because sflow's own logging runs on the thread that was stuck. Created only when a stall actually happens.
- **Console output is length-capped at 2000 characters per line**, on *every* backend. A single unbounded line was measured at ~6.3µs and ~300 bytes of RSS per character — a 48MB line cost ~5 CPU-minutes and ~14GB and froze the driver's event loop. **`<task>.log` is unaffected and still holds every byte**; probes, `result:` parsing, and `output:` all read the file, so only the terminal changes.
- **Progress bars that end on a carriage return keep their final frame.** A bar whose last redraw ended in `\r` used to vanish from both the console and `<task>.log`, and two consecutive redraw-terminated reads spliced into `50%60%` — in the log file, not just the console.
- **DAG cycle errors name the loop edge by edge** in `depends_on` phrasing, with a separate "Waiting behind it:" list, instead of a bare `Graph contains a cycle`.
- **`--dry-run`** now lists every planned monitor report folder with its group and, for marker windows, the patterns — so a marker typo surfaces before the run rather than after, as an empty report.
- **New doc page:** [Understanding run output](https://nvidia.github.io/nv-sflow/docs/user/run-output) — the output tree, a "what do you want to know?" lookup table, and a five-step failure triage. `sflow run` now prints the run directory on failure too, as soon as the folder exists.

### 9. Kubernetes MPI: per-rank CPU binding

`mpi.cpu_bind` (`core` | `numa` | `none`, default `core`) injects per-rank CPU binding, but **only when several ranks share a pod**, and never over a binding your recipe already passes. `core` gives each rank an isolated core slice — the tightest cap on the LLVM/OpenMP thread pools that `OMP_NUM_THREADS` alone does not reach.

`mpi.cpu_bind_cores_per_rank` (default `8`, `0` = uncapped) bounds that slice: the launch-time value is `min(cores-in-cpuset / ranks-per-pod, this)`, and if the cpuset has fewer cores than ranks the binding is skipped rather than failing the launch.

---

## Improvements

- **Docker:** CPU-only tasks no longer see the host's GPUs (many CUDA images bake `NVIDIA_VISIBLE_DEVICES=all`). Container names carry the driver PID (`sflow-p<pid>-<task>-<node>`) so concurrent runs never collide, and orphaned containers from a dead driver are reaped once per run. `--gpus device=0,1` is now quoted — docker used to parse the trailing `1` as a *count* and die with *"cannot set both Count and DeviceIDs on device request"*. A raw `--gpus` grant in `extra_args` now warns, because docker *accumulates* device requests and would widen the container past its reservation.
- **Slurm:** `extra_args` no longer silently drops repeated values — a bare value in `["-G","1","-N","1"]` used to delete an earlier identical one and produce a wrong allocation. A new driver-side topology probe records the index→UUID map per node.
- **`sflow batch`:** compound expressions such as `${{ variables.NUM_NODES * 2 }}` now resolve, so the generated sbatch no longer diverges from the dry run; a `--set` node override reaches `#SBATCH --nodes`; config errors keep their full multi-line pydantic detail instead of being cut to `Configuration validation failed:`; and `.cache` is excluded from the source-tree copy, which used to make rsync exit 24 and kill the job seconds in.
- **`--skip-artifact-check`** (`sflow run` and `sflow batch`) — a missing `fs://` path warns instead of failing, and is left alone rather than created as an empty directory. For paths that exist only where the task runs. `sflow batch` forwards it into the job, which is where the check actually runs.
- **Container image preflight was loosened** — an unrecognized reference now warns instead of aborting, and the regex accepts pyxis/enroot forms such as `nvcr.io#nvidia/ai-dynamo/sglang-runtime:1.2.0`. **Recipes that failed preflight on v0.3.0 now run.**
- **A readiness probe written with `match_pattern` no longer fails its own dry run.** The validator normalizes it into `regex_pattern`, so any dump-and-reload round trip (`sflow compose`, or the temp config `sflow batch` writes) handed both back and tripped the "only one of" check.
- **TUI:** the header no longer clips the elapsed clock and output directory, and ticks once a second so the clock does not freeze on an idle run.
- **Packaging:** the `parse` pin was loosened from `==1.16.0` to `>=1.16,<2`, which had made sflow uninstallable alongside anything needing a newer `parse`. A bare `pytest` no longer pulls images and launches containers.

---

## Breaking Changes

Ordered by how likely they are to affect an existing v0.3.0 recipe.

1. **`monitor.report.enabled` now defaults to `true`.** Any recipe with a bare `monitor:` block now writes report folders — disk and post-processing you did not previously pay for. *Migration:* `report: {enabled: false}` to opt out.
2. **Monitor report paths gained a group segment.** `sflow_monitor/<task>/` → `sflow_monitor/lifecycle/<task>/` (or `windowed/<task>/`). `sflow_monitor/raw/` and `sflow_monitor.log` are unchanged. *Migration:* **any scraper, CI glob, or notebook reading `sflow_monitor/<task>/summary.csv` breaks** — insert the group segment.
3. **`resources.gpus.count` must now divide the assigned node count.** Previously the planner rounded *up* and reserved that many on **every** node: `nodes: 2` + `gpus: 1` silently consumed 2 GPUs, and `nodes: 2` + `gpus: 3` consumed 4; a `count: 10` against `(8, 2)` caps silently under-allocated to 4. Both are now rejected at plan time with a message naming a count that works. *Migration:* multiply by the node count — `nodes: 2` + `gpus: 1` becomes `gpus: 2`. Backend-agnostic.
4. **`fail_fast` now defaults per backend — `true` on Kubernetes.** A Kubernetes shell task whose script had a failing command masked by a later successful one (a trailing `echo`) now fails the task. Local, Docker, and Slurm are unchanged at `false`, and an explicit `fail_fast:` in the task always wins.
5. **`gpus.release_after` and `nodes.release_after` no longer default to `workflow_completion`** — an omitted value is now inferred. A **probe-less** GPU task that used to hold its GPUs for the whole workflow now releases at task completion, so downstream tasks may be packed onto them. An omitted node policy now means *placement only* (may overlap with other planned tasks) rather than exclusive. *Migration:* set `release_after: workflow_completion` explicitly to restore v0.3.0 behavior.
6. **A readiness-probed service that exits before becoming READY is now FAILED, even on exit 0.** It was previously marked COMPLETED, wrongly unblocking dependents against a dead server. sflow forces one final readiness scan first to avoid a false failure. The reason string changes from `process exit` to `service exited before readiness`. All backends. Probe-less tasks, and services that reached READY and then exited cleanly, are unaffected.
7. **Kubernetes `<task>.log` is the streamed log, not a rebuilt one.** A multi-pod task's log is now ordered **chronologically, interleaved across pods**, instead of grouped per pod. `kubectl logs --prefix` still tags every line with its pod. *Migration:* a `result:` pattern that takes the *last* match now takes the last one **in time** across all pods — match on the pod prefix if you need a specific pod's value.
8. **Docker container names changed** — `sflow-<task>-<node>` → `sflow-p<pid>-<task>-<node>`. *Migration:* update any script matching the old name.
9. **Docker GPU tasks now fail fast when the host has no free GPUs**, because reservation is on by default. Foreign workloads above 512 MiB count as busy, so **on a workstation with an attached display every GPU can read busy**. The error names the busy GPUs and the escape hatches. *Migration:* `--wait-for-gpus`, `SFLOW_GPU_IGNORE_FOREIGN=1`, `SFLOW_GPU_BUSY_MEM_MIB`, or `SFLOW_GPU_RESERVATION=0`.
10. **Docker CPU-only containers no longer see GPUs** (`NVIDIA_VISIBLE_DEVICES=void`). Skipped when `extra_args` already grant GPUs.
11. **Slurm no longer exports `NVIDIA_VISIBLE_DEVICES` to srun steps.** Containers see all of the node's GPUs, so NVML consumers (`nvidia-smi`, DCGM) lose device isolation — the trade that makes the planned host-numbered slice addressable at all. Docker keeps isolation via `--gpus device=<uuid>`.
12. **New Slurm in-step abort `exit 97`** when the planned GPUs cannot be honored — see [Feature 2](#2-slurm-the-gpu-plan-is-re-applied-inside-the-step).
13. **Kubernetes rejects `resources.gpus.indices` at plan time**, with a message pointing at `count`.
14. **Relative inline-content `file://` artifact URIs that escape the run output dir now hard-error.** `output_dir / raw` did not previously collapse `..`, so `file://../../x` silently wrote outside the run directory. Narrowly scoped: relative `file://` **with inline content** only — `fs://` model mounts and absolute `file://` are untouched.
15. **`sflow batch --bulk-input` refuses a `--nodes` that disagrees with the CSV node column.** `--nodes` sizes the sbatch allocation while the config's number sizes the workflow, so two different numbers allocate one size and plan another. Config-driven paths warn instead of refusing.
16. **`sflow batch` rejects a CSV data row with a blank `sflow_config_file`** — previously a raw `AttributeError` traceback, so this is strictly better, but it is a new hard failure.
17. **`timeout:` now warns on every load.** It was never enforced — no code path reads it, and `TaskStatus.TIMEOUT` is never assigned — but v0.3.0 said nothing. It is still accepted so existing recipes keep loading. *Migration:* bound the run with the backend's own limit (Slurm `--time`). The README and user docs have been corrected in this release; they previously implied it worked.

---

## Tested Environments & Current Support

**Validated setups**

- vanilla bare-metal Kubernetes
- Google Kubernetes Engine (GKE)
- Slurm (GRES and non-GRES partitions)

**Current limitations / work in progress**

- **Kubernetes hardware monitoring is still not supported.** The built-in bare-node `monitor:` feature covers Local, Docker, and Slurm. On Kubernetes, monitor blocks are skipped because there is no DCGM/DaemonSet collector; sampling the driver host would produce misleading data. Unchanged from v0.3.0.
- **`resources.gpus.indices` is not supported on Kubernetes** — the device plugin or DRA assigns physical devices. Use `count`.
- **The GPU reservation registry covers the Docker backend only.** The Local backend also runs on the host but does not participate, so a Local run and a Docker run on the same box can still overlap.
- **DRA GPU allocation is supported but still WIP** — implemented, not yet broadly validated across Kubernetes distributions, versions, and NVIDIA DRA deployments. The device-plugin path remains the default.
- **Kubernetes execution is driver-attached.** Use interactive `sflow run`; detached Kubernetes batch execution is not supported.
- **The GPU reservation registry requires POSIX `fcntl`** and is inert on Windows.

---

## Documentation

Updated for this release: [Backends](https://nvidia.github.io/nv-sflow/docs/user/backends) (Slurm GPU placement, Docker GPU reservation, `collect_node_local_output`, `mpi.cpu_bind`), [CLI](https://nvidia.github.io/nv-sflow/docs/user/cli) (`sflow upgrade`, `--wait-for-gpus`, `--extra-kubectl-apply-args`, `--skip-artifact-check`, `batch --nodes` conflicts), [Resources](https://nvidia.github.io/nv-sflow/docs/user/resources) (`gpus.indices`), [Monitor](https://nvidia.github.io/nv-sflow/docs/user/monitor) (`window:`, report defaults), [Outputs & logs](https://nvidia.github.io/nv-sflow/docs/user/outputs) (new summary sections, `command_trace.jsonl`, `loop_stalls.txt`), and the new [Understanding run output](https://nvidia.github.io/nv-sflow/docs/user/run-output) page.

Full documentation: [nvidia.github.io/nv-sflow](https://nvidia.github.io/nv-sflow/)

---

## Upgrade Guide

```bash
# From an existing install
sflow upgrade

# Or fresh
uv pip install "sflow @ git+https://github.com/NVIDIA/nv-sflow.git@main"

# Re-plan every recipe without consuming resources -- this surfaces the two
# breaking changes most likely to bite (gpus.count divisibility, gpus.indices on k8s)
sflow run -f your_recipe.yaml --dry-run

# New samples worth a look
sflow sample self_contained/slurm/gpu_indices
sflow sample self_contained/slurm/monitor_mixed
sflow sample self_contained/docker/gpu_monitor
```

**A five-minute upgrade checklist:**

1. `--dry-run` every recipe — `gpus.count` divisibility and `gpus.indices`-on-Kubernetes both fail at plan time, before anything is allocated.
2. Grep your tooling for `sflow_monitor/<task>/` and insert the `lifecycle/` or `windowed/` segment.
3. If you have a bare `monitor:` on a large fan-out, add `report: {enabled: false}`.
4. If a Kubernetes shell task relies on a trailing command masking an earlier failure, set `fail_fast: false` on it.
5. If a probe-less GPU task needs to hold its GPUs for the whole run, set `release_after: workflow_completion` explicitly.
6. If you parse a multi-pod Kubernetes `<task>.log` for a *last* match, match on the pod prefix.

Repository: [https://github.com/NVIDIA/nv-sflow](https://github.com/NVIDIA/nv-sflow)
