---
name: sflow-error-analysis
metadata:
  author: Roger Liu <rogliu@nvidia.com>
description: >-
  Diagnose and troubleshoot sflow workflow errors from log output, error messages, and
  task failures. Covers config validation errors, expression resolution failures, backend
  issues (Slurm, Docker, Kubernetes/kubectl/RBAC), probe timeouts, task crashes, S3
  storage/upload failures, and batch submission problems. Use when the user encounters an
  sflow error, pastes error output, asks to debug a failed workflow, or asks about sflow
  troubleshooting.
---

# sflow Error Analysis

Diagnose a failure in a fixed order: **capture → classify → locate → fix → re-verify.**
The first move depends on *when* it broke:

- **Launch / validation failure** — nothing ran (bad config, YAML, expression, artifact,
  merge, CSV). Caught by `--dry-run`; the fix is in the YAML.
- **Runtime failure** — a task started and died (backend launch, probe timeout, crash,
  upload). You have a run dir; the fix is in the logs.

Figure out which you have, then follow that path below.

## Instructions

Work through Steps 1–5 in order; don't propose a fix before Step 3 has located the evidence.

## Step 1 — Capture the failure

- **A task ran and failed (you have a run dir)** → start with the one-shot triage. It walks
  per-task status → orchestrator `sflow.log` → each failed `<task>/<task>.log` and prints a
  single root cause + fix. Drill in from its output:

  ```bash
  python scripts/triage.py <output_dir>/<run_id>/
  ```

- **It failed at launch (no run dir)** → reproduce with a dry-run and parse it:

  ```bash
  sflow run -f config.yaml --dry-run
  python scripts/parse_sflow_errors.py - < <(sflow run -f config.yaml --dry-run 2>&1)
  ```

- **Only have a pasted error?** Match its marker in Step 2 to pick the path.

- **The run went through `--ssh`?** The run dir is on the remote host; what came back is
  under `sflow_output/ssh-<timestamp>-<id>/` next to a `receipt.json`. `sflow fetch
  <receipt> --all` copies the complete remote output dir (then triage it as above);
  `sflow watch <receipt>` tails a run that is still going. A dropped connection did not
  cancel the remote run.

## Step 2 — Classify (marker → where to look)

| Category            | Marker in the output                                     | Path    | Where to look             |
|---------------------|----------------------------------------------------------|---------|---------------------------|
| Config loading      | `Configuration error:`, `File not found:`                | launch  | CLI output / sflow.log    |
| YAML syntax         | `Error parsing YAML`                                     | launch  | CLI output                |
| Expression          | `Undefined variable`, `Invalid expression syntax`        | launch  | CLI output                |
| Artifact            | `Artifact path validation failed`                        | launch  | CLI output / dry-run      |
| Schema              | `type 'docker' is not valid` (use `docker_run`), `mutually exclusive` | launch | CLI output (validation) |
| Merge conflict      | `Version conflict` (name/dup-task now warn+merge)        | launch  | CLI output (multi-file)   |
| Batch / CSV         | `CSV file`, `sflow_config_file column`                   | launch  | CLI output                |
| Docker plan/pre-flight | `asks for N nodes but declares no 'hosts:'`, `docker host '<h>' needs an 'address'`, `Pre-flight validation failed for Docker backend` | launch | CLI output (names the host) |
| Install / upgrade   | `is not a valid PyPI version specifier`, `(development) install`, `install failed with exit code` | launch | `sflow upgrade` / batch bootstrap output |
| SLURM               | `salloc failed`, `sbatch failed`, `salloc: error:`, `srun: error:`, `No devices were found` | runtime | CLI / sbatch stderr / `<task>.log` |
| Kubernetes          | `kubectl`, RBAC/preflight, pod `Pending`/`ImagePullBackOff` | runtime | CLI / sflow.log / `kubectl describe` |
| Docker              | `docker run` errors, remote host, NVIDIA toolkit         | runtime | `<task>/<task>.log`       |
| Storage / uploads   | `requires boto3`, `No AWS credentials`, upload failures  | runtime | CLI / dry-run / sflow.log |
| Probe timeout       | Task stuck waiting                                       | runtime | `<task>/<task>.log`       |
| Task failure        | `Traceback`, non-zero exit                              | runtime | `<task>/<task>.log`       |
| Green but empty     | aiperf `Processed 0 valid requests`, empty CSV            | runtime | `<task>/<task>.log`, the CSV |

## Step 3 — Locate the evidence

```text
<output_dir>/<run_id>/
  sflow.log                    # orchestrator log (launch + status lines)
  sflow_summary.log            # live per-task status + failure hints
  *_cmds.log                   # command-only launch logs, by command family
  results.json                 # workflow-level parsed metrics (aggregated from task result: blocks)
  sflow_monitor.log            # hardware monitor overview (if monitor: enabled — local/slurm/docker only, NOT k8s)
  sflow_monitor/               # detailed monitor reports (csv/svg/png) + raw/ samples
  <task_name>/<task_name>.log  # task stdout/stderr — the crash lives here
  <task_name>/result.json      # that task's parsed metrics (if it declares a result: block)
  <task_name>_0/               # replica 0
```

### Which sink holds which information

Each sink has a fixed scope and lines are **never mirrored between them**, so reading the
wrong file looks exactly like "sflow logged nothing". Pick by what you are looking for:

| Looking for | Read | Deliberately absent from |
|-------------|------|--------------------------|
| The workload's own stdout/stderr — traceback, CUDA OOM, server banner, probe markers | `<task>/<task>.log` | `sflow.log` |
| Orchestration — backend allocation, pod scheduling waits, probe firing, k8s output-collection, teardown, RBAC denials | `sflow.log` | `<task>/<task>.log` |
| Per-task status, failure hints, **Probe Traces**, External Command Health, timeline | `sflow_summary.log` | — |
| The exact command sflow launched | `*_cmds.log` (per family, e.g. `bash_cmds.log`) | — |
| Failed / slow external calls with timings | `command_trace.jsonl` | — |
| The manifest actually applied (k8s) | `<task>/<task>.k8s.yaml` | — |
| Parsed metrics | `<task>/result.json`, run-level `results.json` | — |

The split is by *writer*, not by topic: `<task>.log` receives only that task's own logger
plus (on k8s) the `kubectl logs -f` redirect, while every driver-side module logger writes
to `sflow.log`. The console shows both — task output echoed there is tagged so `sflow.log`
drops it.

Two consequences that cost real debugging time:

- **A `[task_name]` prefix does not mean the line is in `<task>.log`.** Driver-side lines
  are prefixed with the task name for readability but still go to `sflow.log`. On k8s that
  includes the whole output-collection handshake — `still waiting for the collect-ready
  marker`, `pod reached a terminal phase without either collect sentinel appearing`,
  `output-collect window ... expired`, `collection ABANDONED`. Grep `sflow.log` for those,
  not the task log.
- **`log_watch` probes only ever see `<task>.log`.** They cannot match orchestration lines,
  so a readiness/failure pattern must appear in the workload's own output.

- **Parsed results:** a task with a `result:` block writes `<task>/result.json`
  (`ok`/`values`/`errors`) and updates the run-level `results.json`. Result parsing is
  **best-effort** — a missed regex or failed `required:` spec logs a warning and sets
  `ok=false`, but does not by itself fail the task; check `errors` there if a metric is empty.
  Parsing runs at task finalize, so a task shows `COMPLETED` only *after* it (and any
  `uploads:`) finish.
- **No monitor output?** Hardware monitoring is unsupported on **kubernetes**
  (`supports_host_monitoring=False`) — sflow logs "node-level hardware monitoring is not
  implemented for this backend yet" and produces no `sflow_monitor*`. On local/slurm/docker,
  an empty monitor usually means the `used_by_tasks` target resolved to no nodes.

- **Batch jobs:** also read the generated `.sh` and sbatch stdout/stderr
  (`sflow_output/%j-sflow-submit.out` / `.err`).
- **Kubernetes:** `kubectl describe pod <pod>` and `kubectl get events` for
  scheduling/image issues the pod log can't show.
- **Kubernetes — a short `<task>.log` may be a delivery gap, not a quiet task.** That file
  is exactly what `kubectl logs -f` delivered; there is no post-run re-fetch (a re-fetch
  only returns the *current* container log, so it used to replace an hour of output with
  its last few seconds). A container-log rotation can stall the follow on kubelet < 1.29,
  and content written before the follow attached is unreachable. If output you know the pod
  produced is missing, check `sflow.log` for the preflight kubelet-version warning and for
  a rotation note before suspecting the workload.

## Step 4 — Root-cause & fix

### 4a. Launch / validation — fix in the YAML, re-run `--dry-run`

| Error text                                               | Fix                                                        |
|----------------------------------------------------------|------------------------------------------------------------|
| `Configuration file not found: <path>`                   | Check the `-f` path                                        |
| `Error parsing YAML configuration: <detail>`             | Fix YAML syntax at the indicated line                      |
| `Configuration validation failed: <detail>`              | Check field types/values (schema-reference.md)             |
| `Operator type 'docker' is not valid`                    | Use `type: docker_run` for the operator (backend is `type: docker`) |
| `Variable '<key>' ... is not defined`                    | Declare the variable in YAML before `--set`-ing it         |
| `Undefined variable in expression`                       | Fix spelling; ensure the variable is declared              |
| `Invalid expression syntax`                              | Fix the ``${{ }}`` Jinja2 syntax                             |
| `Artifact path validation failed`                        | Fix the `fs://` path, create it, or `--skip-artifact-check` |
| `Version conflict`                                       | Use `"0.1"` wherever `version` is declared, or omit it     |
| `CSV file must contain a 'sflow_config_file' column`     | Add the required column to the CSV                         |
| `Docker backend '<b>' asks for N nodes but declares no 'hosts:'` | Several machines → list them under `hosts:`; one machine → `nodes: 1` |
| `declares H host(s) but asks for nodes: N`               | `hosts:` is the node list — drop `nodes:` or match the pool size |
| `docker host '<h>' needs an 'address'` / `has address '<v>': drop the scheme` | One bare `address` per machine; login in `user`; `docker_host`/`context` only as overrides |
| `Pre-flight validation failed for Docker backend '<b>':` + per-host lines | `unreachable:` → fix non-interactive ssh / the daemon on that host; `mount source(s) not found on that machine` → create the path or fix `mounts:` there (catalog 11.1) |
| `--sflow-version '<x>' is not a valid PyPI version specifier` | Git ref → `sflow upgrade --branch <ref>` (`--repo` for a fork); `--sflow-version` is PEP 440 only |

### 4b. Runtime — read the logs (backend launch, then task crash)

| Symptom                                                  | Fix                                                        |
|----------------------------------------------------------|------------------------------------------------------------|
| `salloc failed` / `sbatch failed`                        | Check partition/account/availability (`sinfo`, `sacctmgr`) |
| `nvidia-smi: No devices were found` / `No CUDA GPUs are available`, `sflow_gpus.log` says `visible_gpu_count=0` | GRES partition, no GPUs requested: `--extra-salloc-args '--gpus-per-node=N'` (sflow `gpus_per_node` is planning-only) |
| `salloc: error: Job allocation N has been revoked` / `Nodes … are still not ready` | Node prolog/boot failure, node drained — re-run, `--exclude-nodes` it |
| `srun: error: Unable to confirm allocation for job N` / `Expired or invalid job` | One backend's `time:` ran out while the partner was queued — raise `time:` |
| `enroot-mount: failed to mount: /dev/nvidia-caps-imex-channels` + `couldn't start container` | Cold enroot cache race; retry, or pre-import once per node (`srun -N<n> --ntasks-per-node=1 --container-image=<img> true`) |
| `ptxas fatal : Value 'sm_103a' is not defined` / `no kernel image is available` | Image toolchain older than the GPU — CUDA 13 image for GB300 (`DYNAMO_IMAGE`) |
| kube preflight: missing permission / namespace           | Fix RBAC or `--kube-namespace`; `SFLOW_SKIP_K8S_PREFLIGHT=1` to bypass |
| pod stuck `Pending`                                      | `kubectl describe pod`: no node has the GPUs — check `gpus_per_node`, `scheduling` mode, node-selector/tolerations, and that `gpus.count` is a multiple of the node count |
| pod `ImagePullBackOff` / container not found             | Wrong `image` or missing `image_pull_secrets`; verify registry access |
| k8s readiness timeout after `OOM during general warmup` / `ncclMemAlloc … unhandled cuda error`, right after another run's pods were deleted | GPU memory not released yet on that node — wait a minute or two, or `--include-nodes` another |
| Docker GPU task fails                                    | Install the NVIDIA container toolkit on the host; set `gpus_per_node` |
| Docker multi-node task `FAILED` with one member's exit code | First non-zero member fails the task, peers are stopped; find the rank label in the merged `<task>.log` |
| aiperf `Processed 0 valid requests and N errors` (task still `COMPLETED`) | aiperf exits 0 regardless; the CSV is empty — read its error breakdown (404 → endpoint/model name; refused → frontend) |
| Probe timeout (task hangs waiting)                       | Read the **Probe Traces (last attempt)** section of `sflow_summary.log` — it shows what each probe last saw (the last log line for `log_watch`, the endpoint result for `tcp_port`/`http`). Then compare server output to `log_watch` — it matches **literally** unless prefixed `re:`/`regex:`; adjust the pattern or `timeout` |
| `Traceback` in `<task>.log`                              | Read the traceback for the real cause (below are the usual ones) |
| CUDA OOM                                                 | Reduce batch size, TP/DP, or model size                    |
| `Address already in use` / port conflict                | Kill stale processes or offset the port                    |
| NCCL / communication errors                              | Set `NCCL_SOCKET_IFNAME`; check the inter-node network     |
| `S3 storage requires boto3` / `no AWS credentials` / upload fails | `pip install 'nv-sflow[s3]'`; provide creds via the boto3 chain (`AWS_*` env vars / AWS shared credentials file / IAM — never in YAML); verify bucket/region |

For a pattern-by-pattern catalog (exact message, cause, and fix per error), see
[error-catalog.md](error-catalog.md).

## Step 5 — Re-verify

Re-run the relevant capture command — clean exit means fixed:

```bash
sflow run -f config.yaml --dry-run          # launch fixes: exit 0 = ready
python scripts/triage.py <run_dir>/          # runtime fixes: no failed tasks
```

## Diagnostic commands (reference)

```bash
sflow run -f config.yaml --dry-run                       # validate + resolve without executing
sflow compose f1.yaml f2.yaml --resolve -o merged.yaml   # see the merged + resolved config
python scripts/triage.py <output_dir>/<run_id>/          # one-shot: summary + logs -> root cause
python scripts/summarize_run.py <output_dir>/<run_id>/   #   per-task status + tracebacks only
python scripts/parse_sflow_errors.py <run_id>/sflow.log  #   categorize an orchestrator log
sinfo -p <partition>; sacctmgr show account <name>       # Slurm environment
sinfo -R                                                 # drained nodes + reason (Prolog error)
kubectl -n <ns> get pods && kubectl -n <ns> describe pod <pod>   # Kubernetes scheduling/image
kubectl -n <ns> get events --sort-by=.lastTimestamp
ssh <address> docker version                             # what Docker pre-flight does per host
sflow watch <receipt.json>; sflow fetch <receipt.json> --all     # --ssh runs: tail / copy back
```

## Examples

- Dry-run prints `Operator type 'docker' is not valid` → config error; the operator is
  `type: docker_run`, the backend is `type: docker` (Step 4 table).
- `Configuration file not found: <path>` → check the `-f` path (Step 4 table).
- Task log ends in `RuntimeError: No CUDA GPUs are available` and `<task>/sflow_gpus.log`
  has `visible_gpu_count=0`, `nvidia_visible_devices_inherited=all` → GRES partition with no
  GPU request; add `--gpus-per-node=N` via `--extra-salloc-args` (catalog 5.4).
- `Pre-flight validation failed for Docker backend 'pool':` / `- host 'dgx-b' (ssh://dgx-b)
  unreachable: Host key verification failed` → ssh to it once by hand so the key is known;
  pre-flight runs with `BatchMode=yes` (catalog 11.1).

## Additional resources

- Exhaustive error patterns → [error-catalog.md](error-catalog.md)
- Docs → [faq](https://nvidia.github.io/nv-sflow/docs/user/faq),
  [cli](https://nvidia.github.io/nv-sflow/docs/user/cli),
  [configuration](https://nvidia.github.io/nv-sflow/docs/user/configuration),
  [backends](https://nvidia.github.io/nv-sflow/docs/user/backends)
