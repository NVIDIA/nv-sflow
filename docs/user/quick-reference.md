---
title: Quick Reference
sidebar_position: 2.5
---

All `sflow.yaml` config fields at a glance. The `Required` column indicates mandatory fields.

For detailed explanations and examples, see [Configuration](./configuration.md).

## Root-Level

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `version` | Yes | string | — | Schema version. Must be `"0.1"`. |
| `variables` | | dict / list | — | Global variables available to expressions and task env. |
| `artifacts` | | dict / list | — | Named resources referenced by URI. |
| `backends` | | dict / list | — | Compute backends (`local`, `slurm`, `docker`, `kubernetes`). |
| `operators` | | dict / list | — | Task execution operators (`bash`, `srun`, `docker_run`, `ssh`, `python`, `kubernetes`). |
| `workflow` | Yes | object | — | Workflow definition containing name and tasks. |

## Variables

> YAML path: `variables.<name>`

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `value` | Yes | any | — | Variable value (int, float, bool, string, or list). |
| `description` | | string | `null` | Human-readable description. |
| `domain` | | list | `null` | Allowed values; enables replica variable sweeps. `value` must be in domain if set. |
| `type` | | string | `"string"` | Type hint (`string`, `integer`, etc.). |

## Artifacts

> YAML path: `artifacts.<name>`

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `uri` | Yes | string | — | Resource URI with scheme (`fs://`, `file://`, `http://`, `s3://`). |
| `description` | | string | `null` | Human-readable description. |
| `content` | | string | `null` | Inline file content. Only valid with `file://` URI. |

## Backends — Common Fields

> YAML path: `backends.<name>`

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `type` | Yes | string | — | `local`, `slurm`, `docker`, or `kubernetes`. |
| `default` | | bool | `false` | Mark as the default backend (only one allowed). |
| `gpus_per_node` | | int / expr | `null` | GPUs per node for sflow planning / packing. Does not add Slurm GPU allocation flags. |

## Backends — Local

> Additional fields when `type: local`

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `nodes` | | int / expr | `1` | Number of synthetic local nodes. |

## Backends — Slurm

> Additional fields when `type: slurm`

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `account` | Yes | string / expr | — | Slurm account. |
| `partition` | Yes | string / expr | — | Slurm partition. |
| `time` | Yes | string / expr | — | Time limit (e.g. `00:30:00`). |
| `nodes` | Yes | int / expr | — | Number of nodes. |
| `gpus_per_node` | Yes | int / expr | — | GPUs per node for planning. Set to `0` for CPU-only partitions; tasks that request `resources.gpus` against a zero-capacity backend will be rejected. |
| `extra_args` | | list[string] | `null` | Extra `salloc` arguments (e.g. `--exclusive`, `--gpus-per-node=8`). |
| `job_name` | | string | `null` | Job name; defaults to workflow name. |

## Operators — Common Fields

> YAML path: `operators.<name>`

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `type` | Yes | string | — | Operator type: `bash`, `srun`, `docker_run`, `ssh`, `python`, or `kubernetes`. |

## Operators — srun

> Additional fields when `type: srun`

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `job_id` | | string | `null` | Existing Slurm job ID. |
| `nodes` | | int / string | `null` | Node count. |
| `nodelist` | | list[string] | `[]` | Node list. |
| `partition` | | string | `null` | Slurm partition. |
| `account` | | string | `null` | Slurm account. |
| `qos` | | string | `null` | QOS. |
| `reservation` | | string | `null` | Reservation. |
| `time` | | string | `null` | Time limit. |
| `constraint` | | string | `null` | Slurm constraint. |
| `exclusive` | | bool | `false` | Exclusive node allocation. |
| `chdir` | | string | `null` | Working directory. |
| `cpus_per_task` | | int / string | `null` | CPUs per task. |
| `gpus` | | string | `null` | GPU spec (e.g. `all`, `1`, `device=0`). |
| `gpus_per_task` | | string | `null` | GPUs per task. |
| `gres` | | string | `null` | Generic resource spec. |
| `mem` | | string | `null` | Memory. |
| `mem_per_cpu` | | string | `null` | Memory per CPU. |
| `ntasks` | | int / string | `null` | Number of tasks. |
| `ntasks_per_node` | | int / string | `null` | Tasks per node. |
| `export` | | string | `"ALL"` | Environment export setting. |
| `label` | | bool | `true` | Prefix output with task label. |
| `unbuffered` | | bool | `true` | Unbuffered output. |
| `kill_on_bad_exit` | | bool | `false` | Kill job on non-zero task exit. |
| `overlap` | | bool | `true` | Allow step overlap. |
| `wait` | | int / string | `null` | Wait time. |
| `container_image` | | string | `null` | Container image (Pyxis). Mutually exclusive with `container_name`. |
| `container_name` | | string | `null` | Existing container name (Pyxis). Mutually exclusive with `container_image`. |
| `container_mount_home` | | bool | `false` | Mount home directory in container. |
| `container_writable` | | bool | `true` | Writable container filesystem. |
| `container_mounts` | | list[string] | `[]` | Bind mounts (e.g. `"/host:/ctr:rw"`). |
| `container_workdir` | | string | `null` | Container working directory. |
| `container_remap_root` | | bool | `false` | Remap root inside container. |
| `mpi` | | string | `null` | MPI type (e.g. `pmix`, `ucx`). |
| `extra_args` | | list[string] | `[]` | Extra CLI arguments. |

## Operators — Docker Run

> Additional fields when `type: docker_run`

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `image` | Yes | string | — | Docker image. |
| `workdir` | | string | `null` | Working directory inside container. |
| `mounts` | | list[string] | `[]` | Bind mounts (e.g. `"/host:/ctr:rw"`). |
| `gpus` | | string | `null` | GPU spec (e.g. `all`, `device=0`). |
| `extra_args` | | list[string] | `[]` | Extra `docker run` arguments. |
| `pass_envs` | | bool | `true` | Forward host environment variables. |

## Operators — SSH

> Additional fields when `type: ssh`

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `host` | Yes | string | — | SSH host. |
| `user` | | string | `null` | SSH user. |
| `port` | | int | `null` | SSH port. |
| `identity_file` | | string | `null` | Path to identity file. |
| `extra_args` | | list[string] | `[]` | Extra SSH arguments. |

## Operators — Python

> Additional fields when `type: python`

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `python_exec` | | string | `"python"` | Python executable. |
| `extra_args` | | list[string] | `[]` | Extra Python arguments. |

## Workflow

> YAML path: `workflow`

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `name` | Yes | string | — | Workflow name. |
| `tasks` | Yes | list | — | List of task definitions (must be non-empty). |
| `timeout` | | string / int | `null` | Workflow-level timeout (e.g. `1h`, `115m`). |
| `variables` | | dict / list | `null` | Workflow-scoped variables (same format as root `variables`). |

## Tasks

> YAML path: `workflow.tasks[]`

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `name` | Yes | string | — | Task name (must be unique). |
| `script` | Yes | list[string] | — | Script lines to execute (non-empty). |
| `operator` | | string / object | `null` | Operator name, or inline operator override object. |
| `backend` | | string / dict | `null` | Backend name, or inline backend override. |
| `depends_on` | | list[string] | `null` | Names of tasks this task depends on. |
| `timeout` | | int / string | `null` | Task-level timeout. |
| `variables` | | dict / list | `null` | Task-scoped variables. |
| `resources` | | object | `null` | Node / GPU resource requirements. |
| `replicas` | | object | `null` | Replication configuration. |
| `retries` | | object | `null` | Retry configuration. |
| `probes` | | object | `null` | Readiness and failure probes. |
| `outputs` | | list | `null` | Output parsing configuration (legacy MVP). |
| `result` | | map / object | `null` | Consolidated result parsing (regex map, `patterns`, or `file`). Writes `result.json` + workflow `results.json`. |

## Task Resources

> YAML path: `workflow.tasks[].resources`

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `nodes.indices` | | list[int / expr] | `null` | Specific node indices (e.g. `[0]`). |
| `nodes.count` | | int / expr | `null` | Number of nodes. |
| `nodes.exclude` | | int / list[int] / expr | `null` | Node indices to remove from the placement pool before `indices`, `count`, or GPU packing. |
| `nodes.release_after` | | string | inferred | When node reservations can be reused: `workflow_completion`, `task_ready`, or `task_completion`. |
| `gpus.count` | If `gpus` is set | int / expr | — | Number of GPUs (sets `CUDA_VISIBLE_DEVICES`). |
| `gpus.release_after` | | string | inferred | When GPU reservations can be reused: `workflow_completion`, `task_ready`, or `task_completion`. |

For nodes, `release_after` only creates an exclusive node reservation when explicitly set; omitted `nodes.indices` and `nodes.count` are placement constraints and may overlap with other planned tasks. For GPUs, omitted `release_after` is inferred: tasks without readiness probes release GPUs after task completion for downstream dependents, while tasks with readiness probes keep GPUs until workflow completion unless explicitly set to `task_ready`. `task_ready` releases after readiness succeeds. `task_completion` releases after any terminal task status (`COMPLETED`, `FAILED`, `TIMEOUT`, or `CANCELLED`). Dry-run rehearses these resource lifetimes across the DAG.

## Task Replicas

> YAML path: `workflow.tasks[].replicas`

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `count` | | int / expr | `null` | Number of replicas. |
| `policy` | | string / expr | `"parallel"` | `"parallel"` or `"sequential"`. |
| `variables` | | list[string] | `null` | Variable names for sweeps (Cartesian product of domains). |

## Task Retries

> YAML path: `workflow.tasks[].retries`

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `count` | Yes | int / expr | — | Number of retries. |
| `interval` | Yes | int / expr | — | Delay between retries (seconds). |
| `backoff` | | int / expr | `1` | Backoff multiplier. |

## Task Probes (Readiness / Failure)

> YAML path: `workflow.tasks[].probes.readiness` or `workflow.tasks[].probes.failure`

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `delay` | | int / expr | `0` | Initial delay before probing (seconds). |
| `timeout` | | int / expr | `1200` | Max readiness wait time (seconds). Failure probes do not use this as an overall deadline. |
| `each_check_timeout` | | int / expr | `30` | Timeout for a single probe check attempt. |
| `interval` | | int / expr | `5` | Check interval (seconds). |
| `success_threshold` | | int / expr | `1` | Consecutive successes required. |
| `failure_threshold` | | int / expr | `3` | Consecutive failures before failing. |

Exactly one probe type must be set per probe:

| Probe Type | Required Fields | Optional Fields | Description |
|------------|-----------------|-----------------|-------------|
| `tcp_port` | `port` | `host`, `on_node` (`"first"` / `"each"`) | TCP connection check. |
| `http_get` | `url` | `headers` | HTTP GET health check. |
| `http_post` | `url` | `headers`, `body` | HTTP POST health check. |
| `log_watch` | `regex_pattern` or `match_pattern` | `logger`, `match_count` | Match pattern in task logs. Literal by default; prefix with `re:` or `regex:` for regular expressions. |

## Task Outputs

> YAML path: `workflow.tasks[].outputs[]`

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `pattern` | Yes | string | — | Parse pattern (e.g. `"TTFT: {ttft:f} ms"`). |
| `source` | | string | `"stdout"` | Log source: `stdout` or `stderr`. |
| `metrics.<key>.description` | | string | `null` | Metric description. |
| `metrics.<key>.type` | | string | `null` | Metric type. |
| `metrics.<key>.aggregate` | | string | `null` | Aggregation hint. |

## Task Result

> YAML path: `workflow.tasks[].result`. Accepts a simple `name: regex` map, an object with `patterns:`, or an object with `file:`. See [Results](./results.md).

> `patterns` and `file` are mutually exclusive.

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `patterns` | | list | `null` | Advanced regex patterns (see below). Mutually exclusive with `file`. |
| `file` | | string | `null` | Relative path to a JSON source file ending in `.json` (e.g. `result.json`). Normalized into `result.json`. |
| `source` | | string | `"log"` | Source selector. Only `log` is implemented. |

`result.patterns[]`:

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `name` | Yes | string | — | Result key under `values`. |
| `regex` | Yes | string | — | Python regex; prefer one capture group or a named `value` group. |
| `type` | | string | `"auto"` | `auto`, `string`, `int`, `float`, `bool`, or `json`. |
| `unit` | | string | `null` | Optional metadata unit (e.g. `ms`). |
| `aggregate` | | string | `"last"` | `first`, `last`, `list`, `count`, `min`, `max`, `avg`, or `sum`. |
| `required` | | bool | `false` | If `true`, a missing match marks the result `ok: false`. |
| `source` | | string | inherited | Per-pattern source override; only `log` is implemented. |
| `group` | | string / int | `value` then `1` | Capture group name or index to extract. |

A simple `name: regex` map is shorthand for `patterns` with `type: auto`, `aggregate: last`, `required: false`, `source: log`.

To publish a metric literally named `file`, use `patterns:`; a top-level `file:` key is reserved for JSON file-source results.

## Expression Syntax

Fields marked **int / expr** or **string / expr** support `${{ ... }}` expressions:

| Expression | Example |
|------------|---------|
| Variable | `${{ variables.MY_VAR }}` |
| Variable domain | `${{ variables.MY_VAR.domain }}` |
| Backend node IP | `${{ backends.slurm_cluster.nodes[0].ip_address }}` |
| Artifact path | `${{ artifacts.model_dir.path }}` |
| Task node IP | `${{ task.server.nodes[0].ip_address }}` |

## Reserved Environment Variables

### Injected by sflow into task environments

These are automatically set by sflow and available in every task script.

| Variable | Description |
|----------|-------------|
| `SFLOW_WORKSPACE_DIR` | Absolute path to the project workspace root. |
| `SFLOW_OUTPUT_DIR` | Global output root directory (default `./sflow_output`). |
| `SFLOW_WORKFLOW_OUTPUT_DIR` | Output directory for the current workflow run (e.g. `sflow_output/<run-id>`). |
| `SFLOW_TASK_OUTPUT_DIR` | Output directory for the current task replica (e.g. `sflow_output/<run-id>/my_task_0`). |
| `SFLOW_TASK_RESULT_FILE` | Canonical per-task result file path (`${SFLOW_TASK_OUTPUT_DIR}/result.json`). Write JSON here to publish results directly. |
| `SFLOW_WORKFLOW_RESULT_FILE` | Workflow-level results index path (`${SFLOW_WORKFLOW_OUTPUT_DIR}/results.json`). |
| `SFLOW_REPLICA_INDEX` | Zero-based replica index (`0`, `1`, `2`, ...). |
| `SFLOW_TASK_ASSIGNED_NODE_NAMES` | Comma-separated hostnames of nodes assigned to this task. |
| `SFLOW_TASK_ASSIGNED_NODE_IPS` | Comma-separated IP addresses of nodes assigned to this task. |
| `SFLOW_BACKEND_JOB_ID` | Backend allocation/job id when available. For Slurm this mirrors `SLURM_JOB_ID` / `SLURM_JOBID`. |
| `SFLOW_BACKEND_NODELIST` | Backend allocation nodelist when available. For Slurm this mirrors `SLURM_JOB_NODELIST` / `SLURM_NODELIST`. |
| `SFLOW_BACKEND_NUM_NODES` | Number of nodes in the backend allocation when available. For Slurm this mirrors `SLURM_NNODES`. |
| `CUDA_VISIBLE_DEVICES` | Comma-separated GPU indices allocated to this task (set when `resources.gpus.count` is used). |

In addition, all resolved `variables` and `artifacts` paths are injected as environment variables accessible via `${VAR_NAME}` in scripts.

> Avoid naming a variable after a reserved `SFLOW_*` / `CUDA_VISIBLE_DEVICES` env var above — sflow injects and owns these at launch, so a same-named variable collides and causes undefined behavior. `sflow run --dry-run` prints a **Reserved env collisions** section listing any such variables so you can rename them before a real run.

### Read by sflow from the host environment

sflow reads these to detect an existing Slurm allocation and skip `salloc`.
When a Slurm controller provides `SLURM_*` / `SLURMD_*` variables, sflow preserves those controller values in task environments even if workflow variables use the same names.

| Variable | Description |
|----------|-------------|
| `SLURM_JOB_ID` / `SLURM_JOBID` | Current Slurm job ID. Used to detect an existing allocation. |
| `SLURM_JOB_NODELIST` / `SLURM_NODELIST` | Node list for the current Slurm allocation. |

### Provided by Slurm at runtime

These are set by Slurm (not by sflow) and commonly used in task scripts. The `srun` operator also maps common Slurm step/rank variables into backend-agnostic aliases.

| Slurm variable | SFLOW alias | Description |
|----------------|-------------|-------------|
| `SLURM_STEP_ID` | `SFLOW_BACKEND_STEP_ID` | Slurm step id for the current `srun` step. |
| `SLURMD_NODENAME` | `SFLOW_TASK_NODE_NAME` | Hostname of the node running the task process. |
| `SLURM_NODEID` | `SFLOW_TASK_NODE_INDEX` | Node index within the allocation (useful for `NODE_RANK`). |
| `SLURM_PROCID` | `SFLOW_TASK_PROCESS_ID` | Global process/rank id within the step. |
| `SLURM_LOCALID` | `SFLOW_TASK_LOCAL_PROCESS_ID` | Local process/rank id on the node. |
| `SLURM_NTASKS` | `SFLOW_TASK_NUM_PROCESSES` | Number of tasks/processes in the step. |
| `SLURM_SUBMIT_DIR` | — | Directory from which the job was submitted. |
