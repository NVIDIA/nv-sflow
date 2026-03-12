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
| `backends` | | dict / list | — | Compute backends (`local`, `slurm`). |
| `operators` | | dict / list | — | Task execution operators (`bash`, `srun`, `docker`, `ssh`, `python`). |
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
| `type` | Yes | string | — | `local` or `slurm`. |
| `default` | | bool | `false` | Mark as the default backend (only one allowed). |
| `gpus_per_node` | | int / expr | `null` | GPUs per node for allocation / packing. |

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
| `gpus_per_node` | Yes | int / expr | — | GPUs per node. |
| `extra_args` | | list[string] | `null` | Extra `salloc` arguments (e.g. `--exclusive`). |
| `job_name` | | string | `null` | Job name; defaults to workflow name. |

## Operators — Common Fields

> YAML path: `operators.<name>`

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `type` | Yes | string | — | Operator type: `bash`, `srun`, `docker`, `ssh`, or `python`. |

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

## Operators — Docker

> Additional fields when `type: docker`

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
| `outputs` | | list | `null` | Output parsing configuration. |

## Task Resources

> YAML path: `workflow.tasks[].resources`

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `nodes.indices` | | list[int / expr] | `null` | Specific node indices (e.g. `[0]`). |
| `nodes.count` | | int / expr | `null` | Number of nodes. |
| `gpus.count` | Yes | int / expr | — | Number of GPUs (sets `CUDA_VISIBLE_DEVICES`). |

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
| `timeout` | | int / expr | `60` | Max wait time (seconds). |
| `interval` | | int / expr | `5` | Check interval (seconds). |
| `success_threshold` | | int / expr | `1` | Consecutive successes required. |
| `failure_threshold` | | int / expr | `3` | Consecutive failures before failing. |

Exactly one probe type must be set per probe:

| Probe Type | Required Fields | Optional Fields | Description |
|------------|-----------------|-----------------|-------------|
| `tcp_port` | `port` | `host`, `on_node` (`"first"` / `"each"`) | TCP connection check. |
| `http_get` | `url` | `headers` | HTTP GET health check. |
| `http_post` | `url` | `headers`, `body` | HTTP POST health check. |
| `log_watch` | `regex_pattern` | `logger`, `match_count` | Match pattern in task logs. |

## Task Outputs

> YAML path: `workflow.tasks[].outputs[]`

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `pattern` | Yes | string | — | Parse pattern (e.g. `"TTFT: {ttft:f} ms"`). |
| `source` | | string | `"stdout"` | Log source: `stdout` or `stderr`. |
| `metrics.<key>.description` | | string | `null` | Metric description. |
| `metrics.<key>.type` | | string | `null` | Metric type. |
| `metrics.<key>.aggregate` | | string | `null` | Aggregation hint. |

## Expression Syntax

Fields marked **int / expr** or **string / expr** support `${{ ... }}` expressions:

| Expression | Example |
|------------|---------|
| Variable | `${{ variables.MY_VAR }}` |
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
| `SFLOW_REPLICA_INDEX` | Zero-based replica index (`0`, `1`, `2`, ...). |
| `SFLOW_TASK_ASSIGNED_NODE_NAMES` | Comma-separated hostnames of nodes assigned to this task. |
| `SFLOW_TASK_ASSIGNED_NODE_IPS` | Comma-separated IP addresses of nodes assigned to this task. |
| `CUDA_VISIBLE_DEVICES` | Comma-separated GPU indices allocated to this task (set when `resources.gpus.count` is used). |

In addition, all resolved `variables` and `artifacts` paths are injected as environment variables accessible via `${VAR_NAME}` in scripts.

### Read by sflow from the host environment

sflow reads these to detect an existing Slurm allocation and skip `salloc`.

| Variable | Description |
|----------|-------------|
| `SLURM_JOB_ID` / `SLURM_JOBID` | Current Slurm job ID. Used to detect an existing allocation. |
| `SLURM_JOB_NODELIST` / `SLURM_NODELIST` | Node list for the current Slurm allocation. |

### Provided by Slurm at runtime

These are set by Slurm (not by sflow) and commonly used in task scripts.

| Variable | Description |
|----------|-------------|
| `SLURM_NODEID` | Node index within the allocation (useful for `NODE_RANK`). |
| `SLURMD_NODENAME` | Hostname of the node running the task. |
| `SLURM_SUBMIT_DIR` | Directory from which the job was submitted. |
