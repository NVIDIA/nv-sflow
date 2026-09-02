---
title: Quick Reference
sidebar_position: 2.5
---

All `sflow.yaml` config fields at a glance. The `Required` column indicates mandatory fields.

For detailed explanations and examples, see [Configuration](./configuration.md).

## Root-Level

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `version` | No | string | `"0.1"` | Schema version. If declared, must be `"0.1"`. |
| `variables` | | dict / list | — | Global variables available to expressions and task env. |
| `artifacts` | | dict / list | — | Named resources referenced by URI. |
| `backends` | | dict / list | — | Compute backends (`local`, `slurm`, `docker`, `kubernetes`). |
| `operators` | | dict / list | — | Task execution operators (`bash`, `srun`, `docker_run`, `ssh`, `python`, `k8s`, `k8s_mpi`). |
| `storage` | | dict / list | — | Post-execution upload targets (S3, …) referenced by task `uploads` / `upload_all` (see [Storage](#storage)). |
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

## Storage

> YAML path: `storage.<name>` — post-execution upload targets referenced by task `uploads` /
> `upload_all`. Credentials come from the provider's default chain (e.g. boto3 for S3) —
> **never inline secrets in YAML**. See [Uploads](./uploads.md).

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `name` | Yes | string | — | Target name (referenced by `uploads.target` / `upload_all.target`). |
| `type` | Yes | string | — | Target type. Currently `s3`. |

**S3 target** (`type: s3`) adds `bucket` (Yes, string / expr) plus optional `region`,
`prefix`, `endpoint_url` (S3-compatible stores such as MinIO / Ceph RGW), `storage_class`,
and `addressing_style` (`auto` / `virtual` / `path`).

## Backends — Common Fields

> YAML path: `backends.<name>`

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `type` | Yes | string | — | `local`, `slurm`, `docker`, or `kubernetes`. |
| `default` | | bool | `false` | Mark as the default backend (only one allowed). |
| `gpus_per_node` | | int / expr | `null` | GPUs per node for sflow planning / packing. Does not add Slurm GPU allocation flags. |
| `include_nodes` | | list[string] | `null` | Restrict allocation to these node hostnames (unioned with `--include-nodes`). Translated per backend: Slurm `--nodelist`, k8s `nodeAffinity` In, Docker host filter. |
| `exclude_nodes` | | list[string] | `null` | Exclude these node hostnames (unioned with `--exclude-nodes`). |

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
| `time` | Yes | string / int / expr | — | Time limit — `HH:MM:SS` string or integer minutes. |
| `nodes` | Yes | int / expr | — | Number of nodes. |
| `gpus_per_node` | Yes | int / expr | — | GPUs per node for planning. Set to `0` for CPU-only partitions; tasks that request `resources.gpus` against a zero-capacity backend will be rejected. |
| `extra_args` | | list[string] | `null` | Extra `salloc` arguments (e.g. `--exclusive`, `--gpus-per-node=8`). |
| `job_name` | | string | `null` | Job name; defaults to workflow name. |

## Backends — Kubernetes

> Additional fields when `type: kubernetes`. The workload **image lives on the
> operator** (`k8s` / `k8s_mpi`), not the backend. Cluster/context/namespace are
> selected with `sflow run` flags (`--kubeconfig`, `--kube-context`,
> `--kube-namespace`), never in YAML. See [Backends](./backends.md#kubernetes-backend).

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `namespace` | | string / expr | `null` | Namespace for all pods; **must already exist**. One namespace per backend. Overridable with `--kube-namespace`. |
| `nodes` | | int / expr | `1` | Number of nodes to reserve. |
| `scheduling` | | `device_plugin`\|`dra` | `device_plugin` | GPU request mode: NVIDIA device plugin (`nvidia.com/gpu`) or DRA (K8s 1.34+, WIP). |
| `gpu_resource_name` | | string / expr | `nvidia.com/gpu` | Device-plugin resource name (e.g. MIG `nvidia.com/mig-1g.5gb`, or another vendor). Ignored under `dra`. |
| `gpu_product_label_key` | | string / expr | `nvidia.com/gpu.product` | Node label carrying the GPU product string (NVLink-scope auto-detection). |
| `host_network` | | bool | `true` | All pods use host networking (pod IP == node IP). Privileged; disable on CNI-routable clusters. |
| `host_ipc` | | bool | `false` | Share node IPC namespace + hostPath `/dev/shm` for cross-pod CUDA IPC over NVLink. Privileged; opt-in. |
| `merge_colocated_gpu_pods` | | bool \| `auto`\|`on`\|`disable` | `auto` | Merge co-located single-node GPU tasks into one pod for intra-node NVLink/cuda_ipc. |
| `nvlink_domain` | | `auto`\|`node`\|`rack`\|`disable` | `auto` | Advisory NVLink-domain scope; set only to correct a wrong auto-detection. Not the MNNVL switch (see `compute_domain`). |
| `rdma` | | `auto`\|`disable`\|`gke`\|`shared_device_plugin`\|`host_device` | `auto` | RDMA/IB provisioning for GPU pods; `auto` detects the provider, `disable` forces sockets. |
| `node_selector` | | map | `null` | Node-selector labels for all pods. Merge/override with `--kube-node-selector`. |
| `tolerations` | | list[map] | tolerate `nvidia.com/gpu` | Pod tolerations for placeholder + task pods. |
| `image_pull_policy` | | string / expr | `null` | Default pod pull policy (`Always`/`IfNotPresent`/`Never`); else cluster default. |
| `probe_pod_image` | | string / expr | `curlimages/curl:latest` | Image for the in-cluster TCP/HTTP probe pod (mirror for air-gapped). |
| `cpu_per_gpu` | | int / expr | `null` | CPU request per GPU for GPU task pods (requests-only; opt-in). |
| `cpu_request` | | int / expr | `null` | CPU request for pods without GPUs (requests-only; opt-in). |
| `collect_max_file_size` | | int / string | `null` (10 MiB) | Per-file cap for collecting a task's node-local output back to the driver; `0` disables. |
| `collect_grace_seconds` | | int / expr | `120` | Grace period (s) to copy node-local output before pod teardown. |
| `collect_node_local_output` | | bool | `true` | Master switch for the node-local output collect. Left on, the DAG waits on a sentinel the pod writes to its log, so a task's output can be lost to log-delivery lag; `false` runs tasks with **no** sflow collect machinery in the pod — completion then depends only on pod status, probes and the merge-pod marker, and outputs must reach the driver via a shared filesystem, `uploads:` or a PVC. |
| `gib_installer_namespace` | | string / expr | `kube-system` | Namespace of the GKE gIB (`nccl-rdma-installer`) DaemonSet. |
| `extra_args` | | list[string] | `null` | Extra backend arguments (reserved). |
| `dra` | | object | `null` | DRA GPU-allocation options (see below); used when `scheduling: dra`. |
| `compute_domain` | | object | `null` | Multi-Node NVLink (IMEX ComputeDomain) options (see below). |
| `volumes` | | list[object] | `null` | PVC / `emptyDir` volumes mounted into **every** task pod (see below). |
| `reservation` | | object | `null` | Node-reservation tuning (see below). |

`gpus_per_node` (see [Common Fields](#backends--common-fields)) is derived from node capacity when unset and validated against real GPU capacity at pre-flight; set `0` for CPU-only. PVCs referenced under `volumes:` must already exist in the namespace — `sflow run --kube-skip-pvc` drops PVC-backed volumes for debugging.

### Kubernetes — `volumes[]`

> YAML path: `backends.<name>.volumes[]`. Set **exactly one** of `claim` or `empty_dir`.

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `name` | Yes | string | — | Pod volume name (DNS-1123). |
| `mount_path` | Yes | string / expr | — | Absolute mount path inside each task pod. |
| `claim` | one of | string / expr | `null` | Existing PVC name. Mutually exclusive with `empty_dir`. |
| `empty_dir` | one of | object | `null` | Ephemeral per-pod scratch volume. Mutually exclusive with `claim`. |
| `sub_path` | | string / expr | `null` | Path within the volume to mount (`volumeMount.subPath`). |
| `read_only` | | bool | PVC `true` / emptyDir `false` | Mount read-only. |
| `ensure_writable` | | bool | `false` | Inject a root initContainer to `chmod` the subPath. Requires a `claim` and `read_only: false`. |

`volumes[].empty_dir` fields: `medium` (`""`\|`Memory`, default `""` = node disk; `Memory` = tmpfs), `size_limit` (string / expr, e.g. `50Gi`).

### Kubernetes — `dra`

> YAML path: `backends.<name>.dra` (used when `scheduling: dra`; DRA is WIP).

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `gpu_device_class` | | string / expr | `gpu.nvidia.com` | DeviceClass GPUs are requested from. |
| `device_selectors` | | list[string] | `null` | CEL expressions narrowing eligible devices (by product/memory). |
| `rdma_device_class` | | string / expr | `null` | Opt-in GPU↔NIC co-allocation NIC DeviceClass. |
| `rdma_match_attribute` | | string / expr | `resource.kubernetes.io/pcieRoot` | Attribute the GPU and NIC requests must share (PCIe root). |
| `nvlink_domain_label_key` | | string | `null` | Node label key identifying the physical NVLink domain (multi-domain placement). |

### Kubernetes — `compute_domain`

> YAML path: `backends.<name>.compute_domain` — Multi-Node NVLink (IMEX). Independent of `scheduling`.

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `channel` | | string / expr | `null` | Join an existing IMEX channel template; `auto` = the sole existing domain; `off`/empty = no claim. |
| `create` | | bool | `false` | Create a fresh ComputeDomain CR instead of joining. Privileged; needs `computedomains` RBAC. |

### Kubernetes — `reservation`

> YAML path: `backends.<name>.reservation` — tuning for the node reservation.

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `timeout` | | int / expr | `600` | Seconds to wait for every placeholder pod to be scheduled. |
| `handoff` | | `auto`\|`create_before_destroy`\|`destroy_before_create` | `auto` | Placeholder→task handoff order (`auto` = destroy-before-create when a GPU ResourceQuota is present). |
| `placeholder_image` | | string / expr | `bash:5` | Image for reservation placeholder pods (mirror for air-gapped). |

## Operators — Common Fields

> YAML path: `operators.<name>`

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `type` | Yes | string | — | Operator type: `bash`, `srun`, `docker_run`, `ssh`, `python`, `k8s`, or `k8s_mpi`. |

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

## Operators — Kubernetes (`k8s`)

> Additional fields when `type: k8s`. The workload **image lives on the operator**
> (the `kubernetes` backend has none). See [Backends](./backends.md#kubernetes-backend)
> for the complete pod/container passthrough list.

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `image` | Yes | string | — | Container image for the pod. |
| `image_pull_policy` | | string | `null` | `Always` / `IfNotPresent` / `Never` (else cluster default). |
| `restart` | | string | `"Never"` | Pod `restartPolicy`. |
| `host_network` | | bool | inherit backend | Pod IP == node IP (needed by some RDMA paths). |
| `node_selector` | | map | inherit backend | Constrain the pod to nodes with these labels. |
| `run_as_root` | | bool | `false` | Force `runAsUser/Group=0` (root-owned PVCs, MPI/sshd bootstrap). |
| `pass_envs` | | bool | `true` | Create + mount the env Secret. |
| `shm_size` | | string | `null` | Cap for RAM-backed `/dev/shm` (e.g. `64Gi`); unset = node-RAM-bounded tmpfs. |
| `tty` | | bool | `false` | Allocate a pseudo-TTY so `\r` progress bars stream live. |
| `cpu` | | int / string | `null` | Container CPU **request**. |
| `memory` | | string | `null` | Container memory **request** (e.g. `16Gi`). |
| `cpu_limit` | | int / string | `null` | Optional CPU hard cap (unset = requests-only). |
| `memory_limit` | | string | `null` | Optional memory hard cap (unset = requests-only). |
| `env` | | list[dict] | `null` | Raw k8s `EnvVar` entries (only way to use `valueFrom`). |
| `image_pull_secrets` | | list[string] | `null` | Secret name(s) for pulling from a private registry (e.g. `nvcr.io`). |
| `service_account` | | string | `null` | Pod service account (RBAC / cloud workload identity). |
| `working_dir` | | string | `null` | Container working directory. |
| `ports` | | list[dict] | `null` | Container `ports` entries. |
| `tolerations` | | list[dict] | inherit backend | Pod tolerations (added to the backend's). |
| `collect_max_file_size` | | int/string | inherit backend | Per-file cap for collecting node-local output back to the driver; `0` disables. |
| `security_context` | | map | `null` | Container `securityContext` (deep-merged; caps union-merged). |
| `device_class` / `device_selectors` | | string/list | inherit backend | DRA overrides (ignored under `device_plugin`). |
| `container_overrides` / `pod_overrides` | | map | `null` | Raw escape hatches, deep-merged **last** (they win). |

## Operators — Kubernetes MPI (`k8s_mpi`)

> Additional fields when `type: k8s_mpi` — any MPI (`mpirun`) job, single- or multi-node.
> Inherits every `k8s` field above and adds an `mpi:` block. Write an explicit
> `mpirun -np N ... <workload>` and sflow builds the MPI world either way, injecting the
> SSH keypair, hostfile, and launcher wiring. Use plain `k8s` for non-MPI, single-pod workloads.

| `mpi.` field | Type | Default | Description |
|--------------|------|---------|-------------|
| `route` | `auto`\|`operator`\|`pods` | `auto` | MPIJob CR (Kubeflow MPI Operator) vs plain-pods bootstrap; `auto` detects the CRD. |
| `slots_per_worker` | int/string | per-node GPU count | Slots advertised per worker. |
| `run_launcher_as_worker` | bool | `true` | Launcher pod also runs a rank. |
| `launcher_creation_policy` | `AtStartup`\|`WaitForWorkersReady` | `WaitForWorkersReady` | When the launcher starts. |
| `ssh_port` | int / expr | `2222` | Port for the injected sshd. |
| `mpi_implementation` | `OpenMPI`\|`Intel`\|`MPICH` | `OpenMPI` | Selects hostfile/env conventions. |
| `ensure_sshd` | bool | `true` | Install `openssh-server` if the image lacks sshd. |
| `forward_env_prefixes` | list[string] | `[]` | Extra env-namespace prefixes forwarded to remote ranks. |
| `omp_num_threads` | int | `8` | Injected `OMP_NUM_THREADS` per rank; `null`/`0` keeps the image default. |
| `cpu_bind` | `core`\|`numa`\|`none` | `core` | Per-rank CPU binding, injected only when several ranks share a pod. |
| `cpu_bind_cores_per_rank` | int | `8` | Max cores bound per rank under `cpu_bind: core`; `0` = uncapped. |
| `worker_setup_timeout_seconds` | int / string | `900` | Per-node worker setup / readiness budget before the launcher starts. |
| `launcher_discovery_timeout` | int / string | `600` | Max wait for the MPI operator to create the launcher pod (`route: operator`). |

## Workflow

> YAML path: `workflow`

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `name` | Yes | string | — | Workflow name. |
| `tasks` | Yes | list | — | List of task definitions (must be non-empty). |
| `timeout` | | string / int | `null` | **Not enforced** — accepted and merged, but nothing reads it; a workflow that sets it runs unbounded and `sflow run` warns. Use the backend's own limit (Slurm `--time`). |
| `variables` | | dict / list | `null` | Workflow-scoped variables (same format as root `variables`). |
| `upload_all` | | object | `null` | Zip the whole workflow output dir and upload it to a `storage` target (see [Workflow Upload-All](#workflow-upload-all)). |
| `monitor` | | object | `null` | Workflow-level hardware monitor (see [Monitor](#monitor)). |

## Tasks

> YAML path: `workflow.tasks[]`

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `name` | Yes | string | — | Task name (must be unique). |
| `script` | Yes | list[string] | — | Script lines to execute (non-empty). |
| `operator` | | string / object | `null` | Operator name, or inline operator override object. |
| `backend` | | string / dict | `null` | Backend name, or inline backend override. |
| `depends_on` | | list[string] | `null` | Names of tasks this task depends on. |
| `required_by` | | list[string] | `null` | Reverse dependency: `A required_by: [B]` is folded into `B depends_on: [A]` at load (lets an optional fragment attach to a hub without editing it). |
| `timeout` | | int / string | `null` | **Not enforced** — see [Workflow](#workflow). Bound the task with the backend's own limit instead. |
| `variables` | | dict / list | `null` | Task-scoped variables. |
| `resources` | | object | `null` | Node / GPU resource requirements. |
| `replicas` | | object | `null` | Replication configuration. |
| `retries` | | object | `null` | Retry configuration. |
| `probes` | | object | `null` | Readiness and failure probes. |
| `outputs` | | list | `null` | Output parsing configuration (legacy MVP). |
| `result` | | map / object | `null` | Consolidated result parsing (regex map, `patterns`, or `file`). Writes `result.json` + workflow `results.json`. |
| `fail_fast` | | bool | backend | Prepend `set -e` to the script (shell operators) so the task fails on the first failing command. Unset takes the backend default: `true` on Kubernetes, `false` on Slurm/local/docker. Set explicitly to override. |
| `uploads` | | list | `null` | Per-task file uploads to a `storage` target, fired on `COMPLETED` (see [Task Uploads](#task-uploads)). |
| `ports` | | list | `null` | Service ports the task exposes, feeding `${{ task.<name>.service }}` expressions (see [Task Ports](#task-ports)). |
| `monitor` | | object | `null` | Per-task hardware monitor (see [Monitor](#monitor)). |

## Task Resources

> YAML path: `workflow.tasks[].resources`

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `nodes.indices` | | list[int / expr] | `null` | Specific node indices (e.g. `[0]`). |
| `nodes.count` | | int / expr | `null` | Number of nodes. |
| `nodes.exclude` | | int / list[int] / expr | `null` | Node indices to remove from the placement pool before `indices`, `count`, or GPU packing. |
| `nodes.release_after` | | string | `null` | When node reservations can be reused: `workflow_completion`, `task_ready`, or `task_completion`. sflow reserves nodes **only when this is set explicitly**; omitted, `nodes.indices`/`count` are non-exclusive placement constraints (see note below). |
| `gpus.count` | One of `count` / `indices` | int / expr | `null` | Number of GPUs (sets `CUDA_VISIBLE_DEVICES`). Index-agnostic on its own: the planner packs the task into any contiguous idle GPU run on one node. |
| `gpus.indices` | One of `count` / `indices` | list[int / expr] | `null` | Pin the task to specific **0-based, non-negative, unique** device ids. Combined with `count`, `count` is the total across nodes and `indices` the per-node slice, so the task fans out over `count / len(indices)` nodes. |
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

## Task Uploads

> YAML path: `workflow.tasks[].uploads[]` — copy files to a `storage` target when the task
> reaches `COMPLETED`. See [Uploads](./uploads.md).

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `target` | Yes | string | — | Name of a `storage` target. |
| `from` | Yes | string | — | Local file or glob to upload. |
| `to` | | string | `null` | Remote key/prefix; **must end with `/`** when `from` is a glob. |
| `on_error` | | `warn` \| `fail` | `warn` | What to do if the upload fails. |

## Workflow Upload-All

> YAML path: `workflow.upload_all` — zip the entire workflow output dir and upload it.

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `target` | Yes | string | — | Name of a `storage` target. |
| `to` | | string | `${{ workflow.run_id }}.zip` | Remote key for the uploaded zip. |
| `on_error` | | `warn` \| `fail` | `warn` | What to do if the upload fails. |

## Task Ports

> YAML path: `workflow.tasks[].ports[]` — service ports the task exposes, feeding
> `${{ task.<name>.service }}` expressions.

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `port` | Yes | int / expr | — | Port the task serves on. |
| `name` | | string | `null` | Optional port name. |

## Monitor

> YAML path: `workflow.monitor` or `workflow.tasks[].monitor` — hardware sampling during the
> run. Monitor fields are **concrete ints only** (not `${{ }}`-resolved). See [Monitor](./monitor.md).

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `interval` | | int | `5000` | Sampling interval (ms) for built-in scopes without their own. |
| `scopes` | | object | `null` (all) | Which scopes to collect — `cpu`, `gpu`, `memory`, `disk`, `network`, `custom`; omit ⇒ all built-ins active. |
| `resources` | | object | `null` | Which hardware to target — `nodes` / `gpus` (like task resources) or `used_by_tasks: [names]`. |
| `report` | | object | `{enabled: true}` | Post-run report. `format`: any of `csv`, `svg`, `png` (`png` needs the `sflow[monitor]` extra). |
| `report.enabled` | | bool | `true` | **On by default** whenever `monitor:` is set. Set `false` to opt out — worth doing on large fan-outs, since report cost scales with samples × views. Raw samples and the overview are still written. |
| `window` | | object | `null` | `{start, end}` task-log markers bounding the reported window, so the report covers the benchmark rather than the whole task lifecycle. Task monitors only (rejected on `workflow.monitor`) and requires `report.enabled: true`. |

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
