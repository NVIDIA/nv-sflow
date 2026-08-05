# sflow YAML Schema Reference

Complete field-by-field reference for sflow v0.1 YAML configuration.

> For narrative docs with examples:
> [configuration](https://nvidia.github.io/nv-sflow/docs/user/configuration),
> [quick-reference](https://nvidia.github.io/nv-sflow/docs/user/quick-reference),
> [variables](https://nvidia.github.io/nv-sflow/docs/user/variables),
> [artifacts](https://nvidia.github.io/nv-sflow/docs/user/artifacts),
> [backends](https://nvidia.github.io/nv-sflow/docs/user/backends),
> [operators](https://nvidia.github.io/nv-sflow/docs/user/operators),
> [probes](https://nvidia.github.io/nv-sflow/docs/user/probes),
> [replicas](https://nvidia.github.io/nv-sflow/docs/user/replicas),
> [resources](https://nvidia.github.io/nv-sflow/docs/user/resources),
> [monitor](https://nvidia.github.io/nv-sflow/docs/user/monitor),
> [uploads](https://nvidia.github.io/nv-sflow/docs/user/uploads),
> [modular-workflows](https://nvidia.github.io/nv-sflow/docs/user/modular-workflows),
> [outputs](https://nvidia.github.io/nv-sflow/docs/user/outputs),
> [results](https://nvidia.github.io/nv-sflow/docs/user/results),
> [cli](https://nvidia.github.io/nv-sflow/docs/user/cli).

## Top-Level Fields

```yaml
version: "0.1"          # Required. Only "0.1" is valid (schema version, not release).
variables: {}            # Optional. Dict or list of variable declarations.
artifacts: []            # Optional. List of named URI resources.
backends: []             # Optional. List of compute backends.
operators: []            # Optional. List of task execution methods.
storage: []              # Optional. List of post-run upload targets (S3).
workflow: {}             # Required. Workflow definition with tasks.
```

Extra top-level keys are **rejected** (`extra="forbid"`).

---

## Variables

Variables can be declared as a dict or list. Dict form is more common:

```yaml
variables:
  MY_VAR:
    description: "Human-readable description"   # optional
    type: integer                                # optional: integer, string
    value: 42                                    # required
    domain: [16, 32, 64]                         # optional: valid values for sweeps
```

List form:

```yaml
variables:
  - name: MY_VAR
    value: 42
```

### Field Reference

| Field         | Type              | Required | Description                                    |
|---------------|-------------------|----------|------------------------------------------------|
| `description` | string            | No       | Human-readable description                     |
| `type`        | `integer`/`string`| No       | Type hint; default inferred from value         |
| `value`       | any               | Yes      | Default value (can be an expression)           |
| `domain`      | list              | No       | Allowed values; `value` must be in `domain`    |

### Expression Syntax

Use `${{ }}` in YAML fields (resolved before execution):

```yaml
value: "${{ variables.TP_SIZE * variables.DP_SIZE }}"
value: "${{ [variables.X, 1] | max }}"
value: "${{ 'yes' if variables.FLAG else 'no' }}"
```

Available expression contexts:

| Context      | Available In       | Example                                              |
|--------------|--------------------|------------------------------------------------------|
| `variables`  | YAML fields        | `${{ variables.TP_SIZE }}`                           |
| `backends`   | YAML fields        | `${{ backends.slurm_cluster.nodes[0].ip_address }}`  |
| `artifacts`  | YAML fields        | `${{ artifacts.MODEL.path }}`                        |
| `workflow`   | YAML fields        | `${{ workflow.name }}`                               |
| `task`       | Scripts only       | `${{ task.server.nodes[0].ip_address }}`             |

### Backend Node Properties

```
backends.<name>.nodes[i].name         # hostname
backends.<name>.nodes[i].ip_address   # IP address
backends.<name>.nodes[i].index        # 0-based index
backends.<name>.nodes[i].num_gpus     # GPUs on that node
```

### Task Context (Scripts Only)

```
task.<name>.nodes[i].name
task.<name>.nodes[i].ip_address
task.<name>.gpus               # assigned GPU indices
task.<name>.backend            # backend name
task.<name>.operator           # operator name
```

For replicated tasks, use `task.<name>_0` or `task.<name>[0]`.

### Reserved Environment Variables

| Variable                         | Description                             |
|----------------------------------|-----------------------------------------|
| `SFLOW_WORKSPACE_DIR`           | Workspace root directory                |
| `SFLOW_OUTPUT_DIR`              | Output root directory                   |
| `SFLOW_WORKFLOW_OUTPUT_DIR`     | Workflow-specific output directory      |
| `SFLOW_TASK_OUTPUT_DIR`         | Task-specific output directory          |
| `SFLOW_TASK_RESULT_FILE`             | Per-task result file (`${SFLOW_TASK_OUTPUT_DIR}/result.json`) |
| `SFLOW_WORKFLOW_RESULT_FILE`   | Workflow results index (`${SFLOW_WORKFLOW_OUTPUT_DIR}/results.json`) |
| `SFLOW_REPLICA_INDEX`           | Replica index (0-based)                 |
| `SFLOW_TASK_ASSIGNED_NODE_NAMES`| Comma-separated node hostnames          |
| `SFLOW_TASK_ASSIGNED_NODE_IPS`  | Comma-separated node IP addresses       |
| `SFLOW_BACKEND_JOB_ID`          | Backend allocation/job id (Slurm job id when using Slurm) |
| `SFLOW_BACKEND_NODELIST`        | Backend allocation nodelist             |
| `SFLOW_BACKEND_NUM_NODES`       | Number of nodes in the backend allocation |
| `SFLOW_BACKEND_STEP_ID`         | Backend step id (Slurm step id when using `srun`) |
| `SFLOW_TASK_NODE_NAME`          | Runtime node hostname for this task process |
| `SFLOW_TASK_NODE_INDEX`         | Runtime node index for this task process |
| `SFLOW_TASK_PROCESS_ID`         | Runtime process/rank id for this task process |
| `SFLOW_TASK_LOCAL_PROCESS_ID`   | Runtime local process/rank id on the node |
| `SFLOW_TASK_NUM_PROCESSES`      | Runtime task/process count for this launch |

---

## Artifacts

```yaml
artifacts:
  - name: LOCAL_MODEL_PATH
    uri: fs:///absolute/path/to/model
  - name: CONFIG_FILE
    uri: file://generated_config.yaml
    content: |
      setting: ${{ variables.SOME_VAR }}
```

### Field Reference

| Field     | Type   | Required | Description                                        |
|-----------|--------|----------|----------------------------------------------------|
| `name`    | string | Yes      | Unique identifier; becomes env var name            |
| `uri`     | string | Yes      | URI with scheme (fs://, file://, http://, etc.)    |
| `content` | string | No       | Inline content; only valid with `file://` scheme   |

### URI Schemes

| Scheme    | Behavior                                                         |
|-----------|------------------------------------------------------------------|
| `fs://`   | Resolved to local path; validated in dry-run (must exist)        |
| `file://` | Generated at runtime; supports `content` field with expressions  |
| Others    | Kept as string; no download or validation                        |

Access patterns:
- YAML: `${{ artifacts.LOCAL_MODEL_PATH.path }}`
- Scripts: `${LOCAL_MODEL_PATH}` (env var)

### Path Resolution & Auto-Mount

An artifact resolves a path **and** makes it reachable inside container tasks, so it is the
preferred way to hand any path (script, config, model, dataset) to a workflow.

**Resolution.** A relative `fs://` / `file://` URI is resolved against `SFLOW_WORKSPACE_DIR`
(the dir where you invoke `sflow`, or `--workspace-dir`); absolute URIs are used as-is. The
resolved absolute path is what `${{ artifacts.NAME.path }}` / `${NAME}` expand to. Generated
`file://` files with a *relative* URI are written under the workflow output dir (keeping the
workspace clean); an *absolute* `file://` is written at that path.

**Auto-mount (same-path).** Declaring an artifact auto-mounts it into container tasks at the
**identical absolute path**, so the resolved path works both on the host and inside the
container — no manual mount wiring. Every declared artifact is **global**: it is mounted into
*every* container/pod, regardless of whether that task's script references it.

| Backend                | `fs://` (existing path)                                          | `file://` + `content`                                   |
|------------------------|-----------------------------------------------------------------|---------------------------------------------------------|
| Slurm (srun/pyxis)     | bind-mount `{path}:{path}:rw` (parent dir when path is a file; `.sqsh` skipped) | file written to output dir, its dir bind-mounted same-path |
| Docker (`docker_run`)  | same as Slurm (skipped for **remote** `hosts:` via `docker_host`/`context`) | same as Slurm                                           |
| Kubernetes             | **hostPath** at the same path (`type: Directory`/`File`, so a missing node path fails loudly) — *skipped when a declared PVC `mount_path` covers the path* | injected as a **ConfigMap**, subPath-mounted read-only at the resolved path |
| local / bash / ssh     | no container — the path is used directly on the node            | file written; used directly                             |

sflow also auto-mounts the four `SFLOW_*` directories (`SFLOW_WORKSPACE_DIR`,
`SFLOW_OUTPUT_DIR`, `SFLOW_WORKFLOW_OUTPUT_DIR`, `SFLOW_TASK_OUTPUT_DIR`) into Slurm/Docker
container tasks at the same path, so a task can write `${SFLOW_TASK_OUTPUT_DIR}` from inside
the container. On Kubernetes these live on a writable `emptyDir` mounted at `SFLOW_OUTPUT_DIR`
(see the kubernetes backend `volumes` notes and Uploads for persistence).

**Shared-filesystem note (Slurm/Lustre/GPFS/NFS).** The same-path bind mount is what makes a
single `fs://` artifact work across every node of a multi-node allocation: if the data lives on
shared storage mounted at the same absolute path on all nodes, the mount resolves identically
everywhere. Put models, datasets, the workspace, and the output dir on shared storage for
multi-node Slurm.

### Best Practice: Use `file://` Artifacts for Helper Scripts

When a task needs a helper script (Python, shell, etc.), always use a `file://` artifact
with `content` instead of embedding multi-line code inline in the task script. Inline code
via heredocs or `python3 -c` breaks due to YAML -> shell -> Python quoting conflicts:

```yaml
artifacts:
  - name: MY_PROXY_SCRIPT
    uri: file://proxy.py
    content: |
      import uvicorn
      from fastapi import FastAPI
      app = FastAPI()
      # ... full script with f-strings, quotes, etc. -- all safe here
      uvicorn.run(app, host="0.0.0.0", port=8000)

workflow:
  tasks:
    - name: proxy
      script:
        - python3 ${{ artifacts.MY_PROXY_SCRIPT.path }}
```

---

## Backends

Four backend types: `local`, `slurm`, `docker`, `kubernetes`. If `backends:` is omitted
entirely, a default `local` backend is created.

### Common Fields (all backend types)

| Field           | Type      | Required | Default | Description                          |
|-----------------|-----------|----------|---------|--------------------------------------|
| `name`          | string    | Yes      |         | Unique backend identifier            |
| `type`          | string    | Yes      |         | `local` / `slurm` / `docker` / `kubernetes` |
| `default`       | bool      | No       | false   | Mark as default (max one allowed)    |
| `gpus_per_node` | int/expr  | No       |         | GPUs/node for sflow planning & GPU index assignment. `0` = CPU-only. **Does not** add a Slurm `--gpus-per-node` flag |
| `include_nodes` | list      | No       | []      | Restrict the node pool to these hostnames (also `--include-nodes`) |
| `exclude_nodes` | list      | No       | []      | Remove these hostnames from the node pool (also `--exclude-nodes`) |

Node include/exclude is backend-agnostic: also settable via `--include-nodes` /
`--exclude-nodes` on `sflow run` / `sflow batch` (CLI values union over the recipe). A host
may not appear in both. Each backend translates it natively (Slurm `--nodelist`/`--exclude`,
Kubernetes `hostname` In/NotIn nodeAffinity, Docker `hosts:` pool filtering; ignored with a
warning on local / hostless docker).

Default operator per backend: `local` → `bash`, `slurm` → `srun`, `docker` → `docker_run`,
`kubernetes` → none (must declare a `k8s` operator).

### local

```yaml
backends:
  - { name: local, type: local, default: true, nodes: 1 }
```

`nodes` (int/expr, default `1`) creates synthetic nodes (`localhost`, `localhost-1`, ...).

### slurm

Uses an existing allocation (`SLURM_JOB_ID`) if present; otherwise runs `salloc`.

```yaml
backends:
  - name: slurm_cluster
    type: slurm
    default: true
    account: my_account       # required
    partition: my_partition   # required
    time: "01:00:00"          # required; Slurm time string (or int minutes)
    nodes: 4                  # required
    gpus_per_node: 8          # required (0 for CPU-only)
    extra_args:               # extra salloc/#SBATCH flags (de-duped by option; CLI wins)
      - --exclusive
      - --gpus-per-node=8
    # job_name: my_job        # optional; defaults to workflow name
```

### docker

Launches via `docker run` (operator `docker_run`). Uses a synthetic local allocation, or an
explicit remote `hosts:` pool.

```yaml
backends:
  - name: docker
    type: docker
    default: true
    image: ubuntu:22.04       # required (default image for docker_run)
    nodes: 1                  # default 1 (ignored when `hosts:` is set)
    gpus_per_node: 0
    # mounts: ["/data:/data:ro"]
    # workdir: /workspace
    # extra_args: ["--shm-size=16g"]
    # hosts:                  # optional multi-host pool (each needs docker_host OR context)
    #   - { name: dgx-a, docker_host: "ssh://dgx-a", ip_address: 10.0.0.11, gpus_per_node: 8 }
    #   - { name: dgx-b, context: dgx-b-context,    ip_address: 10.0.0.12, gpus_per_node: 8 }
```

### kubernetes

"Allocate-first" model: reserves real nodes with placeholder pods, discovers their names/IPs
(`backends.<name>.nodes[*].ip_address`), then runs each task as scheduler-placed pod(s). The
**workload image lives on the `k8s` operator, not the backend**. Cluster access is via CLI
flags (`--kubeconfig` / `--kube-context` / `--kube-namespace`), never YAML.

```yaml
backends:
  - name: k8s
    type: kubernetes
    default: true
    namespace: default        # one namespace per backend (injected into operators)
    nodes: 2
    gpus_per_node: 8
    scheduling: device_plugin  # default (nvidia.com/gpu) | dra (WIP; needs nvidia-dra-driver-gpu, K8s 1.34+)
    # gpu_resource_name: nvidia.com/gpu   # device-plugin GPU resource name (override for MIG/other plugins)
    # host_network: true        # default true (privileged): pod IP == node IP
    # host_ipc: true            # default false (privileged): cross-pod CUDA IPC / same-node NVLink KV
    # rdma: auto               # auto | disable | gke | shared_device_plugin | host_device
    # tolerations: [...]
    # node_selector: { pool: gpu }
    # volumes:                 # mounted into every task pod (exactly one source each)
    #   - { name: models, claim: my-model-pvc, mount_path: /models, read_only: true }
    #   - { name: cache,  empty_dir: {}, mount_path: /cache }
    # dra: { gpu_device_class: gpu.nvidia.com, rdma_device_class: rdma.nvidia.com }
    # compute_domain: { channel: auto }   # Multi-Node NVLink (IMEX); top-level, not under dra
    # probe_pod_image: curlimages/curl:latest
    # collect_max_file_size: 10Mi         # cap for syncing pod output back to driver (0 disables)
    # collect_grace_seconds: 120          # grace window (s) for copying node-local outputs back
    # reservation: { placeholder_image: my-mirror/bash:5 }   # air-gapped placeholder-pod image
    # gpu_product_label_key: nvidia.com/gpu.product   # node label read for NVLink-scope detection
    # gib_installer_namespace: kube-system            # namespace of the GKE gIB (nccl-rdma-installer) DaemonSet
```

On `device_plugin` clusters sflow derives/validates `gpus_per_node` against the nodes'
real GPU capacity at preflight (unset → derived; set-too-high → early warning).

Kubernetes has many more backend fields (RDMA/InfiniBand, ComputeDomain/MNNVL,
`merge_colocated_gpu_pods`, `host_ipc`, `cpu_per_gpu`/`cpu_request`, `nvlink_domain`,
`gpu_resource_name`, `gpu_product_label_key`, `gib_installer_namespace`,
`collect_grace_seconds`, `reservation.placeholder_image`, ...).
See [backends](https://nvidia.github.io/nv-sflow/docs/user/backends) for the full set.

#### Kubernetes `volumes:` (PVC / emptyDir shared storage)

A backend `volumes:` entry is **workflow-wide storage**: it is mounted into *every* task pod of
the backend (as a pod volume + container `volumeMount` at `mount_path`), regardless of whether a
task's script names that path. This is how cluster-resident data (e.g. a model on shared storage)
reaches pods. Set **exactly one** source per entry — `claim` (an existing PVC) or `empty_dir`.

```yaml
backends:
  - name: k8s
    type: kubernetes
    volumes:
      - name: models                # existing PVC (shared, read-only model store)
        claim: model-store-pvc       # the PVC + its data must already exist in the namespace
        mount_path: /models
        read_only: true              # PVC default; keeps one PVC shareable across pods/nodes
      - name: outputs                # writable PVC to persist run output (see note below)
        claim: run-output-pvc
        mount_path: /sflow_output
        read_only: false
        ensure_writable: true        # root initContainer mkdir+chmod 0777 the mounted subPath
      - name: cache                  # per-pod ephemeral scratch (JIT/kernel cache)
        empty_dir: { medium: "", size_limit: 50Gi }
        mount_path: /root/.cache
```

| Field            | Type            | Required | Description                                                                 |
|------------------|-----------------|----------|-----------------------------------------------------------------------------|
| `name`           | string          | Yes      | Pod volume name (DNS-1123); links volume ↔ volumeMount                       |
| `claim`          | string          | one-of   | Name of an **existing** PersistentVolumeClaim (`spec.volumes[].persistentVolumeClaim.claimName`). Mutually exclusive with `empty_dir` |
| `empty_dir`      | dict            | one-of   | Ephemeral per-pod scratch: `medium` (`""` node disk \| `"Memory"` tmpfs), `size_limit` (e.g. `"50Gi"`). Mutually exclusive with `claim` |
| `mount_path`     | string (abs)    | Yes      | Absolute in-pod mount path                                                   |
| `sub_path`       | string          | No       | `volumeMount.subPath` (mount a subdir of the volume)                         |
| `read_only`      | bool            | No       | Default per source: **PVC `true`**, emptyDir `false`. Set explicitly to override |
| `ensure_writable`| bool            | No       | PVC + `read_only: false` only: inject a root initContainer that `mkdir -p` + `chmod 0777` the mounted path, fixing the "subPath created root-owned → non-root pod can't write" gotcha. Rejected on an emptyDir or a read-only PVC |

**PVC covers an `fs://` artifact.** When an `fs://` artifact path equals or sits under a declared
PVC `mount_path`, the PVC serves it and sflow **skips the per-artifact hostPath fallback** — so
`${{ artifacts.MODEL.path }}` resolves to the in-pod path on the PVC. That path matching governs
only the hostPath fallback, never whether the PVC mounts (it always does).

**K8S output dirs are ephemeral.** `SFLOW_OUTPUT_DIR` (and its `WORKFLOW`/`TASK` subdirs) is a
writable `emptyDir` mounted at the same path so driver-host paths are valid + writable in the pod.
It is **node-local and per-pod** — it does not persist past the pod and is not shared across pods.
After a task, files ≤ `collect_max_file_size` (backend/operator field, default 10 MiB, `0`
disables) are `kubectl cp`'d back to the driver; larger or must-persist outputs need a writable
PVC mounted at the output path or an `uploads:` target.

**Debug flag.** `sflow run --kube-skip-pvc` drops every PVC-backed backend volume for the run
(keeps `empty_dir`) so pods schedule on a cluster lacking the recipe's PVCs — the PVC data is not
mounted, so it is for quick plumbing checks only.

---

## Operators

Operator `type` is one of: `bash`, `srun`, `docker_run`, `python`, `ssh`, `k8s`, `k8s_mpi`.
Every operator has `name` (required) and `type` (required). Type-specific fields follow.

> **Docker naming:** the Docker launch operator is `type: docker_run` (the Docker *backend*
> is `type: docker`). A stray `type: docker` operator is rejected with a hint to use
> `docker_run`.

### srun (Slurm)

```yaml
operators:
  - name: my_operator
    type: srun
    container_image: nvcr.io/nvidia/pytorch:24.07-py3
    container_name: my_container  # optional Pyxis container name (reuse; see below)
    container_writable: true      # default true
    mpi: pmix                     # MPI implementation (pmix, pmi2, ...)
    extra_args:
      - --container-image=${{ variables.IMAGE }}
```

Common fields: `container_image` / `container_name` (mutually exclusive),
`container_writable`, `container_mounts` (list, e.g. `"/host:/ctr:rw"`), `container_workdir`,
`mpi`, `ntasks`, `ntasks_per_node`, `gres`, `mem`, `cpus_per_task`, `extra_args`, plus many
raw Slurm knobs (`partition`, `account`, `qos`, `reservation`, `time`, `exclusive`, ...).
See [operators](https://nvidia.github.io/nv-sflow/docs/user/operators) for the full list.

### docker_run

| Field       | Type   | Required | Default | Description                              |
|-------------|--------|----------|---------|------------------------------------------|
| `image`     | string | Yes      |         | Docker image (defaults to backend `image` if set) |
| `workdir`   | string | No       | null    | Working dir inside the container         |
| `mounts`    | list   | No       | []      | Bind mounts (`"/host:/ctr:rw"`)          |
| `gpus`      | string | No       | null    | GPU spec (`all`, `device=0`); narrowed to the planned slice |
| `pass_envs` | bool   | No       | true    | Forward host env vars                    |
| `extra_args`| list   | No       | []      | Extra `docker run` args                  |

### python / ssh / bash

- `python`: `python_exec` (default `"python"`), `extra_args`.
- `ssh`: `host` (required), `user`, `port`, `identity_file`, `extra_args`.
- `bash`: no extra fields (runs the script directly on the assigned node).

### k8s

Renders each task into pinned pod(s) and `kubectl apply`s them. Required on the `kubernetes`
backend (carries the workload `image`).

```yaml
operators:
  - name: server_op
    type: k8s
    image: my/server:1.0          # required (workload image)
    # image_pull_policy: IfNotPresent
    # image_pull_secrets: [my-regcred]
    # host_network: true
    # node_selector: { pool: gpu }
    # tty: true                   # live \r progress bars (aiperf/pip) via kubectl logs
    # cpu: 16                     # container cpu request (default: none -> BestEffort)
    # memory: 32Gi                # container memory request
    # cpu_limit: 24 / memory_limit: 48Gi
    # env: [ { name: POD_IP, valueFrom: { fieldRef: { fieldPath: status.podIP } } } ]
    # security_context / pod_security_context / lifecycle / service_account / labels / ...
    # container_overrides / pod_overrides   # raw dicts, deep-merged last (win)
```

The `k8s` operator also maps common `v1.Container` / `v1.PodSpec` fields (`working_dir`,
`ports`, `runtime_class`, `priority_class`, `termination_grace_period`, `annotations`, ...);
see the field table in [backends](https://nvidia.github.io/nv-sflow/docs/user/backends).

### Container Reuse with Named Containers

Combine `--container-image` in `extra_args` with `container_name` to import the image once
and reuse the named container for all subsequent tasks. The first task (e.g. `load_image`)
pays the import cost; all later tasks using the same `container_name` start instantly:

```yaml
operators:
  - name: my_runtime
    type: srun
    container_name: my_container        # attach by name (skips import if exists)
    container_writable: true
    mpi: pmix
    extra_args:
      - --container-image=${{ variables.MY_IMAGE }}  # imports only when name not found
```

### Task-Level Operator Override

Tasks can override operator settings inline:

```yaml
operator:
  name: my_operator       # reference named operator
  ntasks: 4               # total MPI tasks
  ntasks_per_node: 1      # tasks per node
  extra_args:             # additional flags for this task only
    - --mem=64G
```

**Override keys are strict.** Every key here must be a real field of the operator this task
resolves to. An unknown key — a typo, or a setting for a *different* backend's operator
(e.g. an srun `ntasks_per_node` on a `kubernetes` operator) — is a **hard error**, not a
silent drop. For a setting that should only apply under one backend, define it on the
operator **inside that backend's fragment** (operators deep-merge by name per backend) so
the task itself stays backend-portable.

### Kubernetes MPI (`type: k8s_mpi`)

**Use `k8s_mpi` for any MPI workload** (anything launched with `mpirun` — TRT-LLM
`trtllm-llmapi-launch`, multi-GPU TP/PP serving) on a `kubernetes` backend, and keep a
plain **explicit** `mpirun -np N ... <workload>` in the task script. sflow reads that
command to build the correct MPI world for **both single-node and multi-node** (a lone
launcher pod when it fits one node, launcher + worker pods when it spans nodes), so the
same recipe scales from `-np 8` to `-np 16` with no SSH/hostfile glue. sflow injects the
SSH keypair, hostfile, `sshd`, wait-for-workers, and `-x` env forwarding driver-side.
Two interchangeable routes are auto-selected:

- `operator` — emit a Kubeflow `MPIJob` CR (requires the `mpijobs.kubeflow.org`
  CRD + mpi-operator controller; sflow does not install them).
- `pods` — plain pods with sflow's injected MPI bootstrap preamble.
- `auto` (default) — use the operator when its CRD is present, else pods.

```yaml
operators:
  - name: mpi_server
    type: k8s_mpi
    image: ${{ variables.CONTAINER_IMAGE }}
    shm_size: 64Gi
    run_as_root: true            # sshd host keys / ~/.ssh need root
    cpu: 16                      # optional cpu request (default: none, unless the backend sets cpu_per_gpu/cpu_request)
    memory: 32Gi                 # optional memory request (default: unset)
    # cpu_limit: 24              # optional hard caps (default: unset -> requests-only)
    # memory_limit: 48Gi
    mpi:
      route: auto                # auto | operator | pods
      slots_per_worker: 8        # hostfile slots/node; default = per-node GPUs
      run_launcher_as_worker: true
      launcher_creation_policy: WaitForWorkersReady   # or AtStartup
      ssh_port: 2222
      mpi_implementation: OpenMPI # OpenMPI | Intel | MPICH
      ensure_sshd: true          # apt-install openssh-server if the image lacks it
      forward_env_prefixes: [TRTLLM_, TLLM_, FLASHINFER_]  # ADDITIVE (see below)
      omp_num_threads: 8         # per-rank OMP_NUM_THREADS default (null/0 to disable)
      worker_setup_timeout_seconds: 900   # per-node setup budget before a worker is reaped (~15 min)
      launcher_discovery_timeout: 600     # seconds to wait for the mpi-operator to create the launcher pod
```

The task then carries only its launch line (no keypair/hostfile/sshd/wait/`-x`):

```yaml
    - name: server
      operator: mpi_server
      resources:
        gpus: { count: 16 }      # 16 GPUs -> spans 2 nodes (8/node), one MPI world
      script:
        - mpirun -np 16 --allow-run-as-root trtllm-llmapi-launch trtllm-serve /model --port 8336
```

**`mpi` field reference**

| Field                     | Type      | Default              | Description |
|---------------------------|-----------|----------------------|-------------|
| `route`                   | enum      | `auto`               | `auto` / `operator` / `pods` |
| `slots_per_worker`        | int       | per-node GPU count   | Hostfile slots per node / MPIJob `slotsPerWorker` |
| `run_launcher_as_worker`  | bool      | `true`               | Launcher node also runs ranks (the HTTP server) |
| `launcher_creation_policy`| enum      | `WaitForWorkersReady`| `AtStartup` / `WaitForWorkersReady` |
| `ssh_port`                | int       | `2222`               | sshd port (avoids `:22` under `host_network`) |
| `mpi_implementation`      | enum      | `OpenMPI`            | `OpenMPI` / `Intel` / `MPICH` |
| `ensure_sshd`             | bool      | `true`               | Install `openssh-server` at startup if missing |
| `forward_env_prefixes`    | list[str] | `[]`                 | ADDITIVE app env prefixes to forward to ranks |
| `omp_num_threads`         | int\|null | `8`                  | Per-rank `OMP_NUM_THREADS` (pod env, forwarded to ranks). Caps OpenMP so co-located ranks don't exhaust pthreads at model load; a recipe `export OMP_NUM_THREADS=...` overrides it; `null`/`0` disables |
| `worker_setup_timeout_seconds` | int/expr | `900` | Per-node setup budget (image apt-install + weight staging) before a worker's readiness probe reaps it (operator route). Rendered as a probe with a fixed 5s poll and `failureThreshold = ceil(timeout/5)`. Raise for a large first-time weight download over slow storage |
| `launcher_discovery_timeout`   | int/expr | `600` | Seconds to wait for the mpi-operator controller to create the launcher pod after the MPIJob is applied (operator route) |

**Env forwarding.** `mpirun` over SSH gives remote ranks a *bare* environment, so
env is forwarded with `-x`. sflow forwards a built-in transport/system set (`NCCL_`,
`UCX_`, `GLOO_`, `NVSHMEM_`, `OMPI_MCA_`, `SFLOW_` + `PATH`, `LD_LIBRARY_PATH`,
`OMP_NUM_THREADS`) AND every var the recipe itself `export`s (auto-detected via a pre-recipe snapshot), so a
plain `export FOO=bar` in the recipe reaches workers with no extra config.
`forward_env_prefixes` is only needed for vars that come from the *pod* env (declared
task env) rather than a recipe `export`. Forwarding is transparent (a PATH-shadowing
`mpirun` wrapper); the same recipe runs on both routes.

**Worker parity.** The recipe's setup (everything except the final `mpirun`) runs on
*every* node, so filesystem side-effects (`mkdir`, writing files) exist on the
workers too -- not just env. Only the `mpirun` launch is leader-only, and the leader
never launches until each worker's setup finished. Keep per-node setup idempotent and
avoid writing to shared storage concurrently from every node.

Write a plain `mpirun ...` launch: sflow auto-`exec`s it (so `mpirun` becomes the
container's main process for clean SIGTERM handling) -- you do not add `exec`
yourself. An explicit `exec mpirun` is still honored (not double-`exec`ed).

Notes: use `type: k8s_mpi` for any `mpirun`-launched workload (single- **or** multi-node —
the explicit `mpirun -np N` sets up the world either way); reserve plain `type: k8s` for
non-MPI, single-pod workloads. `image` must provide `sshd` (or keep `ensure_sshd: true`). The curated
pod/container fields and `container_overrides` / `pod_overrides` escape hatches from the
`### k8s` section above apply on the `pods` route (they are ignored on the `MPIJob` route).

---

## Workflow

```yaml
workflow:
  name: my_workflow           # required
  timeout: 120m               # optional: workflow-level timeout
  variables:                  # optional: workflow-scoped variables (override global)
    HEAD_NODE_IP:
      value: "${{ backends.slurm_cluster.nodes[0].ip_address }}"
  tasks: [...]                # required: list of task definitions
```

### Field Reference

| Field       | Type   | Required | Description                                    |
|-------------|--------|----------|------------------------------------------------|
| `name`      | string | Yes      | Workflow name                                  |
| `timeout`   | string | No       | Timeout (e.g. `"120m"`, `"2h"`)               |
| `variables` | dict   | No       | Workflow-scoped variables (override global)    |
| `tasks`     | list   | Yes      | Non-empty list of task definitions             |
| `monitor`   | dict   | No       | Hardware monitor over the whole pool, run for the full workflow lifetime (see Monitor) |
| `upload_all`| dict   | No       | Zip the whole run output dir and upload to a storage target after the run (see Storage & Uploads) |

---

## Tasks

```yaml
- name: my_task                 # required, unique within workflow
  operator: operator_name       # optional: string or inline dict
  backend: backend_name         # optional: string or inline dict (else the default backend)
  script:                       # required, non-empty list
    - echo "hello"
    - python run.py
  depends_on:                   # optional
    - other_task
  required_by:                  # optional (reverse of depends_on)
    - downstream_task
  resources: {}                 # optional
  replicas: {}                  # optional
  probes: {}                    # optional
  retries: {}                   # optional
  timeout: 30m                  # optional: per-task timeout
  fail_fast: false              # optional: unset=backend default (true on K8s, false on Slurm/local/docker); prepends `set -e` when effective-true
  ports: []                     # optional: named service ports the task exposes
  variables: {}                 # optional: task-scoped variables
  outputs: []                   # optional (legacy MVP)
  result: {}                    # optional (consolidated result parsing)
  uploads: []                   # optional (post-run S3 uploads; see Storage & Uploads)
  monitor: {}                   # optional (hardware monitor bound to this task)
```

### Task Field Reference

| Field        | Type        | Required | Description                                |
|--------------|-------------|----------|--------------------------------------------|
| `name`       | string      | Yes      | Unique task name (no duplicates allowed)   |
| `operator`   | string/dict | No       | Operator name or inline override           |
| `backend`    | string/dict | No       | Backend name or inline override (else the default backend) |
| `script`     | list[str]   | Yes      | Shell commands (non-empty)                 |
| `depends_on` | list[str]   | No       | Task names this task depends on            |
| `required_by`| list[str]   | No       | Reverse of `depends_on`: downstream tasks that must run after this one. Absent targets are skipped (ideal for modular fragments) |
| `resources`  | dict        | No       | Node and GPU requirements                  |
| `replicas`   | dict        | No       | Replica count, policy, sweep variables     |
| `probes`     | dict        | No       | Readiness and failure probes               |
| `retries`    | dict        | No       | Retry policy                               |
| `timeout`    | int/str     | No       | Per-task timeout (e.g. `1800`, `"30m"`)    |
| `fail_fast`  | bool        | No       | **Default depends on the backend** — `true` on Kubernetes, `false` on Slurm/local/docker (shell default: only the LAST command's exit code counts). Leave unset for the backend default, or set explicitly (`true`/`false`) to override per task. When effective-true, runs the task's shell script with `set -e` so a failed command fails the task instead of a later successful command (e.g. a trailing `echo`) masking it. Ignored by the `python` operator (its script is Python source). |
| `ports`      | list        | No       | Named service ports (`{ port, name }`) the task exposes |
| `variables`  | dict/list   | No       | Task-scoped variables (same format as global) |
| `outputs`    | list        | No       | Output parsing patterns (legacy MVP)       |
| `result`     | map/dict    | No       | Consolidated result parsing (regex map, `patterns`, or `file`) |
| `uploads`    | list        | No       | Post-run uploads to a storage target (see Storage & Uploads) |
| `monitor`    | dict        | No       | Hardware monitor bound to this task's lifetime (see Monitor) |

### Resources

```yaml
resources:
  gpus:
    count: 4                    # CUDA_VISIBLE_DEVICES slicing (total across nodes)
    indices: [0, 1]             # optional: pin to specific GPU device ids per node
    release_after: workflow_completion   # optional: when the GPU reservation frees
  nodes:
    indices: [0]                # pin to specific node indices
    count: 2                    # OR request N nodes (mutually exclusive with indices)
    exclude: [0]                # optional: node indices removed before indices/count/packing
    release_after: workflow_completion   # optional
```

GPU allocation (at least one of `count` / `indices` is required):
- `count` alone is index-agnostic: the task gets any contiguous idle run of `count`
  GPUs, and a run never straddles a node boundary (only GPUs 2,3 free on node0 and
  the task needs 4 → it skips to node1's GPU 0)
- `indices` alone pins those device ids on the first node where all of them are free;
  the node scan restarts at node 0 each time, so later tasks backfill idle slots
- `count` + `indices`: `count` is the total, `indices` the per-node slice, so the task
  spans `count / len(indices)` nodes using the same slots on each. `count` must be a
  positive multiple of `len(indices)`
- GPU indices must be non-negative (no Python-style `-1`), and `CUDA_VISIBLE_DEVICES`
  preserves the order written
- GPUs are sliced sequentially across replicas on each node
- Total GPUs across all tasks must not exceed `nodes * gpus_per_node`

`release_after` (`workflow_completion` | `task_ready` | `task_completion`) controls when a
reservation can be reused by other tasks. For GPUs it is inferred when omitted (tasks with a
readiness probe hold until workflow completion; probe-less tasks release after completion).
Dry-run rehearses these lifetimes across the DAG.

**Kubernetes:** `gpus.indices` is rejected at plan time (the device plugin/DRA picks the
physical devices, so a pin cannot be honored — use `count`).
`gpus.count` is a per-task total split evenly across the assigned nodes (must
be a multiple of the node count); for an ordinary pod the device-plugin/DRA assigns physical
devices, so `CUDA_VISIBLE_DEVICES` is left unset — but when co-located GPU tasks are merged
into one pod (`merge_colocated_gpu_pods`, default `auto`) sflow sets a per-member
`CUDA_VISIBLE_DEVICES` to pin each task's slice. `gpus.release_after: task_ready` is coerced to
`task_completion` (a running pod's GPUs cannot be shared).

### Replicas

```yaml
replicas:
  count: 4                      # number of replicas
  policy: "parallel"            # "parallel" or "sequential"
  variables:                    # sweep variable names (must have domain)
    - CONCURRENCY
```

| Field       | Type   | Default    | Description                                  |
|-------------|--------|------------|----------------------------------------------|
| `count`     | int    | 1          | Number of replicas                           |
| `policy`    | string | "parallel" | `"parallel"` or `"sequential"`               |
| `variables` | list   | []         | Variable names with `domain` for sweeps      |

Naming: replicas are named `task_0`, `task_1`, etc. Sweep replicas use Cartesian product indices.

### Probes

`readiness` and `failure` each accept **one probe or a list of probes** (multi-probe).

```yaml
probes:
  readiness:                      # one probe, or a list of probes
    log_watch:
      regex_pattern: "Server ready"  # literal by default; prefix re:/regex: for regex
      match_count: 2                 # wait for N matches (optional, default 1)
      logger: other_task             # watch another task's log (optional)
    timeout: 600                     # readiness deadline (s); default 1200
    interval: 10                     # seconds between checks; default 5
  failure:
    - log_watch: { regex_pattern: "Traceback (most recent call last)" }
    - log_watch: { regex_pattern: "CUDA out of memory" }
```

Exactly one probe **type** must be set per probe entry:

| Type        | Required | Optional                    | Description                       |
|-------------|----------|-----------------------------|-----------------------------------|
| `tcp_port`  | `port`   | `host`, `on_node` (`first`/`each`) | Port accepts TCP connections |
| `http_get`  | `url`    | `headers`                   | HTTP GET health check             |
| `http_post` | `url`    | `headers`, `body`           | HTTP POST health check            |
| `log_watch` | `regex_pattern` or `match_pattern` | `match_count`, `logger` | Pattern in task log     |

Common timing fields (all optional, expressions allowed): `delay` (0), `timeout` (1200,
readiness deadline; not an overall deadline for `failure`), `each_check_timeout` (30),
`interval` (5), `success_threshold` (1), `failure_threshold` (3).

**`log_watch` matches literally by default** — the value of `regex_pattern` (or its alias
`match_pattern`) is treated as a plain substring (parentheses, `.`, `*`, etc. are not
special). Prefix with `re:` or `regex:` for true regex. `logger` watches another task's
`<task>.log`; `match_count` requires N cumulative matches.

**Kubernetes:** `tcp_port` / `http_get` / `http_post` probes run from an in-cluster probe pod
(the `sflow run` host usually cannot route to the pod network). `log_watch` reads the local
`<task>.log` and is unaffected.

### Retries

```yaml
retries:
  count: 3          # max retry attempts
  interval: 30      # seconds between retries
  backoff: 2        # multiplier for interval
```

### Outputs

```yaml
outputs:
  - pattern: "accuracy: {accuracy:f}"
```

Parsed values written to `outputs.json` after task success.

### Result

Consolidated, regex-first result parsing. Successor to `outputs`. Three shapes:

```yaml
# 1. Simple regex map (first capture group; type=auto, aggregate=last)
result:
  ttft: 'TTFT:\s*([0-9.]+)\s*ms'
  tps: 'tok/s:\s*([0-9.]+)'

# 2. Advanced patterns (type cast, aggregation, units, required)
result:
  patterns:
    - name: ttft
      regex: 'TTFT:\s*(?P<value>[0-9.]+)\s*ms'
      type: float
      unit: ms
      aggregate: last        # first|last|list|count|min|max|avg|sum
      required: true

# 3. File source (task writes JSON; sflow normalizes it)
result:
  file: result.json          # relative to SFLOW_TASK_OUTPUT_DIR; or write $SFLOW_TASK_RESULT_FILE
```

- `patterns` and `file` are mutually exclusive.
- `result.file` must end in `.json`; use `patterns:` for a metric literally
  named `file`.
- Runs after task success; a task is not `COMPLETED` until parsing finishes, so
  dependents can read the files below.
- Writes `${SFLOW_TASK_OUTPUT_DIR}/result.json` and updates
  `${SFLOW_WORKFLOW_OUTPUT_DIR}/results.json`. Env vars `SFLOW_TASK_RESULT_FILE` and
  `SFLOW_WORKFLOW_RESULT_FILE` point at these.
- Best-effort: a missing `required` match sets `ok: false` but does not fail the
  workflow in this release.

---

## Monitor

Hardware resource monitoring. Supported at two scopes:

- `workflow.monitor` — covers the whole pool by default; runs for the full
  workflow lifetime.
- `tasks[*].monitor` — bound to a task; starts when the task starts and stops
  when the task's process exits (a long-running `READY` service is monitored
  until workflow teardown).

```yaml
workflow:
  monitor:                       # workflow-level: whole pool, full run
    interval: 5000               # default sample interval (ms)
    scopes:                      # optional; omit to collect ALL built-ins
      gpu: { interval: 1000 }    # cpu | gpu | memory | disk | network
      cpu: {}
      custom:                    # extra user commands run on the node
        script:
          - my_probe.sh >> ${SFLOW_TASK_OUTPUT_DIR}/custom.log
    report:                      # optional detailed per-consumer report
      enabled: true
      format: [csv, svg]         # default; both pure-stdlib. add `png` for raster
                                 # (needs the optional `sflow[monitor]` / matplotlib extra)
  tasks:
    - name: aiperf
      script: [ ... ]
      depends_on: [prefill_server, decode_server]
      monitor:
        resources:               # default: the task's own assigned nodes/GPUs
          used_by_tasks: [prefill_server, decode_server]  # monitor THESE instead
          gpus: { count: 8 }     # reporting filter (GPUs to include)
        scopes:
          gpu: {}
        report: { enabled: true }
```

### How it works

- **Bare host, passive:** collectors run directly on the node host (no container)
  via `nvidia-smi` + `/proc`, overlapping the workload (`srun --overlap`) without
  reserving GPUs.
- **Singleton per node:** at most one collector runs per physical node; overlapping
  monitors (workflow + tasks, overlapping `used_by_tasks`) reuse it and collect the
  union of requested scopes. GPU/scope/time-window subsets are applied at report time.
- **Zero workload overhead:** only lightweight sampling happens during the run; all
  aggregation/plots run once AFTER the workflow finishes.

### Field Reference

| Field       | Type        | Description                                             |
|-------------|-------------|---------------------------------------------------------|
| `interval`  | int         | Default sample interval in ms (default `5000`, min `100`) |
| `scopes`    | dict        | Active scopes; omit to enable all built-ins             |
| `resources` | dict        | `nodes` / `gpus` (like `tasks.resources`) + `used_by_tasks` |
| `report`    | dict        | `enabled` (bool) + `format` (default `[csv, svg]`; `png` is optional) |

Scopes: `cpu`, `gpu` (accepts `fields:` to override the `nvidia-smi` query),
`memory`, `disk`, `network` (each accepts `enabled`/`interval`), and `custom`
(`script: [...]`). When `scopes` is omitted entirely, all built-ins are enabled.

### Outputs

- `<run>/sflow_monitor.log` — terminal-friendly overview (min/avg/max tables +
  ASCII sparkline timelines), sibling to `sflow_summary.log`.
- `<run>/sflow_monitor/raw/` — raw per-node CSV samples.
- `<run>/sflow_monitor/<consumer>/` — detailed timeline + summary CSVs and an SVG
  timeline for each monitor with `report.enabled` (`format` defaults to
  `[csv, svg]`, both pure-stdlib). Add `png` for a raster chart (needs the
  optional `sflow[monitor]` / matplotlib extra; skipped gracefully if absent).

---

## Storage & Uploads

Post-execution upload of task files to cloud storage (S3 today). Declare named top-level
`storage:` targets, then attach per-task `uploads:` (run on task `COMPLETED`) and/or
`workflow.upload_all` (zip the whole run at the end). Requires the `sflow[s3]` extra.
**Credentials come from the boto3 default credential chain — never put secrets in YAML.**

```yaml
storage:
  - name: results_bucket
    type: s3
    bucket: my-bench-results
    region: us-west-2                       # optional
    prefix: "runs/${{ variables.RUN_ID }}/" # optional; prepended to every remote key
    # endpoint_url: http://minio.local:9000  # S3-compatible stores (MinIO, Ceph RGW)
    # addressing_style: path                 # auto|virtual|path (defaults to path when endpoint_url set)
    # storage_class: STANDARD_IA

workflow:
  upload_all:                    # optional: zip the whole run output dir and upload
    target: results_bucket
    to: "archive/${{ workflow.run_id }}.zip" # optional; default <run_id>.zip
    on_error: warn
  tasks:
    - name: benchmark
      script: [ ... ]
      uploads:
        - target: results_bucket
          from: "${{ task.output_dir }}/results.csv"   # local path or glob; ${{ }} allowed
          to: "main/results.csv"                        # optional; relative to prefix
          on_error: fail                                # warn (default) | fail
        - target: results_bucket
          from: "${{ task.output_dir }}/*.json"         # glob -> `to` must end with '/' or be omitted
          on_error: warn
```

### Storage target fields

| Field              | Required | Description                                              |
|--------------------|----------|----------------------------------------------------------|
| `name`             | Yes      | Referenced by each upload spec's `target:`               |
| `type`             | Yes      | Plugin discriminator (`s3` today)                        |
| `bucket`           | Yes (s3) | S3 bucket name                                           |
| `region`           | No       | AWS region (else boto3's chain)                          |
| `prefix`           | No       | Key prefix prepended to every uploaded object            |
| `endpoint_url`     | No       | S3-compatible endpoint (MinIO, Ceph RGW, ...)            |
| `addressing_style` | No       | `auto`/`virtual`/`path`; defaults to `path` when `endpoint_url` is set |
| `storage_class`    | No       | S3 storage class (e.g. `STANDARD_IA`)                    |

### Upload spec fields (`tasks[].uploads[]`)

| Field      | Required | Description                                                        |
|------------|----------|--------------------------------------------------------------------|
| `target`   | Yes      | Name of a declared storage target                                  |
| `from`     | Yes      | Local file path or glob; may contain `${{ }}` (resolved at task completion) |
| `to`       | No       | Remote key relative to the target `prefix`; when `from` is a glob it must end in `/` or be omitted. For replicated tasks the basename is auto-suffixed with the replica name unless `to` references `${{ task.name }}` |
| `on_error` | No       | `warn` (default; task stays `COMPLETED`) or `fail` (task → `FAILED`, triggers fail-fast). Uploads are not retried by `retries:` |

`workflow.upload_all` uses `target`, `to` (optional, default `<run_id>.zip`), and `on_error`.
`--dry-run` lists every planned upload plus an offline S3 credential preflight. See
[uploads](https://nvidia.github.io/nv-sflow/docs/user/uploads).

---

## Modular Composition

Multiple YAML files are merged with `sflow compose` or `sflow run -f`:

### Merge Rules

Files are combined with a **recursive deep merge**. The only key that
distinguishes entries in a collection is their `name`.

1. `version` must match across all files (hard error on conflict)
2. `variables`, `artifacts`, `backends`, `operators`, `storage` merge **by name**; entries with the same name are deep-merged, so a definition can be *scattered* across files (e.g. an operator's `type` in one file, an override in another)
3. `workflow.name` is no longer restricted: the last non-null name wins (a warning is logged if it changes)
4. `workflow.tasks` merge **by name** (same-name tasks are deep-merged), preserving first-seen order — the same task can be defined in one file and extended in another
5. `workflow.variables` merge by name
6. `workflow.timeout` / `upload_all` / `monitor` and any other keys are deep-merged
7. Conflicts occur only at **leaf values** (scalars and value-lists such as `script` or `depends_on`): the last file wins, with an override warning

### Reverse Dependencies (`required_by`) — preferred for modular fragments

`required_by` is the reverse pointer of `depends_on`:

- `A depends_on: [B]` → A runs after B.
- `A required_by: [B]` → B runs after A (equivalent to `B depends_on: [A]`).

It is folded into the targets' `depends_on` at load time, and **targets that are
absent from the merged workflow are skipped silently**. This lets a shared hub
task (e.g. `benchmark`) stay dependency-free while each optional server fragment
declares `required_by: [benchmark]` on itself:

```yaml
# agg_server.yaml (one of several interchangeable server fragments)
- name: agg_server
  script: [ ... ]
  required_by: [benchmark]   # no cross-file reference to prefill/decode needed
```

Composing `base.yaml + agg_server.yaml + benchmark.yaml` needs no
`--missable-tasks`, because omitted fragments simply never add their edge.

### Missable Tasks

`required_by` is preferred, but for existing configs that keep `depends_on` on a
hub task, `--missable-tasks` still works. When composing configurations where
some tasks may be absent (e.g., using `agg.yaml` without `prefill.yaml`):

```bash
sflow run -f base.yaml -f agg.yaml -f benchmark.yaml \
  --missable-tasks prefill_server --missable-tasks decode_server
```

This removes `prefill_server` and `decode_server` from all `depends_on` lists and probe `logger` references.

### Variable Overrides

```bash
sflow run -f config.yaml --set TP_SIZE=8 --set CONCURRENCY=[16,32,64]
sflow run -f config.yaml --artifact LOCAL_MODEL_PATH=fs:///new/path
```

### Common CLI flags (run / batch)

- `sflow --version` / `-V`: print version + runtime/install details (no subcommand).
- `--dry-run`, `--tui`: validate/plan; Rich TUI.
- `--include-nodes` / `--exclude-nodes`: backend-agnostic node pool control (comma-separated,
  quoted whitespace, or repeatable; union over recipe `include_nodes`/`exclude_nodes`).
- Extra backend args (de-duped by option, CLI wins): `--extra-args` / `-e` (generic — Slurm
  `salloc`, docker `docker run`, and every `kubectl` call), `--extra-salloc-args` (Slurm only),
  `--extra-docker-args` (docker only), `--extra-kubectl-args` (kubectl only).
- Kubernetes access (keep out of YAML): `--kubeconfig`, `--kube-context`, `--kube-namespace`,
  `--kube-node-selector`, `--kube-compute-domain-channel`,
  `--kube-compute-domain-create` / `--no-kube-compute-domain-create`.
- Kubernetes debug: `--kube-skip-pvc` drops every PVC-backed backend volume (`volumes:` with a
  `claim`) for the run (keeps `empty_dir`), so pods schedule on a cluster lacking the recipe's
  PVCs without editing the recipe. The PVC data is not mounted — quick plumbing checks only.

---

> For detailed explanations and more examples, see:
> - [modular-workflows](https://nvidia.github.io/nv-sflow/docs/user/modular-workflows) -- merge rules, missable tasks, bulk input
> - [cli](https://nvidia.github.io/nv-sflow/docs/user/cli) -- all CLI commands and flags
> - [quick-reference](https://nvidia.github.io/nv-sflow/docs/user/quick-reference) -- all fields at a glance
