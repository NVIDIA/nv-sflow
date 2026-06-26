# sflow YAML Schema Reference

Complete field-by-field reference for sflow v0.1 YAML configuration.

> For narrative docs with examples:
> [configuration](https://nvidia.github.io/nv-sflow/docs/user/configuration),
> [variables](https://nvidia.github.io/nv-sflow/docs/user/variables),
> [artifacts](https://nvidia.github.io/nv-sflow/docs/user/artifacts),
> [backends](https://nvidia.github.io/nv-sflow/docs/user/backends),
> [operators](https://nvidia.github.io/nv-sflow/docs/user/operators),
> [probes](https://nvidia.github.io/nv-sflow/docs/user/probes),
> [replicas](https://nvidia.github.io/nv-sflow/docs/user/replicas),
> [resources](https://nvidia.github.io/nv-sflow/docs/user/resources),
> [modular-workflows](https://nvidia.github.io/nv-sflow/docs/user/modular-workflows),
> [outputs](https://nvidia.github.io/nv-sflow/docs/user/outputs),
> [results](https://nvidia.github.io/nv-sflow/docs/user/results).

## Top-Level Fields

```yaml
version: "0.1"          # Required. Only "0.1" is valid.
variables: {}            # Optional. Dict or list of variable declarations.
artifacts: []            # Optional. List of named URI resources.
backends: []             # Optional. List of compute backends.
operators: []            # Optional. List of task execution methods.
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

```yaml
backends:
  - name: slurm_cluster
    type: slurm            # "slurm" or "local"
    default: true           # at most one backend can be default
    nodes: 4
    partition: my_partition
    account: my_account
    time: 120               # minutes
    gpus_per_node: 8        # positive integer
    extra_args:             # additional sbatch/salloc flags
      - --exclusive
      - --gpus-per-node=8
```

### Field Reference

| Field           | Type    | Required | Default | Description                          |
|-----------------|---------|----------|---------|--------------------------------------|
| `name`          | string  | Yes      |         | Unique backend identifier            |
| `type`          | string  | Yes      |         | `"slurm"` or `"local"`              |
| `default`       | bool    | No       | false   | Mark as default (max one allowed)    |
| `nodes`         | int/expr| No       | 1       | Number of nodes to allocate          |
| `partition`     | string  | No       |         | SLURM partition name                 |
| `account`       | string  | No       |         | SLURM account name                   |
| `time`          | int/str | No       |         | SLURM time limit (minutes)           |
| `gpus_per_node` | int     | No       |         | GPUs per node (must be positive)     |
| `extra_args`    | list    | No       | []      | Additional backend arguments         |

**local backend**: Creates synthetic nodes (`localhost`). Default operator is `bash`.

**slurm backend**: Uses existing allocation (`SLURM_JOB_ID`) if available; otherwise runs `salloc`. Default operator is `srun`.

---

## Operators

```yaml
operators:
  - name: my_operator
    type: srun                    # "srun" or "bash"
    container_image: nvcr.io/nvidia/pytorch:24.07-py3
    container_name: my_container  # optional Pyxis container name
    container_writable: true      # writable container filesystem
    mpi: pmix                     # MPI implementation
    extra_args:                   # additional srun flags
      - --container-image=${{ variables.IMAGE }}
```

### Field Reference

| Field                 | Type   | Required | Description                              |
|-----------------------|--------|----------|------------------------------------------|
| `name`                | string | Yes      | Unique operator identifier               |
| `type`                | string | Yes      | `"srun"` or `"bash"`                    |
| `container_image`     | string | No       | Container image URI (Pyxis)              |
| `container_name`      | string | No       | Named container for reuse                |
| `container_writable`  | bool   | No       | Allow writes to container filesystem     |
| `mpi`                 | string | No       | MPI type (`pmix`, `pmi2`, etc.)          |
| `extra_args`          | list   | No       | Additional srun/bash flags               |

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

---

## Tasks

```yaml
- name: my_task                 # required, unique within workflow
  operator: operator_name       # optional: string or inline dict
  script:                       # required, non-empty list
    - echo "hello"
    - python run.py
  depends_on:                   # optional
    - other_task
  resources: {}                 # optional
  replicas: {}                  # optional
  probes: {}                    # optional
  retries: {}                   # optional
  outputs: []                   # optional (legacy MVP)
  result: {}                    # optional (consolidated result parsing)
```

### Task Field Reference

| Field        | Type        | Required | Description                                |
|--------------|-------------|----------|--------------------------------------------|
| `name`       | string      | Yes      | Unique task name (no duplicates allowed)   |
| `operator`   | string/dict | No       | Operator name or inline override           |
| `script`     | list[str]   | Yes      | Shell commands (non-empty)                 |
| `depends_on` | list[str]   | No       | Task names this task depends on            |
| `resources`  | dict        | No       | Node and GPU requirements                  |
| `replicas`   | dict        | No       | Replica count, policy, sweep variables     |
| `probes`     | dict        | No       | Readiness and failure probes               |
| `retries`    | dict        | No       | Retry policy                               |
| `outputs`    | list        | No       | Output parsing patterns (legacy MVP)       |
| `result`     | map/dict    | No       | Consolidated result parsing (regex map, `patterns`, or `file`) |
| `monitor`    | dict        | No       | Hardware monitor bound to this task's lifetime (see Monitor) |

### Resources

```yaml
resources:
  gpus:
    count: 4                    # CUDA_VISIBLE_DEVICES slicing
  nodes:
    indices: [0]                # pin to specific node indices
    count: 2                    # OR request N nodes (mutually exclusive with indices)
```

GPU allocation:
- Each task/replica gets `count` GPUs via `CUDA_VISIBLE_DEVICES`
- GPUs are sliced sequentially across replicas on each node
- Total GPUs across all tasks must not exceed `nodes * gpus_per_node`

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

```yaml
probes:
  readiness:
    log_watch:
      regex_pattern: "Server ready"
      match_count: 2             # wait for N matches (optional)
      logger: other_task         # watch another task's log (optional)
    timeout: 600                 # seconds
    interval: 10                 # seconds between checks
  failure:
    log_watch:
      regex_pattern: "Traceback (most recent call last)"
    interval: 10
```

Probe types (exactly one per probe):

| Type       | Fields               | Description                           |
|------------|----------------------|---------------------------------------|
| `tcp_port` | `port`               | Port accepts TCP connections          |
| `log_watch`| `regex_pattern`, optional `match_count`, `logger` | Pattern in task log |

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

## Modular Composition

Multiple YAML files are merged with `sflow compose` or `sflow run -f`:

### Merge Rules

1. `version` must match across all files
2. `variables`, `artifacts`, `backends`, `operators` merge by name (later file wins)
3. `workflow.name` must match (if set in multiple files)
4. `workflow.tasks` are concatenated in file order
5. `workflow.variables` are merged by name

### Missable Tasks

When composing configurations where some tasks may be absent (e.g., using `agg.yaml` without `prefill.yaml`):

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

---

> For detailed explanations and more examples, see:
> - [modular-workflows](https://nvidia.github.io/nv-sflow/docs/user/modular-workflows) -- merge rules, missable tasks, bulk input
> - [cli](https://nvidia.github.io/nv-sflow/docs/user/cli) -- all CLI commands and flags
> - [quick-reference](https://nvidia.github.io/nv-sflow/docs/user/quick-reference) -- all fields at a glance
