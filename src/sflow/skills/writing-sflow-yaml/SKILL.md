---
name: writing-sflow-yaml
description: >-
  Write, create, and modify sflow YAML workflow configuration files. Covers schema structure,
  variable declarations, task DAGs, backends, operators, probes, replicas, artifacts, and
  modular composition. Use when the user asks to create an sflow YAML, configure a workflow,
  set up inference serving, or asks about sflow YAML syntax.
---

# Writing sflow YAML Configurations

## Quick Start

Every sflow YAML starts with `version: "0.1"` and defines a `workflow` with `tasks`.
The five top-level sections are: `variables`, `artifacts`, `backends`, `operators`, `workflow`.

### Minimal Local Workflow

```yaml
version: "0.1"

variables:
  WHO:
    value: "World"

workflow:
  name: hello
  tasks:
    - name: greet
      script:
        - echo "Hello ${WHO}"
```

### Minimal SLURM Workflow

```yaml
version: "0.1"

variables:
  SLURM_ACCOUNT:
    value: my_account
  SLURM_PARTITION:
    value: my_partition

backends:
  - name: slurm_cluster
    type: slurm
    default: true
    nodes: 1
    partition: ${{ variables.SLURM_PARTITION }}
    account: ${{ variables.SLURM_ACCOUNT }}
    gpus_per_node: 8

operators:
  - name: my_container
    type: srun
    container_image: nvcr.io/nvidia/pytorch:24.07-py3
    container_writable: true
    mpi: pmix

workflow:
  name: my_workflow
  tasks:
    - name: train
      operator: my_container
      script:
        - python train.py
```

## Top-Level Schema

| Section     | Required | Purpose                                           |
|-------------|----------|---------------------------------------------------|
| `version`   | Yes      | Must be `"0.1"`                                   |
| `variables` | No       | Declare parameters (dict or list form)            |
| `artifacts` | No       | Named URIs for models, configs, data              |
| `backends`  | No       | Compute providers: `local` or `slurm`             |
| `operators` | No       | How tasks run: `bash` or `srun` (with containers) |
| `workflow`  | Yes      | Workflow name, timeout, and task list              |

## Variables

Two access patterns:
- **In YAML** (resolved before execution): `${{ variables.NAME }}`
- **In scripts** (as env var at runtime): `${NAME}`

```yaml
variables:
  TP_SIZE:
    description: "Tensor parallel size"
    type: integer          # optional: integer, string (default inferred)
    value: 4
  CONCURRENCY:
    description: "Request concurrency"
    value: 16
    domain: [16, 32, 64]   # for replica sweeps
```

Expressions support Jinja2 filters and Python-like logic:
```yaml
value: "${{ variables.TP_SIZE * variables.DP_SIZE }}"
value: "${{ [variables.GPUS // variables.GPUS_PER_NODE, 1] | max }}"
value: "${{ 8180 if variables.NUM_FRONTENDS > 1 else 8000 }}"
```

Workflow-level variables override global ones. Runtime contexts available:
- `backends.<name>.nodes[i].ip_address` / `.name` / `.index` / `.num_gpus`
- `task.<name>.nodes[i].ip_address` (in scripts only)

## Artifacts

```yaml
artifacts:
  - name: LOCAL_MODEL_PATH
    uri: fs:///path/to/model     # fs:// = validated local path
  - name: CONFIG_FILE
    uri: file://config.yaml      # file:// = generated at runtime
    content: |
      key: ${{ variables.SOME_VAR }}
```

Access: `${{ artifacts.LOCAL_MODEL_PATH.path }}` in YAML, `${LOCAL_MODEL_PATH}` in scripts.

## Backends

```yaml
backends:
  - name: slurm_cluster
    type: slurm
    default: true
    nodes: ${{ variables.SLURM_NODES }}
    partition: ${{ variables.SLURM_PARTITION }}
    account: ${{ variables.SLURM_ACCOUNT }}
    time: ${{ variables.SLURM_TIMELIMIT }}
    gpus_per_node: ${{ variables.GPUS_PER_NODE }}
    extra_args:                    # optional sbatch/salloc flags
      - --exclusive
```

## Operators

```yaml
operators:
  - name: dynamo_sglang
    type: srun                     # srun (SLURM) or bash (local)
    container_image: ${{ variables.DYNAMO_IMAGE }}
    container_writable: true
    mpi: pmix
    extra_args:
      - --container-image=${{ variables.DYNAMO_IMAGE }}
```

### Container Reuse Tip

Use `--container-image` in `extra_args` combined with `container_name` on the operator to
import the image once (in a `load_image` task) and reuse it for all subsequent tasks without
re-importing. The first task pulls and names the container; later tasks reference it by name
and start instantly:

```yaml
operators:
  - name: my_runtime
    type: srun
    container_name: my_container        # reuse by name after first import
    container_writable: true
    mpi: pmix
    extra_args:
      - --container-image=${{ variables.MY_IMAGE }}  # only imports if name not found
```

The `load_image` task triggers the initial import on all nodes. All downstream tasks that
use the same operator skip the import and attach to the already-running named container,
which is significantly faster.

Task-level operator override:
```yaml
operator:
  name: dynamo_sglang
  ntasks: 4
  ntasks_per_node: 1
```

## Tasks

Every task needs `name` and `script` (list of shell commands).

```yaml
tasks:
  - name: server
    operator: my_operator
    script:
      - echo "starting"
      - python -m my_server --port 8000
    depends_on:
      - infra_task
    resources:
      gpus:
        count: 4                   # CUDA_VISIBLE_DEVICES slicing
      nodes:
        indices: [0]               # pin to specific node
        count: 2                   # or request N nodes
    replicas:
      count: ${{ variables.NUM_SERVERS }}
      policy: "parallel"           # or "sequential"
      variables:                   # sweep variables
        - CONCURRENCY
    probes:
      readiness:
        log_watch:
          regex_pattern: "ready"
        timeout: 600
        interval: 10
      failure:
        log_watch:
          regex_pattern: "Traceback (most recent call last)"
        interval: 10
    retries:
      count: 3
      interval: 30
      backoff: 2
```

### Probes

| Type | Trigger | Fields |
|------|---------|--------|
| `tcp_port` | Port accepts connections | `port`, `timeout`, `interval` |
| `log_watch` | Pattern found in log | `regex_pattern`, `timeout`, `interval`, optional `logger`, `match_count` |

`logger` references another task's log. `match_count` waits for N matches.

### Replicas

- `policy: "parallel"` -- all replicas start together; downstream waits for all
- `policy: "sequential"` -- replicas run one after another; good for sweeps
- `variables` with `domain` -- Cartesian product sweep

Each replica gets `SFLOW_REPLICA_INDEX` env var and unique GPU slice.

## Modular Composition

Split configs across files and merge with `sflow compose` or `sflow run -f`:

```bash
# Compose and inspect
sflow compose slurm_config.yaml common_workflow.yaml sglang/prefill.yaml \
  sglang/decode.yaml benchmark_aiperf.yaml --resolve -o merged.yaml

# Run directly
sflow run -f slurm_config.yaml -f common_workflow.yaml -f sglang/agg.yaml \
  -f benchmark_aiperf.yaml --missable-tasks prefill_server decode_server
```

Merge rules:
- `version` must match across files
- Variables/artifacts/backends/operators merge by name (later file wins)
- `workflow.tasks` are concatenated in file order
- `--missable-tasks` removes absent tasks from `depends_on` references

## GPU Resource Planning

Total GPUs needed = sum of all tasks' GPU counts across all replicas.
```
GPUs per worker = TP_SIZE * DP_SIZE * PP_SIZE
Total GPU slots = GPUs_per_worker * num_replicas (for each task type)
Nodes needed = ceil(total_GPU_slots / GPUS_PER_NODE)
```

Infrastructure tasks (etcd, nats, frontend) typically use `nodes.indices: [0]` with no GPU claim.

## General Rules

1. **Use `file://` artifacts for generated scripts** -- when a task needs a helper script
   (Python, shell, etc.), define it as a `file://` artifact with inline `content` rather than
   embedding it in the task script via heredocs or `python3 -c`. This avoids quoting and
   escaping issues that break when YAML -> shell -> Python nesting gets too deep:

```yaml
artifacts:
  - name: MY_SCRIPT
    uri: file://my_helper.py
    content: |
      import sys
      print(f"Hello from {sys.argv[1]}")

workflow:
  tasks:
    - name: run_helper
      script:
        - python3 ${{ artifacts.MY_SCRIPT.path }} "world"
```

2. **Never embed multi-line Python in YAML `>` or `-c`** -- shell quoting of single quotes,
   f-strings, and braces will break. Always write to a `file://` artifact or a temp file first.

## Common Pitfalls

1. **Missing `version: "0.1"`** -- every file needs it, even modular fragments
2. **Expression in wrong context** -- `${{ task.X.nodes }}` only works in scripts, not YAML fields
3. **`--set` with undeclared variable** -- variable must exist in config first
4. **Forgetting `depends_on`** -- tasks run immediately if no dependency declared
5. **GPU oversubscription** -- sum of GPU counts must not exceed `nodes * gpus_per_node`
6. **YAML multiline gotcha** -- use `>` for folded (joins lines), `|` for literal (preserves newlines)
7. **Probe on short-lived tasks** -- don't add readiness probes to tasks that exit quickly (e.g. benchmarks)
8. **Missing `--missable-tasks`** -- when composing agg-mode without prefill/decode files, mark them missable

## Validation

Run the validation script to check common issues before submitting:
```bash
python skills/writing-sflow-yaml/scripts/validate_sflow_yaml.py my_workflow.yaml
```

Or use sflow's built-in dry-run:
```bash
sflow run -f my_workflow.yaml --dry-run
```

## Additional Resources

- For detailed field reference, see [schema-reference.md](schema-reference.md)
- For annotated examples, see [examples.md](examples.md)
- If the above don't answer your question, fetch the relevant page from the online docs
  at https://nvidia.github.io/nv-sflow/docs/user/configuration -- also see `variables`,
  `probes`, `replicas`, `artifacts`, `operators`, `backends`, `resources` (same URL pattern)
