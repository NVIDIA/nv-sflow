---
name: writing-sflow-yaml
description: >-
  Write, create, and modify sflow YAML workflow configuration files. Covers schema structure,
  variable declarations, task DAGs, backends, operators, probes, replicas, artifacts, and
  modular composition. Use when the user asks to create an sflow YAML, configure a workflow,
  set up inference serving, or asks about sflow YAML syntax.
---

# Writing sflow YAML Configurations

## Minimal Examples

**Local:**

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

**SLURM with container:**

```yaml
version: "0.1"
variables:
  SLURM_ACCOUNT: { value: my_account }
  SLURM_PARTITION: { value: my_partition }
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

## Key Concepts

| Section     | Required | Purpose                                    |
|-------------|----------|--------------------------------------------|
| `version`   | Yes      | Must be `"0.1"`                            |
| `variables` | No       | Parameters: `${{ variables.X }}` / `${X}`  |
| `artifacts` | No       | Named URIs (`fs://`, `file://`)            |
| `backends`  | No       | Compute: `local` or `slurm`               |
| `operators` | No       | Execution: `bash` or `srun` (containers)  |
| `workflow`  | Yes      | Name, timeout, and task list               |

For complete field reference, see [schema-reference.md](schema-reference.md).
For full docs: [configuration](https://nvidia.github.io/nv-sflow/docs/user/configuration),
[variables](https://nvidia.github.io/nv-sflow/docs/user/variables),
[artifacts](https://nvidia.github.io/nv-sflow/docs/user/artifacts),
[backends](https://nvidia.github.io/nv-sflow/docs/user/backends),
[operators](https://nvidia.github.io/nv-sflow/docs/user/operators),
[probes](https://nvidia.github.io/nv-sflow/docs/user/probes),
[replicas](https://nvidia.github.io/nv-sflow/docs/user/replicas),
[resources](https://nvidia.github.io/nv-sflow/docs/user/resources),
[modular-workflows](https://nvidia.github.io/nv-sflow/docs/user/modular-workflows).

## Essential Patterns

### Variables: Two Access Modes

- **YAML** (plan-time): `${{ variables.NAME }}`
- **Scripts** (runtime env var): `${NAME}`

Expressions support Jinja2 filters: `${{ variables.TP_SIZE * variables.DP_SIZE }}`,
`${{ [variables.X, 1] | max }}`, `${{ 8180 if variables.Y > 1 else 8000 }}`.

When using arithmetic or comparisons in expressions, set the variable `type` correctly
(`integer`, `string`, etc.) -- otherwise values default to strings and operations like
`*` or `>` will produce unexpected results.

### file:// Artifacts for Helper Scripts

Always use `file://` artifacts with `content` for multi-line scripts. Never embed Python
in YAML via heredocs or `python3 -c` -- quoting breaks when YAML -> shell -> Python nests.

```yaml
artifacts:
  - name: MY_SCRIPT
    uri: file://helper.py
    content: |
      import sys
      print(f"Hello from {sys.argv[1]}")
workflow:
  tasks:
    - name: run_helper
      script:
        - python3 ${{ artifacts.MY_SCRIPT.path }} "world"
```

### Container Reuse

Use `container_name` + `--container-image` in `extra_args`. The first task pulls the image;
subsequent tasks attach by name instantly:

```yaml
operators:
  - name: my_runtime
    type: srun
    container_name: my_container
    container_writable: true
    mpi: pmix
    extra_args:
      - --container-image=${{ variables.MY_IMAGE }}
```

### Probes

| Type       | Use For              | Key Fields                            |
|------------|----------------------|---------------------------------------|
| `tcp_port` | Infra services       | `port`, `timeout`, `interval`         |
| `log_watch`| Server readiness     | `regex_pattern`, `timeout`, `interval`|

Always add `failure.log_watch` for `"Traceback (most recent call last)"` on server tasks.

### Replicas & Sweeps

- `policy: "parallel"` -- all at once; downstream waits for all
- `policy: "sequential"` -- one after another; good for sweeps
- `variables` with `domain` -- Cartesian product sweep

### Modular Composition

```bash
sflow run -f slurm.yaml -f common.yaml -f sglang/agg.yaml -f bench.yaml \
  --missable-tasks prefill_server --missable-tasks decode_server
```

Merge rules: version must match, named items merge (later wins), tasks concatenate.

### GPU Resource Planning

```
GPUs per worker = TP * DP * PP  (common default, but varies by framework)
Total GPU slots = GPUs_per_worker * replicas (per task)
Nodes needed    = ceil(total_slots / gpus_per_node)
```

The `TP * DP * PP` formula is a common pattern but not universal -- some frameworks
calculate GPU requirements differently (e.g., attention DP, expert parallelism). When
unsure, ask the user or check the framework's documentation for the correct GPU count.

## Common Pitfalls

1. Missing `version: "0.1"` in every file (including fragments)
2. Using `${{ task.X.nodes }}` in YAML fields (only works in scripts)
3. `--set` with undeclared variable (must exist in config first)
4. Forgetting `depends_on` (tasks run immediately without it)
5. GPU oversubscription (sum of GPU counts > nodes * gpus_per_node)
6. Probe on short-lived tasks (don't probe benchmarks)
7. Missing `--missable-tasks` when composing without all task files

## Validation

```bash
python scripts/validate_sflow_yaml.py my_workflow.yaml
python scripts/check_gpu_plan.py my_workflow.yaml
sflow run -f my_workflow.yaml --dry-run
```

## Additional Resources

- Field reference: [schema-reference.md](schema-reference.md)
- Annotated examples: [examples.md](examples.md)
- Full docs: https://nvidia.github.io/nv-sflow/docs/user/intro
