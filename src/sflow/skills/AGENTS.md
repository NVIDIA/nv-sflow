# sflow Agent Guidelines

Always reference the skills under `skills/` for domain knowledge:
- **Writing/modifying YAML configs**: read `skills/writing-sflow-yaml/SKILL.md` first, then
  `skills/writing-sflow-yaml/schema-reference.md` and `skills/writing-sflow-yaml/examples.md` as needed
- **Debugging errors**: read `skills/sflow-error-analysis/SKILL.md` first, then
  `skills/sflow-error-analysis/error-catalog.md` for the full error pattern catalog
- **Validating configs**: run `python skills/writing-sflow-yaml/scripts/validate_sflow_yaml.py <file>`
- **Parsing error logs**: run `python skills/sflow-error-analysis/scripts/parse_sflow_errors.py <log>`
- **Full user documentation**: if the skills don't cover your question, fetch the relevant
  page from the online docs. URL pattern: `https://nvidia.github.io/nv-sflow/docs/user/<page>`
  Example: https://nvidia.github.io/nv-sflow/docs/user/configuration
  Available pages: `intro`, `quickstart`, `quick-reference`, `configuration`, `variables`,
  `artifacts`, `backends`, `operators`, `resources`, `replicas`, `probes`, `outputs`,
  `modular-workflows`, `cli`, `samples`, `faq`

## Workflow

Follow these steps in order when helping users create or modify sflow workflows.

### Step 1: Gather Cluster Information

Before writing any config, ask the user for:
- **SLURM account** (`-A`)
- **SLURM partition** (`-p`)
- **GPUs per node** (e.g. 4, 8)
- **GPU type / architecture** (e.g. H100, GB200, GB300 -- determines container image arch)
- **Extra sbatch/srun args** (e.g. `--exclusive`, `--exclude=<node>`, `--segment`)
- **Container image** they want to use
- **Model path** on the cluster filesystem

Do not assume defaults for these values.

### Step 2: Write a Minimal Plain-Text Config

Start with a **hardcoded, minimal config** -- no variables, no expressions, no modular files.
The goal is to validate the deployment recipe itself before adding abstraction:

```yaml
version: "0.1"

backends:
  - name: slurm_cluster
    type: slurm
    default: true
    nodes: 1
    partition: actual_partition
    account: actual_account
    gpus_per_node: 4

operators:
  - name: my_runtime
    type: srun
    container_name: my_container
    container_writable: true
    mpi: pmix
    extra_args:
      - --container-image=actual_image:tag

workflow:
  name: my_workflow
  tasks:
    - name: my_task
      operator: my_runtime
      script:
        - echo "hello"
```

Keep it simple. Use literal values. Focus on getting the task DAG, probes, and scripts correct.

### Step 3: Validate with Dry-Run

Before every actual run, validate the config:

```bash
sflow run -f my_config.yaml --dry-run
```

If dry-run fails, fix the error before proceeding. Use the error analysis skill to diagnose.
Only proceed to `sflow run -f my_config.yaml --tui` after dry-run passes.

### Step 4: Run and Debug

Run the workflow and monitor for failures:

```bash
sflow run -f my_config.yaml --tui
```

If tasks fail, check the task logs at `<output_dir>/<run_id>/<task>/<task>.log`.
Common issues to watch for:
- Container image pull failures (wrong arch, `/tmp` full, registry access)
- Port conflicts between tasks
- Probe patterns that don't match actual server output
- `ldconfig` / Triton segfaults on ARM clusters (fix: `TRITON_LIBCUDA_PATH`, `--enforce-eager`)
- `min_tokens > max_tokens` conflicts in disaggregated proxy setups

Iterate until the workflow completes successfully.

### Step 5: Extract Variables

Once you have a working plain-text config, identify values that users will want to change
and extract them into `variables`:

- SLURM config: account, partition, nodes, time limit, GPUs per node
- Model config: model path, served model name
- Parallelism: TP size, DP size, PP size, number of servers
- Benchmark params: ISL, OSL, concurrency
- Container images

Replace hardcoded values with `${{ variables.NAME }}` references.
Run `sflow run -f config.yaml --dry-run` again to verify the parameterized config still works.

### Step 6: Ask About Variance and Modularize

Ask the user:
- Which variables will they change between runs?
- Do they need multiple configurations (e.g. disaggregated vs aggregated, different frameworks)?
- Do they want to run bulk sweeps from CSV?

If they need variance, consider splitting into modular files:
- `slurm_config.yaml` -- backend and SLURM variables
- `common_workflow.yaml` -- shared infrastructure tasks
- `<framework>/prefill.yaml`, `decode.yaml`, `agg.yaml` -- framework-specific tasks
- `benchmark.yaml` -- benchmark task and variables

Use `sflow compose` or `sflow run -f` with multiple files.
Use `--missable-tasks` for optional tasks (e.g. `prefill_server` when using aggregated mode).

## General Rules

- Always use `file://` artifacts with `content` for helper scripts -- never embed multi-line
  Python/shell in YAML via heredocs or `python3 -c`
- Use `container_name` + `--container-image` in `extra_args` for container reuse after initial pull
- Always add `probes.failure.log_watch` for `"Traceback (most recent call last)"` on server tasks
- Always add `probes.readiness` on long-running server tasks (use `tcp_port` or `log_watch`)
- Run the validation script before submitting: `python skills/writing-sflow-yaml/scripts/validate_sflow_yaml.py config.yaml`
- Run `sflow run --dry-run` before every actual run
