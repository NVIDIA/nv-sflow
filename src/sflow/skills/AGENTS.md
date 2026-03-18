# sflow Agent Guidelines

## Skill & Doc References

- **Writing/modifying YAML configs**: read `skills/writing-sflow-yaml/SKILL.md`, then
  `schema-reference.md` and `examples.md` as needed
- **Debugging errors**: read `skills/sflow-error-analysis/SKILL.md`, then
  `error-catalog.md` for the full error pattern catalog
- **Full docs** (when skills don't cover it): https://nvidia.github.io/nv-sflow/docs/user/intro
  Available pages: [quickstart](https://nvidia.github.io/nv-sflow/docs/user/quickstart),
  [quick-reference](https://nvidia.github.io/nv-sflow/docs/user/quick-reference),
  [configuration](https://nvidia.github.io/nv-sflow/docs/user/configuration),
  [variables](https://nvidia.github.io/nv-sflow/docs/user/variables),
  [artifacts](https://nvidia.github.io/nv-sflow/docs/user/artifacts),
  [backends](https://nvidia.github.io/nv-sflow/docs/user/backends),
  [operators](https://nvidia.github.io/nv-sflow/docs/user/operators),
  [resources](https://nvidia.github.io/nv-sflow/docs/user/resources),
  [replicas](https://nvidia.github.io/nv-sflow/docs/user/replicas),
  [probes](https://nvidia.github.io/nv-sflow/docs/user/probes),
  [outputs](https://nvidia.github.io/nv-sflow/docs/user/outputs),
  [modular-workflows](https://nvidia.github.io/nv-sflow/docs/user/modular-workflows),
  [cli](https://nvidia.github.io/nv-sflow/docs/user/cli),
  [samples](https://nvidia.github.io/nv-sflow/docs/user/samples),
  [faq](https://nvidia.github.io/nv-sflow/docs/user/faq)

## Utility Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `validate_sflow_yaml.py` | Static validation of YAML configs | `python scripts/validate_sflow_yaml.py <file>` |
| `check_gpu_plan.py` | GPU allocation plan & oversubscription check | `python scripts/check_gpu_plan.py <file> [<file>...]` |
| `parse_sflow_errors.py` | Categorize errors from logs | `python scripts/parse_sflow_errors.py <log>` |
| `summarize_run.py` | Summarize task status from output dir | `python scripts/summarize_run.py <output_dir>/<run_id>/` |

## Workflow

### Step 1: Gather Cluster Information

Before writing any config, ask the user for:
- SLURM account, partition, GPUs per node, GPU type (H100/GB200/GB300)
- Container image, model path on cluster
- Extra sbatch/srun args (`--exclusive`, `--segment`, etc.)

Do not assume defaults for these values.

### Step 2: Write a Minimal Plain-Text Config

Start hardcoded, no variables. Validate the recipe before adding abstraction.

### Step 3: Validate

```bash
python scripts/validate_sflow_yaml.py config.yaml
python scripts/check_gpu_plan.py config.yaml
sflow run -f config.yaml --dry-run
```

Fix errors before proceeding. Use the error analysis skill to diagnose.

### Step 4: Run and Debug

```bash
sflow run -f config.yaml --tui
```

If tasks fail, check `<output_dir>/<run_id>/<task>/<task>.log`. Use:
```bash
python scripts/summarize_run.py <output_dir>/<run_id>/
python scripts/parse_sflow_errors.py <output_dir>/<run_id>/sflow.log
```

### Step 5: Extract Variables

Once working, extract changeable values into `variables`. Re-run `--dry-run`.

### Step 6: Modularize

Ask about variance needs. Split into modular files if needed:
- `slurm_config.yaml`, `common_workflow.yaml`, `<framework>/*.yaml`, `benchmark.yaml`
- Use `sflow compose` or `sflow run -f` with `--missable-tasks`

## Rules

- Use `file://` artifacts with `content` for helper scripts (never heredocs/`python3 -c`)
- Use `container_name` + `--container-image` in `extra_args` for container reuse
- Add `probes.failure.log_watch` for `"Traceback (most recent call last)"` on server tasks
- Add `probes.readiness` on long-running server tasks
- Run validation and `--dry-run` before every actual run
