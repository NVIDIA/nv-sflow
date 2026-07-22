# sflow Agent Guidelines

> Install these skills into a coding agent's auto-discovered folder with
> `sflow skill --target all` (Claude Code → `.claude/skills`, Cursor → `.cursor/skills`,
> Codex → `.codex/skills`); use `-t claude`/`-t cursor`/`-t codex` for one, `--global` for
> the user-level dir.

## Skill & Doc References

- **Writing/modifying YAML configs**: read `skills/writing-sflow-yaml/SKILL.md`, then
  `schema-reference.md` and `examples.md` as needed
- **Debugging errors**: read `skills/sflow-error-analysis/SKILL.md`, then
  `error-catalog.md` for the full error pattern catalog
- **Reviewing sflow code changes** (for sflow developers): read
  `skills/sflow-code-review/SKILL.md`, then `reference.md` for the layer map, test
  conventions, and defect hotspots
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
  [monitor](https://nvidia.github.io/nv-sflow/docs/user/monitor),
  [uploads](https://nvidia.github.io/nv-sflow/docs/user/uploads),
  [outputs](https://nvidia.github.io/nv-sflow/docs/user/outputs),
  [results](https://nvidia.github.io/nv-sflow/docs/user/results),
  [modular-workflows](https://nvidia.github.io/nv-sflow/docs/user/modular-workflows),
  [cli](https://nvidia.github.io/nv-sflow/docs/user/cli),
  [samples](https://nvidia.github.io/nv-sflow/docs/user/samples),
  [faq](https://nvidia.github.io/nv-sflow/docs/user/faq)

## Utility Scripts

Each skill ships helpers under `<skill>/scripts/`. Two **drivers** chain the others into a
single command — reach for them first:

| Script | Purpose | Usage |
|--------|---------|-------|
| **`preflight.py`** | One-shot pre-run gate: static validation + GPU plan + `sflow --dry-run`, and auto-explains a dry-run failure | `python writing-sflow-yaml/scripts/preflight.py <file.yaml> [more.yaml ...] [sflow flags]` |
| **`triage.py`** | One-shot failure triage: run summary + orchestrator & failed-task log analysis → one consolidated root cause | `python sflow-error-analysis/scripts/triage.py <output_dir>/<run_id>/` |
| `validate_sflow_yaml.py` | Static validation of a YAML config (schema, refs, artifacts, k8s volumes) | `python writing-sflow-yaml/scripts/validate_sflow_yaml.py <file>` |
| `check_gpu_plan.py` | GPU allocation plan & oversubscription check | `python writing-sflow-yaml/scripts/check_gpu_plan.py <file> [<file>...]` |
| `parse_sflow_errors.py` | Categorize errors from a log file or stdin (`-`) | `python sflow-error-analysis/scripts/parse_sflow_errors.py <log>` |
| `summarize_run.py` | Summarize task status from an output dir | `python sflow-error-analysis/scripts/summarize_run.py <output_dir>/<run_id>/` |

Run scripts with the same Python that has sflow installed (they import nothing from sflow,
but the drivers invoke `sflow` via `python -m sflow`). Paths above are from the skills root;
drop the `<skill>/` prefix when your shell is already inside that skill.

## Workflow

### Step 1: Gather Backend & Cluster Information

First pick the backend: `local`, `slurm`, `docker`, or `kubernetes`. Then ask the user for the
backend-specific details (do not assume defaults):

- **All**: container/workload image, model path, GPUs per node, GPU type (H100/GB200/GB300).
- **slurm**: account, partition, time limit, node count; extra `salloc`/`#SBATCH` args
  (`--exclusive`, `--segment`, ...) via backend `extra_args` or `--extra-salloc-args`.
- **kubernetes**: namespace, node count, `scheduling` mode (`dra` vs `device_plugin`); cluster
  access is passed at run time via `--kubeconfig` / `--kube-context` / `--kube-namespace` (not
  YAML). The workload image goes on the `k8s` operator, not the backend.
- **docker**: image, GPU availability (NVIDIA container toolkit), and any remote `hosts:`.

### Step 2: Write a Minimal Plain-Text Config

Start hardcoded, no variables. Validate the recipe before adding abstraction.

### Step 3: Preflight (before every run)

One command runs static validation + GPU plan + `sflow --dry-run`, and explains any dry-run
failure (root cause + fix). Exit 0 means ready to run:

```bash
python writing-sflow-yaml/scripts/preflight.py config.yaml
python writing-sflow-yaml/scripts/preflight.py slurm.yaml common.yaml sglang/agg.yaml --set TP_SIZE=8
```

Fix everything it flags first. The individual steps (`validate_sflow_yaml.py`,
`check_gpu_plan.py`, `sflow run -f ... --dry-run`) still work standalone if you want just one.

### Step 4: Run and Debug

```bash
sflow run -f config.yaml --tui
```

If a task fails, triage the whole run dir in one command — task status, plus the orchestrator
(`sflow.log`) and each failed `<task>/<task>.log` categorized into a single root cause + fix:

```bash
python sflow-error-analysis/scripts/triage.py <output_dir>/<run_id>/
```

Open the `<task>/<task>.log` it points at. See the error-analysis skill for the full catalog.

### Step 5: Extract Variables

Once working, extract changeable values into `variables`. Re-run `--dry-run`.

### Step 6: Modularize

Ask about variance needs. Split into modular files if needed:
- `slurm_config.yaml`, `common_workflow.yaml`, `<framework>/*.yaml`, `benchmark.yaml`
- Use `sflow compose` or `sflow run -f` with `--missable-tasks`

## Rules

- Use `file://` artifacts with `content` for helper scripts (never heredocs/`python3 -c`)
- Use `container_name` + `--container-image` in `extra_args` for container reuse (srun/Slurm)
- Backends: `local` / `slurm` / `docker` / `kubernetes`. Default operator per backend:
  local→`bash`, slurm→`srun`, docker→`docker_run`, kubernetes→none (declare a `k8s` operator).
  For Docker, the launch operator is `type: docker_run` (the backend is `type: docker`).
- Kubernetes: the workload `image` goes on the `k8s` operator, not the backend; pass cluster
  access via `--kubeconfig` / `--kube-context` / `--kube-namespace`; `resources.gpus.count` is a
  per-task total split across nodes (must be a multiple of the node count).
- Add `probes.readiness` on long-running server tasks and `probes.failure.log_watch` for
  `"Traceback (most recent call last)"`. `log_watch` matches **literally** unless prefixed
  with `re:` / `regex:`. `readiness`/`failure` may be a list (multi-probe).
- For post-execution uploads to S3, declare a `storage:` block with `type: s3` and per-task `uploads:` specs (needs the `sflow[s3]` extra); credentials come from the boto3 default chain — never in YAML. Use `on_error: fail` only when the upload IS the deliverable.
- `--extra-args` / `-e` is backend-agnostic (Slurm/docker/kubectl), de-duped by option (CLI
  wins); use `--extra-salloc-args` / `--extra-docker-args` / `--extra-kubectl-args` to target one.
- Run validation and `--dry-run` before every actual run
