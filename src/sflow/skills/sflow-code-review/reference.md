# sflow Code Review — Reference

Detailed lookups for the `sflow-code-review` skill. Read this when you need the full
layer map, test conventions, defect hotspots, or commands.

## Source layout & responsibility (`src/sflow/`)

| Package / module | Responsibility | Logic that belongs here |
|------------------|----------------|--------------------------|
| `cli/` | Typer commands: `run`, `batch`, `compose`, `sample`, `visualize`, `skill` | Argument parsing, flag wiring, delegating to `app/`. No domain logic. |
| `app/` | Application layer: `SflowApp` facade, `assembly.py` (task graph, backends, variables), resource/monitor planners, run support | Orchestration, assembly, high-level run flow. |
| `config/` | `loader.py` (YAML load/merge), `schema.py` (Pydantic), `resolver.py` (expressions) | Config parsing, schema, `${...}` resolution. Cross-cutting. |
| `core/` | DAG, task graph, orchestrator, launcher, registries, state, results, uploads, `kubectl_config.py` | Domain model and execution engine. Backend-agnostic. |
| `plugins/backends/` | `local`, `slurm`, `docker`, `kubernetes` | Backend-specific launch/allocation/flag construction. |
| `plugins/operators/` | `bash`, `python`, `srun`, `docker_run`, `ssh`, `k8s` (+ `_k8s_*` internals) | How a task is launched. |
| `plugins/probes/` | `http`, `tcp_port`, `log_watch` | Readiness/failure detection. |
| `plugins/artifacts/` | `local.py` | Artifact resolution. |
| `plugins/storage/` | S3 plugin | Upload/storage (boto3 default credential chain). |
| `monitoring/` | Hardware monitor + timeline postprocessing | Monitoring/telemetry. |
| `ui/` | `rich_tui.py` | Terminal UI. |
| `utils/` | Shared helpers (see below) | Reusable non-domain helpers used across layers. |
| `samples/` | Packaged example YAMLs (`sflow sample`) | Keep in sync with `examples/`. |
| `skills/` | Agent skills + bundled validation scripts | Authoring/validation helpers. |
| top-level | `resolution.py`, `runtime_info.py`, `logging.py`, `exceptions.py` | Cross-cutting primitives. |

Layering direction: `cli/` → `app/` → `core/` → `plugins/`. `config/` and `utils/` are
cross-cutting. A change that puts backend logic in `core/`, or domain logic in `cli/`,
violates by-purpose splitting.

## Shared utilities (`src/sflow/utils/`)

Consolidation targets — prefer extending these over re-implementing:

| Module | Purpose |
|--------|---------|
| `extra_args.py` | `extra_arg_key`, `dedup_merge_extra_args` — de-dup/merge backend `extra_args` (used by CLI `run`, `batch`, docker). The canonical "consolidated module" example. |
| `slurm.py` | Slurm messaging helpers (e.g. `gpus_per_node` warnings). |
| `container.py` | Container mount extraction/validation from extra args. |
| `gpu.py` | GPU parsing (`parse_cuda_visible_devices`). |
| `logging.py` | Dry-run log formatting helpers. |
| `parser.py` / `script.py` | General parsing / script helpers. |

## Test conventions

Pattern: `tests/unit/test_{layer}_{subject}.py`. Functions `test_*`, classes `Test*`.
Every test file carries the SPDX header.

| Source area | Test prefix | Example |
|-------------|-------------|---------|
| `cli/` | `test_cli_*` | `tests/unit/test_cli_batch.py` |
| `config/` | `test_config_*` | `tests/unit/test_config_schema.py` |
| `app/`, `core/` | `test_app_assembly_*`, `test_app_*`, `test_core_*` | `tests/unit/test_app_assembly_build_task_graph.py` |
| `plugins/backends/` | `test_plugin_backends_*` | `tests/unit/test_plugin_backends_slurm_backend.py` |
| `plugins/operators/` | `test_plugin_operators_*` | `tests/unit/test_plugin_operators_k8s.py` |
| `plugins/storage/` | `test_plugins_storage_*` | `tests/unit/test_plugins_storage_s3.py` |
| `utils/` | `test_utils_*` | `tests/unit/test_utils_container.py` |
| artifacts | `test_artifacts_*` | — |
| preflight | `test_preflight_*` | `tests/unit/test_preflight_validate_task_graph.py` |

Test tiers:

| Tier | Location | Mechanism |
|------|----------|-----------|
| Unit | `tests/unit/test_*.py` | pytest; Slurm cmds faked via `fake_process` (pytest-subprocess) in `tests/unit/conftest.py`; CLI via `CliRunner` + `patch("sflow.cli.run._sflow_app")`. |
| Integration | `tests/integration/test_*.py` | `CliRunner().invoke(app, [...])` against real `examples/` YAMLs (often `--dry-run`). |
| Shell e2e / preflight | `scripts/full_sample_tests.sh`, `tests/e2e_tests/*.sh` | Bash invoking the real `sflow` CLI. `test_full_sample_tests_script.py` lint-checks the harness text. |

CLI integration test shape (`tests/integration/`):

```python
from typer.testing import CliRunner
from sflow.cli import app

def test_run_dry_run_threads_flag(tmp_path):
    result = CliRunner().invoke(
        app,
        ["run", "-f", "examples/kubernetes_hello_world.yaml",
         "--kube-namespace", "team-ns", "--dry-run"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "team-ns" in result.output
```

## Functional defect hotspots (extended)

- CLI flag threading: parsed in `cli/` but must reach `app/sflow.py` → assembly →
  backend/operator and appear in `--dry-run` output.
- Resolver/variables: `config/resolver.py`, `resolution.py` — `${...}`, precedence,
  `--set`/CSV overrides, missing-variable handling.
- Schema compatibility: `config/schema.py` — renamed/removed Pydantic fields break old
  YAML; changed defaults change behavior silently.
- Task graph: `app/assembly.py`, `core/` — dependency edges, `missable_tasks`, replicas,
  resource placement, retry/orchestration.
- Backends: `plugins/backends/` — `salloc`/`srun`/`#SBATCH` flag strings, `extra_args`
  merge order, container mounts, GPU/node math.
- Probes: `plugins/probes/` — readiness/failure thresholds, timeouts, `log_watch` regex.
- Artifacts/storage: `file://` content vs heredocs, S3 boto3 default credential chain
  (no hard-coded creds).
- Dry-run parity: logic branching only on `dry_run` can mask real-run bugs.

### Tracing a suspected defect

1. Grep for the symptom/concept (error string, function name) across `src/` to find the
   relevant code path.
2. For a suspect symbol, list its callers and callees: `grep -rn "symbol" src tests`.
3. Follow the call chain by hand (CLI → app → core → plugins) to the failing step.
4. Check every direct caller of a changed signature/default — those are what an
   incompatible change breaks.

## Command cheat-sheet

```bash
# Setup (always activate venv first)
source .venv/bin/activate
uv pip install -e ".[dev]"          # or ".[dev,s3]" for S3 paths

# Scope the diff
git diff --stat main...HEAD
git diff main...HEAD                 # or --staged

# Focused + full unit/integration tests
pytest tests/unit/test_cli_batch.py -v
pytest tests/integration/ -v
pytest

# Coverage (matches CI on py3.12)
pytest --cov=src --cov-report=term-missing

# e2e preflight (dry-run, no Slurm submit)
MODEL_PATH=/tmp/dummy_model bash scripts/full_sample_tests.sh -P
bash scripts/full_sample_tests.sh -s -P   # self-contained examples
bash scripts/full_sample_tests.sh -m -P   # modular examples

# Plan parity for a changed example
sflow run -f examples/<changed>.yaml --dry-run --verbose

# YAML validation (after: sflow skill -o .cursor/skills)
python .cursor/skills/writing-sflow-yaml/scripts/validate_sflow_yaml.py examples/<changed>.yaml
```

CI reference (`.gitlab-ci.yml`): pytest matrix on py3.10–3.14; coverage + S3 on py3.12;
`preflight_cli` runs `scripts/full_sample_tests.sh -P`. No mypy/black/isort; Ruff is
configured but not enforced in CI.

## Change policy (verbatim intent from `CONTRIBUTING.md`)

- Do not modify existing unit tests just to make new behavior pass.
- Do not modify existing e2e cases in `scripts/full_sample_tests.sh` just to make new
  behavior pass.
- Add new test coverage for new behavior.
- Add/update an `examples/` workflow; keep `src/sflow/samples/` in sync if packaged.
- Update user docs and release notes for user-facing changes.
- Only an intentional breaking change may alter existing tests/e2e, and the PR must state:
  what breaks, why compatibility is not preserved, which tests changed and why, and how
  users migrate.
