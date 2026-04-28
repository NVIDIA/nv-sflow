# Contributing to sflow

Thank you for contributing to sflow. sflow is a declarative workflow descriptor that separates what to deploy from where to deploy it.

Contributors describe a workflow once in portable YAML -- tasks, dependencies, resources, launch methods, probes, artifacts, replicas, and sweeps -- and sflow executes the DAG through swappable backends. The current focus is Slurm, where sflow fills the workflow orchestration gap around `salloc`, `srun`, resource placement, and batch submission. Docker and Kubernetes backends are planned.

The repository also carries production-ready examples for NVIDIA Dynamo and LLM inference benchmarking, including modular SGLang, vLLM, and TensorRT-LLM workflows.

This guide explains how to keep changes reviewable, tested, and compatible with downstream co-development workflows.

## Contribution Scope

This project does not accept NVIDIA-external code contributions at this time. If you are an external user and have a bug report, feature request, documentation gap, or other issue that needs attention, please file an issue so maintainers can triage it.

NVIDIA-internal co-development is allowed. Internal contributors should follow the applicable internal engineering, review, and release process documentation in addition to the project-specific rules below.

## Issue Tracking

All enhancement requests, bug reports, documentation gaps, and behavior-change proposals should start with an issue or an internal tracking item.

- External users should file a GitHub issue with enough detail for maintainers to reproduce or understand the request.
- NVIDIA-internal contributors should link the relevant internal task or release tracking item when applicable.
- Feature work should be reviewed by maintainers before code review if it changes user-facing behavior, sample workflows, CLI semantics, or release behavior.
- If a change might break existing behavior, mark it clearly as a breaking change in the issue and pull request.

## Repository Layout

- `src/sflow/`: Python package source.
- `src/sflow/cli/`: CLI commands such as `run`, `batch`, `compose`, `sample`, and `visualize`.
- `src/sflow/app/`: application assembly and high-level workflow execution.
- `src/sflow/config/`: YAML loading, schema validation, and expression resolution.
- `src/sflow/core/`: core DAG, task, probe, backend, operator, artifact, and orchestration logic.
- `src/sflow/plugins/`: built-in backends, operators, probes, and artifact handlers.
- `examples/`: user-facing workflow examples used for local and Slurm regression coverage.
- `src/sflow/samples/`: packaged copies of sample workflows exposed by `sflow sample`.
- `tests/`: unit tests.
- `scripts/full_sample_tests.sh`: end-to-end and preflight regression coverage for shipped examples.
- `docs/`: user documentation and release notes.

## Development Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
```

Always activate the project virtual environment before running commands.

## Coding Guidelines

Keep changes narrowly scoped to the behavior you intend to modify. Prefer existing patterns in the surrounding code over new abstractions.

Please also:

- Avoid committing commented-out code.
- Avoid unrelated formatting churn.
- Keep pull requests focused on one concern. If several unrelated changes are needed, split them into separate pull requests and describe any dependency between them.
- Use clear commit and pull request titles. NVIDIA-internal changes should include the relevant internal tracking ID in the title when applicable.
- Target the branch requested by the relevant internal task or release process. Do not assume every fix belongs on `main`.

## Change Policy

For any feature change:

- Do not modify existing unit tests just to make the new behavior pass.
- Do not modify existing end-to-end cases in `scripts/full_sample_tests.sh` just to make the new behavior pass.
- Add new test coverage for the new behavior.
- Add or update a matching example under `examples/` so co-developed features are covered by future regression runs.
- If the example is meant to be available through `sflow sample`, keep the packaged copy under `src/sflow/samples/` in sync.
- Update user docs and release notes when the behavior is user-facing.

The only exception is an intentional breaking change. In that case, the pull request must clearly explain:

- What old behavior is being broken.
- Why compatibility is not preserved.
- Which existing tests or e2e cases were changed and why.
- How users should migrate.

## Tests and Examples

Every feature change should include focused tests near the changed behavior:

- CLI behavior: add or extend tests under `tests/unit/test_cli_*.py`.
- Config schema or resolver behavior: add or extend tests under `tests/unit/test_config_*.py`.
- Task graph, resource, replica, or probe behavior: add or extend tests under `tests/unit/test_app_assembly_*.py`, `tests/unit/test_core_*.py`, or probe-specific tests.
- Artifact behavior: add or extend tests under `tests/unit/test_artifacts_*.py`.

Add examples that exercise the feature in the same style users will copy:

- Local-only examples should be runnable without Slurm.
- Slurm examples should use variable defaults that can be overridden by `--set` or CSV columns.
- Modular examples should document required `missable_tasks` values when some tasks may be absent.
- Keep `examples/` and `src/sflow/samples/` aligned for packaged samples.

Before submitting a feature change, run the focused tests for your area and the relevant sample regression path:

```bash
pytest tests/unit/<targeted_test_file>.py
scripts/full_sample_tests.sh -P
```

For changes that affect sample workflows, also run the relevant mode:

```bash
scripts/full_sample_tests.sh -s -P  # self-contained examples
scripts/full_sample_tests.sh -m -P  # modular examples
```

Use `-S` only when you intend to submit real Slurm jobs.

## Documentation

Update documentation in the same change when behavior changes. Common locations:

- `docs/user/cli.md` for CLI flags and modes.
- `docs/user/configuration.md` and `docs/user/quick-reference.md` for YAML schema changes.
- `docs/user/resources.md`, `docs/user/probes.md`, `docs/user/variables.md`, or `docs/user/replicas.md` for feature-specific behavior.
- `docs/release_notes/` for release-facing summaries.

Do not add large generated or presentation artifacts to release notes unless they are intentionally part of the release.

## Pull Request Checklist

Before opening an NVIDIA-internal pull request:

- The issue or internal tracking item is linked.
- The change is scoped to one feature or fix.
- Existing behavior is preserved unless the PR explicitly declares a breaking change.
- New behavior has focused unit coverage.
- User-facing behavior has an example under `examples/`.
- Packaged samples under `src/sflow/samples/` are updated when applicable.
- Relevant docs and release notes are updated.
- Focused tests pass.
- Relevant `scripts/full_sample_tests.sh` preflight path passes or any skipped validation is explained.
- Performance, compatibility, or release risks are called out in the pull request description.
