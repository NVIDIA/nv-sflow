# sflow v0.2.1 Release Notes

**Release date:** April 2026
**Previous release:** v0.2.0 (March 2026)

---

## Highlights

sflow v0.2.1 is a documentation and workflow polish release for the InfMax v3 migration path. It documents the branch behavior for CSV-driven execution, self-contained YAML batch submission, replica variable domains, node placement, and probe orchestration.

---

## User-Facing Changes

### CLI and Batch Workflows

- **`sflow run --bulk-input`** now has documented single-row CSV execution. Use `--row` with exactly one selector to run a specific CSV row.
- **Advanced `--row` selectors** are documented for `run`, `compose`, and `batch`: repeated flags, comma lists, Python-style slices with exclusive end, open-ended slices, and negative indices such as `--row=-1`.
- **`sflow batch --bulk-submit`** is documented for submitting self-contained YAML files, folders, or glob patterns without CSV merging.
- **Auto-derived node counts** are documented. Single-job and bulk-submit batch modes can derive `--nodes` from the Slurm backend; bulk-input mode requires either `--nodes` or a CSV node-count column.
- **`--sflow-version`** is documented for pinning the git ref installed by generated sbatch scripts.
- **Expression-aware `--sbatch-extra-args`** is documented. Extra sbatch directives can resolve `${{ variables.X }}` or shorthand `${{ X }}` from config defaults, CLI `--set`, and CSV row values.

### Variables and Replica Sweeps

- **Variable domain metadata** is documented through `${{ variables.NAME.domain }}`.
- **Replica sweep behavior** is clarified: `${{ variables.NAME }}` resolves to the per-replica value, while `${{ variables.NAME.domain }}` remains the full domain list.
- **Domain overrides via `--set`** are documented: JSON-style list values update the variable `domain`, and the variable value becomes the first list item.

### Resources and Placement

- **`resources.nodes.exclude`** is documented for removing nodes from the placement pool before applying `indices`, `count`, or GPU packing.
- **Negative node indices** are clarified, including the fact that negative `indices` are resolved after `exclude` filtering.
- **Default Slurm placement** is documented: when a task does not set `resources.nodes`, sflow passes the full backend allocation to `srun`.
- **GPU packing behavior** is documented, including multi-node expansion when a GPU request is an exact multiple of `gpus_per_node`.

### Probes

- **Probe timing defaults** are documented, including `timeout: 1200` for readiness probes and `each_check_timeout: 30`.
- **HTTP probes** (`http_get` and `http_post`) are documented with examples.
- **Multiple readiness probes** are documented as AND semantics: all readiness probes must trigger before a task becomes ready.
- **Failure probes** are documented as fail-fast signals that mark tasks as failed by probe and cancel downstream work.
- **Replica HTTP probe deduplication** is documented for parallel replicas with identical HTTP probes.

---

## Documentation Updated

- `docs/user/cli.md`
- `docs/user/variables.md`
- `docs/user/resources.md`
- `docs/user/probes.md`
- `docs/user/quick-reference.md`
- `docs/user/configuration.md`
- `docs/user/architecture.md`
