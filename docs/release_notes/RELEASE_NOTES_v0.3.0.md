# sflow v0.3.0 Release Notes

**Release date:** June 2026
**Previous release:** v0.2.2 (May 2026)

---

## Highlights

sflow v0.3.0 refines how CLI-provided backend extra args are routed. `sflow run` now has
backend-specific extra-arg flags in addition to the generic `--extra-args`, and all of
them de-duplicate by option (CLI wins) so a CLI value cleanly overrides a recipe default.

---

## User-Facing Changes

### CLI extra args

- **`--extra-args, -e` is now backend-agnostic.** Its values are forwarded to whichever
  backend the recipe uses: merged into each Slurm backend's `salloc`, each docker
  backend's `docker run`, and every `kubectl` call's global flags. Whichever backend the
  recipe contains picks the args up.
- **New backend-specific flags** for when you want to target one backend kind only:
  - `--extra-salloc-args` — Slurm `salloc` only (e.g. `--gpus-per-node=4`).
  - `--extra-docker-args` — docker `docker run` only (e.g. `--shm-size=16g`).
  - `--extra-kubectl-args` — kubectl global flags only (e.g. `--request-timeout=30s`).
- **De-dup by option (CLI wins).** CLI extra args now override a recipe backend's
  `extra_args` on a conflicting option instead of both being passed (e.g. CLI
  `--gres=gpu:4` overrides a recipe `--gres=gpu:8`). Repeatable `key=value` flags such as
  `--env=FOO=1` / `--env=BAR=2` are preserved as distinct entries. A more specific
  `--extra-<type>-args` wins over the generic `--extra-args` on a conflicting option.

### Breaking changes and migration

- **`--kubectl-arg` was renamed to `--extra-kubectl-args`.** Update any scripts:
  - `sflow run --kubectl-arg=--request-timeout=30s` → `sflow run --extra-kubectl-args=--request-timeout=30s`
  - or use the generic form: `sflow run --extra-args=--request-timeout=30s`.
- `--extra-args` itself is unchanged in spelling and remains backward compatible; it is now
  generic (applies to Slurm, docker, and kubectl) rather than Slurm-only.

---

## Documentation Updated

- `docs/user/cli.md`
- `docs/user/backends.md`
- `docs/user/architecture.md`
