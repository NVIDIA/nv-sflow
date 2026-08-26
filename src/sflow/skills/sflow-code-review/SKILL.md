---
name: sflow-code-review
description: >-
  Review sflow code changes for functional defects, modular by-purpose structure,
  duplicated logic that should be consolidated, adequate unit + e2e CLI test
  coverage, and over-engineering that should simply be deleted (ponytail pass).
  Use when reviewing an sflow diff, branch, PR, staged changes, or when the user
  asks for a code review of sflow.
---

# sflow Code Review

Structured review of sflow changes against five aspects: functional defects, modular
structure, duplication, test coverage, and over-engineering. Findings are advisory —
report them, do not silently rewrite the author's code unless asked.

## When to use

- The user asks to review a diff, branch, PR, staged changes, or "my sflow changes".
- Before committing or opening a PR in this repo.
- After finishing a feature/fix and you want a self-check.

## Review workflow

Copy this checklist into TodoWrite and work through it in order:

```
- [ ] Step 0: Scope the change (collect the diff + changed symbols)
- [ ] Aspect 1: Functional defects in the changed lines
- [ ] Aspect 2: Modular / by-purpose structure
- [ ] Aspect 3: Redundant logic that should be consolidated
- [ ] Aspect 4: Test coverage + test-integrity policy
- [ ] Aspect 5: Over-engineering (ponytail pass) — what can be deleted
- [ ] Verify: run focused tests / coverage / dry-run
- [ ] Report findings grouped by aspect and severity
```

Do not skip Step 0 — every later aspect needs the exact set of changed files and symbols.

## Step 0: Scope the change

Activate the venv first (project rule), then collect the diff. Pick the base ref that
matches what is being reviewed (`main`, a PR base, `--staged`, or `HEAD~1`).

```bash
source .venv/bin/activate            # Windows: .venv\Scripts\activate
git status
git diff --stat main...HEAD          # files touched
git diff main...HEAD                  # full change (or --staged for staged review)
```

Then, for each non-trivial changed function/class/method, find its callers (the blast
radius) with a quick search — e.g. `grep -rn "changed_symbol" src tests` — so you know
what an incompatible change would break. Confirm the change set only touches what the
author intended, and that every direct caller of a changed signature/default was updated.
Note the area each file belongs to (see `reference.md` layer map) — this drives Aspects 2–5.

## Aspect 1 — Functional defects

Read every changed hunk and ask "what input makes this wrong?" Prioritize the changed
lines and their direct callers (found in Step 0).

General checks:

- Edge cases: `None`/empty/missing keys, empty lists, zero/negative counts, off-by-one.
- Mutable default args, shared mutable state, accidental aliasing of dicts/lists.
- Error handling: swallowed exceptions, wrong exception type, missing `sflow`
  `exceptions.py` usage, unclear messages.
- Control flow: inverted conditions, early `return`/`continue` skipping cleanup.
- Backward compatibility: changed function signatures, renamed YAML keys, changed
  defaults — confirm all direct callers were updated.

sflow-specific defect hotspots (verify the behavior end to end, not just locally):

- **CLI flag threading** (`cli/run.py`, `cli/batch.py`): a new flag must flow CLI →
  `app/sflow.py` → assembly → backend/operator. A flag parsed but never passed down is a
  classic silent defect — confirm it reaches the backend and shows in `--dry-run`.
- **Expression/variable resolution** (`config/resolver.py`, `resolution.py`): `${...}`
  expansion, variable precedence, `--set` / CSV overrides.
- **Task graph / DAG** (`app/assembly.py`, `core/`): dependencies, `missable_tasks`,
  replicas, resource placement.
- **Backends** (`plugins/backends/`): `salloc`/`srun`/`#SBATCH` flag construction,
  `extra_args` merge/de-dup, container mounts.
- **Probes** (`plugins/probes/`): readiness/failure conditions, timeouts, log-watch regex.
- **Artifacts** (`plugins/artifacts/`): `file://` content, same-path auto-mount across backends.
- **Results & uploads** (`core/results.py`, `core/uploads.py`, `plugins/storage/`): `result:`
  parsing (regex map / `patterns` / JSON `file:` — `patterns` and `file` are mutually exclusive)
  writing `result.json` + `results.json`; `uploads:` / `upload_all` fire at task finalize
  **before** `COMPLETED`; the S3 credential chain (boto3) must never be hard-coded in YAML.
- **Hardware monitor** (`app/monitor_planner.py`, `monitoring/`): `monitor:` scopes / interval /
  report; monitors are **passive** (never reserve nodes/GPUs); backend
  `capabilities.supports_host_monitoring` gates it (kubernetes = unsupported); a new
  `--enable-workflow-monitor` / `--enable-task-monitor` path must thread into a planned monitor.
- **Dry-run vs real-run parity**: logic gated only on `dry_run` can hide real-run bugs.

See `reference.md` for the full hotspot list.

## Aspect 2 — Modular / by-purpose structure

Confirm new code lives in the layer that matches its purpose. sflow's layering:

`cli/` (thin arg parsing) → `app/` (assembly + orchestration) → `core/` (domain model)
→ `plugins/` (backends, operators, probes, artifacts, storage). `config/` is
cross-cutting; `utils/` holds reusable non-domain helpers.

Flag these structural smells:

- Domain or backend logic added inside a `cli/` command (CLI should parse + delegate).
- Backend-specific behavior (Slurm/Docker/K8s) leaking into `core/` instead of a plugin.
- A new, distinct concern bolted onto an unrelated existing file instead of its own module.
- A function/class doing several unrelated jobs — should be split by purpose.
- Helper used by multiple layers but defined privately in one of them — belongs in `utils/`.

Full responsibility table for each package: `reference.md`.

## Aspect 3 — Redundant logic to consolidate

> Sibling of Aspect 5: this one is *the same logic written twice* (unify it);
> Aspect 5 is *logic that should not exist at all* (delete it).

Look for near-duplicate snippets with the same purpose scattered across files. They
should become one dedicated module/function called the same way everywhere.

How to detect:

- Grep for repeated string literals, flag names, or parsing patterns across `cli/run.py`,
  `cli/batch.py`, and `plugins/backends/*` (common duplication sites).
- Search for similarly named helpers (`_merge_*`, `_parse_*`, `_extract_*`) doing the
  same job in different modules.
- Search the codebase for the concept (a keyword/phrase describing what the snippet does)
  to find sibling implementations.

The fix pattern (canonical example in this repo): `extra_args` merging was consolidated
into `src/sflow/utils/extra_args.py` (`extra_arg_key`, `dedup_merge_extra_args`) and is
now called uniformly by CLI `run`, `batch`, and the docker backend. New shared logic
should follow the same shape:

- Cross-cutting helper → `src/sflow/utils/<name>.py`.
- Domain logic → `src/sflow/core/`.
- Backend-/operator-specific shared code → within that `plugins/` subpackage.

Report each duplication with: the locations, the proposed home module, and which call
sites should be switched to the unified API.

## Aspect 4 — Test coverage + test integrity

This repo has a strict change policy (`CONTRIBUTING.md`). Enforce it:

> Do not modify existing unit tests or existing e2e cases in
> `scripts/full_sample_tests.sh` just to make new behavior pass. Add new coverage instead.
> The only exception is an intentional breaking change, which must document what old
> behavior breaks, why, which tests changed, and how users migrate.

Two-part check:

1. **New behavior is covered.**
   - Unit tests near the change, named by area: `cli/`→`test_cli_*`, `config/`→
     `test_config_*`, `app`+`core`→`test_app_assembly_*` / `test_core_*`,
     `plugins/backends/`→`test_plugin_backends_*`, `plugins/operators/`→
     `test_plugin_operators_*`, `utils/`→`test_utils_*`, artifacts→`test_artifacts_*`.
   - e2e/CLI coverage: an `examples/` YAML exercised by `scripts/full_sample_tests.sh -P`
     (dry-run preflight), and/or an integration test using Typer `CliRunner` against a
     real example. User-facing changes also need the `src/sflow/samples/` copy kept in sync.
   - Use coverage to find untested new lines: `pytest --cov=src --cov-report=term-missing`.

2. **Existing tests were not weakened.** In the diff, inspect every change under `tests/`
   and to `scripts/full_sample_tests.sh`. For each, decide:
   - Is it a NEW test/assertion? Good.
   - Is it an EDIT to an existing test that loosens/deletes an assertion or changes
     expected values so the new code passes? **Flag it** unless the PR explicitly declares
     an intentional behavior/breaking change with migration notes. Tests must not be bent
     to fit the implementation.

## Aspect 5 — Over-engineering (ponytail pass)

Aspects 1–4 ask "is this correct, well-placed, non-duplicated, tested?" This one asks the
cheaper question: **should this code exist at all?** Run it last — the ladder shortens the
solution, never the reading. The best diff is the one that deletes.

Walk the ladder against the change; stop at the first rung that holds:

1. **Does it need to exist?** Speculative need, a flag nobody asked for, config for a
   value that never changes → cut it, say so in one line. (YAGNI)
2. **Already in this repo?** A `utils/` helper, an existing plugin/base class, an
   established convention. Re-implementing what lives a few files over is the most common
   form of slop here — check the `utils/` table in `reference.md` before accepting a new
   helper.
3. **Stdlib does it?** `pathlib`, `dataclasses`, `itertools`, `functools.lru_cache`,
   `shlex`, `re`, `sorted(key=...)` — before any hand-rolled equivalent.
4. **Already-installed dependency?** pydantic, typer, rich, jinja2, boto3, parse are
   already here. Never add a new dependency for what a few lines can do.
5. **Can it be one line?** Then it should be.

Flag these — one line each, `<file>:<line> — cut X, use Y instead (−N lines)`:

- **Speculative abstraction**: an interface/base class with one implementation, a factory
  for one product, a registry with one entry, a hook nothing calls.
- **Dead flexibility**: a parameter every caller passes the same value for, a branch no
  config can reach, an `Optional[...]` that is never `None`.
- **Dead on arrival**: a helper the same diff adds and then makes unreachable. The tell is
  `pytest --cov=src --cov-report=term-missing` — brand-new lines at 0% coverage.
- **Reinvented stdlib / reinvented sflow**: a hand-rolled merge, sort key, or arg-dedup
  where `utils/extra_args.py` or a stdlib call already exists.
- **Boilerplate for later**: scaffolding, placeholder branches, unused config keys.
- **A second way to do an existing thing**: a new YAML key or CLI flag that overlaps one
  already in `config/schema.py`. Two spellings of one concept is a support burden forever.

**Never simplify away** (these are not findings): input validation at trust boundaries
(`config/schema.py` validators), error handling that prevents data loss, the S3 credential
chain and other security paths, backend/hardware calibration knobs a minimal model cannot
see, or anything the author was explicitly asked to build.

**Two things that look like findings but are not:**

- *Dense rationale comments.* sflow deliberately comments the non-obvious "why" at length.
  That is house style. Flag prose only when it defends complexity that should be deleted.
- *Marked shortcuts.* A `ponytail:` comment names a known ceiling and its upgrade path —
  that is a tracked decision, not debt to report. An **unmarked** corner-cut with a real
  ceiling (global lock, O(n²) scan, naive heuristic) IS a finding: ask for the marker.

Report severity by what it costs to carry, not by line count: 🟡 for an abstraction that
will shape future code (a base class, a new YAML key), 🟢 for a local simplification.

## Verify

Run the focused checks for the touched areas before concluding (venv active):

```bash
pytest tests/unit/test_<area>_*.py -v                 # focused unit tests
pytest --cov=src --cov-report=term-missing            # coverage (matches CI)
MODEL_PATH=/tmp/dummy_model bash scripts/full_sample_tests.sh -P   # e2e preflight (no Slurm)
sflow run -f examples/<changed_example>.yaml --dry-run             # plan parity
```

Report what you ran and the result. If you could not run something, say so explicitly.

## Reporting format

Group findings by aspect, ordered by severity. Use file:line references.

```
## sflow code review — <scope>

### Summary
<1-2 sentences: overall risk + headline issues>

### Aspect 1 — Functional defects
- 🔴 Critical: <file>:<line> — <issue> → <suggested fix>
- 🟡 Major: ...

### Aspect 2 — Structure
- 🟡 Major: ...

### Aspect 3 — Duplication
- 🟢 Minor: <locations> — consolidate into <module>, switch <call sites>

### Aspect 4 — Tests
- 🔴 Critical: existing test <file> weakened without declared breaking change
- 🟡 Major: new behavior in <file> lacks unit/e2e coverage

### Aspect 5 — Over-engineering
- 🟡 Major: <file>:<line> — cut <abstraction>, <what replaces it> (−N lines)
- 🟢 Minor: ...

### Verification
- pytest ...: <pass/fail>
- full_sample_tests.sh -P: <pass/fail/not run + reason>
```

Severity: 🔴 Critical (must fix — bug, broken caller, weakened test) · 🟡 Major (should
fix — missing coverage, misplaced logic, abstraction that shapes future code) ·
🟢 Minor/Nit (optional improvement, local simplification).

## Additional resources

- `reference.md` — full source-layout responsibilities, test-file map, defect hotspots,
  and the command cheat-sheet.
