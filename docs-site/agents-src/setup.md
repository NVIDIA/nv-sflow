---
sidebar_position: 2
sidebar_label: Setup (agent runbook)
title: Set up sflow with your AI agent
---

This page is a runbook written for AI coding agents (Claude Code, Codex, Cursor, Copilot, Gemini CLI, ...). A person shares one link, and the agent installs sflow, checks that it runs, and installs the sflow skills into the harness it is running in.

**For people:** paste this into your agent:

```text
Set up sflow for me by following
https://raw.githubusercontent.com/NVIDIA/nv-sflow/main/docs-site/agents-src/setup.md
step by step, then tell me what you installed and where.
```

The same page, rendered: https://nvidia.github.io/nv-sflow/docs/agents/setup

---

## For the agent

Work through the steps in order. Run every command yourself, check its output, and do not move on after a failure. Ask the user only where a step says to. Setup never submits cluster jobs; the smoke test in step 3 runs locally.

References, if you need more than this page:

- Repository: https://github.com/NVIDIA/nv-sflow
- Docs: https://nvidia.github.io/nv-sflow/ (index for agents: https://nvidia.github.io/nv-sflow/llms.txt)
- CLI reference: https://nvidia.github.io/nv-sflow/docs/user/cli
- Backends: https://nvidia.github.io/nv-sflow/docs/user/backends

### 1. Pick the machine

Install sflow where workflows will be launched from: the user's workstation, a Slurm login node, a Docker GPU host, or any machine with `kubectl` access to the cluster. A workstation can also drive a Slurm login node over SSH (`sflow run ... --ssh user@login`), so installing locally is a fine default.

```bash
uname -s             # Linux or Darwin
python3 --version    # 3.10 or newer
```

### 2. Install sflow

sflow is published on PyPI as **`nv-sflow`**; the command it installs is `sflow`. Never install the PyPI package named `sflow`: it is an unrelated project.

First check whether it is already there:

```bash
sflow -V
```

- `command not found`: install it (below).
- `install : installed` is a release build: skip the install and go to step 3. Upgrading replaces the version the user's existing workflows run on, so ask before running `sflow upgrade`; if you cannot ask, leave it and mention the version in the report. On a version before 0.3.2 the first `sflow upgrade` reports landing on the GitHub build; run it once more to reach the PyPI release.
- `install : editable` or `source-tree` is someone's development checkout: leave it alone and ask the user whether to install a release build that replaces it on `PATH`.

Install it as a standalone tool with [uv](https://docs.astral.sh/uv/), so `sflow` is on `PATH` in every directory:

```bash
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh   # no curl? pip install --user uv
uv tool install nv-sflow
uv tool update-shell   # only if `sflow` is not on PATH afterwards; then open a new shell
```

If `uv tool install` stops with `Executable already exists: sflow`, an older `sflow` that uv did not install owns that name, and nothing was installed. Show the user the `sflow -V` output and, once they agree, rerun it as `uv tool install --force nv-sflow`.

If the user wants sflow inside a project virtualenv instead, use `uv venv && source .venv/bin/activate && uv pip install nv-sflow`. Optional extras: `nv-sflow[s3]` (S3 uploads), `nv-sflow[monitor]` (PNG hardware charts).

### 3. Verify

```bash
sflow -V
mkdir -p /tmp/sflow-smoke
sflow sample self_contained/local/hello_world -o /tmp/sflow-smoke/hello.yaml --force
sflow run -f /tmp/sflow-smoke/hello.yaml --output-dir /tmp/sflow-smoke/out
```

The run must exit 0 and its output must contain `Workflow completed`. It runs on the local backend, so it needs no GPU and no cluster. Stay in the user's project directory: step 4 installs project-scoped skills relative to it.

### 4. Install the skills for your harness

The skills teach you to write sflow YAML, validate and dry-run a workflow, and debug failed runs. They ship inside the installed package, so they always match the sflow version. Install into the folder your harness discovers skills from:

| You are running in | Command | Installs to |
|---|---|---|
| Claude Code | `sflow skill --target claude --global` | `~/.claude/skills/` |
| Codex CLI | `sflow skill --target codex --global` | `~/.codex/skills/` |
| Cursor | `sflow skill --target cursor` (run from the project root) | `.cursor/skills/` |
| Several of the above | `sflow skill --target all` (run from the project root) | `.claude/`, `.cursor/` and `.codex/skills/` |

`--global` makes the skills available in every project. Drop it to install into the current project only, for example so the skills can be committed for a team. Cursor has no user-level skills folder, so its skills always go into the project.

**Any other harness** (GitHub Copilot, Gemini CLI, Windsurf, OpenCode, ...):

- If it discovers `SKILL.md` skill folders, install there: `sflow skill -o PATH_TO_ITS_SKILLS_DIR --yes`.
- Otherwise install next to the project with `sflow skill -o .sflow-skills --yes`. Then add this line to the harness's instruction file (`AGENTS.md`, `GEMINI.md`, `.github/copilot-instructions.md`, ...): "For sflow work, read `.sflow-skills/AGENTS.md` first."

Re-running is safe: the copy merges, and existing files are kept unless `--force` is given. The command prints the folder it installed into (`✓ claude → …/skills`); that folder must now contain `writing-sflow-yaml/SKILL.md`. It also contains `AGENTS.md` and the other bundled skills. Each skill is self-contained, so a harness that discovers skill folders needs no extra wiring for `AGENTS.md`; it is only an index.

If that line ends in `(merged, existing preserved)`, the folder already held sflow skills and any older copies were kept as they were. Run the same command again with `--force` to bring them up to this sflow version. `--force` overwrites only sflow's own skill files; other skills in the folder are untouched.

### 5. Check the user's backend (ask first)

Ask the user which backend they will run on, then check only that one. If you cannot ask (a one-shot or non-interactive run), check only `local` and say so in the report:

| Backend | Check | If it fails |
|---|---|---|
| local | nothing to check | |
| Docker | `docker info`; for GPUs, `nvidia-smi` on the host | install Docker plus the NVIDIA Container Toolkit |
| Slurm | `sinfo -s` on the login node, or `ssh user@login sinfo -s` | get cluster access; from a workstation, add `--ssh user@login` to `sflow run` |
| Kubernetes | `kubectl config current-context` and `kubectl auth can-i create pods -n NAMESPACE` | get a kubeconfig with pod rights in the namespace |

Report what failed and how to fix it, but do not change cluster or system configuration on your own.

### 6. Report to the user

Tell the user:

- the sflow version and where it is installed (`sflow -V` shows both);
- which skills folder you installed into;
- the result of the smoke test and of the backend check;
- that a new agent session may be needed before the skills are picked up;
- how to start: `sflow sample --list` lists ready-made workflows, and the quickstart is https://nvidia.github.io/nv-sflow/docs/user/quickstart.

## Keeping it current

```bash
sflow upgrade                                  # newest PyPI release (add --dry-run to preview)
sflow skill --target claude --global --force   # same target as step 4, refreshed to the new version
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `uv tool install` fails with `Executable already exists: sflow` | an older `sflow` script is in uv's bin dir; with the user's agreement, `uv tool install --force nv-sflow` |
| `sflow: command not found` after `uv tool install` | `uv tool update-shell`, then open a new shell, or add uv's bin dir (`uv tool dir --bin`) to `PATH` |
| `sflow --version` shows an old version, or a `bin` path other than the one just installed | an older `sflow` is earlier on `PATH`; find it with `which -a sflow` and remove it, or run `sflow upgrade` |
| the agent does not use the skills | start a new session; confirm the folder from step 4 contains `writing-sflow-yaml/SKILL.md` |
| no `curl` and no `pip` on a locked-down host | install sflow on a workstation instead and use `sflow run --ssh user@login` |
