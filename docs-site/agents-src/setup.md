---
sidebar_position: 2
sidebar_label: Setup
title: Set up sflow agent skills
---

The skills are bundled with the `sflow` package and copied into your project with the [`sflow skill`](/docs/user/cli) command, so they always match your installed version.

## 1. List what's available

```bash
sflow skill --list
```

Lists the bundled skills (`writing-sflow-yaml`, `sflow-error-analysis`) and notes the bundled `AGENTS.md` workflow guidelines.

## 2. Copy them into your project

```bash
# Default: copy into ./skills
sflow skill

# Copy into a custom directory
sflow skill -o .cursor/skills

# Overwrite existing bundled skill files when merging
sflow skill -o .cursor/skills --force
```

`sflow skill` merges into an existing directory: by default existing files are preserved, and `--force` overwrites them.

### What gets written

```text
<output-dir>/
  AGENTS.md
  writing-sflow-yaml/
    SKILL.md
    schema-reference.md
    examples.md
    scripts/{validate_sflow_yaml.py, check_gpu_plan.py}
  sflow-error-analysis/
    SKILL.md
    error-catalog.md
    scripts/{parse_sflow_errors.py, summarize_run.py}
```

## 3. Wire it into your agent

Pick the directory your agent already reads:

| Tool | Command |
|------|---------|
| **Cursor** | `sflow skill -o .cursor/skills` |
| **Claude Code** | `sflow skill -o .claude/skills` |
| **Any agent using `AGENTS.md`** | `sflow skill -o .` then keep the generated `AGENTS.md` at your repo root |

:::tip
`AGENTS.md` is the entry point: it tells the agent which skill to read for writing YAML vs. debugging, lists the helper scripts, and encodes the hard rules (validate + `--dry-run` before every run, use `file://` artifacts for helper scripts, add failure probes on server tasks, never put credentials in YAML).
:::

## 4. Keep them current

Re-run `sflow skill -o <dir> --force` after upgrading sflow to refresh the bundled skills to the new version.

## See also

- [Agent guidelines (AGENTS.md)](./agents-guidelines.md)
- [Writing sflow YAML](./writing-sflow-yaml/index.md)
- [Error analysis](./sflow-error-analysis/index.md)
- [`sflow skill` CLI reference](/docs/user/cli)
