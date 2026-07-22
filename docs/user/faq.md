---
title: FAQ
sidebar_position: 12
---

## Why don’t `--resume` / `--task` work?

Because the current CLI/implementation explicitly marks them as “not implemented yet”; calling them errors immediately:

- `sflow run --resume ...`: `NotImplementedError`
- `sflow run --task ...`: `typer.BadParameter`

This guide treats them as planned features, not currently supported behavior.

## Why does `--set` say the variable doesn't exist?

Today, `--set KEY=VALUE` **only overrides variables already declared in the config file**. If you want a new variable, you must declare it under `variables:` first (even with a default value).

## Does dry-run create output directories?

No. Dry-run computes planned output paths but does not `mkdir` or write log files.

## Do artifacts automatically download from HuggingFace / pull Docker images?

In v0.1, only `fs://` / `file://` are resolved to real local paths. Other schemes are kept as strings (no download/pull behavior yet).

## Visualization to `png/svg/pdf` fails: `dot` not found?

That output format depends on Graphviz. You can:

- Install Graphviz (provides `dot`)
- Or use `--format mermaid` / `--format dot`

## How do I check which sflow version I'm running?

Run `sflow --version` (or `sflow -V`). It prints the package version plus runtime/build details — binary path, Python path, install mode, source label, repo path, and git branch/commit when available. Each `sflow run` logs the same details, which helps confirm whether a local editable checkout, a branch build, or a released package is running.

## How do I restrict a run to (or away from) specific nodes?

Use `--include-nodes` to pin a run to a set of hosts, or `--exclude-nodes` to keep it off flaky ones. Both work on `sflow run` and `sflow batch`, accept comma-separated and/or repeated values, and apply to whatever backend the recipe uses. You can also set `include_nodes` / `exclude_nodes` on a backend in YAML; CLI values are unioned with the recipe's. See [Backends](./backends.md#selecting-or-excluding-nodes-all-backends).

## How do I pass extra flags to the backend (salloc / docker run / kubectl)?

Use the generic `--extra-args` / `-e` to forward a flag to whichever backend the recipe uses — it is merged into each Slurm backend's `salloc`, each docker backend's `docker run`, and every `kubectl` call. For backend-specific flags use `--extra-salloc-args`, `--extra-docker-args`, or `--extra-kubectl-args`; the more specific form wins on a conflicting option, and CLI values win over the recipe.

## How do I connect to a Kubernetes cluster?

Kube access is provided by CLI flags, so the recipe stays cluster-agnostic: `--kubeconfig`, `--kube-context`, and `--kube-namespace` select the cluster and namespace, while `--kube-node-selector`, `--kube-compute-domain-channel`, and `--kube-compute-domain-create` / `--no-kube-compute-domain-create` tune placement and Multi-Node NVLink. Credentials always live in the kubeconfig, never in YAML. See [Backends](./backends.md#connecting-to-the-cluster).

## How do I install sflow from a private PyPI index in batch mode?

Pass `--sflow-index-url <url>` to `sflow batch` (e.g. an Artifactory registry). When it's set, `--sflow-version` is interpreted as a PyPI version specifier: a bare version is pinned (`0.3.0` → `sflow==0.3.0`), an operator spec (`>=0.3,<0.4`) is passed through, and omitting it installs the latest available. Credentials must be available on the compute node via `~/.netrc` or a credential helper (URLs with embedded credentials are rejected).

## What does `--sflow-venv-path` do in `sflow batch`?

It's the *parent* directory under which each Slurm job builds its own fresh, disposable per-job venv (`.sflow_venv-<job id>/`), installs sflow into it, and removes it when the job exits. The venv is built on the compute node with the node's `python3`, so it always matches the node architecture (x86/arm). It defaults to compute-node-local scratch; pass a shared-filesystem path to override.

## Which optional extras can I install?

- `pip install 'sflow[s3]'` — S3 artifact/result uploads (credentials from the boto3 default chain).
- `pip install 'sflow[monitor]'` — PNG charts for the hardware monitor.

## How do I get the AI-agent skills for writing sflow YAML?

Run `sflow skill` to copy the bundled agent skills into your project (`sflow skill --list` to see them, `-o <dir>` to choose the output directory). These teach coding agents how to author sflow YAML and diagnose failed runs.
