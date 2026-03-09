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
