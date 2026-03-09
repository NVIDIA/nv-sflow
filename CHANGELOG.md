# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-03-09

### Added

- Initial open-source release
- Declarative YAML workflow definition with DAG support
- Pluggable backend system (local and Slurm)
- Operator framework (bash, srun, docker, ssh, python)
- Artifact management (local, HTTP, HuggingFace, Docker)
- Readiness and failure probes (TCP port, log watch, HTTP)
- Replica support with parallel and sequential policies
- Variable system with CLI overrides and domain sweeps
- GPU resource management with CUDA_VISIBLE_DEVICES slicing
- Rich TUI for interactive workflow monitoring
- `sflow run` for interactive execution
- `sflow batch` for Slurm batch job generation and submission
- `sflow visualize` for DAG visualization (Mermaid, DOT, PNG, SVG, PDF)
- `sflow sample` for listing and copying sample workflows
- Dry-run validation mode
- Comprehensive sample workflows for local, Slurm, Dynamo, and TRT-LLM setups
