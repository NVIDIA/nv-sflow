# sflow

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/NVIDIA/nv-sflow/actions/workflows/ci.yml/badge.svg)](https://github.com/NVIDIA/nv-sflow/actions/workflows/ci.yml)

A Python CLI workflow orchestrator with **pluggable backends** (e.g. local, Slurm) for running declarative YAML DAGs, collecting logs, and organizing outputs consistently.

Define _what to run_ in a `sflow.yaml` — tasks, dependencies, how to launch each task, and required resources. `sflow` executes the DAG in order, collects logs, and organizes outputs into a consistent directory structure. Example of a dynamo PD disaggregation LLM inference service workflow:

<p align="center">
  <img src="docs-site/static/img/workflow-dag.png" alt="Workflow DAG Example" width="700">
</p>

## Documentation

Full user documentation is available at: **https://nvidia.github.io/nv-sflow/**

Start here:

- [Introduction](https://nvidia.github.io/nv-sflow/docs/user/intro)
- [Quickstart](https://nvidia.github.io/nv-sflow/docs/user/quickstart)
- [Configuration](https://nvidia.github.io/nv-sflow/docs/user/configuration)
- [CLI Reference](https://nvidia.github.io/nv-sflow/docs/user/cli)
- [Sample Workflows](https://nvidia.github.io/nv-sflow/docs/user/samples)
## Quickstart

If you just want to validate the workflow engine locally (no Slurm required):

```bash
uv venv
source .venv/bin/activate
uv pip install "sflow @ git+https://github.com/NVIDIA/nv-sflow.git@main"

sflow run --file examples/local_hello_world.yaml --tui
```

The `local_hello_world.yaml` file looks like this:

```yaml
version: "0.1"

variables:
  WHO:
    description: "who to greet"
    value: Nvidia

workflow:
  name: hello_local
  tasks:
    - name: hello
      script:
        - echo "Hello ${WHO}"
```

## Development Setup

This guide will help you set up the development environment for contributing to `sflow`.

### Prerequisites

- **Python 3.10 or higher**

  ```bash
  python --version  # Check your Python version
  ```

- **uv** (Python package installer and resolver)

  If you don't have `uv` installed, you can install it using:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

  Or via pip:

  ```bash
  pip install uv
  ```

### Cloning the Repository and Installing in Development Mode

1. **Clone the repository**

   ```bash
   git clone https://github.com/NVIDIA/nv-sflow.git
   cd nv-sflow
   ```

2. **Create a virtual environment**

   ```bash
   uv venv
   ```

3. **Activate the virtual environment**

   ```bash
   source .venv/bin/activate
   ```

4. **Install the project with development dependencies**

   ```bash
   uv pip install -e ".[dev]"
   ```

   This will install:

   - The `sflow` package in editable mode
   - All runtime dependencies (typer, pydantic, pyyaml, etc.)
   - Development tools (pytest, pytest-cov, ipython, etc.)

5. **Run unit tests to validate your setup**

   ```bash
   pytest
   ```

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
