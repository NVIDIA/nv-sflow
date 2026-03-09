# sflow

A Python CLI workflow orchestrator with **pluggable backends** (e.g. local, Slurm) for running declarative YAML DAGs, collecting logs, and organizing outputs consistently.

## Introduction

Start here:

- **User docs**: `docs/user/intro.md`
## Quickstart

If you just want to validate the workflow engine locally (no Slurm required):

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

sflow run --file examples/hello_local.yaml --tui
```

The `hello_local.yaml` file looks like this:

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
      operator: local_bash
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

### Getting Started

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd sflow
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
