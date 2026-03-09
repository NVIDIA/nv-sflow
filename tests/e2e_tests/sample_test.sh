#!/bin/bash

set -x
# Create a compute node sflow env and install latest sflow
WORKSPACE_DIR=$(pwd)
srun -N 1 \
     -p your_slurm_partition \
     -A your_slurm_account \
     -t 00:10:00 \
     --job-name=sflow_runtime_venv \
     bash -c "
         set -x && \
         cd $WORKSPACE_DIR && \
         rm -rf .sflow_venv && \
         /usr/bin/python3 -m venv .sflow_venv && \
         source .sflow_venv/bin/activate && \
         pip install uv && \
         cd ../../ && \
         uv pip install -e ".[dev]"
     "

# Run sflow batch on all YAML samples in src/sflow/sample

for yaml_file in ../../src/sflow/samples/slurm_*.yaml; do
    echo "Running sflow batch on $yaml_file"
    if [[ "$yaml_file" == *replica* ]]; then
        n_val=6
    elif [[ "$yaml_file" == *infmax* ]]; then
        n_val=3
    else
        n_val=1
    fi
    yaml_basename=$(basename "$yaml_file" .yaml)
    sflow batch -f "$yaml_file" -J "$yaml_basename" -p your_slurm_partition -A your_slurm_account -N "$n_val" -o sflow.sh --sflow-venv-path "$WORKSPACE_DIR" --submit
    sleep 2
done

rm -rf sflow.sh