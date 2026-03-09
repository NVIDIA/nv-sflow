#!/bin/bash

#SBATCH --account=your_slurm_account
#SBATCH --partition=your_slurm_partition
#SBATCH --job-name=sflow-test
#SBATCH --nodes=2
#SBATCH --time=00:30:00
#SBATCH --output=sflow-%j.out
#SBATCH --error=sflow-%j.err


# Set up environment
export SFLOW_OUTPUT_DIR=${SLURM_SUBMIT_DIR}/sflow_output
mkdir -p ${SFLOW_OUTPUT_DIR}

# Change to the directory where the job was submitted
cd ${SLURM_SUBMIT_DIR}

# Run sflow - it will detect the existing Slurm allocation
source .venv/bin/activate
python -m sflow run --file sflow.yaml