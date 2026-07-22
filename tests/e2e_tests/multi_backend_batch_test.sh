#!/bin/bash

set -euo pipefail

usage() {
    echo "Usage: $0 -A <account> -a <partition_a> -b <partition_b> -m <model_path> [options]"
    echo ""
    echo "Submit the multi-backend Slurm sample via 'sflow batch' and verify the"
    echo "per-backend-salloc fix: each backend gets its OWN Slurm job (the leader"
    echo "reuses the driver sbatch; the other backend sallocs its own), so Pyxis/enroot"
    echo "containers start on BOTH partitions and task_a/task_b land on distinct nodes."
    echo "Run it WITH container images (the default) to exercise the path that was broken."
    echo ""
    echo "Required unless --check is used:"
    echo "  -A, --account       Slurm account"
    echo "  -a, --partition-a   Slurm partition for cluster_a (the driver/leader)"
    echo "  -b, --partition-b   Slurm partition for cluster_b (sallocated at runtime)"
    echo "  -m, --model-path    Local model path artifact value"
    echo ""
    echo "Options:"
    echo "  --check JOB_ID      Skip submit and check an existing driver job id"
    echo "  --image-a IMAGE     Container image for task_a on cluster_a"
    echo "                      (default: python:3.12-slim; empty = use sample default)"
    echo "  --image-b IMAGE     Container image for task_b on cluster_b"
    echo "                      (default: python:3.11-slim; empty = use sample default)"
    echo "  -t, --time MINUTES  Slurm time limit in minutes (default: 10)"
    echo "  -o, --output-dir DIR  sflow output directory (default: <repo>/sflow_output)"
    echo "  -e, --extra ARG     Extra sflow batch/sbatch arg, repeatable (example: -e '--exclusive')"
    echo "  -h, --help          Show this help"
    echo ""
    echo "Examples:"
    echo "  # Verify containers work on both partitions (the fix):"
    echo "  $0 -A sflow-gitlab-ci -a genesisq -b gamoraq -m /path/to/model"
    echo "  # Re-check an already-submitted run by its driver job id:"
    echo "  $0 --check JOB_ID"
    exit "${1:-1}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$REPO_DIR/examples/self_contained/slurm/multi_backend.yaml"
OUTPUT_DIR="$REPO_DIR/sflow_output"
ACCOUNT=""
PARTITION_A=""
PARTITION_B=""
MODEL_PATH=""
TIME_LIMIT="10"
# Default to container images so a bare run exercises Pyxis/enroot on BOTH
# partitions (the path the per-backend-salloc change fixes). Matches the sample's
# defaults. Override with --image-a/--image-b or MULTI_BACKEND_IMAGE_A/B.
IMAGE_A="${MULTI_BACKEND_IMAGE_A:-python:3.12-slim}"
IMAGE_B="${MULTI_BACKEND_IMAGE_B:-python:3.11-slim}"
CHECK_JOB_ID=""
EXTRA_BATCH_ARGS=()

abs_path() {
    local path="$1"
    if [ -d "$path" ]; then
        (cd "$path" && pwd)
        return
    fi
    case "$path" in
        /*) printf '%s\n' "$path" ;;
        *) printf '%s/%s\n' "$PWD" "$path" ;;
    esac
}

workflow_output_dir_for_job() {
    local jid="$1"
    local out_dir
    out_dir=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "${jid}-*" 2>/dev/null | head -1)
    if [ -n "$out_dir" ]; then
        abs_path "$out_dir"
    fi
}

# Each task echoes "task_<x> backend=cluster_<x> job=<job id> nodes=<assigned nodes>"
# from inside its container; grep it out of the workflow output dir.
task_line() {
    local dir="$1" task="$2"
    grep -rhoE "${task} backend=cluster_${task#task_} job=[0-9]+ nodes=[^[:space:]]+" \
        "$dir" 2>/dev/null | head -1 || true
}

multi_backend_run_ok() {
    # Correct when BOTH tasks ran their container (each echoed its line) and
    # landed on DIFFERENT nodes (distinct per-backend allocations => the leader
    # reused the driver job and cluster_b sallocated its own on the other
    # partition). A Pyxis/enroot failure on either partition leaves that task's
    # log without the echo, so its node is missing and this returns non-zero.
    local dir="$1"
    [ -n "$dir" ] || return 1
    local a_nodes b_nodes
    a_nodes=$(task_line "$dir" task_a | sed -E 's/.*nodes=//')
    b_nodes=$(task_line "$dir" task_b | sed -E 's/.*nodes=//')
    [ -n "$a_nodes" ] && [ -n "$b_nodes" ] && [ "$a_nodes" != "$b_nodes" ]
}

print_task_evidence() {
    local dir="$1"
    local a_line b_line a_job b_job
    a_line=$(task_line "$dir" task_a)
    b_line=$(task_line "$dir" task_b)
    echo "  task_a (cluster_a): ${a_line:-<no output - container may have failed>}"
    echo "  task_b (cluster_b): ${b_line:-<no output - container may have failed>}"
    a_job=$(printf '%s' "$a_line" | sed -nE 's/.*job=([0-9]+).*/\1/p')
    b_job=$(printf '%s' "$b_line" | sed -nE 's/.*job=([0-9]+).*/\1/p')
    if [ -n "$a_job" ] && [ -n "$b_job" ]; then
        if [ "$a_job" != "$b_job" ]; then
            echo "  -> distinct Slurm job ids (task_a=$a_job, task_b=$b_job): separate per-backend allocations, as expected."
        else
            echo "  -> WARNING: task_a and task_b share job id $a_job (expected separate per-backend allocations)."
        fi
    fi
}

scan_pyxis_enroot_error() {
    # Surface the exact failure this change fixes, if it is still present.
    local dir="$1" hit
    hit=$(grep -rhiE 'enroot|pyxis|spank' "$dir" 2>/dev/null \
        | grep -iE 'permission denied|could ?n.t start container|failed to import|/run/enroot' \
        | head -8 || true)
    if [ -n "$hit" ]; then
        echo ""
        echo "  Detected Pyxis/enroot error (the failure this change fixes):"
        printf '%s\n' "$hit" | sed 's/^/    /'
    fi
}

job_state() {
    local jid="$1"
    local state
    state=$(sacct -j "$jid" --noheader -o State -X 2>/dev/null | head -1 | tr -d ' ' || true)
    if [ -n "$state" ]; then
        printf '%s\n' "$state"
        return
    fi
    state=$(squeue -j "$jid" -h -o "%T" 2>/dev/null | head -1 | tr -d ' ' || true)
    printf '%s\n' "${state:-UNKNOWN}"
}

wait_for_job() {
    local jid="$1"
    local poll_seconds="${POLL_SECONDS:-30}"
    local timeout_seconds="${TIMEOUT_SECONDS:-3600}"
    local start now state out_dir
    start=$(date +%s)

    echo "Waiting for Slurm job $jid (timeout=${timeout_seconds}s, poll=${poll_seconds}s)..."
    while true; do
        state=$(job_state "$jid")
        out_dir=$(workflow_output_dir_for_job "$jid")
        if [ -n "$out_dir" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') $jid state=${state:-UNKNOWN} output=$out_dir"
        else
            echo "$(date '+%Y-%m-%d %H:%M:%S') $jid state=${state:-UNKNOWN} output=(not found yet)"
        fi

        case "$state" in
            PENDING|RUNNING|CONFIGURING|COMPLETING|UNKNOWN|"")
                ;;
            *)
                return 0
                ;;
        esac

        now=$(date +%s)
        if [ $((now - start)) -ge "$timeout_seconds" ]; then
            echo "ERROR: timed out waiting for job $jid"
            return 1
        fi
        sleep "$poll_seconds"
    done
}

check_job() {
    local jid="$1"
    local out_dir status

    wait_for_job "$jid"
    out_dir=$(workflow_output_dir_for_job "$jid")
    if [ -z "$out_dir" ]; then
        echo "FAIL: output folder not found under $(abs_path "$OUTPUT_DIR")"
        return 1
    fi

    status=$(grep -m1 -E '^Status' "$out_dir/sflow_summary.log" 2>/dev/null \
        | sed -E 's/^Status[[:space:]]*:[[:space:]]*//' || true)

    echo ""
    echo "===== Result for driver job $jid ====="
    echo "Output          : $out_dir"
    echo "Workflow status : ${status:-UNKNOWN}"
    print_task_evidence "$out_dir"

    if multi_backend_run_ok "$out_dir"; then
        echo ""
        echo "PASS: both backends ran their container on distinct nodes via separate"
        echo "      per-backend allocations (cluster_a reused the driver job; cluster_b"
        echo "      sallocated its own). The Pyxis/enroot multi-backend issue is fixed."
        return 0
    fi

    echo ""
    echo "FAIL: task_a and/or task_b did not run on distinct backend nodes."
    scan_pyxis_enroot_error "$out_dir"
    if [ -f "$out_dir/sflow_summary.log" ]; then
        echo ""
        echo "===== sflow_summary.log ====="
        sed -n '1,120p' "$out_dir/sflow_summary.log"
    fi
    if [ -f "$out_dir/slurm_cmds.log" ]; then
        echo ""
        echo "===== slurm_cmds.log ====="
        sed -n '1,160p' "$out_dir/slurm_cmds.log"
    fi
    return 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        -A|--account)
            [ $# -ge 2 ] || usage
            ACCOUNT="$2"
            shift 2
            ;;
        -a|--partition-a)
            [ $# -ge 2 ] || usage
            PARTITION_A="$2"
            shift 2
            ;;
        -b|--partition-b)
            [ $# -ge 2 ] || usage
            PARTITION_B="$2"
            shift 2
            ;;
        -m|--model-path)
            [ $# -ge 2 ] || usage
            MODEL_PATH="$2"
            shift 2
            ;;
        --check)
            [ $# -ge 2 ] || usage
            CHECK_JOB_ID="$2"
            shift 2
            ;;
        --image-a)
            [ $# -ge 2 ] || usage
            IMAGE_A="$2"
            shift 2
            ;;
        --image-b)
            [ $# -ge 2 ] || usage
            IMAGE_B="$2"
            shift 2
            ;;
        -t|--time)
            [ $# -ge 2 ] || usage
            TIME_LIMIT="$2"
            shift 2
            ;;
        -o|--output-dir)
            [ $# -ge 2 ] || usage
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -e|--extra)
            [ $# -ge 2 ] || usage
            EXTRA_BATCH_ARGS+=("-e" "$2")
            shift 2
            ;;
        -h|--help)
            usage 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(abs_path "$OUTPUT_DIR")"

if [ -n "$CHECK_JOB_ID" ]; then
    check_job "$CHECK_JOB_ID"
    exit $?
fi

if [ -z "$ACCOUNT" ] || [ -z "$PARTITION_A" ] || [ -z "$PARTITION_B" ] || [ -z "$MODEL_PATH" ]; then
    usage
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: config not found: $CONFIG_FILE" >&2
    exit 1
fi

# Pass image overrides only when non-empty: a `--set IMAGE_X=` (empty) would
# resolve to an empty container_image. Omitting it lets the sample's default
# image apply instead.
IMAGE_SET_ARGS=()
[ -n "$IMAGE_A" ] && IMAGE_SET_ARGS+=(--set "IMAGE_A=$IMAGE_A")
[ -n "$IMAGE_B" ] && IMAGE_SET_ARGS+=(--set "IMAGE_B=$IMAGE_B")

echo "Submitting multi-backend Slurm driver job (sflow batch)..."
echo "  config       : $CONFIG_FILE"
echo "  account      : $ACCOUNT"
echo "  partition A  : $PARTITION_A"
echo "  partition B  : $PARTITION_B"
echo "  image A      : ${IMAGE_A:-<sample default>}"
echo "  image B      : ${IMAGE_B:-<sample default>}"
echo "  output dir   : $OUTPUT_DIR"
echo "  note         : the driver sbatch wraps the heaviest backend; the rest salloc at runtime (see plan below)"

submit_output=$(sflow batch "$CONFIG_FILE" \
    --set "SLURM_ACCOUNT=$ACCOUNT" \
    --set "PARTITION_A=$PARTITION_A" \
    --set "PARTITION_B=$PARTITION_B" \
    --set "TIME_LIMIT=$TIME_LIMIT" \
    "${IMAGE_SET_ARGS[@]}" \
    --artifact "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
    --workspace-dir "$REPO_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --sflow-venv-path "$REPO_DIR" \
    -p "$PARTITION_A" \
    -A "$ACCOUNT" \
    --job-name "multi_backend_slurm" \
    --submit \
    "${EXTRA_BATCH_ARGS[@]}" 2>&1)

echo "$submit_output"

# Guard: confirm the per-backend-salloc code path is active. An old sflow build
# (without this change) would instead emit a Slurm heterogeneous job.
if printf '%s\n' "$submit_output" | grep -qiE 'heterogeneous job|#SBATCH hetjob'; then
    echo ""
    echo "WARNING: sflow generated a heterogeneous (hetjob) submission, not a"
    echo "         per-backend-salloc driver. You are likely running an sflow build"
    echo "         WITHOUT this fix; the result below may reproduce the original failure."
fi

job_id=$(echo "$submit_output" | sed -n 's/.*Submitted batch job \([0-9]\+\).*/\1/p' | tail -1)
if [ -z "$job_id" ]; then
    echo "ERROR: no Slurm job id found in sflow batch output"
    exit 1
fi

echo "Submitted multi-backend driver job id: $job_id"
check_job "$job_id"
