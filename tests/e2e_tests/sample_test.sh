#!/bin/bash

# set -x

usage() {
    echo "Usage: $0 -p <partition> -A <account> -m <model_path> [-G <gpus_per_node>] [-t s|m|a] [--submit] [--check JOB_IDS] [-- <extra args>]"
    echo ""
    echo "  -t s   Self-contained examples only (--bulk-submit examples/)"
    echo "  -t m   Modular examples only (--bulk-input inference_x_v2/bulk_input.csv)"
    echo "  -t a   Both single and multi (default)"
    echo ""
    echo "  --check JOB_IDS   Skip submission, only check results"
    echo "                    Accepts: comma-separated IDs and/or [START:END] ranges"
    echo ""
    echo "  Examples:"
    echo "    $0 -p gpu -A myacct -m /path/to/model --submit"
    echo "    $0 -p gpu -A myacct -m /path/to/model -t s --submit"
    echo "    $0 -p gpu -A myacct -m /path/to/model -t m --submit"
    echo "    $0 -p gpu -A myacct -m /path/to/model -- -e '--exclusive'"
    echo "    $0 --check 267005,267006,267007,267008"
    echo "    $0 --check '[267005:267008]'"
    echo "    $0 --check '267001,[267005:267008],267020'"
    exit 1
}

SUBMIT=""
CHECK_JOBS=""
TEST_TYPE="a"
ARGS=()
EXTRA_BATCH_ARGS=()
SEEN_DASHDASH=false
while [ $# -gt 0 ]; do
    if $SEEN_DASHDASH; then
        EXTRA_BATCH_ARGS+=("$1")
        shift
        continue
    fi
    case "$1" in
        --)
            SEEN_DASHDASH=true
            shift
            ;;
        --submit)
            SUBMIT="--submit"
            shift
            ;;
        --check)
            CHECK_JOBS="$2"
            shift 2
            ;;
        --type|-t)
            TEST_TYPE="$2"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${ARGS[@]}"

# Expand Python-like range expressions [START:END] into individual IDs.
# Supports: "267005", "[267005:267008]", "267001,[267005:267008],267020"
expand_job_ids() {
    local input="$1"
    local -a result=()
    IFS=',' read -ra tokens <<< "$input"
    for token in "${tokens[@]}"; do
        token="${token// /}"
        if [[ "$token" =~ ^\[([0-9]+):([0-9]+)\]$ ]]; then
            local start="${BASH_REMATCH[1]}" end="${BASH_REMATCH[2]}"
            for (( i=start; i<=end; i++ )); do
                result+=("$i")
            done
        elif [[ "$token" =~ ^[0-9]+$ ]]; then
            result+=("$token")
        else
            echo "ERROR: invalid job ID token: '$token'" >&2
            exit 1
        fi
    done
    echo "${result[@]}"
}

# --check mode: skip everything, just check results
if [ -n "$CHECK_JOBS" ]; then
    set +x
    JOB_IDS=()
    read -ra JOB_IDS <<< "$(expand_job_ids "$CHECK_JOBS")"

    echo ""
    echo "===== Checking ${#JOB_IDS[@]} jobs ====="

    # Jump directly to the results section
else

GPUS_PER_NODE=4
while getopts "p:A:m:G:" opt; do
    case $opt in
        p) PARTITION="$OPTARG" ;;
        A) ACCOUNT="$OPTARG" ;;
        m) MODEL_PATH="$OPTARG" ;;
        G) GPUS_PER_NODE="$OPTARG" ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))
EXTRA_BATCH_ARGS+=("$@")

if [ -z "$PARTITION" ] || [ -z "$ACCOUNT" ] || [ -z "$MODEL_PATH" ]; then
    usage
fi

if [ "$TEST_TYPE" != "s" ] && [ "$TEST_TYPE" != "m" ] && [ "$TEST_TYPE" != "a" ]; then
    echo "ERROR: -t must be 's', 'm', or 'a', got '$TEST_TYPE'"
    usage
fi

if [ ${#EXTRA_BATCH_ARGS[@]} -gt 0 ]; then
    echo "Extra sflow batch args: ${EXTRA_BATCH_ARGS[*]}"
fi

WORKSPACE_DIR=$(pwd)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$SCRIPT_DIR/../../examples"
SAMPLES_DIR="$SCRIPT_DIR/../../src/sflow/samples"

# Sync examples/ to src/sflow/samples/ so packaged samples stay up to date
echo "Syncing examples/ -> src/sflow/samples/ ..."
rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' --exclude='__init__.py' --exclude='sflow_output' \
    "$EXAMPLES_DIR/" "$SAMPLES_DIR/"
echo "Done."

if [ -n "$SUBMIT" ] && [ ! -d "$WORKSPACE_DIR/.sflow_venv" ]; then
    srun -N 1 \
         -p "$PARTITION" \
         -A "$ACCOUNT" \
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
             uv pip install -e '.[dev]'
         "
elif [ -n "$SUBMIT" ]; then
    echo "Skipping venv creation: $WORKSPACE_DIR/.sflow_venv already exists"
fi

JOB_IDS=()
CSV_FILE="$EXAMPLES_DIR/inference_x_v2/bulk_input.csv"

# =============================================================================
# Part 1: Self-contained examples (--bulk-submit)
# =============================================================================
if [ "$TEST_TYPE" = "s" ] || [ "$TEST_TYPE" = "a" ]; then
    echo ""
    echo "===== Part 1: Self-contained examples (--bulk-submit) ====="
    echo ""

    output=$(sflow batch \
        --bulk-submit "$EXAMPLES_DIR" \
        -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
        -G "$GPUS_PER_NODE" \
        -p "$PARTITION" \
        -A "$ACCOUNT" --log-level warn \
        --sflow-venv-path "$WORKSPACE_DIR" \
        $SUBMIT \
        "${EXTRA_BATCH_ARGS[@]}" 2>&1)

    if [ $? -ne 0 ]; then
        echo "ERROR: sflow batch --bulk-submit failed"
        echo "$output"
        exit 1
    fi
    echo "$output"

    while IFS= read -r job_id; do
        if [ -n "$job_id" ]; then
            JOB_IDS+=("$job_id")
        fi
    done < <(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+')
fi

# =============================================================================
# Part 2: Modular examples (--bulk-input)
# =============================================================================
if [ "$TEST_TYPE" = "m" ] || [ "$TEST_TYPE" = "a" ]; then
    echo ""
    echo "===== Part 2: Modular examples (--bulk-input) ====="
    echo ""

    if [ ! -f "$CSV_FILE" ]; then
        echo "WARNING: CSV file not found: $CSV_FILE, skipping modular examples"
    else
        output=$(sflow batch \
            --bulk-input "$CSV_FILE" \
            -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
            -G "$GPUS_PER_NODE" \
            -A "$ACCOUNT" \
            -p "$PARTITION" \
            --sflow-venv-path "$WORKSPACE_DIR" \
            $SUBMIT --log-level warn \
            "${EXTRA_BATCH_ARGS[@]}" 2>&1)

        if [ $? -ne 0 ]; then
            echo "ERROR: sflow batch --bulk-input failed"
            echo "$output"
            exit 1
        fi
        echo "$output"

        while IFS= read -r job_id; do
            if [ -n "$job_id" ]; then
                JOB_IDS+=("$job_id")
            fi
        done < <(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+')
    fi
fi

set +x

echo ""
echo "===== Submitted Jobs ====="

fi  # end of --check else block


if [ ${#JOB_IDS[@]} -eq 0 ]; then
    echo "No job IDs captured."
    exit 0
fi

echo "Job IDs: ${JOB_IDS[*]}"
echo "Monitor: squeue -j $(IFS=,; echo "${JOB_IDS[*]}")"

# Wait for all jobs to finish, polling every 30s
echo ""
echo "===== Waiting for jobs to complete ====="
while true; do
    RUNNING=0
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Job status:"
    for jid in "${JOB_IDS[@]}"; do
        state=$(sacct -j "$jid" --noheader -o State -X 2>/dev/null | head -1 | tr -d ' ')
        jobname=$(sacct -j "$jid" --noheader -o JobName%20 -X 2>/dev/null | head -1 | tr -d ' ')
        elapsed=$(sacct -j "$jid" --noheader -o Elapsed -X 2>/dev/null | head -1 | tr -d ' ')
        nnodes=$(sacct -j "$jid" --noheader -o NNodes -X 2>/dev/null | head -1 | tr -d ' ')
        echo "  $jid ($jobname): ${state:-UNKNOWN}  nodes: ${nnodes:-?}  elapsed: ${elapsed:-N/A}"
        if [ -z "$state" ] || [[ "$state" == "PENDING" ]] || [[ "$state" == "RUNNING" ]] || [[ "$state" == "CONFIGURING" ]]; then
            RUNNING=$((RUNNING + 1))
        fi
    done
    if [ "$RUNNING" -eq 0 ]; then
        echo "All jobs finished."
        break
    fi
    echo "  $RUNNING jobs still in progress..."
    echo ""
    sleep 30
done

# Check results in output folders
echo ""
echo "===== Results ====="
TOTAL=0
PASSED=0
for jid in "${JOB_IDS[@]}"; do
    TOTAL=$((TOTAL + 1))
    out_dir=$(find sflow_output -maxdepth 1 -type d \( -name "${jid}_*" -o -name "${jid}-*" \) 2>/dev/null | grep -v "sflow-submit" | head -1)
    if [ -z "$out_dir" ]; then
        echo "  Job $jid: output folder not found"
        continue
    fi
    # Check for various success indicators across different workflow types
    #   aiperf benchmark: '0 errors' in benchmark log
    #   aiperf template:  '0 valid' in benchmark log
    #   infmax benchmark: 'Successful requests:' with non-zero value
    #   auto_replica:     'Client Task Nodes' in client task log
    count_aiperf_errors=$(find "$out_dir" -type f -name 'benchmark*.log' -exec grep -l "0 errors" {} + 2>/dev/null | wc -l)
    count_aiperf_valid=$(find "$out_dir" -type f -name 'benchmark*.log' -exec grep -l "0 valid" {} + 2>/dev/null | wc -l)
    count_zero_success=$(find "$out_dir" -type f -name 'benchmark*.log' -exec grep -lP "Successful requests:\s+0\s*$" {} + 2>/dev/null | wc -l)
    count_any_success=$(find "$out_dir" -type f -name 'benchmark*.log' -exec grep -l "Successful requests:" {} + 2>/dev/null | wc -l)
    count_replica=$(find "$out_dir" -type f -name 'client*.log' -exec grep -l "Client Task Nodes" {} + 2>/dev/null | wc -l)

    if [ "$count_zero_success" -gt 0 ]; then
        echo "  Job $jid: FAIL ('Successful requests: 0' found in $out_dir)"
    elif [ "$count_aiperf_errors" -gt 0 ] || [ "$count_aiperf_valid" -gt 0 ] || [ "$count_any_success" -gt 0 ] || [ "$count_replica" -gt 0 ]; then
        PASSED=$((PASSED + 1))
        echo "  Job $jid: PASS (under $out_dir)"
    else
        echo "  Job $jid: FAIL (no success indicator found in $out_dir)"
    fi
done

echo ""
echo "===== Summary ====="
echo "$PASSED/$TOTAL jobs passed"
