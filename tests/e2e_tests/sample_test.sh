#!/bin/bash

# set -x

usage() {
    echo "Usage: $0 -p <partition> -A <account> -m <model_path> [-G <gpus_per_node>] [-t s|m|inf|a|smoke] [--submit] [--check JOB_IDS] [-- <extra args>]"
    echo ""
    echo "  -t s   Self-contained examples only (--bulk-submit examples/)"
    echo "  -t m   Modular examples only (--bulk-input inference_x_v2/bulk_input.csv)"
    echo "  -t inf Infmax multi-node batch suites only"
    echo "  -t a   Both single and multi (default)"
    echo "  -t smoke Curated Slurm smoke subset with broad coverage"
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

abs_path() {
    local path="$1"
    if [ -d "$path" ]; then
        (cd "$path" && pwd)
        return
    fi
    if [ -e "$path" ]; then
        local dir base
        dir="$(dirname "$path")"
        base="$(basename "$path")"
        printf '%s/%s\n' "$(cd "$dir" && pwd)" "$base"
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
    out_dir=$(find sflow_output -maxdepth 2 -type d \( -name "${jid}_*" -o -name "${jid}-*" \) 2>/dev/null | grep -v "sflow-submit" | head -1)
    if [ -n "$out_dir" ]; then
        abs_path "$out_dir"
    fi
}

is_colon_task_result() {
    local jid="$1"
    local out_dir="${2:-}"
    local colon_jid
    for colon_jid in "${COLON_JOB_IDS[@]:-}"; do
        if [ "$colon_jid" = "$jid" ]; then
            return 0
        fi
    done
    case "$out_dir" in
        */colon_in_task_script/*|*-colon_in_task_script-*)
            return 0
            ;;
    esac
    return 1
}

colon_task_log_has_markers() {
    local out_dir="$1"
    find "$out_dir" -mindepth 2 -maxdepth 2 -type f -name '*.log' -exec sh -c '
        for log_file do
            if grep -Fq "My GPUs:" "$log_file" && grep -Fq "COLON_SCRIPT_E2E_PASS" "$log_file"; then
                exit 0
            fi
        done
        exit 1
    ' sh {} + 2>/dev/null
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

if [ "$TEST_TYPE" != "s" ] && [ "$TEST_TYPE" != "m" ] && [ "$TEST_TYPE" != "inf" ] && [ "$TEST_TYPE" != "a" ] && [ "$TEST_TYPE" != "smoke" ]; then
    echo "ERROR: -t must be 's', 'm', 'inf', 'a', or 'smoke', got '$TEST_TYPE'"
    usage
fi

if [ ${#EXTRA_BATCH_ARGS[@]} -gt 0 ]; then
    echo "Extra sflow batch args: ${EXTRA_BATCH_ARGS[*]}"
fi

WORKSPACE_DIR=$(pwd)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXAMPLES_DIR="$SCRIPT_DIR/../../examples"
SAMPLES_DIR="$SCRIPT_DIR/../../src/sflow/samples"
INFMAX_DIR="$SCRIPT_DIR/infmax"
INFMAX_CSV_NAME="${INFMAX_CSV_NAME:-1k1k_jobs.csv}"
INFMAX_ROW="${INFMAX_ROW:-1}"
INFMAX_RECIPE_REPO="${INFMAX_RECIPE_REPO:-https://gitlab-master.nvidia.com/rogliu/prenyx-ci-automation.git}"
INFMAX_RECIPE_REF="${INFMAX_RECIPE_REF:-inference_x}"
INFMAX_RECIPE_SUBDIR="${INFMAX_RECIPE_SUBDIR:-recipes/public/infmax}"
INFMAX_REFRESH_RECIPES="${INFMAX_REFRESH_RECIPES:-}"
INFMAX_BENCH_SERVING_DIR="${INFMAX_BENCH_SERVING_DIR:-$INFMAX_DIR/nvidia_submission/sa-bench}"

source "$REPO_DIR/scripts/use_under_dev_sflow.sh"
trap cleanup_under_dev_sflow EXIT
setup_under_dev_sflow "$REPO_DIR"

fetch_infmax_recipes() {
    local need_fetch="$INFMAX_REFRESH_RECIPES"
    local required_path

    for required_path in \
        "$INFMAX_DIR/dsr1-fp8-gb200-multi_node-sglang/$INFMAX_CSV_NAME" \
        "$INFMAX_DIR/dsr1-fp4-gb200-multi_node-trtllm/$INFMAX_CSV_NAME" \
        "$INFMAX_DIR/kimik2.5-fp4-gb200-multi_node-vllm/$INFMAX_CSV_NAME" \
        "$INFMAX_BENCH_SERVING_DIR"; do
        if [ ! -e "$required_path" ]; then
            need_fetch="1"
            break
        fi
    done

    if [ -z "$need_fetch" ]; then
        return
    fi

    if ! command -v git >/dev/null 2>&1; then
        echo "ERROR: git is required to fetch infmax recipes from $INFMAX_RECIPE_REPO" >&2
        exit 1
    fi
    if ! command -v rsync >/dev/null 2>&1; then
        echo "ERROR: rsync is required to sync fetched infmax recipes" >&2
        exit 1
    fi

    local token="${GITLAB_TOKEN:-${GITLAB_PRIVATE_TOKEN:-}}"
    if [ -z "$token" ]; then
        echo "ERROR: infmax recipe files are missing or INFMAX_REFRESH_RECIPES was requested." >&2
        echo "       Set GITLAB_TOKEN or GITLAB_PRIVATE_TOKEN so this script can fetch:" >&2
        echo "       $INFMAX_RECIPE_REPO ($INFMAX_RECIPE_REF:$INFMAX_RECIPE_SUBDIR)" >&2
        exit 1
    fi

    local tmp_dir
    tmp_dir=$(mktemp -d)
    local auth_repo="${INFMAX_RECIPE_REPO/https:\/\//https:\/\/oauth2:${token}@}"

    echo "Fetching infmax recipes ($INFMAX_RECIPE_REF:$INFMAX_RECIPE_SUBDIR) ..."
    if ! GIT_TERMINAL_PROMPT=0 git clone --depth 1 --branch "$INFMAX_RECIPE_REF" "$auth_repo" "$tmp_dir/repo" >/dev/null 2>&1; then
        rm -rf "$tmp_dir"
        echo "ERROR: failed to clone infmax recipe repo." >&2
        echo "       Check that GITLAB_TOKEN/GITLAB_PRIVATE_TOKEN has access to $INFMAX_RECIPE_REPO" >&2
        exit 1
    fi

    local source_dir="$tmp_dir/repo/$INFMAX_RECIPE_SUBDIR"
    if [ ! -d "$source_dir" ]; then
        rm -rf "$tmp_dir"
        echo "ERROR: infmax recipe subdir not found in fetched repo: $INFMAX_RECIPE_SUBDIR" >&2
        exit 1
    fi

    rsync -a --delete --exclude='batch_test.sh' "$source_dir/" "$INFMAX_DIR/"
    rm -rf "$tmp_dir"
}

set_infmax_suite_overrides() {
    INFMAX_SUITE_OVERRIDES=(-s "CONCURRENCY=[16,32]")
    case "$1" in
        kimik2.5-fp4-gb200-multi_node-vllm)
            INFMAX_SUITE_OVERRIDES+=(-s "DYNAMO_VERSION=1.0.1" -e "--container-remap-root")
            ;;
    esac
}

submit_infmax_batch_suites() {
    if [ ! -d "$INFMAX_BENCH_SERVING_DIR" ]; then
        echo "ERROR: benchmark serving directory not found: $INFMAX_BENCH_SERVING_DIR" >&2
        exit 1
    fi

    local target_dirs=(
        "$INFMAX_DIR/dsr1-fp8-gb200-multi_node-sglang"
        "$INFMAX_DIR/dsr1-fp4-gb200-multi_node-trtllm"
        "$INFMAX_DIR/kimik2.5-fp4-gb200-multi_node-vllm"
    )
    local suite_dir csv_file suite_name output job_id

    for suite_dir in "${target_dirs[@]}"; do
        csv_file="$suite_dir/$INFMAX_CSV_NAME"
        suite_name="$(basename "$suite_dir")"
        set_infmax_suite_overrides "$suite_name"

        if [ ! -f "$csv_file" ]; then
            echo "ERROR: CSV file not found for $suite_name: $csv_file" >&2
            exit 1
        fi

        echo ""
        echo "===== Infmax batch: $suite_name ($INFMAX_CSV_NAME row $INFMAX_ROW) ====="
        echo ""

        output=$(sflow batch \
            --bulk-input "$csv_file" \
            --row "$INFMAX_ROW" \
            -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
            -a "BENCH_SERVING_DIR=fs://$INFMAX_BENCH_SERVING_DIR" \
            -G "$GPUS_PER_NODE" \
            -A "$ACCOUNT" \
            -p "$PARTITION" \
            --sflow-venv-path "$WORKSPACE_DIR" \
            "${INFMAX_SUITE_OVERRIDES[@]}" \
            $SUBMIT \
            "${EXTRA_BATCH_ARGS[@]}" 2>&1)

        if [ $? -ne 0 ]; then
            echo "ERROR: sflow batch failed for $suite_name"
            echo "$output"
            exit 1
        fi
        echo "$output"

        while IFS= read -r job_id; do
            if [ -n "$job_id" ]; then
                JOB_IDS+=("$job_id")
            fi
        done < <(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+')
    done
}

submit_smoke_infmax_suites() {
    if [ ! -d "$INFMAX_BENCH_SERVING_DIR" ]; then
        echo "ERROR: benchmark serving directory not found: $INFMAX_BENCH_SERVING_DIR" >&2
        exit 1
    fi

    local target_dirs=(
        "$INFMAX_DIR/dsr1-fp4-gb200-multi_node-trtllm"
        "$INFMAX_DIR/kimik2.5-fp4-gb200-multi_node-vllm"
    )
    local suite_dir csv_file
    local suite_name output job_id

    for suite_dir in "${target_dirs[@]}"; do
        csv_file="$suite_dir/$INFMAX_CSV_NAME"
        suite_name="$(basename "$suite_dir")"
        set_infmax_suite_overrides "$suite_name"

        if [ ! -f "$csv_file" ]; then
            echo "ERROR: CSV file not found for smoke infmax suite $suite_name: $csv_file" >&2
            exit 1
        fi

        echo ""
        echo "===== Smoke infmax batch: $suite_name ($INFMAX_CSV_NAME row $INFMAX_ROW) ====="
        echo ""

        output=$(sflow batch \
            --bulk-input "$csv_file" \
            --row "$INFMAX_ROW" \
            -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
            -a "BENCH_SERVING_DIR=fs://$INFMAX_BENCH_SERVING_DIR" \
            -G "$GPUS_PER_NODE" \
            -A "$ACCOUNT" \
            -p "$PARTITION" \
            --sflow-venv-path "$WORKSPACE_DIR" \
            "${INFMAX_SUITE_OVERRIDES[@]}" \
            $SUBMIT \
            "${EXTRA_BATCH_ARGS[@]}" 2>&1)

        if [ $? -ne 0 ]; then
            echo "ERROR: sflow batch failed for smoke infmax suite $suite_name"
            echo "$output"
            exit 1
        fi
        echo "$output"

        while IFS= read -r job_id; do
            if [ -n "$job_id" ]; then
                JOB_IDS+=("$job_id")
            fi
        done < <(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+')
    done
}

submit_colon_task_script_e2e() {
    if [ -z "${SFLOW_COLON_SCRIPT_FIXTURE:-}" ]; then
        return
    fi
    if [ "$TEST_TYPE" != "s" ] && [ "$TEST_TYPE" != "a" ] && [ "$TEST_TYPE" != "smoke" ]; then
        return
    fi

    local colon_dir="${SFLOW_COLON_SCRIPT_OUTPUT_DIR:-$PWD/sflow_output/colon_in_task_script}"
    local colon_sbatch="$colon_dir/colon_in_task_script.sh"
    local colon_output colon_status colon_job_id
    mkdir -p "$colon_dir"

    echo ""
    echo "===== Focused e2e: colon in task script ====="
    echo ""

    colon_output=$(sflow batch -f "$SFLOW_COLON_SCRIPT_FIXTURE" \
        -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
        --sflow-venv-path "$WORKSPACE_DIR" \
        --output-dir "$colon_dir" \
        --job-name "colon_in_task_script" \
        -e "${SFLOW_COLON_SCRIPT_EXTRA_ARGS:---exclude=gb-nvl-137-compute09,gb-nvl-137-compute16}" \
        -o "$colon_sbatch" \
        $SUBMIT 2>&1)
    colon_status=$?
    echo "$colon_output"
    if [ "$colon_status" -ne 0 ]; then
        echo "ERROR: colon-in-task-script e2e submission failed"
        exit 1
    fi

    colon_job_id=$(echo "$colon_output" | sed -n 's/.*Submitted batch job \([0-9]\+\).*/\1/p' | tail -1)
    if [ -z "$colon_job_id" ]; then
        if [ -n "$SUBMIT" ]; then
            echo "ERROR: colon-in-task-script e2e did not report a Slurm job id"
            exit 1
        fi
        return
    fi

    echo "Colon task script job ID: $colon_job_id"
    JOB_IDS+=("$colon_job_id")
    COLON_JOB_IDS+=("$colon_job_id")
}

# Sync examples/ to src/sflow/samples/ so packaged samples stay up to date
echo "Syncing examples/ -> src/sflow/samples/ ..."
rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' --exclude='__init__.py' --exclude='sflow_output' \
    "$EXAMPLES_DIR/" "$SAMPLES_DIR/"
echo "Done."

if [ -n "$SUBMIT" ]; then
    srun -N 1 \
         -p "$PARTITION" \
         -A "$ACCOUNT" \
         -t 00:10:00 \
         --job-name=sflow_runtime_venv \
         bash -c "
             set -x && \
             cd \"$WORKSPACE_DIR\" && \
             rm -rf .sflow_venv && \
             /usr/bin/python3 -m venv .sflow_venv && \
             source .sflow_venv/bin/activate && \
             pip install uv && \
             cd \"$REPO_DIR\" && \
             uv pip install -e '.[dev]'
         "
fi

JOB_IDS=()
COLON_JOB_IDS=()
CSV_FILE="$EXAMPLES_DIR/inference_x_v2/bulk_input.csv"

# =============================================================================
# Part 1: Self-contained examples (--bulk-submit)
# =============================================================================
if [ "$TEST_TYPE" = "s" ] || [ "$TEST_TYPE" = "a" ] || [ "$TEST_TYPE" = "smoke" ]; then
    echo ""
    if [ "$TEST_TYPE" = "smoke" ]; then
        echo "===== Part 1: Smoke self-contained examples (--bulk-submit selected files) ====="
    else
        echo "===== Part 1: Self-contained examples (--bulk-submit) ====="
    fi
    echo ""

    if [ "$TEST_TYPE" = "smoke" ]; then
        SMOKE_SELF_CONTAINED=()
        SMOKE_BULK_ARGS=()
        for yaml_file in "$EXAMPLES_DIR"/slurm_*.yaml; do
            case "$(basename "$yaml_file")" in
                slurm_dynamo_sglang_agg.yaml|slurm_dynamo_vllm_agg.yaml|slurm_sglang_server_client.yaml)
                    continue
                    ;;
            esac
            SMOKE_SELF_CONTAINED+=("$yaml_file")
            SMOKE_BULK_ARGS+=(--bulk-submit "$yaml_file")
        done
        if [ ${#SMOKE_SELF_CONTAINED[@]} -eq 0 ]; then
            echo "ERROR: no smoke self-contained Slurm YAML files selected"
            exit 1
        fi

        output=$(sflow batch \
            "${SMOKE_BULK_ARGS[@]}" \
            -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
            -G "$GPUS_PER_NODE" \
            -p "$PARTITION" \
            -A "$ACCOUNT" \
            --sflow-venv-path "$WORKSPACE_DIR" \
            $SUBMIT \
            "${EXTRA_BATCH_ARGS[@]}" 2>&1)
    else
        output=$(sflow batch \
            --bulk-submit "$EXAMPLES_DIR" \
            -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
            -G "$GPUS_PER_NODE" \
            -p "$PARTITION" \
            -A "$ACCOUNT" \
            --sflow-venv-path "$WORKSPACE_DIR" \
            $SUBMIT \
            "${EXTRA_BATCH_ARGS[@]}" 2>&1)
    fi

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
if [ "$TEST_TYPE" = "m" ] || [ "$TEST_TYPE" = "a" ] || [ "$TEST_TYPE" = "smoke" ]; then
    echo ""
    if [ "$TEST_TYPE" = "smoke" ]; then
        echo "===== Part 2: Smoke modular examples (--bulk-input selected rows) ====="
    else
        echo "===== Part 2: Modular examples (--bulk-input) ====="
    fi
    echo ""

    if [ ! -f "$CSV_FILE" ]; then
        echo "WARNING: CSV file not found: $CSV_FILE, skipping modular examples"
    else
        MODULAR_ROW_ARGS=()
        if [ "$TEST_TYPE" = "smoke" ]; then
            # Cover disaggregated and aggregated paths across trtllm, sglang, and vllm.
            MODULAR_ROW_ARGS=(--row 1 --row 4 --row 6 --row 8)
        fi
        output=$(sflow batch \
            --bulk-input "$CSV_FILE" \
            "${MODULAR_ROW_ARGS[@]}" \
            -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
            -G "$GPUS_PER_NODE" \
            -A "$ACCOUNT" \
            -p "$PARTITION" \
            --sflow-venv-path "$WORKSPACE_DIR" \
            $SUBMIT \
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

# =============================================================================
# Part 3: Infmax multi-node batch suites
# =============================================================================
if [ "$TEST_TYPE" = "inf" ] || [ "$TEST_TYPE" = "m" ] || [ "$TEST_TYPE" = "a" ] || [ "$TEST_TYPE" = "smoke" ]; then
    echo ""
    if [ "$TEST_TYPE" = "smoke" ]; then
        echo "===== Part 3: Smoke infmax multi-node batch suite ====="
    else
        echo "===== Part 3: Infmax multi-node batch suites ====="
    fi
    echo ""

    fetch_infmax_recipes
    if [ "$TEST_TYPE" = "smoke" ]; then
        submit_smoke_infmax_suites
    else
        submit_infmax_batch_suites
    fi
fi

submit_colon_task_script_e2e

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
        workflow_out=$(workflow_output_dir_for_job "$jid")
        if [ -n "$workflow_out" ]; then
            echo "  $jid ($jobname): ${state:-UNKNOWN}  nodes: ${nnodes:-?}  elapsed: ${elapsed:-N/A}  output: $workflow_out"
        else
            echo "  $jid ($jobname): ${state:-UNKNOWN}  nodes: ${nnodes:-?}  elapsed: ${elapsed:-N/A}  output: (not found under $(abs_path sflow_output))"
        fi
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
    out_dir=$(workflow_output_dir_for_job "$jid")
    if [ -z "$out_dir" ]; then
        echo "  Job $jid: output folder not found under $(abs_path sflow_output)"
        continue
    fi
    if is_colon_task_result "$jid" "$out_dir"; then
        if colon_task_log_has_markers "$out_dir"; then
            PASSED=$((PASSED + 1))
            echo "  Job $jid: PASS (colon-in-task-script markers under $out_dir)"
        else
            echo "  Job $jid: FAIL (colon-in-task-script task log markers missing under $out_dir)"
        fi
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
