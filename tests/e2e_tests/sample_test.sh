#!/bin/bash

# set -x

usage() {
    echo "Usage: $0 -p <partition> -A <account> -m <model_path> [-G <gpus_per_node>] [-t s|m|inf|a|smoke|min] [--submit] [--check JOB_IDS] [-- <extra args>]"
    echo ""
    echo "  -t s   Self-contained examples only (--bulk-submit examples/self_contained/slurm/)"
    echo "  -t m   Modular examples only (--bulk-input modular/inference_x_v2/bulk_input.csv)"
    echo "  -t inf Infmax multi-node batch suites only"
    echo "  -t a   Both single and multi (default)"
    echo "  -t smoke Curated Slurm smoke subset with broad coverage"
    echo "  -t min Minimal representative set (one job per validation type)"
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
E2E_OUTPUT_DIR="${E2E_OUTPUT_DIR:-$REPO_DIR/sflow_output}"

workflow_output_dir_for_job() {
    local jid="$1"
    local out_dir
    local search_dir="${E2E_OUTPUT_DIR:-sflow_output}"
    out_dir=$(find "$search_dir" -maxdepth 2 -type d \( -name "${jid}_*" -o -name "${jid}-*" \) 2>/dev/null | grep -v "sflow-submit" | head -1)
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

is_multi_backend_result() {
    local jid="$1"
    local mbid
    for mbid in "${MULTI_BACKEND_JOB_IDS[@]:-}"; do
        if [ "$mbid" = "$jid" ]; then
            return 0
        fi
    done
    return 1
}

multi_backend_run_ok() {
    # The multi-backend run is correct when task_a and task_b ran on DIFFERENT
    # backend nodes (distinct per-backend allocations => correct per-task backend
    # binding). Each task echoes
    #   "task_<x> backend=cluster_<x> job=<job id> nodes=<assigned node names>".
    # Each backend gets its own salloc job id; node identity is the binding signal.
    local dir="$1"
    [ -n "$dir" ] || return 1
    local a_nodes b_nodes
    a_nodes=$(grep -rhoE 'task_a backend=cluster_a job=[0-9]+ nodes=[^[:space:]]+' "$dir" 2>/dev/null | sed -E 's/.*nodes=//' | head -1)
    b_nodes=$(grep -rhoE 'task_b backend=cluster_b job=[0-9]+ nodes=[^[:space:]]+' "$dir" 2>/dev/null | sed -E 's/.*nodes=//' | head -1)
    [ -n "$a_nodes" ] && [ -n "$b_nodes" ] && [ "$a_nodes" != "$b_nodes" ]
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

if [ "$TEST_TYPE" != "s" ] && [ "$TEST_TYPE" != "m" ] && [ "$TEST_TYPE" != "inf" ] && [ "$TEST_TYPE" != "a" ] && [ "$TEST_TYPE" != "smoke" ] && [ "$TEST_TYPE" != "min" ]; then
    echo "ERROR: -t must be 's', 'm', 'inf', 'a', 'smoke', or 'min', got '$TEST_TYPE'"
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
INFMAX_RECIPE_REPO="${INFMAX_RECIPE_REPO:-}"
INFMAX_RECIPE_REF="${INFMAX_RECIPE_REF:-inference_x}"
INFMAX_RECIPE_SUBDIR="${INFMAX_RECIPE_SUBDIR:-recipes/public/infmax}"
INFMAX_REFRESH_RECIPES="${INFMAX_REFRESH_RECIPES:-}"
INFMAX_BENCH_SERVING_DIR="${INFMAX_BENCH_SERVING_DIR:-$INFMAX_DIR/nvidia_submission/sa-bench}"
INFMAX_MONITOR_CONFIG="${INFMAX_MONITOR_CONFIG:-$INFMAX_DIR/monitor/monitor.yaml}"
BATCH_WORKSPACE_ARGS=(--workspace-dir "$REPO_DIR")
BATCH_OUTPUT_ARGS=(--output-dir "$E2E_OUTPUT_DIR")
# No --sflow-venv-path: each job's venv and per-job source copy default to
# compute-node-local scratch (${TMPDIR:-/tmp}) -- isolated and auto-cleaned.
BATCH_VENV_ARGS=()

source "$REPO_DIR/scripts/use_under_dev_sflow.sh"
trap cleanup_under_dev_sflow EXIT
setup_under_dev_sflow "$REPO_DIR" || exit 1

fetch_infmax_recipes() {
    local need_fetch="$INFMAX_REFRESH_RECIPES"
    local required_path

    for required_path in \
        "$INFMAX_DIR/dsr1-fp8-gb200-multi_node-sglang/$INFMAX_CSV_NAME" \
        "$INFMAX_DIR/dsr1-fp4-gb200-multi_node-trtllm/$INFMAX_CSV_NAME" \
        "$INFMAX_DIR/kimik2.5-fp4-gb200-multi_node-vllm/$INFMAX_CSV_NAME" \
        "$INFMAX_MONITOR_CONFIG" \
        "$INFMAX_BENCH_SERVING_DIR"; do
        if [ ! -e "$required_path" ]; then
            need_fetch="1"
            break
        fi
    done

    if [ -z "$need_fetch" ]; then
        return
    fi

    if [ -z "$INFMAX_RECIPE_REPO" ]; then
        echo "ERROR: INFMAX_RECIPE_REPO is not set; cannot fetch infmax recipes." >&2
        echo "       Set INFMAX_RECIPE_REPO to the git repo hosting $INFMAX_RECIPE_SUBDIR" >&2
        echo "       (ref: $INFMAX_RECIPE_REF)." >&2
        exit 1
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
            -f "$INFMAX_MONITOR_CONFIG" \
            --bulk-input "$csv_file" \
            --row "$INFMAX_ROW" \
            -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
            -a "BENCH_SERVING_DIR=fs://$INFMAX_BENCH_SERVING_DIR" \
            -G "$GPUS_PER_NODE" \
            -A "$ACCOUNT" \
            -p "$PARTITION" \
            "${BATCH_WORKSPACE_ARGS[@]}" \
            "${BATCH_OUTPUT_ARGS[@]}" \
            "${BATCH_VENV_ARGS[@]}" \
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
            -f "$INFMAX_MONITOR_CONFIG" \
            --bulk-input "$csv_file" \
            --row "$INFMAX_ROW" \
            -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
            -a "BENCH_SERVING_DIR=fs://$INFMAX_BENCH_SERVING_DIR" \
            -G "$GPUS_PER_NODE" \
            -A "$ACCOUNT" \
            -p "$PARTITION" \
            "${BATCH_WORKSPACE_ARGS[@]}" \
            "${BATCH_OUTPUT_ARGS[@]}" \
            "${BATCH_VENV_ARGS[@]}" \
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

submit_min_infmax_suite() {
    if [ ! -d "$INFMAX_BENCH_SERVING_DIR" ]; then
        echo "ERROR: benchmark serving directory not found: $INFMAX_BENCH_SERVING_DIR" >&2
        exit 1
    fi

    local suite_dir="$INFMAX_DIR/dsr1-fp4-gb200-multi_node-trtllm"
    local csv_file="$suite_dir/$INFMAX_CSV_NAME"
    local suite_name output job_id
    suite_name="$(basename "$suite_dir")"
    set_infmax_suite_overrides "$suite_name"

    if [ ! -f "$csv_file" ]; then
        echo "ERROR: CSV file not found for min infmax suite $suite_name: $csv_file" >&2
        exit 1
    fi

    echo ""
    echo "===== Min infmax batch: $suite_name ($INFMAX_CSV_NAME row $INFMAX_ROW) ====="
    echo ""

    output=$(sflow batch \
        -f "$INFMAX_MONITOR_CONFIG" \
        --bulk-input "$csv_file" \
        --row "$INFMAX_ROW" \
        -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
        -a "BENCH_SERVING_DIR=fs://$INFMAX_BENCH_SERVING_DIR" \
        -G "$GPUS_PER_NODE" \
        -A "$ACCOUNT" \
        -p "$PARTITION" \
        "${BATCH_WORKSPACE_ARGS[@]}" \
        "${BATCH_OUTPUT_ARGS[@]}" \
        "${BATCH_VENV_ARGS[@]}" \
        "${INFMAX_SUITE_OVERRIDES[@]}" \
        $SUBMIT \
        "${EXTRA_BATCH_ARGS[@]}" 2>&1)

    if [ $? -ne 0 ]; then
        echo "ERROR: sflow batch failed for min infmax suite $suite_name"
        echo "$output"
        exit 1
    fi
    echo "$output"

    while IFS= read -r job_id; do
        if [ -n "$job_id" ]; then
            JOB_IDS+=("$job_id")
        fi
    done < <(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+')
}

submit_colon_task_script_e2e() {
    if [ -z "${SFLOW_COLON_SCRIPT_FIXTURE:-}" ]; then
        return
    fi
    if [ "$TEST_TYPE" != "s" ] && [ "$TEST_TYPE" != "a" ] && [ "$TEST_TYPE" != "smoke" ]; then
        return
    fi

    local colon_dir="${SFLOW_COLON_SCRIPT_OUTPUT_DIR:-$E2E_OUTPUT_DIR/colon_in_task_script}"
    local colon_sbatch="$colon_dir/colon_in_task_script.sh"
    local colon_output colon_status colon_job_id
    mkdir -p "$colon_dir"

    echo ""
    echo "===== Focused e2e: colon in task script ====="
    echo ""

    # This focused e2e submits through its own `sflow batch` (it does NOT receive
    # the shared E2E_BATCH_EXTRA_ARGS), so enable the workflow monitor explicitly --
    # otherwise this job would be the one workflow without an sflow_monitor.log and
    # would trip the independent monitor-coverage gate below.
    colon_output=$(sflow batch -f "$SFLOW_COLON_SCRIPT_FIXTURE" \
        -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
        "${BATCH_WORKSPACE_ARGS[@]}" \
        "${BATCH_VENV_ARGS[@]}" \
        --output-dir "$colon_dir" \
        --job-name "colon_in_task_script" \
        --enable-workflow-monitor \
        -e "${SFLOW_COLON_SCRIPT_EXTRA_ARGS:---exclude=${SLURM_E2E_EXCLUDE_NODES:-gb-nvl-137-compute02,gb-nvl-137-compute14}}" \
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

run_multi_backend_real() {
    # Real multi-backend coverage via `sflow batch`: a >=2-Slurm-backend config
    # makes `sflow batch` emit one driver sbatch sized to the leader backend;
    # inside, the leader reuses that allocation while every other backend runs its
    # own salloc, so each backend binds to its OWN allocation/partition and task_a
    # and task_b land on DIFFERENT nodes (distinct per-backend allocations =>
    # correct per-task backend binding). The backends target two DIFFERENT
    # partitions (each needs >=1 idle node); override the defaults via
    # MULTI_BACKEND_PARTITION_A / MULTI_BACKEND_PARTITION_B. Requires a real
    # cluster (--submit).
    if [ -z "$SUBMIT" ]; then
        return
    fi
    case "$TEST_TYPE" in
        s|a|smoke|min) ;;
        *) return ;;
    esac

    local part_a="${MULTI_BACKEND_PARTITION_A:-genesisq}"
    local part_b="${MULTI_BACKEND_PARTITION_B:-gamoraq}"
    MULTI_BACKEND_RUN_DIR="$E2E_OUTPUT_DIR/multi_backend_real"
    local mb_dir="$MULTI_BACKEND_RUN_DIR"
    local mb_script="$mb_dir/multi_backend_hetjob.sh"
    mkdir -p "$mb_dir"

    echo ""
    echo "===== Real multi-backend run (per-backend salloc via sflow batch) ====="
    echo "      cluster_a partition=$part_a, cluster_b partition=$part_b"
    echo ""

    # `sflow batch` on the two-backend config emits a driver sbatch. The workflow
    # output dir is keyed by the driver job id (SFLOW_RUN_ID_PREFIX=$SLURM_JOB_ID),
    # so capturing "Submitted batch job N" lets it flow through the shared wait
    # (sacct) + validate loop below like every other batched job. The CLI -p/-A
    # are required by the command but the driver is sized to the leader backend
    # (each backend uses its own resolved partition/account).
    local mb_output mb_status
    mb_output=$(sflow batch "$EXAMPLES_DIR/self_contained/slurm/multi_backend.yaml" \
        --set "SLURM_ACCOUNT=$ACCOUNT" \
        --set "PARTITION_A=$part_a" \
        --set "PARTITION_B=$part_b" \
        --set "TIME_LIMIT=10" \
        -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
        -p "$part_a" \
        -A "$ACCOUNT" \
        --job-name "multi_backend_slurm" \
        "${BATCH_WORKSPACE_ARGS[@]}" \
        "${BATCH_OUTPUT_ARGS[@]}" \
        "${BATCH_VENV_ARGS[@]}" \
        -o "$mb_script" \
        $SUBMIT \
        "${EXTRA_BATCH_ARGS[@]}" 2>&1)
    mb_status=$?
    echo "$mb_output"
    if [ "$mb_status" -ne 0 ]; then
        MULTI_BACKEND_LAUNCH_FAILED=1
        echo "  Multi-backend run: FAIL (sflow batch failed, rc=$mb_status; see output above)"
        return
    fi

    # The driver submission reports a single job id; the results loop finds its
    # workflow output dir by that id and validates distinct nodes.
    local mb_job_id
    mb_job_id=$(echo "$mb_output" | sed -n 's/.*Submitted batch job \([0-9]\+\).*/\1/p' | tail -1)
    if [ -n "$mb_job_id" ]; then
        JOB_IDS+=("$mb_job_id")
        MULTI_BACKEND_JOB_IDS+=("$mb_job_id")
        echo "  Multi-backend driver job id: $mb_job_id (script: $mb_script)"
    else
        # No job id was reported; record the failure so it is still counted.
        MULTI_BACKEND_LAUNCH_FAILED=1
        echo "  Multi-backend run: FAIL (no Slurm job id reported by sflow batch)"
    fi
}

# Sync examples/ to src/sflow/samples/ so packaged samples stay up to date
echo "Syncing examples/ -> src/sflow/samples/ ..."
rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' --exclude='__init__.py' --exclude='sflow_output' \
    "$EXAMPLES_DIR/" "$SAMPLES_DIR/"
echo "Done."

# No shared runtime venv to pre-build. Under the per-job venv flow, each
# submitted job creates its own fresh venv and installs this checkout editable
# (--sflow-source-path "$REPO_DIR", injected by use_under_dev_sflow.sh). Each job
# also rsyncs the checkout into its own per-job source dir before installing, so
# concurrent editable builds never share one source tree -- an editable build
# rewrites setuptools-scm's _version.py and src/*.egg-info back into the tree
# (a warm uv cache does NOT skip this), so a shared tree would race. The local
# editable install setup_under_dev_sflow requires still warms uv's package cache,
# which just makes the per-job installs faster.

JOB_IDS=()
COLON_JOB_IDS=()
MULTI_BACKEND_JOB_IDS=()
MULTI_BACKEND_RUN_DIR=""
MULTI_BACKEND_LAUNCH_FAILED=""
CSV_FILE="$EXAMPLES_DIR/modular/inference_x_v2/bulk_input.csv"

# =============================================================================
# Part 1: Self-contained examples (--bulk-submit)
# =============================================================================
if [ "$TEST_TYPE" = "s" ] || [ "$TEST_TYPE" = "a" ] || [ "$TEST_TYPE" = "smoke" ] || [ "$TEST_TYPE" = "min" ]; then
    echo ""
    if [ "$TEST_TYPE" = "min" ]; then
        echo "===== Part 1: Min self-contained examples (--bulk-submit selected files) ====="
    elif [ "$TEST_TYPE" = "smoke" ]; then
        echo "===== Part 1: Smoke self-contained examples (--bulk-submit selected files) ====="
    else
        echo "===== Part 1: Self-contained examples (--bulk-submit) ====="
    fi
    echo ""

    if [ "$TEST_TYPE" = "min" ]; then
        MIN_SELF_CONTAINED=(
            "$EXAMPLES_DIR/self_contained/slurm/auto_replica.yaml"
            "$EXAMPLES_DIR/self_contained/slurm/dynamo_trtllm_disagg.yaml"
            "$EXAMPLES_DIR/self_contained/slurm/resource_release_after.yaml"
            "$EXAMPLES_DIR/self_contained/slurm/trtllm_serve_disagg.yaml"
        )
        MIN_BULK_ARGS=()
        for yaml_file in "${MIN_SELF_CONTAINED[@]}"; do
            if [ ! -f "$yaml_file" ]; then
                echo "ERROR: min self-contained Slurm YAML not found: $yaml_file"
                exit 1
            fi
            MIN_BULK_ARGS+=(--bulk-submit "$yaml_file")
        done
        output=$(sflow batch \
            "${MIN_BULK_ARGS[@]}" \
            -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
            -G "$GPUS_PER_NODE" \
            -p "$PARTITION" \
            -A "$ACCOUNT" \
            "${BATCH_WORKSPACE_ARGS[@]}" \
            "${BATCH_OUTPUT_ARGS[@]}" \
            "${BATCH_VENV_ARGS[@]}" \
            $SUBMIT \
            "${EXTRA_BATCH_ARGS[@]}" 2>&1)
    elif [ "$TEST_TYPE" = "smoke" ]; then
        SMOKE_SELF_CONTAINED=()
        SMOKE_BULK_ARGS=()
        for yaml_file in "$EXAMPLES_DIR"/self_contained/slurm/*.yaml; do
            case "$(basename "$yaml_file")" in
                dynamo_sglang_agg.yaml|dynamo_vllm_agg.yaml|sglang_server_client.yaml)
                    continue
                    ;;
                multi_backend.yaml)
                    # Covered separately by run_multi_backend_real as a `sflow
                    # batch` heterogeneous job; it needs two partitions
                    # (PARTITION_A/PARTITION_B), so skip the single-partition
                    # bulk-submit copy here.
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
            "${BATCH_WORKSPACE_ARGS[@]}" \
            "${BATCH_OUTPUT_ARGS[@]}" \
            "${BATCH_VENV_ARGS[@]}" \
            $SUBMIT \
            "${EXTRA_BATCH_ARGS[@]}" 2>&1)
    else
        # Bulk-submit every example EXCEPT the multi-backend config: it needs two
        # partitions (PARTITION_A/PARTITION_B) and is covered separately by
        # run_multi_backend_real as a `sflow batch` heterogeneous job.
        ALL_BULK_ARGS=()
        for yaml_file in "$EXAMPLES_DIR"/self_contained/slurm/*.yaml; do
            case "$(basename "$yaml_file")" in
                multi_backend.yaml)
                    continue
                    ;;
            esac
            ALL_BULK_ARGS+=(--bulk-submit "$yaml_file")
        done
        output=$(sflow batch \
            "${ALL_BULK_ARGS[@]}" \
            -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
            -G "$GPUS_PER_NODE" \
            -p "$PARTITION" \
            -A "$ACCOUNT" \
            "${BATCH_WORKSPACE_ARGS[@]}" \
            "${BATCH_OUTPUT_ARGS[@]}" \
            "${BATCH_VENV_ARGS[@]}" \
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
if [ "$TEST_TYPE" = "m" ] || [ "$TEST_TYPE" = "a" ] || [ "$TEST_TYPE" = "smoke" ] || [ "$TEST_TYPE" = "min" ]; then
    echo ""
    if [ "$TEST_TYPE" = "min" ]; then
        echo "===== Part 2: Min modular example (--bulk-input selected row) ====="
    elif [ "$TEST_TYPE" = "smoke" ]; then
        echo "===== Part 2: Smoke modular examples (--bulk-input selected rows) ====="
    else
        echo "===== Part 2: Modular examples (--bulk-input) ====="
    fi
    echo ""

    if [ ! -f "$CSV_FILE" ]; then
        echo "WARNING: CSV file not found: $CSV_FILE, skipping modular examples"
    else
        MODULAR_ROW_ARGS=()
        if [ "$TEST_TYPE" = "min" ]; then
            MODULAR_ROW_ARGS=(--row 2)
        elif [ "$TEST_TYPE" = "smoke" ]; then
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
            "${BATCH_WORKSPACE_ARGS[@]}" \
            "${BATCH_OUTPUT_ARGS[@]}" \
            "${BATCH_VENV_ARGS[@]}" \
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
# SFLOW_E2E_SKIP_INFMAX=1 drops this part only. The infmax benchmark suites are run by
# prenyx-ci-automation on real GB200, which owns that hardware plus the Slurm polling,
# result enrichment and public-data diffing. (The trigger lives in the internal pipeline;
# this file is mirrored to OSS, so it does not name the internal script.) Skipping them
# here keeps the two pipelines from submitting the same suites twice; Parts 1 and 2 (the
# sample suite) are unaffected, and that is the coverage this job exists for.
if [ "${SFLOW_E2E_SKIP_INFMAX:-}" = "1" ]; then
    echo ""
    echo "===== Part 3: infmax suites SKIPPED (SFLOW_E2E_SKIP_INFMAX=1; run by prenyx CI) ====="
    echo ""
elif [ "$TEST_TYPE" = "inf" ] || [ "$TEST_TYPE" = "m" ] || [ "$TEST_TYPE" = "a" ] || [ "$TEST_TYPE" = "smoke" ] || [ "$TEST_TYPE" = "min" ]; then
    echo ""
    if [ "$TEST_TYPE" = "min" ]; then
        echo "===== Part 3: Min infmax multi-node batch suite ====="
    elif [ "$TEST_TYPE" = "smoke" ]; then
        echo "===== Part 3: Smoke infmax multi-node batch suite ====="
    else
        echo "===== Part 3: Infmax multi-node batch suites ====="
    fi
    echo ""

    fetch_infmax_recipes
    if [ "$TEST_TYPE" = "min" ]; then
        submit_min_infmax_suite
    elif [ "$TEST_TYPE" = "smoke" ]; then
        submit_smoke_infmax_suites
    else
        submit_infmax_batch_suites
    fi
fi

submit_colon_task_script_e2e

# Real multi-backend run (two concurrent salloc via a direct `sflow run`). Done
# after the async submissions so its salloc job ids join JOB_IDS and flow through
# the shared wait (sacct) + validate loop below.
run_multi_backend_real

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
            echo "  $jid ($jobname): ${state:-UNKNOWN}  nodes: ${nnodes:-?}  elapsed: ${elapsed:-N/A}  output: (not found under $(abs_path "${E2E_OUTPUT_DIR:-sflow_output}"))"
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

# A GPU/CUDA *infrastructure* failure (driver / fabric-manager not ready on the
# node) is a machine issue, not a recipe/code regression. We detect the symbols
# such failures leave in the per-task logs so a would-be FAIL can be EXCUSED from
# the pass/fail threshold -- while emitting a loud warning so a broken node is
# never silently masked as a success.
cuda_infra_failure() {  # <out_dir> -> 0 (true) if the job failed due to CUDA infra
    local out_dir="$1"
    [ -n "$out_dir" ] && [ -d "$out_dir" ] || return 1
    # These are GB200/driver "system not ready" + missing-CUDA-runtime signatures.
    # 'system not yet initialized' covers both torch ('Error 802: system not yet
    # initialized') and cupy ('cudaErrorSystemNotReady: system not yet initialized').
    grep -rIqs --include='*.log' --include='*.out' \
        -e 'system not yet initialized' \
        -e 'cudaErrorSystemNotReady' \
        -e 'CUDA initialization: Unexpected error from cudaGetDeviceCount' \
        -e 'No CUDA runtime is found' \
        -e 'Failed to get device capability: Unexpected error from cudaGetDeviceCount' \
        "$out_dir"
}

# Reclassify a would-be FAIL as a CUDA-infra excuse: bump the excused counter and
# print a prominent warning. Excused jobs are NOT counted as failures by the CI
# threshold, but they did NOT succeed -- the node should be investigated/drained.
mark_cuda_excused() {  # <jid> <out_dir> <detail>
    CUDA_INFRA=$((CUDA_INFRA + 1))
    echo "  Job $1: CUDA-INFRA EXCUSED (GPU/driver init failed on node; NOT a real pass) $3"
    echo "  ⚠ WARNING: Job $1 failed due to a CUDA/GPU infrastructure error on its node (e.g. 'CUDA initialization: Unexpected error from cudaGetDeviceCount()' / 'Error 802: system not yet initialized'). Excused from the pass/fail threshold, but this is NOT a successful run -- investigate/drain the node. See $2"
}

# Check results in output folders
echo ""
echo "===== Results ====="
TOTAL=0
PASSED=0
CUDA_INFRA=0
for jid in "${JOB_IDS[@]}"; do
    TOTAL=$((TOTAL + 1))
    if is_multi_backend_result "$jid"; then
        # The driver workflow output dir is keyed by the submitted job id;
        # validate that task_a and task_b ran on distinct backend nodes.
        mb_out_dir=$(workflow_output_dir_for_job "$jid")
        if [ -n "$mb_out_dir" ] && multi_backend_run_ok "$mb_out_dir"; then
            PASSED=$((PASSED + 1))
            echo "  Job $jid: PASS (multi-backend; task_a/task_b on distinct backend nodes under $mb_out_dir)"
        else
            echo "  Job $jid: FAIL (multi-backend; tasks not on distinct backend nodes under ${mb_out_dir:-not found under $(abs_path "${E2E_OUTPUT_DIR:-sflow_output}")})"
        fi
        continue
    fi
    out_dir=$(workflow_output_dir_for_job "$jid")
    if [ -z "$out_dir" ]; then
        echo "  Job $jid: output folder not found under $(abs_path "${E2E_OUTPUT_DIR:-sflow_output}")"
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
        if cuda_infra_failure "$out_dir"; then
            mark_cuda_excused "$jid" "$out_dir" "('Successful requests: 0' with a CUDA init failure)"
        else
            echo "  Job $jid: FAIL ('Successful requests: 0' found in $out_dir)"
        fi
    elif [ "$count_aiperf_errors" -gt 0 ] || [ "$count_aiperf_valid" -gt 0 ] || [ "$count_any_success" -gt 0 ] || [ "$count_replica" -gt 0 ]; then
        PASSED=$((PASSED + 1))
        echo "  Job $jid: PASS (under $out_dir)"
    elif cuda_infra_failure "$out_dir"; then
        mark_cuda_excused "$jid" "$out_dir" "(no success indicator; CUDA init failure on node)"
    else
        echo "  Job $jid: FAIL (no success indicator found in $out_dir)"
    fi
done

if [ -n "${MULTI_BACKEND_LAUNCH_FAILED:-}" ]; then
    # The multi-backend run never produced a Slurm job id (no allocation granted),
    # so the results loop above did not visit it; count it as one failed job.
    TOTAL=$((TOTAL + 1))
    echo "  Multi-backend run: FAIL (no Slurm allocation granted)"
fi

echo ""
echo "===== Summary ====="
echo "$PASSED/$TOTAL jobs passed"
if [ "${CUDA_INFRA:-0}" -gt 0 ]; then
    echo "$CUDA_INFRA/$TOTAL jobs excused due to CUDA/GPU infrastructure failures (not counted as failures)"
    echo "⚠ WARNING: $CUDA_INFRA job(s) failed because of CUDA/GPU infrastructure errors on their nodes (driver/fabric not ready). They are EXCUSED from the pass/fail threshold but did NOT succeed -- investigate/drain the affected nodes."
fi

# =============================================================================
# Independent monitor coverage check
# =============================================================================
# --enable-workflow-monitor is injected for every submitted e2e workflow, so each
# run must emit <out_dir>/sflow_monitor.log AND that overview must carry a populated
# Metric Summary (a positive sample count + real numeric metric rows). Existence
# alone is NOT enough: a collector that started but never sampled (or whose samples
# failed to parse) would still leave a header-only/empty overview, which must count
# as a failure. Asserted INDEPENDENTLY of the job's own pass/fail, on its own line.

# Return 0 only when an sflow_monitor.log carries a meaningful monitor summary:
# non-empty file, a Metric Summary section, no empty-samples marker, a positive
# Samples count, and at least one numeric utilization metric row (cpu/gpu/memory
# come from /proc + nvidia-smi, so at least cpu_utilization_pct is always present).
monitor_log_has_content() {
    local f="$1"
    [ -s "$f" ] || return 1
    grep -q "Metric Summary" "$f" || return 1
    if grep -q "(no numeric samples collected)" "$f"; then
        return 1
    fi
    local samples
    samples=$(sed -n 's/^Samples[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$f" | head -1)
    [ -n "$samples" ] && [ "$samples" -gt 0 ] 2>/dev/null || return 1
    grep -Eq '(cpu_utilization_pct|gpu_utilization_pct|memory_used_pct)' "$f" || return 1
    return 0
}

echo ""
echo "===== Monitor Coverage ====="
MONITOR_TOTAL=0
MONITOR_PRESENT=0
MONITOR_MISSING_LABELS=""
for jid in "${JOB_IDS[@]}"; do
    MONITOR_TOTAL=$((MONITOR_TOTAL + 1))
    mon_out_dir=$(workflow_output_dir_for_job "$jid")
    mon_file=""
    if [ -n "$mon_out_dir" ]; then
        mon_file=$(find "$mon_out_dir" -maxdepth 2 -type f -name 'sflow_monitor.log' 2>/dev/null | head -1)
    fi
    if [ -z "$mon_file" ]; then
        if [ -n "$mon_out_dir" ]; then
            MONITOR_MISSING_LABELS="$MONITOR_MISSING_LABELS  - $jid (no sflow_monitor.log under $mon_out_dir)\n"
            echo "  Job $jid: MONITOR MISSING (no sflow_monitor.log under $mon_out_dir)"
        else
            MONITOR_MISSING_LABELS="$MONITOR_MISSING_LABELS  - $jid (output dir not found)\n"
            echo "  Job $jid: MONITOR MISSING (output dir not found)"
        fi
    elif monitor_log_has_content "$mon_file"; then
        MONITOR_PRESENT=$((MONITOR_PRESENT + 1))
        mon_samples=$(sed -n 's/^Samples[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$mon_file" | head -1)
        echo "  Job $jid: monitor overview OK ($mon_file; ${mon_samples:-?} samples)"
    else
        MONITOR_MISSING_LABELS="$MONITOR_MISSING_LABELS  - $jid (sflow_monitor.log present but no populated metric summary: $mon_file)\n"
        echo "  Job $jid: MONITOR EMPTY (sflow_monitor.log present but no populated metric summary: $mon_file)"
    fi
done
echo ""
echo "$MONITOR_PRESENT/$MONITOR_TOTAL workflows produced a monitor overview with metrics (sflow_monitor.log)"
if [ -n "$MONITOR_MISSING_LABELS" ]; then
    echo "Workflows missing a populated monitor overview:"
    echo -e "$MONITOR_MISSING_LABELS"
fi

# =============================================================================
# Disagg monitor targeting check
# =============================================================================
# The slurm *disagg* recipes attach a monitor to the `benchmark` task with
# resources.used_by_tasks: [prefill_server, decode_server]. This asserts the
# generated cross reports (<server>__monitored_by__benchmark) sampled the
# SERVERS' resources (matching each server's own monitor view, and NOT the
# benchmark client's), so used_by_tasks targeting is verified end to end.
# Only workflows that produced *__monitored_by__benchmark reports are checked;
# all others are skipped. Reported on its own parseable line, independent of the
# job pass/fail and the monitor-coverage gate above.
DISAGG_MONITOR_CHECK="$SCRIPT_DIR/check_disagg_monitor.py"
echo ""
echo "===== Monitor Targeting (disagg used_by_tasks) ====="
TARGETING_TOTAL=0
TARGETING_OK=0
TARGETING_FAIL_LABELS=""
for jid in "${JOB_IDS[@]}"; do
    tgt_out_dir=$(workflow_output_dir_for_job "$jid")
    [ -n "$tgt_out_dir" ] || continue
    # Only disagg used_by_tasks runs emit *__monitored_by__benchmark reports.
    if ! find "$tgt_out_dir" -type d -name '*__monitored_by__benchmark' 2>/dev/null | grep -q .; then
        continue
    fi
    TARGETING_TOTAL=$((TARGETING_TOTAL + 1))
    if python3 "$DISAGG_MONITOR_CHECK" "$tgt_out_dir"; then
        TARGETING_OK=$((TARGETING_OK + 1))
        echo "  Job $jid: monitor targeting OK ($tgt_out_dir)"
    else
        tgt_rc=$?
        if [ "$tgt_rc" -eq 2 ]; then
            # Not actually applicable after all -> un-count.
            TARGETING_TOTAL=$((TARGETING_TOTAL - 1))
        else
            TARGETING_FAIL_LABELS="$TARGETING_FAIL_LABELS  - $jid ($tgt_out_dir)\n"
            echo "  Job $jid: MONITOR TARGETING FAIL (monitor sampled the wrong resources under $tgt_out_dir)"
        fi
    fi
done
echo ""
echo "$TARGETING_OK/$TARGETING_TOTAL disagg workflows monitored the correct (server) resources"
if [ -n "$TARGETING_FAIL_LABELS" ]; then
    echo "Disagg workflows whose monitor sampled the wrong resources:"
    echo -e "$TARGETING_FAIL_LABELS"
fi
