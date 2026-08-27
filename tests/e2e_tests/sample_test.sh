#!/bin/bash

# set -x

usage() {
    echo "Usage: $0 -p <partition> -A <account> -m <model_path> [-G <gpus_per_node>] [-t s|m|inf|a|smoke|min|one] [--submit] [--check JOB_IDS] [-- <extra args>]"
    echo ""
    echo "  -t s   Self-contained examples only (--bulk-submit examples/self_contained/slurm/)"
    echo "  -t m   Modular examples only (--bulk-input modular/inference_x_v2/bulk_input.csv)"
    echo "  -t inf Infmax multi-node batch suites only"
    echo "  -t a   Both single and multi (default)"
    echo "  -t smoke Curated Slurm smoke subset with broad coverage"
    echo "  -t min Minimal representative set (one job per validation type)"
    echo "  -t one EXACTLY ONE Slurm job -- plumbing smoke, not coverage."
    echo "         Use it to prove a CI/cluster path end to end (submit -> run ->"
    echo "         sflow_output -> summary) before spending nodes on min/smoke."
    echo ""
    echo "  SFLOW_E2E_RECIPE_CLASS=workload|sanity|all (env, default all)"
    echo "         Which HALF of the suite to submit. 'workload' is the"
    echo "         dynamo/trtllm/vllm/sglang/aiperf/infmax recipes (real servers,"
    echo "         model loads, big pulls); 'sanity' is everything else and needs"
    echo "         no model at all. The two halves run on different clusters."
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

workflow_summary_ok() {
    # Every indicator below comes from a benchmark or client log, so a recipe that
    # ships neither (monitor_mixed, gpu_indices, ...) scored FAIL however green it
    # was. sflow already writes the authoritative verdict: sflow_summary.log says
    # `Status : COMPLETED` only when every task reached COMPLETED/READY
    # (core/execution_summary.py::_infer_status), and FAILED/CANCELLED/TIMEOUT on
    # any other outcome. Judge those runs by that instead of per-recipe allowlists.
    local dir="$1"
    [ -n "$dir" ] || return 1
    local summary
    summary=$(find "$dir" -maxdepth 2 -type f -name 'sflow_summary.log' 2>/dev/null | head -1)
    [ -n "$summary" ] || return 1
    grep -Eq '^Status[[:space:]]*:[[:space:]]*COMPLETED[[:space:]]*$' "$summary"
}

aiperf_tally_ok() {  # <run_dir> -> 0 every aiperf run benchmarked, 1 one did not, 2 no aiperf here
    # A benchmark that measured NOTHING is the one failure this suite could not see.
    # aiperf 0.3.0 exits 0 even when every single request failed, so the task rc is
    # 0, sflow's own `Status : COMPLETED` is green, and the run is scored PASS while
    # its CSV holds nothing but `Error Request Count`.
    #
    # Not hypothetical: dynamo >= 1.3.0 dropped `ignore_eos` from its (strictly
    # deserialized) NvExt struct, so one stale `--extra-inputs` 400'd all 1024
    # requests in six workflows -- and this suite printed "11/11 jobs passed".
    #
    # aiperf states the outcome itself, and it is unambiguous:
    #     Processed 1024 valid requests and 0 errors (1024 total).   <- benchmarked
    #     Processed 0 valid requests and 1024 errors (1024 total).   <- measured nothing
    #
    # Both markers this replaces were wrong. "0 valid" was read as a SUCCESS marker
    # when it is precisely what a total failure prints -- that alone scored the six
    # dead runs green -- and unanchored "0 errors" also matches "10 errors" and
    # "1000 errors", so a mostly-failed run passed too.
    #
    # Returns 2, not 0, when no aiperf ran: "no evidence" is a different answer from
    # "good evidence", and only the caller knows whether this recipe owed any. Same
    # lesson as workload_placement_ok() -- a checker that passes on an empty
    # directory scores a workflow that died before it started as a success.
    local dir="$1" valid errors seen=0 bad=0
    [ -n "$dir" ] && [ -d "$dir" ] || return 2
    # grep -o pins the field positions, so the split below cannot drift:
    #   Processed <valid> valid requests and <errors> errors
    while read -r _ valid _ _ _ errors _; do
        seen=$((seen + 1))
        [ "${valid:-0}" -gt 0 ] && [ "${errors:-1}" -eq 0 ] && continue
        bad=$((bad + 1))
        echo "  AIPERF MEASURED NOTHING: ${valid:-?} valid / ${errors:-?} errors under $dir" >&2
    done < <(find "$dir" -type f -name '*.log' \
        -exec grep -hoE 'Processed [0-9]+ valid requests and [0-9]+ errors' {} + 2>/dev/null)
    [ "$seen" -gt 0 ] || return 2
    [ "$bad" -eq 0 ]
}

serving_tally_ok() {  # <run_dir> -> 0 every benchmark_serving run completed, 1 one did not, 2 none here
    # The benchmark_serving.py (InferenceX) half of the same question aiperf_tally_ok
    # asks. The modular dynamo_benchmark rows drive this instead of aiperf, so
    # without it they keep the old, far weaker gate.
    #
    # What the old markers did: PASS on `grep -l "Successful requests:"` -- the mere
    # PRESENCE of the string -- and FAIL only on `Successful requests:\s+0\s*$`.
    # So a run that completed 3 of 512 requests scored a clean PASS, and any
    # zero-count formatted differently (trailing text, padding) slipped the FAIL too.
    #
    # benchmark_serving.py prints its own summary, and the run's own command line is
    # in the same log, so the two can be cross-checked:
    #     python3 ... benchmark_serving.py ... --num-prompts 128 ...
    #     ============ Serving Benchmark Result ============
    #     Successful requests:                     128
    #     Output token throughput (tok/s):         4132.77
    #
    # A real benchmark therefore owes three things: it succeeded at all (got > 0), it
    # generated tokens (throughput > 0 -- a run can "succeed" 512 times with empty
    # responses, which is the same measured-nothing shape ignore_eos produced), and
    # it completed the work it was ASKED for (got == --num-prompts). Every healthy
    # run in CI matches exactly: 16/16, 32/32, 48/48, 128/128, 256/256, 512/512.
    #
    # want == 0 means no command line was captured in this log; the cross-check is
    # then skipped rather than guessed at. Returns 2 for "no benchmark_serving here"
    # for the same reason aiperf_tally_ok does -- no evidence is not good evidence.
    local dir="$1" f seen=0 bad=0
    [ -n "$dir" ] && [ -d "$dir" ] || return 2
    while IFS= read -r f; do
        seen=$((seen + 1))
        awk -v src="$f" '
            match($0, /--num-prompts[= ]+[0-9]+/) {
                s = substr($0, RSTART, RLENGTH); gsub(/[^0-9]/, "", s); want = s + 0
            }
            /Successful requests:/                { got  = $NF + 0 }
            /Output token throughput \(tok\/s\):/ { thpt = $NF + 0 }
            END {
                if (got > 0 && thpt > 0 && (want == 0 || got == want)) exit 0
                printf "  BENCHMARK INCOMPLETE: %d/%d requests succeeded, %g tok/s -- %s\n", \
                       got, want, thpt, src > "/dev/stderr"
                exit 1
            }
        ' "$f" || bad=$((bad + 1))
    done < <(find "$dir" -type f -name '*.log' \
        -exec grep -l 'Serving Benchmark Result' {} + 2>/dev/null)
    [ "$seen" -gt 0 ] || return 2
    [ "$bad" -eq 0 ]
}

recipe_is_client_only() {  # <run_dir> -> 0 when the recipe starts no server of its own
    # aiperf_template is a TEMPLATE: ONE CPU-only client task aimed at
    # ${HEAD_NODE_IP}:8000, an endpoint it deliberately does NOT start -- the reader
    # is meant to point it at a server they already run. Played standalone in CI
    # nothing is listening, every request is ConnectionRefused, and aiperf cannot
    # benchmark. That is the recipe working as designed, not a regression.
    #
    # So the claim here is deliberately narrow. Such a run is still expected to
    # COMPLETE -- sflow `Status : COMPLETED`, benchmark task exit=0 -- it simply
    # owes no metrics. Every OTHER recipe in the workload half still owes a real
    # benchmark, which is the whole point of aiperf_tally_ok().
    #
    # Structural rather than a name allowlist: a workflow declaring a single task
    # cannot have started the server it benchmarks. aiperf_template declares 1;
    # every serving recipe here declares 6-7 (servers + frontend + benchmark), and
    # the modular compositions more. The recipe is copied into the run directory,
    # so this reads what actually ran rather than what is on disk now.
    local dir="$1" total=0 n yml
    [ -n "$dir" ] && [ -d "$dir" ] || return 1
    for yml in "$dir"/*.y*ml; do
        [ -f "$yml" ] || continue
        n=$(sed -n '/^  tasks:/,$p' "$yml" | grep -cE '^    - name:')
        total=$((total + n))
    done
    [ "$total" -eq 1 ]
}

gpu_placement_run_ok() {
    # The placement matrix ships no benchmark: its result IS its assertions. Every
    # task resolves its planned HOST indices through a per-node bare-metal
    # index -> UUID map and exits non-zero on a mismatch, and verify_disjoint sits
    # downstream of all of them -- so its OK line can only appear when every
    # assertion passed on every node.
    #
    # Only per-task logs are searched (mindepth 2): sflow.log echoes each task's
    # script verbatim, FAIL: branches included, so a recursive grep would match
    # text that never ran.
    local dir="$1"
    [ -n "$dir" ] || return 1
    find "$dir" -mindepth 2 -maxdepth 2 -type f -name '*.log' \
        -exec grep -lF "OK: concurrent tasks held disjoint GPUs" {} + 2>/dev/null |
        grep -q . || return 1
    # And the multi-node cases must actually have spanned nodes. A one-node
    # allocation degenerates them into duplicates of the single-node cases, which
    # still pass -- silently dropping the coverage they exist for. Each assertion
    # names the node it proved, so two distinct names is the proof.
    local task nodes found=0
    for task in "$dir"/*multinode*/; do
        [ -d "$task" ] || continue
        found=1
        # Per multi-node TASK. A union over every task in the run is not the same
        # claim: two single-node tasks landing on different nodes satisfy it while
        # both multi-node cases sat on one.
        nodes=$(find "$task" -maxdepth 1 -type f -name '*.log' \
            -exec grep -hoE 'OK: [^ ]+ holds host GPU' {} + 2>/dev/null | sort -u | wc -l)
        [ "${nodes:-0}" -ge 2 ] || return 1
    done
    [ "$found" = 1 ]
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

if [ "$TEST_TYPE" != "s" ] && [ "$TEST_TYPE" != "m" ] && [ "$TEST_TYPE" != "inf" ] && [ "$TEST_TYPE" != "a" ] && [ "$TEST_TYPE" != "smoke" ] && [ "$TEST_TYPE" != "min" ] && [ "$TEST_TYPE" != "one" ]; then
    echo "ERROR: -t must be 's', 'm', 'inf', 'a', 'smoke', 'min', or 'one', got '$TEST_TYPE'"
    usage
fi

# Which HALF of the suite this run submits. A class, not a new -t mode, so it
# composes with every existing mode instead of multiplying them.
#
#   workload -- dynamo / trtllm / vllm / sglang / infmax / aiperf. Real servers,
#               real model loads, multi-GB image pulls, many GPU-hours. Belongs on
#               the cluster that has the quota for it.
#   sanity   -- everything else: GPU placement, replicas, resource release, the
#               monitor and multi-backend recipes. No framework and no model --
#               they only DECLARE LocalModelPath so the harness's `-a` override is
#               accepted, and never read it -- so this half runs on any cluster
#               with GPUs and is cheap enough to play constantly.
#   all      -- both. The default, so a manual or local run is unchanged.
RECIPE_CLASS="${SFLOW_E2E_RECIPE_CLASS:-all}"
if [ "$RECIPE_CLASS" != "all" ] && [ "$RECIPE_CLASS" != "workload" ] && [ "$RECIPE_CLASS" != "sanity" ]; then
    echo "ERROR: SFLOW_E2E_RECIPE_CLASS must be 'all', 'workload' or 'sanity', got '$RECIPE_CLASS'"
    exit 1
fi

is_workload_recipe() {
    # Named by the framework they drive, which is exactly the line the split is
    # drawn on. Everything else is a functionality check.
    case "$(basename "$1")" in
        dynamo_*|trtllm_*|sglang_*|vllm_*|infmax_*|aiperf_*) return 0 ;;
    esac
    return 1
}

recipe_in_class() {
    # -t one is a single named plumbing smoke (the GPU placement matrix), and its
    # job is to prove THIS cluster's path end to end. Both halves want that, so it
    # is never filtered out.
    [ "$TEST_TYPE" = "one" ] && return 0
    case "$RECIPE_CLASS" in
        workload) is_workload_recipe "$1" ;;
        sanity) ! is_workload_recipe "$1" ;;
        *) return 0 ;;
    esac
}

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
    # Functionality check, no framework: it belongs to the sanity half -- which
    # runs via `sflow run` (run_sanity_recipes_with_sflow_run), not batch. So this
    # batch-based path is only for the combined default.
    [ "$RECIPE_CLASS" != "all" ] && return

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
    # Built as an array rather than a nested ${a:-${b:-...}} default: with no nodes
    # to exclude that expression collapses to a bare `--exclude=`, which is not the
    # same as passing nothing.
    COLON_EXTRA_ARGS=()
    if [ -n "${SFLOW_COLON_SCRIPT_EXTRA_ARGS:-}" ]; then
        COLON_EXTRA_ARGS=(-e "$SFLOW_COLON_SCRIPT_EXTRA_ARGS")
    elif [ -n "${SLURM_E2E_EXCLUDE_NODES:-}" ]; then
        COLON_EXTRA_ARGS=(-e "--exclude=$SLURM_E2E_EXCLUDE_NODES")
    fi
    COLON_SEGMENT="${SLURM_E2E_SEGMENT:-}"
    if [ "$COLON_SEGMENT" = "auto" ]; then
        COLON_SEGMENT='${{SLURM_NODES}}'
    fi
    if [ -n "$COLON_SEGMENT" ]; then
        COLON_EXTRA_ARGS+=(-e "--segment=$COLON_SEGMENT")
    fi

    colon_output=$(sflow batch -f "$SFLOW_COLON_SCRIPT_FIXTURE" \
        -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
        "${BATCH_WORKSPACE_ARGS[@]}" \
        "${BATCH_VENV_ARGS[@]}" \
        --output-dir "$colon_dir" \
        --job-name "colon_in_task_script" \
        --enable-workflow-monitor \
        "${COLON_EXTRA_ARGS[@]}" \
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

# Strip a suite-wide `-e --segment=...` for ONE `sflow batch` call, into
# STRIPPED_SEGMENT_ARGS. That value is an sflow expression sized to a recipe's own
# SLURM_NODES, so a config WITHOUT that variable passes it to sbatch VERBATIM as
# "#SBATCH --segment=${{SLURM_NODES}}". Both multi-backend recipes are in that
# position (they fix their backends' node counts instead), and a single salloc-wide
# segment could not be right for two differently sized backends anyway.
#
# One copy, two callers on purpose: a second hand-written twin of this predicate
# drifting is how the verbatim expression reached sbatch to begin with.
strip_suite_segment_args() {
    STRIPPED_SEGMENT_ARGS=()
    local i=0
    local n=${#EXTRA_BATCH_ARGS[@]}
    while [ "$i" -lt "$n" ]; do
        local arg="${EXTRA_BATCH_ARGS[$i]}"
        local next=""
        if [ $((i + 1)) -lt "$n" ]; then
            next="${EXTRA_BATCH_ARGS[$((i + 1))]}"
        fi
        case "$arg:$next" in
            "-e:--segment="*)
                i=$((i + 2))
                continue
                ;;
        esac
        STRIPPED_SEGMENT_ARGS+=("$arg")
        i=$((i + 1))
    done
}

# The second partition every two-backend recipe needs. Defaults to the partition
# this run was given: hardcoded names are cluster-specific and simply do not exist
# elsewhere -- the stale genesisq / gamoraq defaults meant every submission on a new
# cluster died with "no Slurm allocation granted". Two DIFFERENT partitions are
# better coverage, so CI sets SLURM_E2E_PARTITION_B (ptyche: backfill) and an
# operator can still override per recipe.
E2E_PARTITION_B="${SLURM_E2E_PARTITION_B:-$PARTITION}"

# =============================================================================
# The SANITY half: one `sflow run` per recipe, concurrently, no sbatch.
# =============================================================================
# `sflow batch --submit` sbatches a DRIVER that runs ON a compute node, so the
# checkout, the venv and the output dir all have to be visible from there. On a
# cluster whose login-node $HOME is not exported to the compute nodes that is
# simply impossible. `sflow run` keeps the driver on the login node and only
# srun's the task steps out, which is why this half does not batch.
#
# It also makes the verdict honest. A batched job is scored by hunting for a
# "success indicator" in its logs; a subprocess just has an EXIT CODE, and sflow
# already exits non-zero when any task fails. So none of the indicator guessing
# applies here -- rc is the answer.
#
# NOTE: recipes whose tasks exchange files through $SFLOW_TASK_OUTPUT_DIR
# (gpu_placement_matrix reads the node -> UUID map written by another task, and
# verify_disjoint compares two tasks' device lists) still need the OUTPUT DIR on
# storage the compute nodes share. Point -o/E2E_OUTPUT_DIR at shared scratch;
# only the driver moved, the task steps still run out on the nodes.
sanity_recipe_set_args() {
    # `--set` of a variable a config does not declare is a hard error, and so is
    # `--artifact` of an artifact it does not declare -- which is why these are
    # built per recipe rather than shared. The colon fixture needs no model and
    # declares no artifacts at all, so handing it the standard
    # `-a LOCAL_MODEL_PATH=` killed it before the workflow started:
    #   "Artifact 'LOCAL_MODEL_PATH' specified in overrides is not defined".
    if grep -qE '^[[:space:]]*-?[[:space:]]*name:[[:space:]]*LOCAL_MODEL_PATH' "$1"; then
        printf '%s\n' "--artifact" "LOCAL_MODEL_PATH=fs://$MODEL_PATH"
    fi
    # Cap the Slurm allocation. This is the ONLY cap that bites: a recipe's
    # workflow-level `timeout:` is accepted by the schema and enforced by nothing
    # (see TaskConfig.timeout), so without --time a server that never becomes
    # ready holds its nodes for the recipe's own limit -- up to 120 minutes for
    # the workload recipes, which on this cluster cannot serve at all and will
    # always wait the full time.
    #
    # Grepped, not assumed, for the same reason as the model artifact above:
    # `--set` of a variable a config does not declare is a hard error, and the
    # two spellings are NOT interchangeable -- multi_backend.yaml calls it
    # TIME_LIMIT, everything else SLURM_TIMELIMIT, and the colon fixture declares
    # neither. Unset means "leave each recipe's own value alone".
    if [ -n "${SFLOW_E2E_SLURM_TIMELIMIT:-}" ]; then
        if grep -qE '^[[:space:]]*SLURM_TIMELIMIT:' "$1"; then
            printf '%s\n' "--set" "SLURM_TIMELIMIT=$SFLOW_E2E_SLURM_TIMELIMIT"
        elif grep -qE '^[[:space:]]*TIME_LIMIT:' "$1"; then
            printf '%s\n' "--set" "TIME_LIMIT=$SFLOW_E2E_SLURM_TIMELIMIT"
        fi
    fi
    case "$(basename "$1")" in
        multi_backend.yaml)
            printf '%s\n' "--set" "PARTITION_A=$PARTITION" \
                           "--set" "PARTITION_B=${E2E_PARTITION_B:-$PARTITION}" \
                           "--set" "SLURM_ACCOUNT=$ACCOUNT"
            ;;
        monitor_mixed.yaml)
            printf '%s\n' "--set" "PARTITION_A=$PARTITION" \
                           "--set" "PARTITION_B=${E2E_PARTITION_B:-$PARTITION}" \
                           "--set" "SLURM_ACCOUNT=$ACCOUNT" \
                           "--set" "GPUS_PER_NODE=$GPUS_PER_NODE"
            ;;
        *)
            printf '%s\n' "--set" "SLURM_PARTITION=$PARTITION" \
                           "--set" "SLURM_ACCOUNT=$ACCOUNT" \
                           "--set" "GPUS_PER_NODE=$GPUS_PER_NODE"
            ;;
    esac
}

gpu_placement_verified() {  # <out_dir> -> 0 when every GPU task PROVED its placement
    # "The workflow completed" says nothing about WHICH cards it used, and until
    # now only gpu_placement_matrix checked that -- every other recipe (gpu_indices
    # pins devices! auto_replica and resource_release_after pack and re-use them)
    # was scored on completion alone, and the workload half checked no placement at
    # all.
    #
    # Every GPU task now leaves sflow_gpus.log recording what it was PLANNED for
    # and what it actually SELECTED, both as physical UUIDs, so this audits any
    # recipe without the recipe having to assert anything.
    #
    # A task whose record says `fallback`/`unverified` did not prove its placement:
    # the driver could not resolve the plan to UUIDs (probe failed, node names
    # disagree, gpus_per_node larger than the node really has) or the step had no
    # nvidia-smi. On these clusters that is a regression, not a normal mode -- and
    # it is precisely the silent degradation this suite exists to catch.
    local dir="$1"
    [ -n "$dir" ] && [ -d "$dir" ] || return 0
    local rec action planned selected total=0 unproven=0 verified=0
    while IFS= read -r rec; do
        [ -n "$rec" ] || continue
        total=$((total + 1))
        action=$(sed -n 's/^action=//p' "$rec" | head -1)
        planned=$(sed -n 's/^planned_uuids=//p' "$rec" | head -1)
        # The UUID of each device CUDA will really use, in the order it will see them.
        selected=$(sed -n 's/^selected=[0-9?]* //p' "$rec" | paste -sd, -)
        case "$action" in
            verified) verified=$((verified + 1)) ;;
            *)
                # `fallback` (no UUID map, or Slurm granted cards the plan never
                # named) and `unverified` (no nvidia-smi) both mean the placement
                # was not proven against physical devices.
                unproven=$((unproven + 1))
                echo "  GPU PLACEMENT UNPROVEN: $(dirname "$rec" | xargs basename) recorded action='${action:-none}' in $rec" >&2
                continue
                ;;
        esac
        if [ "$selected" != "$planned" ]; then
            unproven=$((unproven + 1))
            echo "  GPU PLACEMENT MISMATCH: $(dirname "$rec" | xargs basename) planned '$planned' but holds '$selected' ($rec)" >&2
        fi
    done < <(find "$dir" -type f -name 'sflow_gpus*.log' 2>/dev/null | sort)

    if [ "$total" -eq 0 ]; then
        # No GPU task in this workflow (or none reached the prelude). Nothing to
        # prove; the recipe's own verdict still applies.
        return 0
    fi
    echo "  GPU placement: $verified/$total task record(s) proven by UUID"
    [ "$unproven" -eq 0 ]
}

recipe_requests_gpus() {  # <recipe.yaml> -> 0 when a TASK asks for GPUs
    # A `gpus:` block inside a task's `resources:`. Deliberately NOT
    # `gpus_per_node:`, which is the BACKEND's allocation shape -- a recipe can
    # size an allocation and still run nothing on a GPU. aiperf_template is
    # exactly that: one CPU-only benchmark-client task (`resources: nodes:`),
    # gpus_per_node on the backend, and no GPU task anywhere.
    grep -qE '^[[:space:]]+gpus:[[:space:]]*$' "$1"
}

workload_placement_ok() {  # <recipe> <run_dir> -> 0 when placement is PROVEN
    # The verdict for a recipe whose application is expected to fail. Placement is
    # the only claim, so it must be a POSITIVE one: gpu_placement_verified()
    # answers 0 when it finds no records at all -- correct for a CPU-only recipe,
    # badly wrong here, because a workload that died before any task started would
    # score PASS on zero evidence. Require records to exist, then require every
    # one of them to be proven.
    local recipe="$1" dir="$2"
    [ -n "$dir" ] && [ -d "$dir" ] || return 1
    # No GPU task => no placement to prove, ever. Demanding a record here would
    # fail such a recipe on every run forever, which is what happened to
    # aiperf_template: it completed cleanly and was scored FAIL for producing
    # evidence it structurally cannot produce. Fall back to its own verdict --
    # the only signal that means anything for a recipe holding no GPU.
    if ! recipe_requests_gpus "$recipe"; then
        workflow_summary_ok "$dir"
        return
    fi
    # It DOES ask for GPUs, so a missing record means no GPU task ever reached the
    # placement prelude -- the workflow died first. That is unproven, not passing.
    find "$dir" -type f -name 'sflow_gpus*.log' -print -quit 2>/dev/null | grep -q . || return 1
    gpu_placement_verified "$dir"
}

sanity_recipe_content_ok() {  # <name> <run_dir> -> 0 when the run PROVED itself
    # rc == 0 only says every task exited 0. It does NOT say the workflow did the
    # thing it exists to prove, and these recipes exist to prove something:
    #   * gpu_placement_matrix can exit 0 while its assertions never ran, or while
    #     a one-node allocation quietly collapsed the multi-node cases -- so read
    #     the by-UUID evidence and the 2-distinct-node proof.
    #   * multi_backend can exit 0 with both backends on the SAME node, which is
    #     precisely the binding it is meant to disprove.
    #   * the colon fixture can exit 0 without ever emitting its markers.
    #   * everything else: sflow's own Status must say COMPLETED.
    local name="$1" dir="$2"
    [ -n "$dir" ] && [ -d "$dir" ] || return 1
    case "$name" in
        gpu_placement_matrix) gpu_placement_run_ok "$dir" ;;
        multi_backend*)       multi_backend_run_ok "$dir" ;;
        colon_in_task_script) colon_task_log_has_markers "$dir" ;;
        *)                    workflow_summary_ok "$dir" ;;
    esac
}

sflow_run_safe_args() {
    # EXTRA_BATCH_ARGS is built for `sflow batch`; two kinds of it do not carry
    # over to `sflow run`.
    #
    #   --sbatch-output / --sbatch-error name the sbatch JOB's stdout/stderr
    #     files, and there is no sbatch job here: the driver runs on the login
    #     node and its output is already captured per recipe. `sflow run` rejects
    #     them outright ("No such option: --sbatch-output"), which killed every
    #     sanity recipe before it started.
    #   -e <flag> is this harness's channel for raw Slurm submission flags
    #     (--segment=, --exclude=). On `sflow run` those are properties of the
    #     ALLOCATION, so they belong to --extra-salloc-args rather than the
    #     generic --extra-args. In practice the list is usually empty here:
    #     --segment is a GB200 requirement and the sanity cluster does not set it.
    local out=() skip=0 want_salloc=0 a
    for a in "$@"; do
        if [ "$skip" = 1 ]; then skip=0; continue; fi
        if [ "$want_salloc" = 1 ]; then
            want_salloc=0
            out+=("--extra-salloc-args" "$a")
            continue
        fi
        case "$a" in
            --sbatch-*=*) ;;        # value is inline; drop this token only
            --sbatch-*) skip=1 ;;   # value is the NEXT token; drop both
            -e|--extra-args) want_salloc=1 ;;
            *) out+=("$a") ;;
        esac
    done
    [ ${#out[@]} -eq 0 ] || printf '%s\n' "${out[@]}"
}

run_sanity_recipes_with_sflow_run() {
    local recipes=()
    local f
    for f in "$EXAMPLES_DIR"/self_contained/slurm/*.yaml; do
        [ "$TEST_TYPE" != "one" ] || [ "${f##*/}" = "gpu_placement_matrix.yaml" ] || continue
        if recipe_in_class "$f"; then
            recipes+=("$f")
        elif [ "${SFLOW_E2E_INCLUDE_WORKLOAD_PLACEMENT:-0}" = "1" ] && is_workload_recipe "$f"; then
            # Workload recipes on a cluster that cannot actually serve. They ride
            # `sflow run` (not batch --submit) because that is the only path this
            # cluster supports, and they are judged on PLACEMENT ALONE -- see the
            # verdict below.
            recipes+=("$f")
        fi
    done
    # The colon-in-task-script fixture is written by full_sample_tests.sh; it is a
    # functionality check like the rest, so it rides this half when present.
    if [ -n "${SFLOW_COLON_SCRIPT_FIXTURE:-}" ] && [ -f "$SFLOW_COLON_SCRIPT_FIXTURE" ]; then
        recipes+=("$SFLOW_COLON_SCRIPT_FIXTURE")
    fi
    if [ ${#recipes[@]} -eq 0 ]; then
        echo "ERROR: no sanity recipes selected"
        exit 1
    fi

    echo ""
    local par_note="all at once"
    [ "${SFLOW_E2E_MAX_PARALLEL:-0}" -gt 0 ] && par_note="${SFLOW_E2E_MAX_PARALLEL} at a time"
    echo "===== Sanity half: ${#recipes[@]} recipe(s) via \`sflow run\` (no sbatch), $par_note ====="
    echo ""

    # All at once by default. Every recipe is its own allocation, so the cluster's
    # own scheduler is what orders them -- a starved salloc queues rather than
    # failing, and the Slurm --time cap (SFLOW_E2E_SLURM_TIMELIMIT) is what stops
    # a hung one holding nodes. SFLOW_E2E_MAX_PARALLEL>0 throttles to waves for a
    # cluster where that is not welcome; 0/unset means no limit.
    local names=() logs=() roots=() kinds=() files=()
    local max_par="${SFLOW_E2E_MAX_PARALLEL:-0}"
    for f in "${recipes[@]}"; do
        local name set_args=()
        name=$(basename "$f" .yaml)
        mapfile -t set_args < <(sanity_recipe_set_args "$f")
        local run_args=()
        mapfile -t run_args < <(sflow_run_safe_args "${EXTRA_BATCH_ARGS[@]+"${EXTRA_BATCH_ARGS[@]}"}")
        local root="$E2E_OUTPUT_DIR/$name"
        local log="$root/${name}.sflow_run.log"
        rm -rf "$root"
        mkdir -p "$root"
        # Guarded on >0: `-ge 0` is always true, and `wait -n` with no children
        # returns immediately, so an unguarded loop would spin instead of launch.
        if [ "$max_par" -gt 0 ]; then
            while [ "$(jobs -rp | wc -l)" -ge "$max_par" ]; do wait -n; done
        fi
        echo "  launching $name"
        # Each run records its OWN exit status. `wait -n` above reaps children as
        # they finish, so a later `wait $pid` would hit "not a child of this
        # shell" and report 127 for a run that actually passed.
        (
            sflow run -f "$f" \
                "${set_args[@]}" \
                --output-dir "$root" \
                --enable-workflow-monitor \
                "${run_args[@]+"${run_args[@]}"}" \
                > "$log" 2>&1
            echo $? > "$root/.rc"
        ) &
        names+=("$name")
        logs+=("$log")
        roots+=("$root")
        files+=("$f")
        if is_workload_recipe "$f"; then kinds+=("workload"); else kinds+=("sanity"); fi
    done
    wait

    echo ""
    echo "===== Scoring ${#names[@]} sflow run(s) ====="
    local i rc
    for i in "${!names[@]}"; do
        # Missing .rc means the subshell never got to write one -- treat as failure.
        rc=$(cat "${roots[$i]}/.rc" 2>/dev/null || echo 1)
        # No job id to look a run up by later, so find where it landed; the
        # content check and the monitor gate below both read it.
        local run_dir
        run_dir=$(ls -d "${roots[$i]}"/*/ 2>/dev/null | head -1)
        SANITY_RUN_DIRS+=("${run_dir:-$E2E_OUTPUT_DIR/${names[$i]}-NOT-FOUND}")
        SANITY_RUN_NAMES+=("${names[$i]}")

        TOTAL=$((TOTAL + 1))
        if [ "${kinds[$i]}" = "workload" ]; then
            # This cluster's GPUs cannot run real LLM inference, so the framework
            # WILL fail and its own verdict answers nothing. These recipes are here
            # for one reason -- to prove GPU placement on a second cluster and a
            # second GPU generation -- so that is the entire test. The app's exit
            # status is deliberately ignored; a placement regression is not.
            # Say which of the two things actually happened. workload_placement_ok
            # passes for two different reasons and one message for both CLAIMED
            # EVIDENCE THAT DOES NOT EXIST: aiperf_template holds no GPU, wrote no
            # record, and still reported "placement proven by UUID" -- the exact
            # kind of line that misleads whoever audits these artifacts later.
            local proved="placement proven by UUID; app rc=$rc ignored on this cluster"
            local unproved="GPU placement not proven"
            if ! recipe_requests_gpus "${files[$i]}"; then
                proved="no GPU task, so no placement to prove; sflow reports COMPLETED"
                unproved="no GPU task to place, and sflow's own verdict is not COMPLETED"
            fi
            # Say what aiperf actually measured here, without gating on it. This
            # cluster cannot serve a real model, so demanding a benchmark would fail
            # these recipes forever -- but a bare PASS next to an aiperf that
            # measured nothing is how the ptyche half stayed green for six workflows.
            # Whoever audits these artifacts should not have to open the CSV to find
            # that out. aiperf_tally_ok() already prints the counts to stderr.
            aiperf_tally_ok "$run_dir"
            case $? in
                0) proved="$proved; aiperf benchmarked" ;;
                1) proved="$proved; aiperf measured nothing (expected here, not gated)" ;;
            esac
            if workload_placement_ok "${files[$i]}" "$run_dir"; then
                PASSED=$((PASSED + 1))
                echo "  ${names[$i]}: PASS ($proved)"
            else
                echo "  ${names[$i]}: FAIL ($unproved; see ${run_dir:-${logs[$i]}})"
            fi
        elif [ "$rc" -ne 0 ]; then
            if cuda_infra_failure "${roots[$i]}"; then
                mark_cuda_excused "${names[$i]}" "${logs[$i]}" "(sflow run rc=$rc with a CUDA init failure)"
            else
                echo "  ${names[$i]}: FAIL (sflow run exited $rc; see ${logs[$i]})"
            fi
        elif sanity_recipe_content_ok "${names[$i]}" "$run_dir" \
             && gpu_placement_verified "$run_dir"; then
            PASSED=$((PASSED + 1))
            echo "  ${names[$i]}: PASS (rc=0 and its own output proves it, under $run_dir)"
        elif cuda_infra_failure "${roots[$i]}"; then
            mark_cuda_excused "${names[$i]}" "${run_dir:-${logs[$i]}}" "(exited 0 but proved nothing; CUDA init failure on node)"
        else
            # The nastiest shape: green process, unproven run. Exactly what a
            # silently-degraded placement or a collapsed two-backend run looks like.
            echo "  ${names[$i]}: FAIL (sflow run exited 0 but its output does not prove the run: ${run_dir:-no run dir found})"
        fi
    done
}

run_monitor_mixed_real() {
    # monitor_mixed.yaml is the broadest single-run regression net in examples/
    # (placement proven by UUID, replicas, release_after GPU reuse, readiness
    # ordering, marker-clipped monitor reports, two Slurm pools). It needs its OWN
    # `sflow batch` call for one reason: `--set PARTITION_A=...` is REJECTED by any
    # config that does not declare that variable ("Variable 'PARTITION_A' ... is not
    # defined"), so it cannot ride the shared bulk-submit args. Without the --set it
    # submitted with the sample's `your_partition_a` placeholder and was a
    # guaranteed sbatch rejection.
    if [ -z "$SUBMIT" ]; then
        return
    fi
    case "$TEST_TYPE" in
        s|a|smoke|min) ;;
        *) return ;;
    esac
    # No framework and no model load: the sanity half owns it, and that half runs
    # through `sflow run` now -- so batch it only in the combined default.
    [ "$RECIPE_CLASS" != "all" ] && return

    local part_a="${MONITOR_MIXED_PARTITION_A:-$PARTITION}"
    local part_b="${MONITOR_MIXED_PARTITION_B:-$E2E_PARTITION_B}"
    local mm_dir="$E2E_OUTPUT_DIR/monitor_mixed_real"
    local mm_script="$mm_dir/monitor_mixed.sh"
    mkdir -p "$mm_dir"

    echo ""
    echo "===== Real monitor_mixed run (all-in-one regression net) ====="
    echo "      gpu_pool partition=$part_a, cpu_pool partition=$part_b"
    echo ""

    # -G is IGNORED for a multi-backend config (each backend uses its own config
    # values), so the per-node GPU count has to go in as a --set or the recipe keeps
    # its own default and mis-plans the 2-node decode server.
    strip_suite_segment_args
    local mm_output mm_status
    mm_output=$(sflow batch "$EXAMPLES_DIR/self_contained/slurm/monitor_mixed.yaml" \
        --set "SLURM_ACCOUNT=$ACCOUNT" \
        --set "PARTITION_A=$part_a" \
        --set "PARTITION_B=$part_b" \
        --set "GPUS_PER_NODE=$GPUS_PER_NODE" \
        -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
        -p "$part_a" \
        -A "$ACCOUNT" \
        --job-name "monitor_mixed_slurm" \
        "${BATCH_WORKSPACE_ARGS[@]}" \
        "${BATCH_OUTPUT_ARGS[@]}" \
        "${BATCH_VENV_ARGS[@]}" \
        -o "$mm_script" \
        $SUBMIT \
        "${STRIPPED_SEGMENT_ARGS[@]}" 2>&1)
    mm_status=$?
    echo "$mm_output"
    if [ "$mm_status" -ne 0 ]; then
        echo "  monitor_mixed run: FAIL (sflow batch failed, rc=$mm_status; see output above)"
        MONITOR_MIXED_LAUNCH_FAILED=1
        return
    fi

    # Join JOB_IDS so the driver job flows through the shared wait (sacct) +
    # validate loop like every other batched job.
    local mm_job_id
    mm_job_id=$(echo "$mm_output" | sed -n 's/.*Submitted batch job \([0-9]\+\).*/\1/p' | tail -1)
    if [ -n "$mm_job_id" ]; then
        JOB_IDS+=("$mm_job_id")
        echo "  monitor_mixed driver job id: $mm_job_id (script: $mm_script)"
    else
        echo "  monitor_mixed run: FAIL (no Slurm job id reported by sflow batch)"
        MONITOR_MIXED_LAUNCH_FAILED=1
    fi
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
    # No framework and no model load: the sanity half owns it, and that half runs
    # through `sflow run` now -- so batch it only in the combined default.
    [ "$RECIPE_CLASS" != "all" ] && return

    # Default BOTH to the partition this run was given. Hardcoded names are
    # cluster-specific and simply do not exist elsewhere -- the stale genesisq /
    # gamoraq defaults meant every submission on a new cluster died with "no Slurm
    # allocation granted". Two DIFFERENT partitions are better coverage, so an
    # operator can still opt in via MULTI_BACKEND_PARTITION_A/_B; with one
    # partition the test still proves what it is for, because each backend gets
    # its OWN allocation and the check is that task_a and task_b land on
    # different NODES.
    local part_a="${MULTI_BACKEND_PARTITION_A:-$PARTITION}"
    local part_b="${MULTI_BACKEND_PARTITION_B:-$E2E_PARTITION_B}"
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
    # multi_backend.yaml declares no SLURM_NODES either -- see strip_suite_segment_args.
    strip_suite_segment_args

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
        "${STRIPPED_SEGMENT_ARGS[@]}" 2>&1)
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

# Sync examples/ to src/sflow/samples/ so packaged samples stay up to date.
# gpu_reservation/ and mlperf/ are deliberately NOT packaged: they are local
# scratch (examples/mlperf/ is even gitignored), so copying them here only
# produced untracked dirs under src/sflow/samples/ that would ship with the
# wheel if anyone committed them. --delete does not clean an --exclude'd path,
# so remove any copy an earlier run already made.
echo "Syncing examples/ -> src/sflow/samples/ ..."
rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' --exclude='__init__.py' --exclude='sflow_output' \
    --exclude='gpu_reservation' --exclude='mlperf' \
    "$EXAMPLES_DIR/" "$SAMPLES_DIR/"
rm -rf "$SAMPLES_DIR/gpu_reservation" "$SAMPLES_DIR/mlperf"
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
SANITY_RUN_DIRS=()
SANITY_RUN_NAMES=()
MULTI_BACKEND_RUN_DIR=""
MULTI_BACKEND_LAUNCH_FAILED=""
MONITOR_MIXED_LAUNCH_FAILED=""
CSV_FILE="$EXAMPLES_DIR/modular/inference_x_v2/bulk_input.csv"

# =============================================================================
# Part 1: Self-contained examples (--bulk-submit)
# =============================================================================
if { [ "$TEST_TYPE" = "s" ] || [ "$TEST_TYPE" = "a" ] || [ "$TEST_TYPE" = "smoke" ] || [ "$TEST_TYPE" = "min" ] || [ "$TEST_TYPE" = "one" ]; } && [ "$RECIPE_CLASS" != "sanity" ]; then
    echo ""
    if [ "$TEST_TYPE" = "one" ]; then
        echo "===== Part 1: Single-job plumbing smoke (--bulk-submit one file) ====="
    elif [ "$TEST_TYPE" = "min" ]; then
        echo "===== Part 1: Min self-contained examples (--bulk-submit selected files) ====="
    elif [ "$TEST_TYPE" = "smoke" ]; then
        echo "===== Part 1: Smoke self-contained examples (--bulk-submit selected files) ====="
    else
        echo "===== Part 1: Self-contained examples (--bulk-submit) ====="
    fi
    echo ""

    if [ "$TEST_TYPE" = "min" ] || [ "$TEST_TYPE" = "one" ]; then
        MIN_SELF_CONTAINED=(
            "$EXAMPLES_DIR/self_contained/slurm/auto_replica.yaml"
            "$EXAMPLES_DIR/self_contained/slurm/dynamo_trtllm_disagg.yaml"
            # One node, seconds of runtime, and it ASSERTS by UUID that every
            # container/bare x offset x concurrency combination held the physical
            # GPUs it was planned for. The recipes above only echo their devices,
            # so they passed while a server held none.
            "$EXAMPLES_DIR/self_contained/slurm/gpu_placement_matrix.yaml"
            "$EXAMPLES_DIR/self_contained/slurm/resource_release_after.yaml"
            "$EXAMPLES_DIR/self_contained/slurm/trtllm_serve_disagg.yaml"
        )
        if [ "$TEST_TYPE" = "one" ]; then
            # Narrow to ONE job: the GPU placement matrix. Two nodes, a small
            # container, no model to load, and seconds of compute -- so it stays
            # cheap enough to play on every CI or cluster change.
            #
            # It is the recipe that ASSERTS rather than echoes. Every combination
            # that can break the slice is covered (bare vs container, zero vs
            # high offset, two tasks concurrent on one node, and multi-node), and
            # each one resolves its planned HOST indices through a bare-metal
            # index -> UUID map taken per node before anything was carved. That is
            # the only way to tell "the right number of GPUs" from "the right
            # GPUs", and it is the failure that actually reaches clusters: a task
            # planned for 2,3 inside a 2-GPU container renumbered to 0,1.
            #
            # It replaced the single-node DISAGG recipe, which probed the same
            # failure but only PRINTED its devices -- so it passed while a server
            # held none -- and cost a multi-gigabyte pull plus a model load, which
            # made a red run ambiguous between a broken CI path and a real
            # regression. Here a red run means placement is genuinely wrong; read
            # SFLOW_GPU_PROBE in the task logs for the planned slice next to the
            # devices the step actually held.
            MIN_SELF_CONTAINED=(
                "$EXAMPLES_DIR/self_contained/slurm/gpu_placement_matrix.yaml"
            )
        fi
        MIN_BULK_ARGS=()
        for yaml_file in "${MIN_SELF_CONTAINED[@]}"; do
            if [ ! -f "$yaml_file" ]; then
                echo "ERROR: min self-contained Slurm YAML not found: $yaml_file"
                exit 1
            fi
            recipe_in_class "$yaml_file" || continue
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
                multi_backend.yaml|monitor_mixed.yaml)
                    # Both declare PARTITION_A/PARTITION_B and are covered
                    # separately (run_multi_backend_real / run_monitor_mixed_real).
                    # They cannot ride the shared bulk args: `--set PARTITION_A=...`
                    # is REJECTED by every config that does not declare it, and
                    # without the --set they submit with the sample's
                    # `your_partition_a` placeholder and sbatch rejects them.
                    continue
                    ;;
            esac
            recipe_in_class "$yaml_file" || continue
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
        # Bulk-submit every example EXCEPT the two-partition configs: they declare
        # PARTITION_A/PARTITION_B, which the shared bulk args cannot set (a --set of
        # a variable a config does not declare is a hard error), and are covered
        # separately by run_multi_backend_real / run_monitor_mixed_real.
        ALL_BULK_ARGS=()
        for yaml_file in "$EXAMPLES_DIR"/self_contained/slurm/*.yaml; do
            case "$(basename "$yaml_file")" in
                multi_backend.yaml|monitor_mixed.yaml)
                    continue
                    ;;
            esac
            recipe_in_class "$yaml_file" || continue
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
if { [ "$TEST_TYPE" = "m" ] || [ "$TEST_TYPE" = "a" ] || [ "$TEST_TYPE" = "smoke" ] || [ "$TEST_TYPE" = "min" ]; } && [ "$RECIPE_CLASS" != "sanity" ]; then
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
elif { [ "$TEST_TYPE" = "inf" ] || [ "$TEST_TYPE" = "m" ] || [ "$TEST_TYPE" = "a" ] || [ "$TEST_TYPE" = "smoke" ] || [ "$TEST_TYPE" = "min" ]; } && [ "$RECIPE_CLASS" != "sanity" ]; then
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
run_monitor_mixed_real

set +x

echo ""
echo "===== Submitted Jobs ====="

fi  # end of --check else block


if [ ${#JOB_IDS[@]} -eq 0 ] && [ "$RECIPE_CLASS" != "sanity" ]; then
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
    # These are GB200/driver "system not ready" signatures -- an ERROR from the
    # driver, on a node that needs draining. 'system not yet initialized' covers
    # both torch ('Error 802: system not yet initialized') and cupy
    # ('cudaErrorSystemNotReady: system not yet initialized').
    #
    # Every pattern here must be something ONLY a broken node produces. Excusing is
    # not a soft verdict -- it removes the job from the pass/fail threshold, so a
    # pattern that also matches healthy output turns real regressions into a green
    # pipeline. 'No CUDA runtime is found' used to be in this list and did exactly
    # that: torch prints it as a routine WARNING when cpp_extension cannot find
    # nvcc for JIT ("No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'"),
    # which healthy runs emit constantly. It excused two genuinely failed disagg
    # jobs and reported "PASS - 0/6 failed".
    grep -rIqs --include='*.log' --include='*.out' \
        -e 'system not yet initialized' \
        -e 'cudaErrorSystemNotReady' \
        -e 'CUDA initialization: Unexpected error from cudaGetDeviceCount' \
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
if [ "$RECIPE_CLASS" = "sanity" ]; then
    run_sanity_recipes_with_sflow_run
fi
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
    case "$out_dir" in
        *-gpu_placement_matrix-*)
            if gpu_placement_run_ok "$out_dir"; then
                PASSED=$((PASSED + 1))
                echo "  Job $jid: PASS (GPU placement proven by UUID on 2+ nodes under $out_dir)"
            elif cuda_infra_failure "$out_dir"; then
                mark_cuda_excused "$jid" "$out_dir" "(placement assertions unproven; CUDA init failure on node)"
            else
                echo "  Job $jid: FAIL (GPU placement not proven under $out_dir; read SFLOW_GPU_PROBE / FAIL: in the task logs)"
            fi
            continue
            ;;
    esac
    # Check for various success indicators across different workflow types
    #   aiperf:           its own "Processed N valid requests and M errors" tally
    #   benchmark_serving: its own "Successful requests:" vs the --num-prompts asked for
    #   auto_replica:     'Client Task Nodes' in client task log
    count_replica=$(find "$out_dir" -type f -name 'client*.log' -exec grep -l "Client Task Nodes" {} + 2>/dev/null | wc -l)
    # Both: 0 = benchmarked, 1 = ran and measured nothing, 2 = that tool not used here.
    aiperf_tally_ok "$out_dir"
    aiperf_state=$?
    serving_tally_ok "$out_dir"
    serving_state=$?

    if [ "$aiperf_state" -eq 1 ] && recipe_is_client_only "$out_dir"; then
        # The ONE expected-not-to-benchmark shape, claimed explicitly so it reads as
        # a decision rather than a hole: this recipe starts no server, so aiperf had
        # nothing to talk to. It still owes a clean completion.
        if workflow_summary_ok "$out_dir"; then
            PASSED=$((PASSED + 1))
            echo "  Job $jid: PASS (client-only recipe: no server to benchmark by design, and it completed; $out_dir)"
        else
            echo "  Job $jid: FAIL (client-only recipe is still expected to COMPLETE, and did not; $out_dir)"
        fi
    elif [ "$aiperf_state" -eq 1 ]; then
        # Checked BEFORE the success indicators: this is the shape where every other
        # signal in the run says green. A dead aiperf is a failed benchmark even when
        # a sibling task in the same workflow reported requests of its own.
        if cuda_infra_failure "$out_dir"; then
            mark_cuda_excused "$jid" "$out_dir" "(aiperf measured nothing; CUDA init failure on node)"
        else
            echo "  Job $jid: FAIL (aiperf ran but measured nothing; see the tally above, $out_dir)"
        fi
    elif [ "$serving_state" -eq 1 ]; then
        if cuda_infra_failure "$out_dir"; then
            mark_cuda_excused "$jid" "$out_dir" "(benchmark_serving did not complete; CUDA init failure on node)"
        else
            echo "  Job $jid: FAIL (benchmark_serving ran but did not complete its requests; see the counts above, $out_dir)"
        fi
    elif [ "$aiperf_state" -eq 0 ] || [ "$serving_state" -eq 0 ] || [ "$count_replica" -gt 0 ]; then
        PASSED=$((PASSED + 1))
        echo "  Job $jid: PASS (under $out_dir)"
    elif [ -z "$(find "$out_dir" -type f -name 'benchmark*.log' -print -quit 2>/dev/null)" ] \
         && workflow_summary_ok "$out_dir"; then
        PASSED=$((PASSED + 1))
        echo "  Job $jid: PASS (no benchmark log; sflow reports Status: COMPLETED under $out_dir)"
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

if [ -n "${MONITOR_MIXED_LAUNCH_FAILED:-}" ]; then
    # Same as above: never submitted, so the loop never visited it. Silence here
    # let the broadest recipe in the suite vanish without touching the verdict.
    TOTAL=$((TOTAL + 1))
    echo "  monitor_mixed run: FAIL (no Slurm job submitted)"
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
for _i in "${!SANITY_RUN_DIRS[@]}"; do
    MONITOR_TOTAL=$((MONITOR_TOTAL + 1))
    _mon=$(find "${SANITY_RUN_DIRS[$_i]}" -maxdepth 2 -type f -name 'sflow_monitor.log' 2>/dev/null | head -1)
    if [ -n "$_mon" ] && monitor_log_has_content "$_mon"; then
        MONITOR_PRESENT=$((MONITOR_PRESENT + 1))
        echo "  ${SANITY_RUN_NAMES[$_i]}: monitor overview OK ($_mon)"
    else
        MONITOR_MISSING_LABELS="$MONITOR_MISSING_LABELS  - ${SANITY_RUN_NAMES[$_i]} (no populated sflow_monitor.log under ${SANITY_RUN_DIRS[$_i]})\n"
        echo "  ${SANITY_RUN_NAMES[$_i]}: MONITOR MISSING/EMPTY (${SANITY_RUN_DIRS[$_i]})"
    fi
done
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

# The verdict, as an exit status. Without this the script ended on an `if` and
# returned 0 however red the run was, which is what made full_sample_tests.sh's
# `exit "$e2e_rc"` unable to fail.
# CUDA-infra excusals are excluded from the threshold, matching what
# summarize_validation() does with the same numbers -- but an ALL-excused run
# proved nothing, so it is not a pass either.
[ "$PASSED" -gt 0 ] \
    && [ $((PASSED + CUDA_INFRA)) -eq "$TOTAL" ] \
    && [ "$MONITOR_PRESENT" -eq "$MONITOR_TOTAL" ] \
    && [ "$TARGETING_OK" -eq "$TARGETING_TOTAL" ]
