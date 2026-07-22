#!/bin/bash

set -uo pipefail

TEST_TYPE="a"
SUBMIT=""
PREFLIGHT_ONLY=""
MAX_JOBS=16
CLI_MODEL_PATH=""
CLI_PARTITION=""
CLI_ACCOUNT=""
usage() {
    echo "Usage: $0 [-a|-s|-m|-inf|--smoke|--min] [-S] [-P] [-j N] [-M model_path] [-p partition] [-A account]"
    echo "  -a  all tests (default)"
    echo "  -s  self-contained examples only"
    echo "  -m  modular examples only"
    echo "  -inf  infmax batch suites only"
    echo "  --smoke  curated Slurm smoke subset with broad coverage"
    echo "  --min  minimal Slurm submit set (one representative per validation type)"
    echo "  -S  submit jobs to Slurm"
    echo "  -P  preflight checks only (skip job submission even if -S is set)"
    echo "  -j  max parallel jobs (default: 16, 0 for unlimited)"
    echo "  -M  model path (default: \$MODEL_PATH or /home/)"
    echo "  -p  Slurm partition (default: dummy_part for preflight, my_partition for e2e)"
    echo "  -A  Slurm account (default: dummy_acct for preflight, user for e2e)"
}

while [ $# -gt 0 ]; do
    case "$1" in
        -a) TEST_TYPE="a" ;;
        -s) TEST_TYPE="s" ;;
        -m) TEST_TYPE="m" ;;
        -inf) TEST_TYPE="inf" ;;
        --smoke) TEST_TYPE="smoke" ;;
        --min) TEST_TYPE="min" ;;
        -S) SUBMIT="--submit" ;;
        -P) PREFLIGHT_ONLY="1" ;;
        -j) [ $# -ge 2 ] || { usage; exit 1; }; shift; MAX_JOBS="$1" ;;
        -M) [ $# -ge 2 ] || { usage; exit 1; }; shift; CLI_MODEL_PATH="$1" ;;
        -p) [ $# -ge 2 ] || { usage; exit 1; }; shift; CLI_PARTITION="$1" ;;
        -A) [ $# -ge 2 ] || { usage; exit 1; }; shift; CLI_ACCOUNT="$1" ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
    shift
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXAMPLES_DIR="$REPO_DIR/examples"
# Packaged samples (a curated SUBSET of examples/, shipped via `sflow sample`). Refreshed
# in place from examples/ before recipe validation -- see the sample-sync step below.
SAMPLES_DIR="$REPO_DIR/src/sflow/samples"
CSV_FILE="$EXAMPLES_DIR/modular/inference_x_v2/bulk_input.csv"
MODEL_PATH="${CLI_MODEL_PATH:-${MODEL_PATH:-/home/}}"
PARTITION="${CLI_PARTITION:-dummy_part}"
ACCOUNT="${CLI_ACCOUNT:-dummy_acct}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
INFMAX_DIR="$REPO_DIR/tests/e2e_tests/infmax"
INFMAX_CSV_NAME="${INFMAX_CSV_NAME:-1k1k_jobs.csv}"
INFMAX_ROW="${INFMAX_ROW:-1}"
INFMAX_RECIPE_REPO="${INFMAX_RECIPE_REPO:-}"
INFMAX_RECIPE_REF="${INFMAX_RECIPE_REF:-}"
INFMAX_RECIPE_SUBDIR="${INFMAX_RECIPE_SUBDIR:-}"
INFMAX_REFRESH_RECIPES="${INFMAX_REFRESH_RECIPES:-}"
INFMAX_BENCH_SERVING_DIR="${INFMAX_BENCH_SERVING_DIR:-$INFMAX_DIR/nvidia_submission/sa-bench}"
INFMAX_MONITOR_CONFIG="${INFMAX_MONITOR_CONFIG:-$INFMAX_DIR/monitor/monitor.yaml}"
PREFLIGHT_SKIP_NOTES=""
INFMAX_RECIPE_SKIP_REASON=""
IS_REAL_SUBMIT=""
if [ -n "$SUBMIT" ] && [ -z "$PREFLIGHT_ONLY" ]; then
    IS_REAL_SUBMIT="1"
fi

STAMP=$(date +%Y%m%d-%H%M%S)
PREFLIGHT_DIR="$REPO_DIR/sflow_output/preflight_$STAMP"
mkdir -p "$PREFLIGHT_DIR"

source "$SCRIPT_DIR/use_under_dev_sflow.sh"
RESULTS_DIR=""
cleanup() {
    if [ -n "$RESULTS_DIR" ]; then
        rm -rf "$RESULTS_DIR"
    fi
    cleanup_under_dev_sflow
}
trap cleanup EXIT
if [ -z "$IS_REAL_SUBMIT" ]; then
    setup_under_dev_sflow "$REPO_DIR" || exit 1
else
    echo "Skipping local under-dev sflow setup for real Slurm submit; sample_test.sh creates the runtime venv on Slurm."
fi

RESULTS_DIR=$(mktemp -d)
TEST_ID=0

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

throttle() {
    if [ "$MAX_JOBS" -gt 0 ]; then
        while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
            sleep 0.1
        done
    fi
}

record_preflight_skip() {
    local note="$1"
    case "$PREFLIGHT_SKIP_NOTES" in
        *"$note"*) ;;
        *) PREFLIGHT_SKIP_NOTES="${PREFLIGHT_SKIP_NOTES}  - ${note}\n" ;;
    esac
}

fetch_infmax_recipes() {
    INFMAX_RECIPE_SKIP_REASON=""
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
        return 0
    fi

    if ! command -v git >/dev/null 2>&1; then
        INFMAX_RECIPE_SKIP_REASON="git is not available"
        echo "WARN: skipping Infmax recipe fetch because git is not available." >&2
        return 1
    fi
    if ! command -v rsync >/dev/null 2>&1; then
        INFMAX_RECIPE_SKIP_REASON="rsync is not available"
        echo "WARN: skipping Infmax recipe fetch because rsync is not available." >&2
        return 1
    fi

    if [ -z "$INFMAX_RECIPE_REPO" ] || [ -z "$INFMAX_RECIPE_REF" ] || [ -z "$INFMAX_RECIPE_SUBDIR" ]; then
        INFMAX_RECIPE_SKIP_REASON="recipe source env is incomplete"
        echo "WARN: skipping Infmax recipe fetch because recipe source env is incomplete." >&2
        echo "      Set INFMAX_RECIPE_REPO, INFMAX_RECIPE_REF, and INFMAX_RECIPE_SUBDIR to enable these checks." >&2
        return 1
    fi

    local tmp_dir
    tmp_dir=$(mktemp -d)

    echo "Fetching infmax recipes ($INFMAX_RECIPE_REF:$INFMAX_RECIPE_SUBDIR) ..."
    if ! GIT_TERMINAL_PROMPT=0 git clone --depth 1 --branch "$INFMAX_RECIPE_REF" "$INFMAX_RECIPE_REPO" "$tmp_dir/repo" >/dev/null 2>&1; then
        rm -rf "$tmp_dir"
        INFMAX_RECIPE_SKIP_REASON="recipe repo clone failed"
        echo "WARN: skipping Infmax checks because recipe repo clone failed." >&2
        return 1
    fi

    local source_dir="$tmp_dir/repo/$INFMAX_RECIPE_SUBDIR"
    if [ ! -d "$source_dir" ]; then
        rm -rf "$tmp_dir"
        INFMAX_RECIPE_SKIP_REASON="recipe subdir was not found"
        echo "WARN: skipping Infmax checks because recipe subdir was not found: $INFMAX_RECIPE_SUBDIR" >&2
        return 1
    fi

    rsync -a --delete --exclude='batch_test.sh' "$source_dir/" "$INFMAX_DIR/"
    rm -rf "$tmp_dir"
    return 0
}

set_infmax_suite_overrides() {
    INFMAX_SUITE_OVERRIDES=(-s "CONCURRENCY=[16,32]")
    case "$1" in
        kimik2.5-fp4-gb200-multi_node-vllm)
            INFMAX_SUITE_OVERRIDES+=(-s "DYNAMO_VERSION=1.0.1" -e "--container-remap-root")
            ;;
    esac
}

run_check() {
    local label="$1"
    shift
    local cmd_str="$*"
    TEST_ID=$((TEST_ID + 1))
    local id
    id=$(printf "%03d" "$TEST_ID")
    local result_file="$RESULTS_DIR/${id}.result"
    local output_file="$RESULTS_DIR/${id}.output"

    # Detect output path from -o / --output-dir / --sbatch-path args
    local out_path=""
    local prev=""
    for arg in "$@"; do
        if [ "$prev" = "-o" ] || [ "$prev" = "--output-dir" ] || [ "$prev" = "--sbatch-path" ]; then
            out_path="$arg"
            break
        fi
        prev="$arg"
    done

    throttle

    (
        local status
        if "$@" >"$output_file" 2>&1; then
            status="OK"
        else
            status="FAIL"
        fi
        {
            echo "STATUS=$status"
            echo "LABEL=$label"
            echo "CMD=$cmd_str"
        } > "$result_file"

        # Save the raw command to the output directory for reference
        if [ -n "$out_path" ]; then
            local cmd_target
            if [ -d "$out_path" ]; then
                cmd_target="$out_path"
            else
                cmd_target=$(dirname "$out_path")
            fi
            if [ -d "$cmd_target" ]; then
                printf '# Test: %s\n# Status: %s\n$ %s\n' "$label" "$status" "$cmd_str" \
                    > "$cmd_target/_command.txt"
            fi
        fi
    ) &
}

# =========================================================================
# Preflight: CLI smoke tests (no jobs submitted)
# =========================================================================
if true; then
    echo ""
    echo "===== Preflight: CLI smoke tests (no Slurm submission) ====="
    echo "===== Running tests in parallel (max_jobs=${MAX_JOBS:-unlimited}) ====="
    echo ""

    VERSION_INFO_LOG="$PREFLIGHT_DIR/sflow_version.log"
    run_check "sflow --version runtime info" \
        bash -c "sflow --version > '$VERSION_INFO_LOG' 2>&1 && \
            grep -F -- 'sflow executable' '$VERSION_INFO_LOG' && \
            grep -F -- 'version :' '$VERSION_INFO_LOG' && \
            grep -F -- 'bin     :' '$VERSION_INFO_LOG' && \
            grep -F -- 'python  :' '$VERSION_INFO_LOG' && \
            grep -F -- 'package :' '$VERSION_INFO_LOG' && \
            grep -F -- 'install :' '$VERSION_INFO_LOG' && \
            ! grep -E '^sflow [0-9]' '$VERSION_INFO_LOG'"

    if [ "$TEST_TYPE" = "inf" ]; then
        INFMAX_BATCH_PREFLIGHT_DIR="$PREFLIGHT_DIR/infmax_batch"
        if fetch_infmax_recipes; then
            INFMAX_TARGET_DIRS=(
                "$INFMAX_DIR/dsr1-fp8-gb200-multi_node-sglang"
                "$INFMAX_DIR/dsr1-fp4-gb200-multi_node-trtllm"
                "$INFMAX_DIR/kimik2.5-fp4-gb200-multi_node-vllm"
            )
            for suite_dir in "${INFMAX_TARGET_DIRS[@]}"; do
                suite_name=$(basename "$suite_dir")
                set_infmax_suite_overrides "$suite_name"
                run_check "infmax multi-node batch $suite_name" \
                    sflow batch \
                        -f "$INFMAX_MONITOR_CONFIG" \
                        --bulk-input "$suite_dir/$INFMAX_CSV_NAME" \
                        --row "$INFMAX_ROW" \
                        -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
                        -a "BENCH_SERVING_DIR=fs://$INFMAX_BENCH_SERVING_DIR" \
                        -G "$GPUS_PER_NODE" \
                        -A "$ACCOUNT" \
                        -p "$PARTITION" \
                        --log-level warn \
                        "${INFMAX_SUITE_OVERRIDES[@]}" \
                        --output-dir "$INFMAX_BATCH_PREFLIGHT_DIR/$suite_name"
            done
        else
            record_preflight_skip "Infmax batch preflight checks skipped: ${INFMAX_RECIPE_SKIP_REASON:-recipe source unavailable}"
            echo "  SKIP: Infmax recipe source unavailable; skipping Infmax batch preflight checks."
        fi
    else

    # -- Refresh the packaged samples (src/sflow/samples, a curated SUBSET of examples/) IN
    #    PLACE from the up-to-date examples/ BEFORE validating, so the shipped `sflow sample`
    #    recipes and the sample-copy checks below track the latest examples. Only sample files
    #    that have an examples/ counterpart are overwritten; sample-only files are left as-is.
    #    Runs synchronously (NOT via run_check, which backgrounds) so it finishes before any
    #    check reads the samples. The refreshed samples show up as uncommitted changes under
    #    src/sflow/samples/ -- commit them alongside the examples/ edits they mirror. --
    echo "----- Syncing packaged samples from examples/ (refresh curated subset) -----"
    if [ -d "$SAMPLES_DIR" ]; then
        _sample_synced=0; _sample_skipped=0; _sample_failed=0
        while IFS= read -r -d '' _sfile; do
            _srel="${_sfile#"$SAMPLES_DIR"/}"
            if [ -f "$EXAMPLES_DIR/$_srel" ]; then
                if cp -f "$EXAMPLES_DIR/$_srel" "$_sfile"; then
                    _sample_synced=$((_sample_synced + 1))
                else
                    _sample_failed=$((_sample_failed + 1))
                    echo "  WARN: failed to sync sample '$_srel' from examples/" >&2
                fi
            else
                _sample_skipped=$((_sample_skipped + 1))
            fi
        done < <(find "$SAMPLES_DIR" -type f -print0)
        echo "  synced $_sample_synced sample file(s) from examples/; kept $_sample_skipped sample-only file(s); $_sample_failed failure(s)."
    else
        echo "  WARN: $SAMPLES_DIR not found; skipping sample sync."
    fi

    # -- sflow run --dry-run: local examples --
    run_check "local_hello_world" \
        sflow run "$EXAMPLES_DIR/self_contained/local/hello_world.yaml" --dry-run
    run_check "local_dag" \
        sflow run "$EXAMPLES_DIR/self_contained/local/dag.yaml" --dry-run
    run_check "local_variable_domain" \
        sflow run "$EXAMPLES_DIR/self_contained/local/variable_domain.yaml" --dry-run
    run_check "local_storage_upload" \
        sflow run "$EXAMPLES_DIR/self_contained/local/storage_upload.yaml" --dry-run
    run_check "local_storage_upload_all" \
        sflow run "$EXAMPLES_DIR/self_contained/local/storage_upload_all.yaml" --dry-run

    # -- sflow run --dry-run: backend-agnostic Docker/Kubernetes examples --
    BACKEND_AGNOSTIC_DIR="$PREFLIGHT_DIR/backend_agnostic_examples"
    DOCKER_HELLO_DRYRUN_LOG="$BACKEND_AGNOSTIC_DIR/docker_hello_world.log"
    DOCKER_MULTI_DRYRUN_LOG="$BACKEND_AGNOSTIC_DIR/docker_multi_node.log"
    KUBERNETES_HELLO_DRYRUN_LOG="$BACKEND_AGNOSTIC_DIR/kubernetes_hello_world.log"
    KUBERNETES_LWS_DRYRUN_LOG="$BACKEND_AGNOSTIC_DIR/kubernetes_multinode.log"
    mkdir -p "$BACKEND_AGNOSTIC_DIR"
    run_check "dry-run docker_hello_world uses docker_run default operator" \
        bash -c "sflow run \"$EXAMPLES_DIR/self_contained/docker/hello_world.yaml\" --dry-run --verbose > \"$DOCKER_HELLO_DRYRUN_LOG\" 2>&1 && \
            grep -F -- 'operator: docker_run' \"$DOCKER_HELLO_DRYRUN_LOG\" && \
            grep -F -- 'Dry-run complete: docker_hello_world' \"$DOCKER_HELLO_DRYRUN_LOG\""
    run_check "dry-run docker_multi_node assigns remote Docker hosts and GPUs" \
        bash -c "sflow run \"$EXAMPLES_DIR/self_contained/docker/multi_node.yaml\" --dry-run --verbose > \"$DOCKER_MULTI_DRYRUN_LOG\" 2>&1 && \
            grep -F -- \"nodes=['dgx-a', 'dgx-b']\" \"$DOCKER_MULTI_DRYRUN_LOG\" && \
            grep -F -- 'operator: docker_run' \"$DOCKER_MULTI_DRYRUN_LOG\" && \
            grep -F -- 'CUDA_VISIBLE_DEVICES: 0' \"$DOCKER_MULTI_DRYRUN_LOG\" && \
            grep -F -- 'gpus: device=0' \"$DOCKER_MULTI_DRYRUN_LOG\""
    run_check "dry-run kubernetes_hello_world uses k8s operator" \
        bash -c "sflow run \"$EXAMPLES_DIR/self_contained/kubernetes/hello_world.yaml\" --dry-run --verbose > \"$KUBERNETES_HELLO_DRYRUN_LOG\" 2>&1 && \
            grep -F -- 'id=kubernetes' \"$KUBERNETES_HELLO_DRYRUN_LOG\" && \
            grep -F -- 'operator: k8s' \"$KUBERNETES_HELLO_DRYRUN_LOG\" && \
            grep -F -- 'Dry-run complete: kubernetes_hello_world' \"$KUBERNETES_HELLO_DRYRUN_LOG\""
    run_check "dry-run kubernetes_multinode uses k8s operator" \
        bash -c "sflow run \"$EXAMPLES_DIR/self_contained/kubernetes/multinode.yaml\" --dry-run --verbose > \"$KUBERNETES_LWS_DRYRUN_LOG\" 2>&1 && \
            grep -F -- 'id=kubernetes' \"$KUBERNETES_LWS_DRYRUN_LOG\" && \
            grep -F -- 'operator: k8s' \"$KUBERNETES_LWS_DRYRUN_LOG\" && \
            grep -F -- 'Dry-run complete: k8s_multinode' \"$KUBERNETES_LWS_DRYRUN_LOG\""

    # NOTE: the mimic-script SMOKE recipes (reservation / cross-task IP / apply +
    # validate / log offload) are test fixtures, not user-facing examples. They live
    # under tests/integration/recipes/kubernetes/ and are dry-run-validated by pytest
    # (tests/integration/test_cli_run_dry_run_k8s_examples.py).

    # -- sflow run --dry-run: native Kubernetes Dynamo recipes (vLLM / SGLang,
    #    agg + disagg). Each plans on the kubernetes backend and renders server
    #    tasks via the plain k8s operator, using the recipe defaults: both SGLang and
    #    vLLM serve the model via a LOCAL_MODEL_PATH fs:// artifact whose default lives in
    #    the always-mounted model-store PVC (overridable with -a LOCAL_MODEL_PATH=...). The
    #    TRT-LLM recipes use the k8s_mpi operator and are covered in the MPI loop below. --
    K8S_DYNAMO_DIR="$BACKEND_AGNOSTIC_DIR/kubernetes_dynamo"
    mkdir -p "$K8S_DYNAMO_DIR"
    for k8s_dynamo_case in \
        "self_contained/kubernetes/dynamo_vllm_agg:dynamo_vllm_agg" \
        "self_contained/kubernetes/dynamo_vllm_disagg:dynamo_vllm_disagg" \
        "self_contained/kubernetes/dynamo_sglang_agg:dynamo_sglang_agg" \
        "self_contained/kubernetes/dynamo_sglang_disagg:dynamo_sglang_disagg"; do
        k8s_dynamo_example="${k8s_dynamo_case%%:*}"
        k8s_dynamo_workflow="${k8s_dynamo_case##*:}"
        k8s_dynamo_log="$K8S_DYNAMO_DIR/$(basename "$k8s_dynamo_example").log"
        run_check "dry-run ${k8s_dynamo_example} uses k8s operator" \
            bash -c "sflow run \"$EXAMPLES_DIR/${k8s_dynamo_example}.yaml\" --dry-run --verbose > \"$k8s_dynamo_log\" 2>&1 && \
                grep -F -- 'id=kubernetes' \"$k8s_dynamo_log\" && \
                grep -F -- 'operator: k8s' \"$k8s_dynamo_log\" && \
                grep -F -- 'Dry-run complete: ${k8s_dynamo_workflow}' \"$k8s_dynamo_log\""
    done

    # NOTE: the standalone + single-node MLPerf (DeepSeek-R1 IFB) k8s recipes were removed
    # until their public date (commit "Remove mlperf recipes until public date"); their
    # dry-run checks are dropped here until the recipes ship again.

    # -- sflow run --dry-run: multi-node K8S MPI recipes (the `k8s_mpi` operator).
    #    ALL K8S + TRT-LLM recipes (Dynamo agg/disagg + bare trtllm-serve agg/disagg)
    #    run their GPU model-server tasks on k8s_mpi by default, alongside plain-k8s
    #    infra/router + benchmark tasks. Preflight asserts the k8s_mpi operator is
    #    planned and each recipe keeps parsing as the MPI wiring evolves. --
    K8S_MPI_DIR="$BACKEND_AGNOSTIC_DIR/kubernetes_mpi"
    mkdir -p "$K8S_MPI_DIR"
    for k8s_mpi_case in \
        "self_contained/kubernetes/dynamo_trtllm_agg:dynamo_trtllm_agg" \
        "self_contained/kubernetes/dynamo_trtllm_disagg:dynamo_trtllm_disagg" \
        "self_contained/kubernetes/trtllm_serve_agg:trtllm_serve_agg" \
        "self_contained/kubernetes/trtllm_serve_disagg:trtllm_serve_disagg"; do
        k8s_mpi_example="${k8s_mpi_case%%:*}"
        k8s_mpi_workflow="${k8s_mpi_case##*:}"
        k8s_mpi_log="$K8S_MPI_DIR/$(basename "$k8s_mpi_example").log"
        run_check "dry-run ${k8s_mpi_example} uses k8s_mpi operator" \
            bash -c "sflow run \"$EXAMPLES_DIR/${k8s_mpi_example}.yaml\" --dry-run --verbose > \"$k8s_mpi_log\" 2>&1 && \
                grep -F -- 'id=kubernetes' \"$k8s_mpi_log\" && \
                grep -F -- 'operator: k8s_mpi' \"$k8s_mpi_log\" && \
                grep -F -- 'Dry-run complete: ${k8s_mpi_workflow}' \"$k8s_mpi_log\""
    done

    # -- K8s recipe GOLDEN MANIFESTS: the dry-run checks above prove each recipe PLANS on
    #    the k8s backend, but never render the pod/List/MPIJob manifests (dry-run emits
    #    none -- rendering needs a cluster allocation). This check renders the representative
    #    recipes OFFLINE with a deterministic placeholder allocation and diffs them against
    #    checked-in goldens (tests/unit/golden/k8s_recipes/), so the preflight_cli CI job
    #    also covers K8s MANIFEST rendering, not just planning. Regenerate after an
    #    intentional recipe/manifest change with:
    #      SFLOW_UPDATE_GOLDEN=1 pytest tests/unit/test_k8s_recipe_golden_manifests.py --
    K8S_GOLDEN_LOG="$K8S_MPI_DIR/k8s_recipe_golden_manifests.log"
    run_check "k8s recipe golden manifests match offline render" \
        bash -c "python -m pytest \"$REPO_DIR/tests/unit/test_k8s_recipe_golden_manifests.py\" -q > \"$K8S_GOLDEN_LOG\" 2>&1"

    # -- sflow run --dry-run: the vLLM recipes with the model + JIT/kernel cache pointed at
    #    NODE-LOCAL paths via the LOCAL_MODEL_PATH / LOCAL_CACHE_PATH fs:// artifact overrides
    #    (the default branch -- both on the always-mounted PVCs -- is covered by the loops
    #    above). A path not under a declared PVC volume is hostPath-mounted, so this exercises
    #    the node-local storage path. Asserts the recipe still plans on the k8s backend. --
    NODELOCAL_DIR="$BACKEND_AGNOSTIC_DIR/model_source_nodelocal"
    mkdir -p "$NODELOCAL_DIR"
    for nl_case in \
        "self_contained/kubernetes/dynamo_vllm_agg:dynamo_vllm_agg:k8s" \
        "self_contained/kubernetes/dynamo_vllm_disagg:dynamo_vllm_disagg:k8s"; do
        nl_example="${nl_case%%:*}"
        nl_rest="${nl_case#*:}"
        nl_workflow="${nl_rest%%:*}"
        nl_operator="${nl_rest##*:}"
        nl_log="$NODELOCAL_DIR/$(basename "$nl_example").log"
        run_check "dry-run ${nl_example} node-local LOCAL_MODEL_PATH/LOCAL_CACHE_PATH override" \
            bash -c "sflow run \"$EXAMPLES_DIR/${nl_example}.yaml\" -a LOCAL_MODEL_PATH=fs:///var/tmp/models/Qwen3-8B-FP8 -a LOCAL_CACHE_PATH=fs:///var/tmp/sflow-cache --dry-run --verbose > \"$nl_log\" 2>&1 && \
                grep -F -- 'id=kubernetes' \"$nl_log\" && \
                grep -F -- 'operator: ${nl_operator}' \"$nl_log\" && \
                grep -F -- 'Dry-run complete: ${nl_workflow}' \"$nl_log\""
    done

    # NOTE: the mlperf MPI recipe's MODEL_SOURCE=pvc branch was covered here, but that recipe
    # was removed until its public date (see above); re-add this check when it ships again.

    # -- sflow run --dry-run + sflow compose: backend-agnostic modular samples
    #    (examples/modular). The SAME workload fragments compose onto plain K8s,
    #    K8s-MPI, or Slurm backends -- each names its backend `cluster` and binds the
    #    logical `server`/`helper`/`client` operators. Also asserts the server's
    #    `required_by: [benchmark]` folded into the benchmark's `depends_on`
    #    (reverse-dependency + scattered-merge features; no --missable-tasks). --
    MODULAR_K8S_DIR="$BACKEND_AGNOSTIC_DIR/modular"
    mkdir -p "$MODULAR_K8S_DIR"
    MODULAR_DIR="$EXAMPLES_DIR/modular/backend_agnostic"

    # <label>|<operator substring>|<extra sflow args>|<-f file list...>
    while IFS='|' read -r m_label m_operator m_extra m_files; do
        [ -n "$m_label" ] || continue
        m_slug=$(echo "$m_label" | tr ' /' '__')
        m_log="$MODULAR_K8S_DIR/${m_slug}.log"
        m_ff=""
        for m_f in $m_files; do
            m_ff="$m_ff -f \"$MODULAR_DIR/$m_f\""
        done
        run_check "dry-run modular ${m_label} (required_by fold)" \
            bash -c "sflow run $m_ff $m_extra --dry-run --verbose > \"$m_log\" 2>&1 && \
                grep -F -- 'operator: ${m_operator}' \"$m_log\" && \
                grep -F -- \"depends_on: ['server']\" \"$m_log\" && \
                grep -F -- 'Dry-run complete: modular_inference' \"$m_log\""
    done <<'MODULAR_CASES'
dynamo_trtllm on k8s|k8s||backends/k8s.yaml workloads/dynamo_common.yaml workloads/dynamo_trtllm.yaml benchmark.yaml
dynamo_trtllm on k8s_mpi|k8s_mpi|-s NUM_NODES=2 -s SERVER_GPUS=16|backends/k8s_mpi.yaml workloads/dynamo_common.yaml workloads/dynamo_trtllm.yaml benchmark.yaml
dynamo_vllm on k8s|k8s||backends/k8s.yaml workloads/dynamo_common.yaml workloads/dynamo_vllm.yaml benchmark.yaml
dynamo_vllm on slurm|srun||backends/slurm.yaml workloads/dynamo_common.yaml workloads/dynamo_vllm.yaml benchmark.yaml
dynamo_sglang on k8s|k8s||backends/k8s.yaml workloads/dynamo_common.yaml workloads/dynamo_sglang.yaml benchmark.yaml
dynamo_sglang on slurm|srun||backends/slurm.yaml workloads/dynamo_common.yaml workloads/dynamo_sglang.yaml benchmark.yaml
trtllm_serve on k8s|k8s||backends/k8s.yaml workloads/trtllm_serve.yaml benchmark.yaml
trtllm_serve on k8s_mpi|k8s_mpi|-s NUM_NODES=2 -s SERVER_GPUS=16|backends/k8s_mpi.yaml workloads/trtllm_serve.yaml benchmark.yaml
trtllm_serve on slurm|srun||backends/slurm.yaml workloads/trtllm_serve.yaml benchmark.yaml
vllm_serve on k8s|k8s||backends/k8s.yaml workloads/vllm_serve.yaml benchmark.yaml
vllm_serve on slurm|srun||backends/slurm.yaml workloads/vllm_serve.yaml benchmark.yaml
sglang_serve on k8s|k8s||backends/k8s.yaml workloads/sglang_serve.yaml benchmark.yaml
sglang_serve on slurm|srun||backends/slurm.yaml workloads/sglang_serve.yaml benchmark.yaml
MODULAR_CASES

    # sflow compose of a modular K8s composition exercises the deep-merge +
    # required_by fold end to end and writes a merged snapshot.
    run_check "compose modular dynamo_trtllm on k8s" \
        sflow compose "$MODULAR_DIR/backends/k8s.yaml" \
            "$MODULAR_DIR/workloads/dynamo_common.yaml" \
            "$MODULAR_DIR/workloads/dynamo_trtllm.yaml" \
            "$MODULAR_DIR/benchmark.yaml" \
            -o "$MODULAR_K8S_DIR/composed_trtllm_k8s.yaml"

    # -- sflow run --dry-run: CLI backend extra-arg routing + kube access flags.
    #    Verifies the new flags reach the plan: --extra-salloc-args merges into the
    #    Slurm backend, --kube-namespace overrides the k8s namespace,
    #    --kube-node-selector merges into the k8s backend nodeSelector,
    #    --kube-compute-domain-channel/--kube-compute-domain-create override the recipe's
    #    Multi-Node NVLink (ComputeDomain) settings, and a generic --extra-args
    #    (Slurm-ism) routed to kubectl surfaces the misrouting warning. --
    CLI_FLAGS_DIR="$PREFLIGHT_DIR/cli_backend_flags"
    SALLOC_ARGS_LOG="$CLI_FLAGS_DIR/extra_salloc_args.log"
    KUBE_NS_LOG="$CLI_FLAGS_DIR/kube_namespace.log"
    KUBE_NODE_SELECTOR_LOG="$CLI_FLAGS_DIR/kube_node_selector.log"
    KUBE_COMPUTE_DOMAIN_LOG="$CLI_FLAGS_DIR/kube_compute_domain.log"
    KUBECTL_MISROUTE_LOG="$CLI_FLAGS_DIR/kubectl_misroute_warning.log"
    mkdir -p "$CLI_FLAGS_DIR"
    run_check "dry-run --extra-salloc-args merges into the Slurm backend" \
        bash -c "sflow run \"$EXAMPLES_DIR/self_contained/slurm/auto_replica.yaml\" --dry-run --verbose --extra-salloc-args=--gpus-per-node=4 > \"$SALLOC_ARGS_LOG\" 2>&1 && \
            grep -F -- \"'slurm': ['--gpus-per-node=4']\" \"$SALLOC_ARGS_LOG\""
    run_check "dry-run --kube-namespace overrides the kubernetes backend namespace" \
        bash -c "sflow run \"$EXAMPLES_DIR/self_contained/kubernetes/hello_world.yaml\" --dry-run --verbose --kube-namespace ns-override > \"$KUBE_NS_LOG\" 2>&1 && \
            grep -F -- 'namespace: ns-override' \"$KUBE_NS_LOG\""
    run_check "dry-run --kube-node-selector merges into the kubernetes backend nodeSelector" \
        bash -c "sflow run \"$EXAMPLES_DIR/self_contained/kubernetes/hello_world.yaml\" --dry-run --verbose --kube-node-selector tenant=gpu-pool,zone=z1 > \"$KUBE_NODE_SELECTOR_LOG\" 2>&1 && \
            grep -F -- \"node_selector: {'tenant': 'gpu-pool', 'zone': 'z1'}\" \"$KUBE_NODE_SELECTOR_LOG\""
    run_check "dry-run --kube-compute-domain-channel/--kube-compute-domain-create override the recipe" \
        bash -c "sflow run \"$EXAMPLES_DIR/self_contained/kubernetes/hello_world.yaml\" --dry-run --verbose --kube-compute-domain-channel auto --kube-compute-domain-create > \"$KUBE_COMPUTE_DOMAIN_LOG\" 2>&1 && \
            grep -F -- 'use_compute_domain_channel: auto' \"$KUBE_COMPUTE_DOMAIN_LOG\" && \
            grep -F -- 'create_compute_domain: True' \"$KUBE_COMPUTE_DOMAIN_LOG\""
    run_check "dry-run generic --extra-args routed to kubectl warns about misrouting" \
        bash -c "sflow run \"$EXAMPLES_DIR/self_contained/kubernetes/hello_world.yaml\" --dry-run -e --gpus-per-node=4 > \"$KUBECTL_MISROUTE_LOG\" 2>&1 && \
            grep -F -- 'applying generic --extra-args as kubectl' \"$KUBECTL_MISROUTE_LOG\""

    # -- sflow run --dry-run: standardized report layout (envelope + section
    #    dividers), compact-vs-verbose Tasks, storage/uploads sections, replica
    #    auto-rename, and sbatch out/err surfaced in the Plan via `sflow batch` --
    DRYRUN_FORMAT_DIR="$PREFLIGHT_DIR/dry_run_format"
    DRYRUN_COMPACT_LOG="$DRYRUN_FORMAT_DIR/local_dag_compact.log"
    DRYRUN_VERBOSE_LOG="$DRYRUN_FORMAT_DIR/local_dag_verbose.log"
    DRYRUN_UPLOADS_LOG="$DRYRUN_FORMAT_DIR/local_storage_upload.log"
    DRYRUN_UPLOAD_ALL_LOG="$DRYRUN_FORMAT_DIR/local_storage_upload_all.log"
    DRYRUN_REPLICA_FIXTURE="$DRYRUN_FORMAT_DIR/replica_uploads.yaml"
    DRYRUN_REPLICA_LOG="$DRYRUN_FORMAT_DIR/replica_uploads.log"
    DRYRUN_BATCH_LOG="$DRYRUN_FORMAT_DIR/sbatch_plan.log"
    mkdir -p "$DRYRUN_FORMAT_DIR"

    # Default dry-run: standardized '── Title ──' sections + one compact line per
    # task + the --verbose hint; full per-task detail (operator config) is hidden.
    run_check "dry-run report uses standardized sections + compact tasks by default" \
        bash -c "sflow run \"$EXAMPLES_DIR/self_contained/local/dag.yaml\" --dry-run > \"$DRYRUN_COMPACT_LOG\" 2>&1 && \
            grep -F -- '── Plan ' \"$DRYRUN_COMPACT_LOG\" && \
            grep -F -- '── Tasks ' \"$DRYRUN_COMPACT_LOG\" && \
            grep -F -- '(backend=local, operator=bash' \"$DRYRUN_COMPACT_LOG\" && \
            grep -F -- '(use --verbose for full per-task details)' \"$DRYRUN_COMPACT_LOG\" && \
            ! grep -F -- 'operator config' \"$DRYRUN_COMPACT_LOG\""

    # --verbose dry-run: full per-task detail (tree + operator config); no hint.
    run_check "dry-run --verbose expands full per-task detail" \
        bash -c "sflow run \"$EXAMPLES_DIR/self_contained/local/dag.yaml\" --dry-run --verbose > \"$DRYRUN_VERBOSE_LOG\" 2>&1 && \
            grep -F -- 'operator config' \"$DRYRUN_VERBOSE_LOG\" && \
            grep -F -- 'task_output_dir:' \"$DRYRUN_VERBOSE_LOG\" && \
            ! grep -F -- '(use --verbose for full per-task details)' \"$DRYRUN_VERBOSE_LOG\""

    # Storage targets + planned uploads sections render in the dry-run plan.
    run_check "dry-run renders storage targets and planned uploads" \
        bash -c "sflow run \"$EXAMPLES_DIR/self_contained/local/storage_upload.yaml\" --dry-run > \"$DRYRUN_UPLOADS_LOG\" 2>&1 && \
            grep -F -- '── Storage targets ' \"$DRYRUN_UPLOADS_LOG\" && \
            grep -F -- '[results_bucket] S3StorageTarget' \"$DRYRUN_UPLOADS_LOG\" && \
            grep -F -- '── Planned uploads ' \"$DRYRUN_UPLOADS_LOG\" && \
            grep -F -- '→ results_bucket:main/results.csv' \"$DRYRUN_UPLOADS_LOG\""

    # Workflow-level upload_all: dry-run should show the target plus the planned
    # whole-workflow archive without contacting S3.
    run_check "dry-run renders planned workflow upload_all archive" \
        bash -c "sflow run \"$EXAMPLES_DIR/self_contained/local/storage_upload_all.yaml\" --dry-run > \"$DRYRUN_UPLOAD_ALL_LOG\" 2>&1 && \
            grep -F -- '── Storage targets ' \"$DRYRUN_UPLOAD_ALL_LOG\" && \
            grep -F -- '[workflow_archive] S3StorageTarget' \"$DRYRUN_UPLOAD_ALL_LOG\" && \
            grep -F -- '── Planned workflow upload ' \"$DRYRUN_UPLOAD_ALL_LOG\" && \
            grep -F -- '→ workflow_archive:archive/\${{ workflow.run_id }}.zip' \"$DRYRUN_UPLOAD_ALL_LOG\""

    # Replica uploads: a literal `to:` is auto-renamed per replica, while a `to:`
    # that already references ${{ task.name }} is left untouched (opt-out honored).
    cat > "$DRYRUN_REPLICA_FIXTURE" <<'EOF'
version: "0.1"
storage:
  - name: bucket
    type: s3
    bucket: my-bucket
workflow:
  name: replica_uploads
  tasks:
    - name: bench
      replicas:
        count: 2
        policy: parallel
      script:
        - echo hi
      uploads:
        - target: bucket
          from: "${{ task.output_dir }}/r.csv"
          to: "main/r.csv"
        - target: bucket
          from: "${{ task.output_dir }}/r.csv"
          to: "${{ task.name }}/r.csv"
EOF
    run_check "dry-run auto-renames literal replica uploads (task.name opt-out honored)" \
        bash -c "sflow run \"$DRYRUN_REPLICA_FIXTURE\" --dry-run > \"$DRYRUN_REPLICA_LOG\" 2>&1 && \
            grep -F -- 'auto-renamed per replica' \"$DRYRUN_REPLICA_LOG\" && \
            grep -F -- '}}/r.csv  (on_error=warn)' \"$DRYRUN_REPLICA_LOG\""

    # `sflow batch` surfaces the actual sbatch stdout/stderr paths in the dry-run Plan.
    run_check "sflow batch dry-run Plan shows sbatch out/err paths" \
        bash -c "sflow batch \"$EXAMPLES_DIR/self_contained/local/dag.yaml\" --partition batch --account acct --nodes 1 > \"$DRYRUN_BATCH_LOG\" 2>&1 && \
            grep -F -- 'sbatch out:' \"$DRYRUN_BATCH_LOG\" && \
            grep -F -- 'sbatch err:' \"$DRYRUN_BATCH_LOG\""

    # -- sflow run (live): verify replica sweep + domain resolution --
    # Note: may fail in sandboxed environments (pty device limits) with many parallel tasks.
    LOCAL_DAG_RUN_DIR="$PREFLIGHT_DIR/run_local_dag"
    run_check "run local_dag (live)" \
        sflow run "$EXAMPLES_DIR/self_contained/local/dag.yaml" \
            --output-dir "$LOCAL_DAG_RUN_DIR"

    DOMAIN_RUN_DIR="$PREFLIGHT_DIR/run_variable_domain"
    run_check "run local_variable_domain (live, optional)" \
        sflow run "$EXAMPLES_DIR/self_contained/local/variable_domain.yaml" \
            --output-dir "$DOMAIN_RUN_DIR"

    # -- sflow run (live): per-task subprocess output is captured in the per-task
    #    <task>.log but kept OUT of sflow.log (orchestration-only). The task PRINTS
    #    a sentinel via `printf` that concatenates two args, so the concatenated
    #    sentinel never appears in the logged command text -- making the sflow.log
    #    exclusion assertion precise. This guards the "firehose floods sflow.log /
    #    Slurm stdout" regression. --
    TASK_LOG_ROUTING_DIR="$PREFLIGHT_DIR/task_log_routing"
    TASK_LOG_ROUTING_FIXTURE="$TASK_LOG_ROUTING_DIR/task_log_routing.yaml"
    TASK_LOG_ROUTING_RUN="$TASK_LOG_ROUTING_DIR/run"
    mkdir -p "$TASK_LOG_ROUTING_DIR"
    cat > "$TASK_LOG_ROUTING_FIXTURE" <<'EOF'
version: "0.1"
workflow:
  name: task_log_routing
  tasks:
    - name: printer
      script:
        - printf '%s%s\n' PERTASKLOG SENTINEL
EOF
    run_check "run per-task output routed to <task>.log not sflow.log (live)" \
        bash -c "set -euo pipefail
            sflow run '$TASK_LOG_ROUTING_FIXTURE' --output-dir '$TASK_LOG_ROUTING_RUN' > '$TASK_LOG_ROUTING_DIR/run.log' 2>&1
            sflow_log=\$(find '$TASK_LOG_ROUTING_RUN' -name sflow.log -print -quit)
            task_log=\$(find '$TASK_LOG_ROUTING_RUN' -name printer.log -print -quit)
            test -n \"\$sflow_log\"
            test -n \"\$task_log\"
            # Per-task subprocess stdout is captured in the per-task log ...
            grep -Fq 'PERTASKLOGSENTINEL' \"\$task_log\"
            # ... and must NOT leak into sflow.log (reserved for orchestration).
            if grep -Fq 'PERTASKLOGSENTINEL' \"\$sflow_log\"; then
                echo 'FAIL: per-task subprocess output leaked into sflow.log'
                exit 1
            fi
            # Orchestration command hints are still recorded in sflow.log.
            grep -Fq '========== Command ==========' \"\$sflow_log\""

    # -- sflow run (live): verify consolidated result parsing from logs and files --
    RESULT_PARSING_DIR="$PREFLIGHT_DIR/result_parsing"
    RESULT_PARSING_SAMPLE="$EXAMPLES_DIR/self_contained/local/result_parsing.yaml"
    RESULT_PARSING_RUN_ROOT="$RESULT_PARSING_DIR/run"
    mkdir -p "$RESULT_PARSING_DIR"
    run_check "run result parsing workflow (live)" \
        bash -c "set -euo pipefail
            sflow run '$RESULT_PARSING_SAMPLE' --output-dir '$RESULT_PARSING_RUN_ROOT' > '$RESULT_PARSING_DIR/run.log' 2>&1
            python - '$RESULT_PARSING_RUN_ROOT' <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
runs = [p for p in root.iterdir() if p.is_dir()]
if len(runs) != 1:
    raise SystemExit(f'expected one result parsing run dir under {root}, got {runs}')
run_dir = runs[0]

log_payload = json.loads((run_dir / 'benchmark_log' / 'result.json').read_text())
file_payload = json.loads((run_dir / 'benchmark_file' / 'result.json').read_text())
index = json.loads((run_dir / 'results.json').read_text())
verify_text = (run_dir / 'verify' / 'verify.txt').read_text()

assert log_payload['schema_version'] == 'sflow.result.v1'
assert log_payload['status'] == 'COMPLETED'
assert log_payload['values'] == {'latency_p99': 88, 'tps': 123.0, 'ttft': 42.5}
assert file_payload['status'] == 'COMPLETED'
assert file_payload['values'] == {'errors': 0, 'throughput': 999.5}
assert index['schema_version'] == 'sflow.results.v1'
assert index['tasks']['benchmark_log']['status'] == 'COMPLETED'
assert index['tasks']['benchmark_log']['values']['ttft'] == 42.5
assert index['tasks']['benchmark_file']['status'] == 'COMPLETED'
assert index['tasks']['benchmark_file']['values']['throughput'] == 999.5
assert verify_text == 'RESULT_PARSING_SAMPLE_PASS\n'
print('result parsing sample PASS')
PY"

    # -- sflow run (live): verify per-task uploads and workflow.upload_all with
    # moto-backed S3. This exercises the real local-backend workflow execution,
    # upload code paths, and summary output without requiring external network or
    # real AWS credentials.
    STORAGE_E2E_DIR="$PREFLIGHT_DIR/storage_uploads_e2e"
    run_check "run storage uploads + workflow upload_all with moto (live)" \
        bash -c "set -euo pipefail
            mkdir -p '$STORAGE_E2E_DIR'
            python - '$REPO_DIR' '$STORAGE_E2E_DIR' <<'PY'
import os
import sys
import zipfile
from pathlib import Path

import boto3
from moto import mock_aws

repo = Path(sys.argv[1])
out_root = Path(sys.argv[2])

# moto ignores these values, but boto3's credential provider chain requires
# something to be present.
os.environ['AWS_ACCESS_KEY_ID'] = 'test'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'test'
os.environ['AWS_SESSION_TOKEN'] = 'test'
os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'

from sflow.app.sflow import SflowApp

bucket = 'my-bucket'
with mock_aws():
    s3 = boto3.client('s3', region_name='us-west-2')
    s3.create_bucket(
        Bucket=bucket,
        CreateBucketConfiguration={'LocationConstraint': 'us-west-2'},
    )

    per_task_out = SflowApp().run(
        file=repo / 'examples' / 'self_contained' / 'local' / 'storage_upload.yaml',
        dry_run=False,
        workspace_dir=repo,
        output_dir=out_root / 'per_task',
    )
    keys = sorted(
        obj['Key']
        for obj in s3.list_objects_v2(Bucket=bucket).get('Contents', [])
    )
    expected_per_task = {
        'sflow-demo/run-001/main/results.csv',
        'sflow-demo/run-001/summary.json',
    }
    missing = expected_per_task.difference(keys)
    if missing:
        raise SystemExit(f'missing per-task upload objects: {sorted(missing)}')
    per_task_summary = (per_task_out / 'sflow_summary.log').read_text()
    if 'Uploads     : uploaded=2' not in per_task_summary:
        raise SystemExit('per-task upload summary missing uploaded=2')

    upload_all_out = SflowApp().run(
        file=repo / 'examples' / 'self_contained' / 'local' / 'storage_upload_all.yaml',
        dry_run=False,
        workspace_dir=repo,
        output_dir=out_root / 'upload_all',
    )
    archive_key = f'sflow-demo/run-001/archive/{upload_all_out.name}.zip'
    keys = sorted(
        obj['Key']
        for obj in s3.list_objects_v2(Bucket=bucket).get('Contents', [])
    )
    if archive_key not in keys:
        raise SystemExit(f'missing workflow archive upload object: {archive_key}')
    archive_path = out_root / 'workflow_archive.zip'
    archive_path.write_bytes(
        s3.get_object(Bucket=bucket, Key=archive_key)['Body'].read()
    )
    with zipfile.ZipFile(archive_path) as zf:
        names = set(zf.namelist())
    required = {
        'sflow.log',
        'sflow_summary.log',
        'produce_results/results.csv',
        'produce_results/summary.json',
        'produce_report/report.txt',
    }
    missing = required.difference(names)
    if missing:
        raise SystemExit(f'workflow archive missing files: {sorted(missing)}')
    upload_all_summary = (upload_all_out / 'sflow_summary.log').read_text()
    if 'Uploads     : uploaded=1' not in upload_all_summary:
        raise SystemExit('workflow upload_all summary missing uploaded=1')

print('storage upload e2e PASS')
PY"

    # -- sflow run (live): verify TUI launch path --
    TUI_RUN_DIR="$PREFLIGHT_DIR/run_local_dag_tui"
    run_check "run local_dag with TUI (live)" \
        sflow run "$EXAMPLES_DIR/self_contained/local/dag.yaml" \
            --tui \
            --output-dir "$TUI_RUN_DIR"

    # -- sflow run (live): per-task log offload ON/OFF x --tui ON/OFF matrix --
    #    All four --offload-task-logs/--no-offload-task-logs x --tui/no-tui combos
    #    must (a) run to completion, (b) ALWAYS write the per-task <task>.log, and
    #    (c) route the console correctly: on a TTY task output is streamed to the
    #    console and offload auto-falls back to streaming; in batch/non-TTY mode
    #    offload writes <task>.log itself and the driver-side diagnostics are
    #    merged INTO that same <task>.log (no scattered <task>.orchestration.log
    #    sidecar), and task output never leaks into sflow.log. The task PRINTS a
    #    sentinel via `printf` that concatenates two args, so the joined sentinel
    #    only appears as real task stdout (never in the logged command text).
    OFFLOAD_TUI_DIR="$PREFLIGHT_DIR/offload_tui_matrix"
    OFFLOAD_TUI_FIXTURE="$OFFLOAD_TUI_DIR/offload_matrix.yaml"
    OFFLOAD_TUI_CHECK="$OFFLOAD_TUI_DIR/offload_tui_matrix_check.sh"
    mkdir -p "$OFFLOAD_TUI_DIR"
    cat > "$OFFLOAD_TUI_FIXTURE" <<'EOF'
version: "0.1"
workflow:
  name: offload_tui_matrix
  tasks:
    - name: printer
      script:
        - printf '%s%s\n' OFFLOADMATRIX SENTINEL
EOF
    cat > "$OFFLOAD_TUI_CHECK" <<'EOF'
#!/usr/bin/env bash
# Verify the per-task log offload x --tui matrix end to end. Args: <fixture> <base_dir>
set -uo pipefail
fixture="$1"
base="$2"
marker="OFFLOADMATRIXSENTINEL"
fail=0

# Pseudo-terminal harness: run argv on a PTY so sflow sees an interactive TTY
# (sys.stdout.isatty() == True), teeing the child's output to a capture file;
# exit with the child's status.
pty_run() {
    local capture="$1"; shift
    python - "$capture" "$@" <<'PY'
import os, pty, sys
capture_path = sys.argv[1]
argv = sys.argv[2:]
with open(capture_path, "wb") as cap:
    def _read(fd):
        data = os.read(fd, 1024)
        cap.write(data)
        cap.flush()
        return data
    status = pty.spawn(argv, _read)
raise SystemExit(os.waitstatus_to_exitcode(status))
PY
}

assert_task_log() {  # <run_dir> <slug>
    local run_dir="$1" slug="$2" task_log
    task_log=$(find "$run_dir" -name printer.log -print -quit)
    if [ -z "$task_log" ] || ! grep -Fq "$marker" "$task_log"; then
        echo "FAIL($slug): per-task printer.log missing the task output"
        return 1
    fi
    return 0
}

# 1) Four combos, batch/non-interactive: each completes, ALWAYS writes the
#    per-task <task>.log, and never leaks task output into sflow.log. Driver-side
#    diagnostics are merged into <task>.log, so there must be NO scattered
#    <task>.orchestration.log sidecar in any mode.
for offload in --no-offload-task-logs --offload-task-logs; do
    for tui in "" --tui; do
        slug="batch${offload}${tui}"; slug=${slug//-/}
        run_dir="$base/run_$slug"
        rm -rf "$run_dir"
        if ! sflow run "$fixture" $offload $tui --output-dir "$run_dir" \
                > "$base/$slug.log" 2>&1; then
            echo "FAIL($slug): sflow run exited non-zero"; fail=1; continue
        fi
        assert_task_log "$run_dir" "$slug" || { fail=1; continue; }
        sflow_log=$(find "$run_dir" -name sflow.log -print -quit)
        if [ -n "$sflow_log" ] && grep -Fq "$marker" "$sflow_log"; then
            echo "FAIL($slug): task output leaked into sflow.log"; fail=1; continue
        fi
        sidecar=$(find "$run_dir" -name '*.orchestration.log' -print -quit)
        if [ -n "$sidecar" ]; then
            echo "FAIL($slug): unexpected scattered orchestration sidecar $sidecar"; fail=1; continue
        fi
        echo "OK($slug)"
    done
done

# 2) On a TTY both offload OFF and ON stream task output to the console (offload
#    auto-falls back to streaming). Verify via a PTY-captured console + the file.
for offload in --no-offload-task-logs --offload-task-logs; do
    slug="tty${offload}"; slug=${slug//-/}
    run_dir="$base/run_$slug"
    cap="$base/$slug.console"
    rm -rf "$run_dir"
    if ! pty_run "$cap" sflow run "$fixture" $offload --output-dir "$run_dir" \
            > "$base/$slug.log" 2>&1; then
        echo "FAIL($slug): sflow run under a PTY exited non-zero"; fail=1; continue
    fi
    if ! grep -Fq "$marker" "$cap"; then
        echo "FAIL($slug): task output not streamed to the console on a TTY"; fail=1; continue
    fi
    assert_task_log "$run_dir" "$slug" || { fail=1; continue; }
    echo "OK($slug)"
done

if [ "$fail" -ne 0 ]; then
    echo "offload/TUI matrix: FAILED"
    exit 1
fi
echo "offload/TUI matrix: all combos PASS"
EOF
    run_check "run per-task log offload x TUI matrix (4 combos: file + console) (live)" \
        bash "$OFFLOAD_TUI_CHECK" "$OFFLOAD_TUI_FIXTURE" "$OFFLOAD_TUI_DIR"

    # -- sflow run/batch: plain script commands containing ':' must stay strings --
    COLON_SCRIPT_DIR="$PREFLIGHT_DIR/colon_in_task_script"
    COLON_SCRIPT_FIXTURE="$COLON_SCRIPT_DIR/colon_in_task_script.yaml"
    COLON_SCRIPT_DRYRUN_LOG="$COLON_SCRIPT_DIR/dry_run.log"
    COLON_SCRIPT_COMPOSED="$COLON_SCRIPT_DIR/colon_in_task_script_composed.yaml"
    COLON_SCRIPT_BATCH="$COLON_SCRIPT_DIR/colon_in_task_script_batch.sh"
    COLON_SCRIPT_BATCH_CONFIG="$COLON_SCRIPT_DIR/colon_in_task_script_batch.yaml"
    mkdir -p "$COLON_SCRIPT_DIR"
    cat > "$COLON_SCRIPT_FIXTURE" <<'EOF'
version: "0.1"

variables:
  SLURM_ACCOUNT:
    value: dummy_acct
  SLURM_PARTITION:
    value: dummy_part
  SLURM_TIMELIMIT:
    value: "00:10:00"
  SLURM_NODES:
    value: 1
  GPUS_PER_NODE:
    value: 4

backends:
  - name: slurm_cluster
    type: slurm
    default: true
    account: ${{ variables.SLURM_ACCOUNT }}
    partition: ${{ variables.SLURM_PARTITION }}
    time: ${{ variables.SLURM_TIMELIMIT }}
    nodes: ${{ variables.SLURM_NODES }}
    gpus_per_node: ${{ variables.GPUS_PER_NODE }}

operators:
  - name: srun_no_container
    type: srun
    ntasks_per_node: 1
    mpi: pmix

workflow:
  name: colon_in_task_script
  tasks:
    - name: worker
      operator: srun_no_container
      resources:
        gpus:
          count: 1
      script:
        - echo "My GPUs: $CUDA_VISIBLE_DEVICES"
        - echo "COLON_SCRIPT_E2E_PASS"
EOF
    run_check "run colon in task script (dry-run)" \
        bash -c "sflow run \"$COLON_SCRIPT_FIXTURE\" --dry-run > \"$COLON_SCRIPT_DRYRUN_LOG\" 2>&1"
    run_check "compose colon in task script" \
        sflow compose "$COLON_SCRIPT_FIXTURE" \
            -o "$COLON_SCRIPT_COMPOSED"
    run_check "batch colon in task script" \
        sflow batch -f "$COLON_SCRIPT_FIXTURE" \
            -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
            -o "$COLON_SCRIPT_BATCH"

    if [ "$TEST_TYPE" = "m" ] || [ "$TEST_TYPE" = "a" ]; then
        INFMAX_BATCH_PREFLIGHT_DIR="$PREFLIGHT_DIR/infmax_batch"
        if fetch_infmax_recipes; then
            INFMAX_TARGET_DIRS=(
                "$INFMAX_DIR/dsr1-fp8-gb200-multi_node-sglang"
                "$INFMAX_DIR/dsr1-fp4-gb200-multi_node-trtllm"
                "$INFMAX_DIR/kimik2.5-fp4-gb200-multi_node-vllm"
            )
            for suite_dir in "${INFMAX_TARGET_DIRS[@]}"; do
                suite_name=$(basename "$suite_dir")
                set_infmax_suite_overrides "$suite_name"
                run_check "infmax multi-node batch $suite_name" \
                    sflow batch \
                        -f "$INFMAX_MONITOR_CONFIG" \
                        --bulk-input "$suite_dir/$INFMAX_CSV_NAME" \
                        --row "$INFMAX_ROW" \
                        -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
                        -a "BENCH_SERVING_DIR=fs://$INFMAX_BENCH_SERVING_DIR" \
                        -G "$GPUS_PER_NODE" \
                        -A "$ACCOUNT" \
                        -p "$PARTITION" \
                        --log-level warn \
                        "${INFMAX_SUITE_OVERRIDES[@]}" \
                        --output-dir "$INFMAX_BATCH_PREFLIGHT_DIR/$suite_name"
            done
        else
            record_preflight_skip "Infmax batch preflight checks skipped: ${INFMAX_RECIPE_SKIP_REASON:-recipe source unavailable}"
            echo "  SKIP: Infmax recipe source unavailable; skipping Infmax batch preflight checks."
        fi
    fi

    # -- sflow run/batch/compose: release_after enables dynamic resource rehearsal --
    RELEASE_AFTER_DIR="$PREFLIGHT_DIR/release_after_resource_rehearsal"
    RELEASE_AFTER_FIXTURE="$RELEASE_AFTER_DIR/release_after_inferred_task_completion.yaml"
    RELEASE_AFTER_NEGATIVE_FIXTURE="$RELEASE_AFTER_DIR/release_after_explicit_workflow_completion.yaml"
    RELEASE_AFTER_NODES_OMITTED_FIXTURE="$RELEASE_AFTER_DIR/nodes_release_after_omitted_overlap.yaml"
    RELEASE_AFTER_NODES_COUNT_NEGATIVE_FIXTURE="$RELEASE_AFTER_DIR/nodes_release_after_explicit_count.yaml"
    RELEASE_AFTER_NODES_INDICES_NEGATIVE_FIXTURE="$RELEASE_AFTER_DIR/nodes_release_after_explicit_indices.yaml"
    RELEASE_AFTER_DRYRUN_LOG="$RELEASE_AFTER_DIR/dry_run.log"
    RELEASE_AFTER_NEGATIVE_LOG="$RELEASE_AFTER_DIR/negative_dry_run.log"
    RELEASE_AFTER_NODES_OMITTED_LOG="$RELEASE_AFTER_DIR/nodes_omitted_dry_run.log"
    RELEASE_AFTER_NODES_COUNT_NEGATIVE_LOG="$RELEASE_AFTER_DIR/nodes_count_negative_dry_run.log"
    RELEASE_AFTER_NODES_INDICES_NEGATIVE_LOG="$RELEASE_AFTER_DIR/nodes_indices_negative_dry_run.log"
    RELEASE_AFTER_COMPOSE_LOG="$RELEASE_AFTER_DIR/compose_validate.log"
    RELEASE_AFTER_COMPOSED="$RELEASE_AFTER_DIR/composed.yaml"
    RELEASE_AFTER_BATCH="$RELEASE_AFTER_DIR/release_after_batch.sh"
    mkdir -p "$RELEASE_AFTER_DIR"
    cat > "$RELEASE_AFTER_FIXTURE" <<'EOF'
version: "0.1"

variables:
  SLURM_ACCOUNT:
    value: dummy_acct
  SLURM_PARTITION:
    value: dummy_part

backends:
  - name: slurm_cluster
    type: slurm
    default: true
    account: ${{ variables.SLURM_ACCOUNT }}
    partition: ${{ variables.SLURM_PARTITION }}
    time: "00:10:00"
    nodes: 1
    gpus_per_node: 8

workflow:
  name: release_after_inferred_task_completion
  tasks:
    - name: check_entire_env
      resources:
        gpus:
          count: 8
      script:
        - echo "$CUDA_VISIBLE_DEVICES"

    - name: worker
      replicas:
        count: 4
        policy: parallel
      resources:
        gpus:
          count: 2
      script:
        - echo "worker GPUs: $CUDA_VISIBLE_DEVICES"
      depends_on:
        - check_entire_env
EOF
    cat > "$RELEASE_AFTER_NEGATIVE_FIXTURE" <<'EOF'
version: "0.1"

backends:
  - name: slurm_cluster
    type: slurm
    default: true
    account: dummy_acct
    partition: dummy_part
    time: "00:10:00"
    nodes: 1
    gpus_per_node: 8

workflow:
  name: release_after_explicit_workflow_completion
  tasks:
    - name: check_entire_env
      resources:
        gpus:
          count: 8
          release_after: workflow_completion
      script:
        - echo "$CUDA_VISIBLE_DEVICES"

    - name: worker
      replicas:
        count: 4
        policy: parallel
      resources:
        gpus:
          count: 2
      script:
        - echo "worker GPUs: $CUDA_VISIBLE_DEVICES"
      depends_on:
        - check_entire_env
EOF
    cat > "$RELEASE_AFTER_NODES_OMITTED_FIXTURE" <<'EOF'
version: "0.1"

backends:
  - name: slurm_cluster
    type: slurm
    default: true
    account: dummy_acct
    partition: dummy_part
    time: "00:10:00"
    nodes: 1
    gpus_per_node: 0

workflow:
  name: nodes_release_after_omitted_overlap
  tasks:
    - name: count_a
      resources:
        nodes:
          count: 1
      script:
        - echo count_a

    - name: count_b
      resources:
        nodes:
          count: 1
      script:
        - echo count_b

    - name: pinned_a
      resources:
        nodes:
          indices: [0]
      script:
        - echo pinned_a

    - name: pinned_b
      resources:
        nodes:
          indices: [0]
      script:
        - echo pinned_b
EOF
    cat > "$RELEASE_AFTER_NODES_COUNT_NEGATIVE_FIXTURE" <<'EOF'
version: "0.1"

backends:
  - name: slurm_cluster
    type: slurm
    default: true
    account: dummy_acct
    partition: dummy_part
    time: "00:10:00"
    nodes: 1
    gpus_per_node: 0

workflow:
  name: nodes_release_after_explicit_count
  tasks:
    - name: exclusive_count
      resources:
        nodes:
          count: 1
          release_after: workflow_completion
      script:
        - echo exclusive_count

    - name: sibling
      resources:
        nodes:
          count: 1
      script:
        - echo sibling
EOF
    cat > "$RELEASE_AFTER_NODES_INDICES_NEGATIVE_FIXTURE" <<'EOF'
version: "0.1"

backends:
  - name: slurm_cluster
    type: slurm
    default: true
    account: dummy_acct
    partition: dummy_part
    time: "00:10:00"
    nodes: 1
    gpus_per_node: 0

workflow:
  name: nodes_release_after_explicit_indices
  tasks:
    - name: exclusive_index
      resources:
        nodes:
          indices: [0]
          release_after: workflow_completion
      script:
        - echo exclusive_index

    - name: sibling
      resources:
        nodes:
          count: 1
      script:
        - echo sibling
EOF
    run_check "run release_after resource rehearsal (dry-run)" \
        bash -c "sflow run \"$RELEASE_AFTER_FIXTURE\" --dry-run > \"$RELEASE_AFTER_DRYRUN_LOG\" 2>&1 && grep -q 'Resource Occupancy' \"$RELEASE_AFTER_DRYRUN_LOG\""
    run_check "batch release_after resource rehearsal" \
        sflow batch -f "$RELEASE_AFTER_FIXTURE" \
            -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
            -o "$RELEASE_AFTER_BATCH"
    run_check "compose release_after resource rehearsal (validate)" \
        bash -c "sflow compose \"$RELEASE_AFTER_FIXTURE\" --validate -o \"$RELEASE_AFTER_COMPOSED\" > \"$RELEASE_AFTER_COMPOSE_LOG\" 2>&1 && grep -q 'Dry-run validation passed' \"$RELEASE_AFTER_COMPOSE_LOG\" && ! grep -q 'WARNING: dry-run validation failed' \"$RELEASE_AFTER_COMPOSE_LOG\""
    run_check "run explicit workflow_completion resource lifetime (expect fail)" \
        bash -c "! sflow run \"$RELEASE_AFTER_NEGATIVE_FIXTURE\" --dry-run > \"$RELEASE_AFTER_NEGATIVE_LOG\" 2>&1 && grep -q 'remain available' \"$RELEASE_AFTER_NEGATIVE_LOG\""
    run_check "run omitted node release_after allows overlap (dry-run)" \
        bash -c "sflow run \"$RELEASE_AFTER_NODES_OMITTED_FIXTURE\" --dry-run > \"$RELEASE_AFTER_NODES_OMITTED_LOG\" 2>&1 && grep -q 'Dry-run complete' \"$RELEASE_AFTER_NODES_OMITTED_LOG\" && ! grep -q 'remain available' \"$RELEASE_AFTER_NODES_OMITTED_LOG\""
    run_check "run explicit node count workflow_completion (expect fail)" \
        bash -c "! sflow run \"$RELEASE_AFTER_NODES_COUNT_NEGATIVE_FIXTURE\" --dry-run > \"$RELEASE_AFTER_NODES_COUNT_NEGATIVE_LOG\" 2>&1 && grep -q 'remain available' \"$RELEASE_AFTER_NODES_COUNT_NEGATIVE_LOG\" && grep -q 'exclusive_count: release_after=workflow_completion' \"$RELEASE_AFTER_NODES_COUNT_NEGATIVE_LOG\""
    run_check "run explicit node indices workflow_completion (expect fail)" \
        bash -c "! sflow run \"$RELEASE_AFTER_NODES_INDICES_NEGATIVE_FIXTURE\" --dry-run > \"$RELEASE_AFTER_NODES_INDICES_NEGATIVE_LOG\" 2>&1 && grep -q 'remain available' \"$RELEASE_AFTER_NODES_INDICES_NEGATIVE_LOG\" && grep -q 'exclusive_index: release_after=workflow_completion' \"$RELEASE_AFTER_NODES_INDICES_NEGATIVE_LOG\""

    # -- sflow run --dry-run: readiness accepts a list and builds multiple readiness probes --
    READINESS_AND_DIR="$PREFLIGHT_DIR/readiness_probe_and"
    READINESS_AND_FIXTURE="$READINESS_AND_DIR/readiness_probe_and.yaml"
    READINESS_AND_DRYRUN_LOG="$READINESS_AND_DIR/dry_run.log"
    READINESS_SINGLE_FIXTURE="$READINESS_AND_DIR/readiness_probe_single.yaml"
    READINESS_SINGLE_DRYRUN_LOG="$READINESS_AND_DIR/single_dry_run.log"
    mkdir -p "$READINESS_AND_DIR"
    cat > "$READINESS_AND_FIXTURE" <<'EOF'
version: "0.1"
workflow:
  name: readiness_probe_and
  tasks:
    - name: service
      script:
        - echo "readiness one"
        - sleep 1
        - echo "readiness two"
        - touch "${SFLOW_WORKFLOW_OUTPUT_DIR}/all_readiness_probes_passed"
        - sleep 2
      probes:
        readiness:
          - log_watch:
              match_pattern: "readiness one"
            interval: 0
            timeout: 10
          - log_watch:
              match_pattern: "readiness two"
            interval: 0
            timeout: 10
    - name: after_ready
      depends_on:
        - service
      script:
        - test -f "${SFLOW_WORKFLOW_OUTPUT_DIR}/all_readiness_probes_passed"
        - echo "after_ready observed all readiness probes"
EOF
    run_check "run readiness probe list (dry-run)" \
        bash -c "sflow run \"$READINESS_AND_FIXTURE\" --dry-run --verbose > \"$READINESS_AND_DRYRUN_LOG\" 2>&1"
    cat > "$READINESS_SINGLE_FIXTURE" <<'EOF'
version: "0.1"
workflow:
  name: readiness_probe_single_compat
  tasks:
    - name: service
      script:
        - echo "single readiness"
      probes:
        readiness:
          log_watch:
            match_pattern: "single readiness"
          interval: 0
          timeout: 10
EOF
    run_check "run single readiness probe compatibility (dry-run)" \
        bash -c "sflow run \"$READINESS_SINGLE_FIXTURE\" --dry-run --verbose > \"$READINESS_SINGLE_DRYRUN_LOG\" 2>&1"

    # -- sflow run --dry-run: user variables colliding with reserved envs warn --
    RESERVED_ENV_DIR="$PREFLIGHT_DIR/reserved_env_collision"
    RESERVED_ENV_FIXTURE="$RESERVED_ENV_DIR/reserved_env_collision.yaml"
    RESERVED_ENV_DRYRUN_LOG="$RESERVED_ENV_DIR/dry_run.log"
    mkdir -p "$RESERVED_ENV_DIR"
    cat > "$RESERVED_ENV_FIXTURE" <<'EOF'
version: "0.1"
variables:
  SFLOW_TASK_OUTPUT_DIR:
    value: "/tmp/should-not-be-a-variable"
  CUDA_VISIBLE_DEVICES:
    value: "0,1"
  MODEL_PATH:
    value: "/models/demo"
backends:
  - name: local
    type: local
    default: true
    nodes: 1
workflow:
  name: reserved_env_collision
  tasks:
    - name: t1
      script:
        - echo hi
EOF
    run_check "run reserved env collision warning (dry-run)" \
        bash -c "sflow run \"$RESERVED_ENV_FIXTURE\" --dry-run > \"$RESERVED_ENV_DRYRUN_LOG\" 2>&1 && \
            grep -F -- 'Reserved env collisions' \"$RESERVED_ENV_DRYRUN_LOG\" && \
            grep -F -- '⚠ SFLOW_TASK_OUTPUT_DIR' \"$RESERVED_ENV_DRYRUN_LOG\" && \
            grep -F -- '⚠ CUDA_VISIBLE_DEVICES' \"$RESERVED_ENV_DRYRUN_LOG\" && \
            ! grep -F -- '⚠ MODEL_PATH' \"$RESERVED_ENV_DRYRUN_LOG\""

    # -- sflow run --dry-run: self-contained slurm examples --
    for f in "$EXAMPLES_DIR"/self_contained/slurm/*.yaml; do
        name=$(basename "$f" .yaml)
        run_check "dry-run $name" \
            sflow run "$f" --dry-run \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH"
    done

    # -- sflow run (fake Slurm): backend gpus_per_node is planning-only; --gpus-per-node is extra_args-only --
    SLURM_GPN_DIR="$PREFLIGHT_DIR/slurm_gpus_per_node_extra_args_only"
    SLURM_GPN_FIXTURE_DIR="$SLURM_GPN_DIR/fixture"
    SLURM_GPN_FAKE_BIN="$SLURM_GPN_DIR/fake_bin"
    SLURM_GPN_LOG_DIR="$SLURM_GPN_DIR/logs"
    mkdir -p "$SLURM_GPN_FIXTURE_DIR" "$SLURM_GPN_FAKE_BIN" "$SLURM_GPN_LOG_DIR"
    cat > "$SLURM_GPN_FAKE_BIN/salloc" <<'EOF'
#!/bin/bash
{
    printf 'salloc'
    for arg in "$@"; do
        printf ' %s' "$arg"
    done
    printf '\n'
} >> "$SFLOW_FAKE_SLURM_LOG_DIR/salloc.args"
echo "salloc: Granted job allocation 424242"
echo "salloc: Nodes fake-node are ready for job"
EOF
    cat > "$SLURM_GPN_FAKE_BIN/scontrol" <<'EOF'
#!/bin/bash
if [ "$1" = "getaddrs" ]; then
    echo "fake-node: 127.0.0.1:123"
    exit 0
fi
exit 1
EOF
    cat > "$SLURM_GPN_FAKE_BIN/srun" <<'EOF'
#!/bin/bash
{
    printf 'srun'
    for arg in "$@"; do
        printf ' %s' "$arg"
    done
    printf '\n'
} >> "$SFLOW_FAKE_SLURM_LOG_DIR/srun.args"
exit 0
EOF
    cat > "$SLURM_GPN_FAKE_BIN/scancel" <<'EOF'
#!/bin/bash
echo "scancel $*" >> "$SFLOW_FAKE_SLURM_LOG_DIR/scancel.args"
exit 0
EOF
    chmod +x "$SLURM_GPN_FAKE_BIN"/salloc "$SLURM_GPN_FAKE_BIN"/scontrol \
        "$SLURM_GPN_FAKE_BIN"/srun "$SLURM_GPN_FAKE_BIN"/scancel
    cat > "$SLURM_GPN_FIXTURE_DIR/no_extra_args.yaml" <<'EOF'
version: "0.1"
backends:
  - name: fake_slurm
    type: slurm
    default: true
    account: acct
    partition: gpu
    time: "00:05:00"
    nodes: 1
    gpus_per_node: 4
workflow:
  name: slurm_no_implicit_gpus_per_node
  tasks:
    - name: worker
      script:
        - echo no implicit gpus-per-node
EOF
    cat > "$SLURM_GPN_FIXTURE_DIR/with_extra_args.yaml" <<'EOF'
version: "0.1"
backends:
  - name: fake_slurm
    type: slurm
    default: true
    account: acct
    partition: gpu
    time: "00:05:00"
    nodes: 1
    gpus_per_node: 4
    extra_args:
      - "--gpus-per-node=4"
workflow:
  name: slurm_explicit_gpus_per_node
  tasks:
    - name: worker
      script:
        - echo explicit gpus-per-node
EOF
    run_check "run fake slurm gpus-per-node extra_args-only" \
        bash -c "set -euo pipefail
            export PATH='$SLURM_GPN_FAKE_BIN':\"\$PATH\"
            export SFLOW_FAKE_SLURM_LOG_DIR='$SLURM_GPN_LOG_DIR'
            # This fake-Slurm check must exercise SlurmBackend's salloc path.
            # If the developer runs preflight from inside a real Slurm allocation,
            # ambient SLURM_* vars would make sflow reuse that allocation and never
            # call the fake salloc, causing this assertion to fail.
            unset SLURM_JOB_ID SLURM_JOBID SLURM_JOB_NODELIST SLURM_NODELIST SLURM_NNODES SLURM_HET_SIZE
            rm -f \"\$SFLOW_FAKE_SLURM_LOG_DIR\"/*.args
            no_extra_log='$SLURM_GPN_DIR/no_extra_run.log'
            with_extra_log='$SLURM_GPN_DIR/with_extra_run.log'
            sflow run '$SLURM_GPN_FIXTURE_DIR/no_extra_args.yaml' \
                --output-dir '$SLURM_GPN_DIR/no_extra_run' \
                > \"\$no_extra_log\" 2>&1
            first_salloc=\$(sed -n '1p' \"\$SFLOW_FAKE_SLURM_LOG_DIR/salloc.args\")
            if printf '%s\n' \"\$first_salloc\" | grep -q -- '--gpus-per-node'; then
                echo \"unexpected implicit --gpus-per-node in: \$first_salloc\"
                exit 1
            fi
            grep -Fq -- 'backend.gpus_per_node=4 is sflow planning only' \"\$no_extra_log\"
            sflow run '$SLURM_GPN_FIXTURE_DIR/with_extra_args.yaml' \
                --output-dir '$SLURM_GPN_DIR/with_extra_run' \
                > \"\$with_extra_log\" 2>&1
            second_salloc=\$(sed -n '2p' \"\$SFLOW_FAKE_SLURM_LOG_DIR/salloc.args\")
            if ! printf '%s\n' \"\$second_salloc\" | grep -Fq -- '--gpus-per-node=4'; then
                echo \"expected explicit --gpus-per-node=4 in second fake salloc, got: \${second_salloc:-<empty>}\"
                echo \"all fake salloc calls:\"
                sed -n '1,20p' \"\$SFLOW_FAKE_SLURM_LOG_DIR/salloc.args\" 2>/dev/null || true
                exit 1
            fi
            grep -Fq -- 'backend.gpus_per_node=4 is sflow planning only' \"\$with_extra_log\""

    # -- sflow run (fake Slurm): controller SLURM_* envs are preserved and mirrored to SFLOW_* aliases --
    SLURM_ENV_DIR="$PREFLIGHT_DIR/slurm_controller_env_aliases"
    SLURM_ENV_FIXTURE_DIR="$SLURM_ENV_DIR/fixture"
    SLURM_ENV_FAKE_BIN="$SLURM_ENV_DIR/fake_bin"
    SLURM_ENV_LOG_DIR="$SLURM_ENV_DIR/logs"
    mkdir -p "$SLURM_ENV_FIXTURE_DIR" "$SLURM_ENV_FAKE_BIN" "$SLURM_ENV_LOG_DIR"
    cat > "$SLURM_ENV_FAKE_BIN/salloc" <<'EOF'
#!/bin/bash
echo "unexpected salloc $*" >> "$SFLOW_FAKE_SLURM_LOG_DIR/salloc.args"
exit 42
EOF
    cat > "$SLURM_ENV_FAKE_BIN/scontrol" <<'EOF'
#!/bin/bash
if [ "$1" = "getaddrs" ]; then
    echo "fake-node: 127.0.0.1:123"
    exit 0
fi
exit 1
EOF
    cat > "$SLURM_ENV_FAKE_BIN/srun" <<'EOF'
#!/bin/bash
{
    printf 'srun'
    for arg in "$@"; do
        printf ' %s' "$arg"
    done
    printf '\n'
} >> "$SFLOW_FAKE_SLURM_LOG_DIR/srun.args"

while [ "$#" -gt 0 ]; do
    if [ "$1" = "bash" ] && [ "${2:-}" = "-c" ]; then
        shift 2
        export SLURM_STEP_ID=3
        export SLURMD_NODENAME=fake-node
        export SLURM_NODEID=0
        export SLURM_PROCID=5
        export SLURM_LOCALID=0
        export SLURM_NTASKS=1
        bash -c "$1"
        exit $?
    fi
    shift
done
exit 0
EOF
    cat > "$SLURM_ENV_FAKE_BIN/scancel" <<'EOF'
#!/bin/bash
echo "scancel $*" >> "$SFLOW_FAKE_SLURM_LOG_DIR/scancel.args"
exit 0
EOF
    chmod +x "$SLURM_ENV_FAKE_BIN"/salloc "$SLURM_ENV_FAKE_BIN"/scontrol \
        "$SLURM_ENV_FAKE_BIN"/srun "$SLURM_ENV_FAKE_BIN"/scancel
    cat > "$SLURM_ENV_FIXTURE_DIR/controller_env_aliases.yaml" <<'EOF'
version: "0.1"
variables:
  - name: SLURM_JOB_ID
    value: workflow-shadow
  - name: SLURM_CUSTOM_CONTROLLER_ENV
    value: workflow-shadow
backends:
  - name: fake_slurm
    type: slurm
    default: true
    account: acct
    partition: gpu
    time: "00:05:00"
    nodes: 1
    gpus_per_node: 0
workflow:
  name: slurm_controller_env_aliases
  tasks:
    - name: worker
      script:
        - 'env | sort > "${SFLOW_TASK_OUTPUT_DIR}/slurm_env.txt"'
        - 'test "$SLURM_JOB_ID" = "777777"'
        - 'test "$SLURM_CUSTOM_CONTROLLER_ENV" = "controller-kept"'
        - 'test "$SFLOW_BACKEND_JOB_ID" = "777777"'
        - 'test "$SFLOW_BACKEND_NODELIST" = "fake-node"'
        - 'test "$SFLOW_BACKEND_NUM_NODES" = "1"'
        - 'test "$SFLOW_BACKEND_STEP_ID" = "3"'
        - 'test "$SFLOW_TASK_NODE_NAME" = "fake-node"'
        - 'test "$SFLOW_TASK_NODE_INDEX" = "0"'
        - 'test "$SFLOW_TASK_PROCESS_ID" = "5"'
        - 'test "$SFLOW_TASK_LOCAL_PROCESS_ID" = "0"'
        - 'test "$SFLOW_TASK_NUM_PROCESSES" = "1"'
EOF
    run_check "run fake slurm controller env inheritance and sflow aliases" \
        bash -c "set -euo pipefail
            export PATH='$SLURM_ENV_FAKE_BIN':\"\$PATH\"
            export SFLOW_FAKE_SLURM_LOG_DIR='$SLURM_ENV_LOG_DIR'
            export SLURM_JOB_ID=777777
            export SLURM_JOB_NODELIST=fake-node
            export SLURM_NNODES=1
            export SLURM_CUSTOM_CONTROLLER_ENV=controller-kept
            rm -f \"\$SFLOW_FAKE_SLURM_LOG_DIR\"/*.args
            sflow run '$SLURM_ENV_FIXTURE_DIR/controller_env_aliases.yaml' \
                --output-dir '$SLURM_ENV_DIR/run' \
                > '$SLURM_ENV_DIR/run.log' 2>&1
            if [ -f \"\$SFLOW_FAKE_SLURM_LOG_DIR/salloc.args\" ]; then
                echo 'unexpected salloc for existing controller allocation'
                exit 1
            fi
            env_report=\$(find '$SLURM_ENV_DIR/run' -path '*/worker/slurm_env.txt' -print -quit)
            test -n \"\$env_report\"
            grep -F -- 'SLURM_JOB_ID=777777' \"\$env_report\"
            grep -F -- 'SLURM_CUSTOM_CONTROLLER_ENV=controller-kept' \"\$env_report\"
            grep -F -- 'SFLOW_BACKEND_JOB_ID=777777' \"\$env_report\"
            grep -F -- 'SFLOW_BACKEND_NODELIST=fake-node' \"\$env_report\"
            grep -F -- 'SFLOW_BACKEND_NUM_NODES=1' \"\$env_report\"
            grep -F -- 'SFLOW_BACKEND_STEP_ID=3' \"\$env_report\"
            grep -F -- 'SFLOW_TASK_NODE_NAME=fake-node' \"\$env_report\"
            grep -F -- 'SFLOW_TASK_PROCESS_ID=5' \"\$env_report\"
            grep -F -- 'SFLOW_TASK_LOCAL_PROCESS_ID=0' \"\$env_report\"
            grep -F -- 'SFLOW_TASK_NUM_PROCESSES=1' \"\$env_report\"
            grep -F -- 'srun --jobid 777777' \"\$SFLOW_FAKE_SLURM_LOG_DIR/srun.args\""

    # -- sflow run (fake Slurm): multi-backend binds each task to its own backend allocation --
    # Two slurm backends (distinct partitions) each get their own salloc job; each
    # task's srun must target its own backend's --jobid/--nodelist. The fake salloc
    # maps partition -> (job id, node) so we can assert the per-backend linkage.
    SLURM_MB_DIR="$PREFLIGHT_DIR/slurm_multi_backend"
    SLURM_MB_FAKE_BIN="$SLURM_MB_DIR/fake_bin"
    SLURM_MB_LOG_DIR="$SLURM_MB_DIR/logs"
    mkdir -p "$SLURM_MB_FAKE_BIN" "$SLURM_MB_LOG_DIR"
    cat > "$SLURM_MB_FAKE_BIN/salloc" <<'EOF'
#!/bin/bash
line="salloc"
for arg in "$@"; do
    line="$line $arg"
done
printf '%s\n' "$line" >> "$SFLOW_FAKE_SLURM_LOG_DIR/salloc.args"
partition=""
prev=""
for arg in "$@"; do
    if [ "$prev" = "--partition" ]; then partition="$arg"; fi
    prev="$arg"
done
case "$partition" in
    *alpha*) echo "salloc: Granted job allocation 5001"; echo "salloc: Nodes alpha-node are ready for job" ;;
    *beta*)  echo "salloc: Granted job allocation 5002"; echo "salloc: Nodes beta-node are ready for job" ;;
    *)       echo "salloc: Granted job allocation 5999"; echo "salloc: Nodes other-node are ready for job" ;;
esac
EOF
    cat > "$SLURM_MB_FAKE_BIN/scontrol" <<'EOF'
#!/bin/bash
if [ "$1" = "getaddrs" ]; then
    echo "$2: 127.0.0.1:123"
    exit 0
fi
exit 1
EOF
    cat > "$SLURM_MB_FAKE_BIN/srun" <<'EOF'
#!/bin/bash
line="srun"
for arg in "$@"; do
    line="$line $arg"
done
printf '%s\n' "$line" >> "$SFLOW_FAKE_SLURM_LOG_DIR/srun.args"
exit 0
EOF
    cat > "$SLURM_MB_FAKE_BIN/scancel" <<'EOF'
#!/bin/bash
echo "scancel $*" >> "$SFLOW_FAKE_SLURM_LOG_DIR/scancel.args"
exit 0
EOF
    chmod +x "$SLURM_MB_FAKE_BIN"/salloc "$SLURM_MB_FAKE_BIN"/scontrol \
        "$SLURM_MB_FAKE_BIN"/srun "$SLURM_MB_FAKE_BIN"/scancel
    run_check "run fake slurm multi-backend binds each task to its own backend" \
        bash -c "set -euo pipefail
            export PATH='$SLURM_MB_FAKE_BIN':\"\$PATH\"
            export SFLOW_FAKE_SLURM_LOG_DIR='$SLURM_MB_LOG_DIR'
            unset SLURM_JOB_ID SLURM_JOBID SLURM_JOB_NODELIST SLURM_NODELIST SLURM_NNODES SLURM_HET_SIZE
            rm -f \"\$SFLOW_FAKE_SLURM_LOG_DIR\"/*.args
            sflow run '$EXAMPLES_DIR/self_contained/slurm/multi_backend.yaml' \
                --set SLURM_ACCOUNT=acct \
                --set PARTITION_A=alpha \
                --set PARTITION_B=beta \
                --set IMAGE_A=nvcr.io/sflow-test/mb:alpha \
                --set IMAGE_B=nvcr.io/sflow-test/mb:beta \
                -a 'LOCAL_MODEL_PATH=fs://$MODEL_PATH' \
                --output-dir '$SLURM_MB_DIR/run' \
                > '$SLURM_MB_DIR/run.log' 2>&1
            salloc_args=\"\$SFLOW_FAKE_SLURM_LOG_DIR/salloc.args\"
            srun_args=\"\$SFLOW_FAKE_SLURM_LOG_DIR/srun.args\"
            scancel_args=\"\$SFLOW_FAKE_SLURM_LOG_DIR/scancel.args\"
            # Each backend allocates its own pool on its own partition.
            grep -- '--partition alpha' \"\$salloc_args\"
            grep -- '--partition beta' \"\$salloc_args\"
            # task_a (default backend cluster_a, operator worker_a) binds to job
            # 5001 / alpha-node and runs IMAGE_A.
            grep -- '--job-name task_a' \"\$srun_args\" | grep -q -- '--jobid 5001'
            grep -- '--job-name task_a' \"\$srun_args\" | grep -q -- '--nodelist alpha-node'
            grep -- '--job-name task_a' \"\$srun_args\" | grep -q -- '--container-image nvcr.io/sflow-test/mb:alpha'
            # task_b (backend cluster_b, operator worker_b) binds to job 5002 /
            # beta-node and runs IMAGE_B.
            grep -- '--job-name task_b' \"\$srun_args\" | grep -q -- '--jobid 5002'
            grep -- '--job-name task_b' \"\$srun_args\" | grep -q -- '--nodelist beta-node'
            grep -- '--job-name task_b' \"\$srun_args\" | grep -q -- '--container-image nvcr.io/sflow-test/mb:beta'
            # Each task targets only its own backend's allocation + image (no
            # cross-talk between the two pools).
            ! grep -- '--job-name task_a' \"\$srun_args\" | grep -q -- '--jobid 5002'
            ! grep -- '--job-name task_b' \"\$srun_args\" | grep -q -- '--jobid 5001'
            ! grep -- '--job-name task_a' \"\$srun_args\" | grep -q -- 'mb:beta'
            ! grep -- '--job-name task_b' \"\$srun_args\" | grep -q -- 'mb:alpha'
            # Both owned allocations are released on completion.
            grep -q '5001' \"\$scancel_args\"
            grep -q '5002' \"\$scancel_args\""

    # -- sflow run (inside fake `sflow batch` driver): leader reuses the driver
    # allocation; every other backend runs its own salloc --
    # Simulates running INSIDE the multi-backend driver sbatch: SLURM_JOB_ID is
    # the driver/leader allocation and the per-backend-salloc markers are set, so
    # the leader backend (cluster_a) reuses it while cluster_b sallocs its own
    # job. Each backend thus gets a distinct Slurm job id, so each task's srun
    # targets its own --jobid and pyxis/enroot keep per-job runtime dirs that
    # match the node. Reuses the fake bin.
    SLURM_MB_DRV_LOG_DIR="$SLURM_MB_DIR/driver_logs"
    mkdir -p "$SLURM_MB_DRV_LOG_DIR"
    run_check "run fake slurm batch driver leader reuses alloc and others salloc" \
        bash -c "set -euo pipefail
            export PATH='$SLURM_MB_FAKE_BIN':\"\$PATH\"
            export SFLOW_FAKE_SLURM_LOG_DIR='$SLURM_MB_DRV_LOG_DIR'
            unset SLURM_JOB_ID SLURM_JOBID SLURM_JOB_NODELIST SLURM_NODELIST SLURM_NNODES SLURM_HET_SIZE
            rm -f \"\$SFLOW_FAKE_SLURM_LOG_DIR\"/*.args
            # Driver sbatch env: leader (cluster_a) allocation + per-backend-salloc markers.
            export SLURM_JOB_ID=6001
            export SLURM_JOB_NODELIST=alpha-node
            export SFLOW_SLURM_MULTI_BACKEND_SALLOC=1
            export SFLOW_SLURM_WRAPPER_BACKEND=cluster_a
            sflow run '$EXAMPLES_DIR/self_contained/slurm/multi_backend.yaml' \
                --set SLURM_ACCOUNT=acct \
                --set PARTITION_A=alpha \
                --set PARTITION_B=beta \
                --set IMAGE_A=nvcr.io/sflow-test/mb:alpha \
                --set IMAGE_B=nvcr.io/sflow-test/mb:beta \
                -a 'LOCAL_MODEL_PATH=fs://$MODEL_PATH' \
                --output-dir '$SLURM_MB_DIR/driver_run' \
                > '$SLURM_MB_DIR/driver_run.log' 2>&1
            salloc_args=\"\$SFLOW_FAKE_SLURM_LOG_DIR/salloc.args\"
            srun_args=\"\$SFLOW_FAKE_SLURM_LOG_DIR/srun.args\"
            scancel_args=\"\$SFLOW_FAKE_SLURM_LOG_DIR/scancel.args\"
            # Only the non-leader backend sallocs (its own partition); the leader
            # reuses the driver allocation and must NOT salloc.
            grep -- '--partition beta' \"\$salloc_args\"
            ! grep -q -- '--partition alpha' \"\$salloc_args\"
            # task_a (leader cluster_a) reuses the driver job 6001, runs IMAGE_A.
            grep -- '--job-name task_a' \"\$srun_args\" | grep -q -- '--jobid 6001'
            grep -- '--job-name task_a' \"\$srun_args\" | grep -q -- '--container-image nvcr.io/sflow-test/mb:alpha'
            # task_b (cluster_b) binds to its own salloc job 5002, runs IMAGE_B.
            grep -- '--job-name task_b' \"\$srun_args\" | grep -q -- '--jobid 5002'
            grep -- '--job-name task_b' \"\$srun_args\" | grep -q -- '--container-image nvcr.io/sflow-test/mb:beta'
            # No cross-talk between backends or images.
            ! grep -- '--job-name task_a' \"\$srun_args\" | grep -q -- '--jobid 5002'
            ! grep -- '--job-name task_b' \"\$srun_args\" | grep -q -- '--jobid 6001'
            ! grep -- '--job-name task_a' \"\$srun_args\" | grep -q -- 'mb:beta'
            ! grep -- '--job-name task_b' \"\$srun_args\" | grep -q -- 'mb:alpha'
            # The leader's reused allocation is NOT owned (the driver sbatch owns
            # it); only the non-leader's own salloc job (5002) is scancelled.
            grep -q '5002' \"\$scancel_args\"
            ! grep -q '6001' \"\$scancel_args\""

    # -- sflow run --dry-run: modular (multi-file) --
    SLURM_CFG="$EXAMPLES_DIR/modular/inference_x_v2/slurm_config.yaml"
    COMMON="$EXAMPLES_DIR/modular/inference_x_v2/common_workflow.yaml"
    BENCH_INFMAX="$EXAMPLES_DIR/modular/inference_x_v2/benchmark_infmax.yaml"
    BENCH_AIPERF="$EXAMPLES_DIR/modular/inference_x_v2/benchmark_aiperf.yaml"
    DYNAMO_IMAGE="${DYNAMO_IMAGE:-nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.0}"
    MODULAR_MISSABLE=(-M agg_server -M prefill_server -M decode_server -M benchmark_infmax -M benchmark_aiperf)
    MODULAR_OVERRIDES=(-a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" -s "DYNAMO_IMAGE=$DYNAMO_IMAGE")
    for framework in trtllm sglang vllm; do
        run_check "dry-run modular $framework/disagg" \
            sflow run "$SLURM_CFG" "$COMMON" \
                "$EXAMPLES_DIR/modular/inference_x_v2/$framework/prefill.yaml" \
                "$EXAMPLES_DIR/modular/inference_x_v2/$framework/decode.yaml" \
                "$BENCH_INFMAX" \
                --dry-run "${MODULAR_MISSABLE[@]}" "${MODULAR_OVERRIDES[@]}"
        run_check "dry-run modular $framework/agg" \
            sflow run "$SLURM_CFG" "$COMMON" \
                "$EXAMPLES_DIR/modular/inference_x_v2/$framework/agg.yaml" \
                "$BENCH_AIPERF" \
                --dry-run "${MODULAR_MISSABLE[@]}" "${MODULAR_OVERRIDES[@]}"
    done

    # -- sflow compose: variable domain access --
    COMPOSE_DOMAIN_DIR="$PREFLIGHT_DIR/compose_domain"
    mkdir -p "$COMPOSE_DOMAIN_DIR"
    run_check "compose variable_domain" \
        sflow compose "$EXAMPLES_DIR/self_contained/local/variable_domain.yaml" -vl -r \
            -o "$COMPOSE_DOMAIN_DIR/resolved.yaml"

    # -- sflow compose: deferred Jinja should keep backend refs but inline resolved vars --
    COMPOSE_DEFERRED_DIR="$PREFLIGHT_DIR/compose_deferred_jinja"
    COMPOSE_DEFERRED_FIXTURE_DIR="$COMPOSE_DEFERRED_DIR/fixture"
    mkdir -p "$COMPOSE_DEFERRED_FIXTURE_DIR"
    cat > "$COMPOSE_DEFERRED_FIXTURE_DIR/vars.yaml" <<'EOF'
version: "0.1"
variables:
  - name: INFRA_NODE_INDEX
    value: 0
backends:
  - name: slurm_cluster
    type: slurm
    default: true
    account: acct
    partition: batch
    time: "00:10:00"
    nodes: 4
    gpus_per_node: 4
EOF
    cat > "$COMPOSE_DEFERRED_FIXTURE_DIR/workflow.yaml" <<'EOF'
version: "0.1"
workflow:
  name: wf
  variables:
    - name: HEAD_NODE_IP
      value: ${{ backends.slurm_cluster.nodes[0].ip_address if variables.INFRA_NODE_INDEX == 0 else backends.slurm_cluster.nodes[-1].ip_address }}
    - name: NATS_SERVER
      value: nats://${{ backends.slurm_cluster.nodes[0].ip_address if variables.INFRA_NODE_INDEX == 0 else backends.slurm_cluster.nodes[-1].ip_address }}:4222
  tasks:
    - name: t1
      script:
        - echo hi
EOF
    run_check "compose deferred_jinja_literal_rewrite" \
        sflow compose "$COMPOSE_DEFERRED_FIXTURE_DIR/vars.yaml" \
            "$COMPOSE_DEFERRED_FIXTURE_DIR/workflow.yaml" -r \
            -o "$COMPOSE_DEFERRED_DIR/resolved.yaml"

    # -- sflow compose: resources.nodes.indices/exclude may be a single expression string resolving to a list --
    COMPOSE_INDICES_DIR="$PREFLIGHT_DIR/compose_indices_expression"
    COMPOSE_INDICES_FIXTURE_DIR="$COMPOSE_INDICES_DIR/fixture"
    COMPOSE_INDICES_DRYRUN_LOG="$COMPOSE_INDICES_DIR/dry_run.log"
    mkdir -p "$COMPOSE_INDICES_FIXTURE_DIR"
    cat > "$COMPOSE_INDICES_FIXTURE_DIR/vars.yaml" <<'EOF'
version: "0.1"
variables:
  - name: INFRA_NODE_INDEX
    value: 0
    type: integer
  - name: NUM_FRONTENDS
    value: 2
    type: integer
backends:
  - name: slurm_cluster
    type: slurm
    default: true
    account: acct
    partition: batch
    time: "00:10:00"
    nodes: 4
    gpus_per_node: 4
EOF
    cat > "$COMPOSE_INDICES_FIXTURE_DIR/workflow.yaml" <<'EOF'
version: "0.1"
workflow:
  name: wf
  tasks:
    - name: frontend_server
      script:
        - echo hi
      resources:
        nodes:
          indices: ${{ range(variables.INFRA_NODE_INDEX, variables.INFRA_NODE_INDEX + variables.NUM_FRONTENDS) | list }}
    - name: worker_server
      script:
        - echo worker
      resources:
        nodes:
          exclude: ${{ range(variables.INFRA_NODE_INDEX, variables.INFRA_NODE_INDEX + variables.NUM_FRONTENDS) | list }}
    - name: ordered_pool
      script:
        - echo ordered
      replicas:
        count: 4
        policy: parallel
      resources:
        nodes:
          indices: [-1, 0, 1, 2]
          count: 1
EOF
    run_check "compose nodes.indices/exclude expression strings resolve to list" \
        sflow compose "$COMPOSE_INDICES_FIXTURE_DIR/vars.yaml" \
            "$COMPOSE_INDICES_FIXTURE_DIR/workflow.yaml" -r \
            -o "$COMPOSE_INDICES_DIR/resolved.yaml"
    run_check "run nodes.indices/exclude expression strings and indices+count ordering (dry-run)" \
        bash -c "sflow run \"$COMPOSE_INDICES_FIXTURE_DIR/vars.yaml\" \"$COMPOSE_INDICES_FIXTURE_DIR/workflow.yaml\" --dry-run --verbose > \"$COMPOSE_INDICES_DRYRUN_LOG\" 2>&1"

    # -- sflow compose: single-file self-contained examples --
    COMPOSE_SINGLE_DIR="$PREFLIGHT_DIR/compose_single"
    mkdir -p "$COMPOSE_SINGLE_DIR"
    for f in "$EXAMPLES_DIR"/self_contained/slurm/*.yaml; do
        name=$(basename "$f" .yaml)
        run_check "compose $name" \
            sflow compose "$f" -vl -r -o "$COMPOSE_SINGLE_DIR/$name.yaml"
    done

    # -- sflow compose: modular (multi-file) --
    COMPOSE_MODULAR_DIR="$PREFLIGHT_DIR/compose_modular"
    mkdir -p "$COMPOSE_MODULAR_DIR"
    for framework in trtllm sglang vllm; do
        run_check "compose modular $framework/disagg" \
            sflow compose "$SLURM_CFG" "$COMMON" \
                "$EXAMPLES_DIR/modular/inference_x_v2/$framework/prefill.yaml" \
                "$EXAMPLES_DIR/modular/inference_x_v2/$framework/decode.yaml" \
                "$BENCH_INFMAX" \
                "${MODULAR_MISSABLE[@]}" -r -vl \
                -o "$COMPOSE_MODULAR_DIR/${framework}_disagg.yaml"
        run_check "compose modular $framework/agg" \
            sflow compose "$SLURM_CFG" "$COMMON" \
                "$EXAMPLES_DIR/modular/inference_x_v2/$framework/agg.yaml" \
                "$BENCH_AIPERF" \
                "${MODULAR_MISSABLE[@]}" -r -vl \
                -o "$COMPOSE_MODULAR_DIR/${framework}_agg.yaml"
    done

    # -- sflow compose --bulk-input (CSV) --
    if [ -f "$CSV_FILE" ]; then
        run_check "compose bulk-input all rows" \
            sflow compose -b "$CSV_FILE" -o "$PREFLIGHT_DIR/compose_bulk_input"

        run_check "compose bulk-input single row" \
            sflow compose -b "$CSV_FILE" --row 1 -o "$PREFLIGHT_DIR/compose_bulk_input_row1"

        run_check "compose bulk-input row range" \
            sflow compose -b "$CSV_FILE" --row 7:10 -o "$PREFLIGHT_DIR/compose_bulk_input_multi_rows"

        # -- negative index and open-ended slice tests --
        run_check "compose bulk-input last row (--row=-1)" \
            sflow compose -b "$CSV_FILE" --row=-1 -o "$PREFLIGHT_DIR/compose_bulk_input_last_row"

        run_check "compose bulk-input negative range (--row=-3:)" \
            sflow compose -b "$CSV_FILE" --row=-3: -o "$PREFLIGHT_DIR/compose_bulk_input_last3"

        run_check "compose bulk-input open-end slice (--row 3:)" \
            sflow compose -b "$CSV_FILE" --row=3: -o "$PREFLIGHT_DIR/compose_bulk_input_3_to_end"

        run_check "compose bulk-input negative slice (--row=-3:-1)" \
            sflow compose -b "$CSV_FILE" --row=-3:-1 -o "$PREFLIGHT_DIR/compose_bulk_input_neg_slice"
    else
        echo "  SKIP: CSV not found at $CSV_FILE"
    fi

    # -- sflow batch -f (single file): self-contained examples --
    BATCH_SINGLE_DIR="$PREFLIGHT_DIR/batch_single"
    mkdir -p "$BATCH_SINGLE_DIR"
    for f in "$EXAMPLES_DIR"/self_contained/slurm/*.yaml; do
        name=$(basename "$f" .yaml)
        run_check "batch single $name" \
            sflow batch -f "$f" \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
                -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
                -o "$BATCH_SINGLE_DIR/$name.sh"
    done

    # -- sflow batch: generate an sbatch script to verify (post-wait) that the
    #    venv bootstrap neutralizes a leaked caller virtualenv and creates the
    #    runtime venv with a resolved system python3 (never a bare PATH python3),
    #    guarding the cross-arch "Exec format error" regression. --
    BATCH_VENV_BOOTSTRAP_DIR="$PREFLIGHT_DIR/batch_venv_bootstrap"
    BATCH_VENV_BOOTSTRAP_SCRIPT="$BATCH_VENV_BOOTSTRAP_DIR/venv_bootstrap.sh"
    mkdir -p "$BATCH_VENV_BOOTSTRAP_DIR"
    run_check "batch sbatch venv bootstrap script generation" \
        sflow batch -f "$EXAMPLES_DIR/self_contained/local/dag.yaml" \
            -p "$PARTITION" -A "$ACCOUNT" --nodes 1 --log-level warn \
            -o "$BATCH_VENV_BOOTSTRAP_SCRIPT"

    # -- sflow run/batch: monitor CLI injection (--enable-workflow-monitor /
    #    --enable-task-monitor) adds default hardware monitors without editing the
    #    recipe. Covers the app-layer pydantic override (run --dry-run), the
    #    split_list_arg comma/space/repeat forms, unknown-task validation, and the
    #    batch threading into both the generated sbatch script and the YAML snapshot. --
    MONITOR_CLI_DIR="$PREFLIGHT_DIR/monitor_cli_injection"
    mkdir -p "$MONITOR_CLI_DIR"
    MONITOR_CLI_WF_LOG="$MONITOR_CLI_DIR/run_workflow_monitor.log"
    MONITOR_CLI_TASK_LOG="$MONITOR_CLI_DIR/run_task_monitors.log"
    MONITOR_CLI_BAD_LOG="$MONITOR_CLI_DIR/run_unknown_task.log"
    MONITOR_CLI_BATCH_SCRIPT="$MONITOR_CLI_DIR/batch_monitor.sh"
    MONITOR_CLI_BATCH_YAML="$MONITOR_CLI_DIR/batch_monitor.yaml"
    MONITOR_CLI_BATCH_LOG="$MONITOR_CLI_DIR/batch_monitor.log"

    # --enable-workflow-monitor injects a whole-pool monitor into the plan.
    run_check "run --enable-workflow-monitor injects a workflow monitor (dry-run)" \
        bash -c "sflow run \"$EXAMPLES_DIR/self_contained/local/dag.yaml\" --enable-workflow-monitor --dry-run > \"$MONITOR_CLI_WF_LOG\" 2>&1 && \
            grep -q 'Enabled default workflow monitor' \"$MONITOR_CLI_WF_LOG\" && \
            grep -q 'Planned monitors' \"$MONITOR_CLI_WF_LOG\""

    # --enable-task-monitor binds per-task monitors; accepts comma-separated,
    # quoted-whitespace, and repeated-flag forms (split_list_arg).
    run_check "run --enable-task-monitor injects per-task monitors (comma/space/repeat list, dry-run)" \
        bash -c "sflow run \"$EXAMPLES_DIR/self_contained/local/dag.yaml\" --enable-task-monitor 'prepare_data,preprocess' --enable-task-monitor 'train export_model' --dry-run > \"$MONITOR_CLI_TASK_LOG\" 2>&1 && \
            grep -q \"Enabled default task monitor for 'prepare_data'\" \"$MONITOR_CLI_TASK_LOG\" && \
            grep -q \"Enabled default task monitor for 'preprocess'\" \"$MONITOR_CLI_TASK_LOG\" && \
            grep -q \"Enabled default task monitor for 'train'\" \"$MONITOR_CLI_TASK_LOG\" && \
            grep -q \"Enabled default task monitor for 'export_model'\" \"$MONITOR_CLI_TASK_LOG\""

    # An unknown task name is rejected up front with a clear error (non-zero exit).
    run_check "run --enable-task-monitor rejects an unknown task" \
        bash -c "! sflow run \"$EXAMPLES_DIR/self_contained/local/dag.yaml\" --enable-task-monitor nope_task --dry-run > \"$MONITOR_CLI_BAD_LOG\" 2>&1 && \
            grep -q 'refers to unknown task' \"$MONITOR_CLI_BAD_LOG\""

    # batch threads the flags into the generated `sflow run` command AND injects the
    # monitor into the composed YAML snapshot saved next to the script.
    run_check "batch threads monitor flags into the sbatch script + composed snapshot" \
        bash -c "sflow batch -f \"$EXAMPLES_DIR/self_contained/local/dag.yaml\" -p \"$PARTITION\" -A \"$ACCOUNT\" --nodes 1 --log-level warn --enable-workflow-monitor --enable-task-monitor train -o \"$MONITOR_CLI_BATCH_SCRIPT\" > \"$MONITOR_CLI_BATCH_LOG\" 2>&1 && \
            grep -q -- '--enable-workflow-monitor' \"$MONITOR_CLI_BATCH_SCRIPT\" && \
            grep -q -- '--enable-task-monitor' \"$MONITOR_CLI_BATCH_SCRIPT\" && \
            grep -q 'monitor:' \"$MONITOR_CLI_BATCH_YAML\""

    # -- sflow batch -f (multi-file): modular examples --
    BATCH_MODULAR_DIR="$PREFLIGHT_DIR/batch_modular"
    mkdir -p "$BATCH_MODULAR_DIR"
    for framework in trtllm sglang vllm; do
        run_check "batch modular $framework/disagg" \
            sflow batch \
                -f "$SLURM_CFG" -f "$COMMON" \
                -f "$EXAMPLES_DIR/modular/inference_x_v2/$framework/prefill.yaml" \
                -f "$EXAMPLES_DIR/modular/inference_x_v2/$framework/decode.yaml" \
                -f "$BENCH_INFMAX" -r \
                "${MODULAR_MISSABLE[@]}" "${MODULAR_OVERRIDES[@]}" \
                -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
                -o "$BATCH_MODULAR_DIR/${framework}_disagg.sh"
        run_check "batch modular $framework/agg" \
            sflow batch \
                -f "$SLURM_CFG" -f "$COMMON" \
                -f "$EXAMPLES_DIR/modular/inference_x_v2/$framework/agg.yaml" \
                -f "$BENCH_AIPERF" \
                "${MODULAR_MISSABLE[@]}" "${MODULAR_OVERRIDES[@]}" \
                -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
                -o "$BATCH_MODULAR_DIR/${framework}_agg.sh"
    done

    # -- sflow batch -e with expression resolution --
    BATCH_EXTRA_ARGS_DIR="$PREFLIGHT_DIR/batch_extra_args_expr"
    mkdir -p "$BATCH_EXTRA_ARGS_DIR"
    EXTRA_ARGS_EXAMPLE="$EXAMPLES_DIR/self_contained/slurm/dynamo_sglang_disagg.yaml"
    if [ -f "$EXTRA_ARGS_EXAMPLE" ]; then
        run_check "batch -e expression resolution" \
            sflow batch -f "$EXTRA_ARGS_EXAMPLE" \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
                -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
                -s "SLURM_NODES=3" \
                -e '--segment=${{ variables.SLURM_NODES }}' \
                -o "$BATCH_EXTRA_ARGS_DIR/expr_test.sh"
        if [ -f "$BATCH_EXTRA_ARGS_DIR/expr_test.sh" ]; then
            if grep -q '#SBATCH --segment=3' "$BATCH_EXTRA_ARGS_DIR/expr_test.sh"; then
                echo "  PASS: -e expression resolved to '--segment=3'"
            else
                echo "  FAIL: -e expression was not resolved (expected '#SBATCH --segment=3')"
                grep '#SBATCH --segment' "$BATCH_EXTRA_ARGS_DIR/expr_test.sh" || echo "    (no --segment directive found)"
            fi
        fi
    fi

    # -- sflow batch under-dev install: should be editable from the local checkout --
    # use_under_dev_sflow.sh injects --sflow-source-path "$REPO_DIR" into batch
    # calls, so the generated script must install sflow editable from the local
    # checkout rather than from a git ref.
    BATCH_DEFAULT_VERSION_DIR="$PREFLIGHT_DIR/batch_default_sflow_version"
    mkdir -p "$BATCH_DEFAULT_VERSION_DIR"
    if [ -z "$IS_REAL_SUBMIT" ] && [ -f "$EXTRA_ARGS_EXAMPLE" ]; then
        run_check "batch under-dev install editable from local checkout" \
            sflow batch -f "$EXTRA_ARGS_EXAMPLE" \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
                -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
                -s "SLURM_NODES=3" \
                -o "$BATCH_DEFAULT_VERSION_DIR/default_version.sh"
    fi

    # -- sflow batch default install version: with neither --sflow-source-path nor
    # --sflow-version, the install link must be the git ref auto-detected from the
    # running sflow env. Bypass the under-dev wrapper (which injects
    # --sflow-source-path) by invoking the module directly, so the no-flag default
    # path is actually exercised. --
    BATCH_AUTOVERSION_DIR="$PREFLIGHT_DIR/batch_auto_version"
    mkdir -p "$BATCH_AUTOVERSION_DIR"
    if [ -z "$IS_REAL_SUBMIT" ] && [ -n "${SFLOW_TEST_PYTHON:-}" ] && [ -f "$EXTRA_ARGS_EXAMPLE" ]; then
        run_check "batch default install version auto-detected from running env" \
            "$SFLOW_TEST_PYTHON" -m sflow batch -f "$EXTRA_ARGS_EXAMPLE" \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
                -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
                -s "SLURM_NODES=3" \
                -o "$BATCH_AUTOVERSION_DIR/auto_version.sh"
    fi

    # -- sflow batch -e with variables.X.domain expression --
    BATCH_DOMAIN_DIR="$PREFLIGHT_DIR/batch_domain_expr"
    mkdir -p "$BATCH_DOMAIN_DIR"
    DOMAIN_EXAMPLE="$EXAMPLES_DIR/self_contained/local/variable_domain.yaml"
    if [ -f "$DOMAIN_EXAMPLE" ]; then
        run_check "batch -e domain expression" \
            sflow batch -f "$DOMAIN_EXAMPLE" \
                -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
                --nodes 1 \
                -e '--comment=${{ variables.CONCURRENCY.domain }}' \
                -o "$BATCH_DOMAIN_DIR/domain_test.sh"
    fi

    # -- sflow batch --bulk-submit (no --submit): self-contained slurm recipes.
    #    --bulk-submit globs the given directory's top-level YAMLs, so point it at a
    #    leaf backend folder (the recipes now live under self_contained/<backend>/). --
    run_check "batch bulk-submit (no submit)" \
        sflow batch --bulk-submit "$EXAMPLES_DIR/self_contained/slurm" \
            -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
            -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
            --output-dir "$PREFLIGHT_DIR/batch_bulk_submit"

    # -- sflow batch --bulk-input (no --submit): CSV --
    if [ -f "$CSV_FILE" ]; then
        run_check "batch bulk-input (no submit)" \
            sflow batch --bulk-input "$CSV_FILE" \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
                -p "$PARTITION" -A "$ACCOUNT" --log-level warn -r \
                --output-dir "$PREFLIGHT_DIR/batch_bulk_input"

        # -- verify sflow_batch_dir column in results.csv --
        # -- negative index and open-ended slice tests --
        run_check "batch bulk-input last row (--row=-1)" \
            sflow batch --bulk-input "$CSV_FILE" --row=-1 \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
                -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
                --output-dir "$PREFLIGHT_DIR/batch_bulk_input_last_row"

        run_check "batch bulk-input last 3 rows (--row=-3:)" \
            sflow batch --bulk-input "$CSV_FILE" --row=-3: \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
                -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
                --output-dir "$PREFLIGHT_DIR/batch_bulk_input_last3"

        run_check "batch bulk-input open-end (--row=3:)" \
            sflow batch --bulk-input "$CSV_FILE" --row=3: \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
                -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
                --output-dir "$PREFLIGHT_DIR/batch_bulk_input_3_to_end"
    else
        echo "  SKIP: CSV not found at $CSV_FILE"
    fi

    # -- sflow batch --bulk-input + CLI -f: CLI config must be prepended to every CSV row --
    BATCH_BULK_CLI_FILES_DIR="$PREFLIGHT_DIR/batch_bulk_input_cli_files"
    BATCH_BULK_CLI_FILES_FIXTURE_DIR="$BATCH_BULK_CLI_FILES_DIR/fixture"
    mkdir -p "$BATCH_BULK_CLI_FILES_FIXTURE_DIR"
    cat > "$BATCH_BULK_CLI_FILES_FIXTURE_DIR/common.yaml" <<'EOF'
version: "0.1"
variables:
  - name: SHARED_VALUE
    value: from_common
EOF
    cat > "$BATCH_BULK_CLI_FILES_FIXTURE_DIR/task.yaml" <<'EOF'
version: "0.1"
workflow:
  name: batch_bulk_input_cli_files
  tasks:
    - name: show_shared
      script:
        - echo "${SHARED_VALUE}"
EOF
    cat > "$BATCH_BULK_CLI_FILES_FIXTURE_DIR/jobs.csv" <<EOF
sflow_config_file
$BATCH_BULK_CLI_FILES_FIXTURE_DIR/task.yaml
EOF
    run_check "batch bulk-input with cli -f prepends config" \
        bash -c "set -euo pipefail
            sflow batch -f '$BATCH_BULK_CLI_FILES_FIXTURE_DIR/common.yaml' \
                --bulk-input '$BATCH_BULK_CLI_FILES_FIXTURE_DIR/jobs.csv' \
                -p '$PARTITION' -A '$ACCOUNT' --nodes 1 --log-level warn \
                --output-dir '$BATCH_BULK_CLI_FILES_DIR/out'
            sh_file=\$(find '$BATCH_BULK_CLI_FILES_DIR/out' -name '*.sh' -print -quit)
            test -n \"\$sh_file\"
            common_path=\$(python - '$BATCH_BULK_CLI_FILES_FIXTURE_DIR/common.yaml' <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve())
PY
)
            task_path=\$(python - '$BATCH_BULK_CLI_FILES_FIXTURE_DIR/task.yaml' <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve())
PY
)
            common_arg=\"--file \$common_path\"
            task_arg=\"--file \$task_path\"
            grep -F -- \"\$common_arg\" \"\$sh_file\"
            grep -F -- \"\$task_arg\" \"\$sh_file\"
            python - \"\$sh_file\" \"\$common_arg\" \"\$task_arg\" <<'PY'
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text()
common = sys.argv[2]
task = sys.argv[3]
if text.index(common) > text.index(task):
    raise SystemExit('CLI -f config appears after CSV row config')
PY"

    # -- sflow batch --bulk-input with -e expression: verify per-row resolution --
    if [ -f "$CSV_FILE" ]; then
        BATCH_BULK_EXPR_DIR="$PREFLIGHT_DIR/batch_bulk_input_expr"
        run_check "batch bulk-input -e expression" \
            sflow batch --bulk-input "$CSV_FILE" \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
                -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
                -e '--segment=${{ variables.SLURM_NODES }}' \
                --output-dir "$BATCH_BULK_EXPR_DIR"
        EXPR_FAIL=0
        for sh_file in "$BATCH_BULK_EXPR_DIR"/bulk_input_*/*.sh; do
            [ -f "$sh_file" ] || continue
            if grep -q '#SBATCH --segment=\${{' "$sh_file"; then
                echo "  FAIL: unresolved expression in $(basename "$sh_file")"
                EXPR_FAIL=1
            elif ! grep -q '#SBATCH --segment=[0-9]' "$sh_file"; then
                echo "  FAIL: missing --segment directive in $(basename "$sh_file")"
                EXPR_FAIL=1
            fi
        done
        if [ "$EXPR_FAIL" -eq 0 ]; then
            echo "  PASS: -e expressions resolved per CSV row in bulk-input"
        fi
    fi

    # -- sflow batch --bulk-input with -s overlapping CSV column: CLI --set must win --
    if [ -f "$CSV_FILE" ]; then
        BATCH_BULK_SET_DIR="$PREFLIGHT_DIR/batch_bulk_input_set_precedence"
        BATCH_BULK_SET_OUT="$RESULTS_DIR/batch_bulk_input_set_precedence.stderr"
        run_check "batch bulk-input -s overrides CSV column" \
            bash -c "sflow batch --bulk-input '$CSV_FILE' \
                -a 'LOCAL_MODEL_PATH=fs://$MODEL_PATH' \
                -p '$PARTITION' -A '$ACCOUNT' --log-level warn \
                -s 'GPUS_PER_NODE=77' \
                --output-dir '$BATCH_BULK_SET_DIR' 2> '$BATCH_BULK_SET_OUT'"
    fi

    # -- sflow compose --bulk-input with --set overlapping CSV column: CLI --set must win --
    if [ -f "$CSV_FILE" ]; then
        COMPOSE_BULK_SET_DIR="$PREFLIGHT_DIR/compose_bulk_input_set_precedence"
        COMPOSE_BULK_SET_OUT="$RESULTS_DIR/compose_bulk_input_set_precedence.stderr"
        run_check "compose bulk-input --set overrides CSV column" \
            bash -c "sflow compose --bulk-input '$CSV_FILE' \
                --set 'GPUS_PER_NODE=77' \
                -o '$COMPOSE_BULK_SET_DIR' 2> '$COMPOSE_BULK_SET_OUT'"
    fi

    # -- sflow run --bulk-input --row (dry-run): CSV row execution --
    # Missable tasks are defined in the CSV's missable_tasks column, not via CLI -M.
    if [ -f "$CSV_FILE" ]; then
        run_check "run bulk-input row 1 (dry-run)" \
            sflow run --bulk-input "$CSV_FILE" --row 1 --dry-run \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH"

        run_check "run bulk-input row 3 (dry-run)" \
            sflow run --bulk-input "$CSV_FILE" --row 3 --dry-run \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH"

        run_check "run bulk-input with cli files (dry-run)" \
            sflow run -f "$SLURM_CFG" --bulk-input "$CSV_FILE" --row 1 --dry-run \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH"

        # -- negative index tests for sflow run --
        run_check "run bulk-input last row (--row=-1, dry-run)" \
            sflow run --bulk-input "$CSV_FILE" --row=-1 --dry-run \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH"

        run_check "run bulk-input negative row (--row=-3, dry-run)" \
            sflow run --bulk-input "$CSV_FILE" --row=-3 --dry-run \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH"

        run_check "run bulk-input missing --row (expect fail)" \
            bash -c '! sflow run --bulk-input '"$CSV_FILE"' --dry-run 2>&1'

        run_check "run --row without bulk-input (expect fail)" \
            bash -c '! sflow run --row 1 --dry-run 2>&1'
    else
        echo "  SKIP: CSV not found at $CSV_FILE"
    fi

    # -- sflow visualize --
    run_check "visualize modular vllm/disagg" \
        sflow visualize "$SLURM_CFG" "$COMMON" \
            "$EXAMPLES_DIR/modular/inference_x_v2/vllm/prefill.yaml" \
            "$EXAMPLES_DIR/modular/inference_x_v2/vllm/decode.yaml" \
            "$BENCH_INFMAX" \
            "${MODULAR_MISSABLE[@]}" "${MODULAR_OVERRIDES[@]}" \
            --format mermaid \
            -o "$PREFLIGHT_DIR/visualize_vllm_disagg.mmd"

    # -- sflow sample --
    run_check "sample list" \
        sflow sample --list
    SAMPLE_SELF_DIR="$PREFLIGHT_DIR/sample_copy_self"
    mkdir -p "$SAMPLE_SELF_DIR"
    run_check "sample copy self-contained" \
        sflow sample self_contained/local/hello_world \
            --output "$SAMPLE_SELF_DIR/local_hello_world.yaml"
    SAMPLE_MODULAR_DIR="$PREFLIGHT_DIR/sample_copy_modular"
    mkdir -p "$SAMPLE_MODULAR_DIR"
    run_check "sample copy modular" \
        sflow sample modular/inference_x_v2 \
            --output "$SAMPLE_MODULAR_DIR/inference_x_v2"

    fi

    # =====================================================================
    # Wait for all parallel tests and aggregate results
    # =====================================================================
    echo "Launched $TEST_ID tests — waiting for completion..."
    echo ""
    wait

    PASS=0
    FAIL=0
    TOTAL=0
    FAILED_LABELS=""
    FAILED_DETAILS=""
    for result_file in "$RESULTS_DIR"/*.result; do
        [ -f "$result_file" ] || continue
        TOTAL=$((TOTAL + 1))
        id=$(basename "$result_file" .result)
        output_file="$RESULTS_DIR/${id}.output"

        status="" label="" cmd=""
        while IFS='=' read -r key value; do
            case "$key" in
                STATUS) status="$value" ;;
                LABEL)  label="$value" ;;
                CMD)    cmd="$value" ;;
            esac
        done < "$result_file"

        if [ "$status" = "OK" ]; then
            PASS=$((PASS + 1))
            echo "  [$id] $label ... OK"
            echo "       \$ $cmd"
            highlights=$(grep -E 'Output directory:|Scripts directory:|Results CSV:|Bulk (submit|input|compose):|topological order:' "$output_file" 2>/dev/null | head -10 || true)
            if [ -n "$highlights" ]; then
                echo "$highlights" | sed 's/^/       /'
            fi
        else
            FAIL=$((FAIL + 1))
            echo "  [$id] $label ... FAIL"
            echo "       \$ $cmd"
            echo "       Reason (captured output):"
            if [ -s "$output_file" ]; then
                sed -n '1,80p' "$output_file" 2>/dev/null | sed 's/^/       /'
                FAILED_DETAILS="$FAILED_DETAILS\n[$id] $label\n  \$ $cmd\n$(sed -n '1,80p' "$output_file" 2>/dev/null | sed 's/^/  /')\n"
            else
                echo "       (no output captured)"
                FAILED_DETAILS="$FAILED_DETAILS\n[$id] $label\n  \$ $cmd\n  (no output captured)\n"
            fi
            FAILED_LABELS="$FAILED_LABELS  - $label\n"
        fi
    done

    # Save test commands and results to the preflight output directory
    TEST_LOG="$PREFLIGHT_DIR/preflight_test_log.txt"
    {
        echo "# Preflight Test Log"
        echo "# Generated: $(date)"
        echo "# Results: $PASS/$TOTAL passed, $FAIL failed"
        echo ""
    } > "$TEST_LOG"
    for result_file in "$RESULTS_DIR"/*.result; do
        [ -f "$result_file" ] || continue
        id=$(basename "$result_file" .result)
        log_status="" log_label="" log_cmd=""
        while IFS='=' read -r key value; do
            case "$key" in
                STATUS) log_status="$value" ;;
                LABEL)  log_label="$value" ;;
                CMD)    log_cmd="$value" ;;
            esac
        done < "$result_file"
        echo "[$id] $log_status  $log_label" >> "$TEST_LOG"
        echo "  \$ $log_cmd" >> "$TEST_LOG"
        echo "" >> "$TEST_LOG"
    done

    if [ "$TEST_TYPE" != "inf" ]; then

    # -- Post-wait: verify replica sweep resolves per-replica values + domain --
    SFLOW_LOG=$(find "$DOMAIN_RUN_DIR" -name "sflow.log" -print -quit 2>/dev/null)
    if [ -f "$SFLOW_LOG" ]; then
        REPLICA_FAIL=0
        # Verify domain resolved in the command log
        if grep -q 'concurrency_domain=\[1, 4, 16\]' "$SFLOW_LOG"; then
            :  # pass
        else
            echo "  FAIL: sflow.log did not contain resolved concurrency domain list"
            REPLICA_FAIL=1
        fi
        if grep -q 'framework_domain=.*sglang.*vllm.*trtllm' "$SFLOW_LOG"; then
            :  # pass
        else
            echo "  FAIL: sflow.log did not contain resolved framework domain list"
            REPLICA_FAIL=1
        fi
        # Verify per-replica value shift for both sweep variables
        if grep -q "echo concurrency=1$" "$SFLOW_LOG" && grep -q "echo concurrency=16$" "$SFLOW_LOG"; then
            :  # pass
        else
            echo "  FAIL: concurrency replica value shift not found (expected 1 and 16)"
            REPLICA_FAIL=1
        fi
        if grep -q "echo framework=sglang$" "$SFLOW_LOG" && grep -q "echo framework=trtllm$" "$SFLOW_LOG"; then
            :  # pass
        else
            echo "  FAIL: framework replica value shift not found (expected sglang and trtllm)"
            REPLICA_FAIL=1
        fi
        if [ "$REPLICA_FAIL" -eq 0 ]; then
            echo "  PASS: replica sweep resolves per-replica values + domain correctly"
        else
            FAIL=$((FAIL + REPLICA_FAIL))
            TOTAL=$((TOTAL + REPLICA_FAIL))
            FAILED_LABELS="$FAILED_LABELS  - replica sweep value/domain resolution\n"
        fi
    fi

    # -- Post-wait: verify live run summary and command-only command logs --
    SFLOW_SUMMARY_LOG=$(find "$DOMAIN_RUN_DIR" -name "sflow_summary.log" -print -quit 2>/dev/null)
    if [ -f "$SFLOW_SUMMARY_LOG" ]; then
        SUMMARY_FAIL=0
        DOMAIN_WORKFLOW_DIR=$(dirname "$SFLOW_SUMMARY_LOG")
        for summary_pattern in \
            'Sflow Summary' \
            'Runtime' \
            'Workflow DAG' \
            'Timeline' \
            'Task Duration Chart' \
            'Counts       :' \
            'FAILED/CANCELLED Tasks :' \
            'SUBMITTED'; do
            if ! grep -Fq "$summary_pattern" "$SFLOW_SUMMARY_LOG"; then
                echo "  FAIL: sflow_summary.log missing '$summary_pattern'"
                SUMMARY_FAIL=1
            fi
        done
        if ! grep -Eq 'COMPLETED|FAILED|CANCELLED|READY' "$SFLOW_SUMMARY_LOG"; then
            echo "  FAIL: sflow_summary.log missing terminal task event"
            SUMMARY_FAIL=1
        fi
        if grep -Fq 'Traceback' "$SFLOW_SUMMARY_LOG"; then
            echo "  FAIL: sflow_summary.log contains traceback"
            SUMMARY_FAIL=1
        fi

        FIRST_CMD_LOG=$(find "$DOMAIN_WORKFLOW_DIR" -name "*_cmds.log" -print -quit 2>/dev/null)
        if [ -z "$FIRST_CMD_LOG" ]; then
            echo "  FAIL: no command log found under $DOMAIN_WORKFLOW_DIR"
            SUMMARY_FAIL=1
        fi
        BASH_CMD_LOG="$DOMAIN_WORKFLOW_DIR/bash_cmds.log"
        if [ -f "$BASH_CMD_LOG" ]; then
            if ! grep -Fq 'bash -c' "$BASH_CMD_LOG"; then
                echo "  FAIL: bash_cmds.log missing bash -c command"
                SUMMARY_FAIL=1
            fi
            if grep -xFq 'concurrency=1' "$BASH_CMD_LOG" || \
               grep -xFq 'framework=sglang' "$BASH_CMD_LOG"; then
                echo "  FAIL: bash_cmds.log contains raw task output lines"
                SUMMARY_FAIL=1
            fi
        else
            echo "  FAIL: bash_cmds.log missing under $DOMAIN_WORKFLOW_DIR"
            SUMMARY_FAIL=1
        fi

        if [ "$SUMMARY_FAIL" -eq 0 ]; then
            echo "  PASS: sflow_summary.log and command logs generated for live run"
        else
            FAIL=$((FAIL + SUMMARY_FAIL))
            TOTAL=$((TOTAL + SUMMARY_FAIL))
            FAILED_LABELS="$FAILED_LABELS  - sflow summary/command logs\n"
        fi
    else
        echo "  FAIL: sflow_summary.log not found under $DOMAIN_RUN_DIR"
        FAIL=$((FAIL + 1))
        TOTAL=$((TOTAL + 1))
        FAILED_LABELS="$FAILED_LABELS  - sflow summary log missing\n"
    fi

    # -- Post-wait: verify ${{ variables.X.domain }} resolved in batch -e --
    BATCH_DOMAIN_SCRIPT="$BATCH_DOMAIN_DIR/domain_test.sh"
    if [ -f "$BATCH_DOMAIN_SCRIPT" ]; then
        if grep -q '#SBATCH --comment=\[1, 4, 16\]' "$BATCH_DOMAIN_SCRIPT"; then
            echo "  PASS: batch -e variables.X.domain resolved to [1, 4, 16]"
        else
            echo "  FAIL: batch -e variables.X.domain not resolved in sbatch script"
            grep '#SBATCH --comment' "$BATCH_DOMAIN_SCRIPT" || echo "    (no --comment directive found)"
            FAIL=$((FAIL + 1))
            TOTAL=$((TOTAL + 1))
            FAILED_LABELS="$FAILED_LABELS  - batch -e variables.X.domain resolution\n"
        fi
    fi

    # -- Post-wait: verify the under-dev batch install is editable from the local checkout --
    # use_under_dev_sflow.sh injects --sflow-source-path, so the generated script
    # must editable-install the checkout (uv pip install -e ".[dev]") and must NOT
    # fall back to a git-ref install.
    BATCH_DEFAULT_SCRIPT="$BATCH_DEFAULT_VERSION_DIR/default_version.sh"
    if [ -z "$IS_REAL_SUBMIT" ] && [ -f "$BATCH_DEFAULT_SCRIPT" ]; then
        if grep -Fq '"$VIRTUAL_ENV/bin/uv" pip install -e ".[dev]"' "$BATCH_DEFAULT_SCRIPT" \
            && ! grep -Fq 'git+https://github.com/NVIDIA/nv-sflow.git@' "$BATCH_DEFAULT_SCRIPT"; then
            echo "  PASS: batch under-dev install is editable from the local checkout"
        else
            echo "  FAIL: batch under-dev install is not editable from the local checkout"
            grep -F 'pip install' "$BATCH_DEFAULT_SCRIPT" || \
                echo "    (no pip install line found)"
            FAIL=$((FAIL + 1))
            TOTAL=$((TOTAL + 1))
            FAILED_LABELS="$FAILED_LABELS  - batch under-dev source-path install\n"
        fi
    fi

    # -- Post-wait: verify the no-flag default install link is the git ref
    # auto-detected from the running sflow env (and not an editable install). --
    BATCH_AUTOVERSION_SCRIPT="$BATCH_AUTOVERSION_DIR/auto_version.sh"
    if [ -z "$IS_REAL_SUBMIT" ] && [ -f "$BATCH_AUTOVERSION_SCRIPT" ]; then
        if grep -Fq '"$VIRTUAL_ENV/bin/uv" pip install '\''sflow @ git+https://github.com/NVIDIA/nv-sflow.git@' "$BATCH_AUTOVERSION_SCRIPT" \
            && grep -Fq -- '--prerelease=allow' "$BATCH_AUTOVERSION_SCRIPT" \
            && ! grep -Fq 'pip install -e ".[dev]"' "$BATCH_AUTOVERSION_SCRIPT"; then
            echo "  PASS: batch default install version auto-detected a git ref from the running env"
        else
            echo "  FAIL: batch default install version did not auto-detect a git ref from the running env"
            grep -F 'pip install' "$BATCH_AUTOVERSION_SCRIPT" || \
                echo "    (no pip install line found)"
            FAIL=$((FAIL + 1))
            TOTAL=$((TOTAL + 1))
            FAILED_LABELS="$FAILED_LABELS  - batch default install version auto-detect\n"
        fi
    fi

    # -- Post-wait: verify the sbatch venv bootstrap (1) neutralizes a leaked
    # caller virtualenv and creates the runtime venv with a resolved system
    # python3 (never a bare `python3` from PATH) -- sbatch defaults to
    # --export=ALL, so a caller venv on PATH would otherwise build the venv and
    # fail with "Exec format error" across login/compute CPU arch -- and (2)
    # creates a fresh per-job venv keyed on $SLURM_JOB_ID instead of sharing one
    # venv under flock, so concurrent jobs never race. --
    BATCH_VENV_BOOTSTRAP_SCRIPT="$PREFLIGHT_DIR/batch_venv_bootstrap/venv_bootstrap.sh"
    if [ -f "$BATCH_VENV_BOOTSTRAP_SCRIPT" ]; then
        VENV_BOOTSTRAP_FAIL=0
        # Caller virtualenv is stripped from PATH and unset before venv creation.
        if ! grep -Fq 'grep -vxF "$VIRTUAL_ENV/bin"' "$BATCH_VENV_BOOTSTRAP_SCRIPT"; then
            echo "  FAIL: sbatch script does not strip the caller VIRTUAL_ENV from PATH"
            VENV_BOOTSTRAP_FAIL=1
        fi
        if ! grep -Fq 'unset VIRTUAL_ENV' "$BATCH_VENV_BOOTSTRAP_SCRIPT"; then
            echo "  FAIL: sbatch script does not unset the inherited VIRTUAL_ENV"
            VENV_BOOTSTRAP_FAIL=1
        fi
        # Venv is created via a resolved interpreter, never a bare python3 from PATH.
        if ! grep -Fq '"$SFLOW_BOOTSTRAP_PYTHON" -m venv "$SFLOW_VENV_DIR"' "$BATCH_VENV_BOOTSTRAP_SCRIPT"; then
            echo "  FAIL: sbatch script does not create the venv with a resolved system python3"
            VENV_BOOTSTRAP_FAIL=1
        fi
        if grep -Fq 'python3 -m venv' "$BATCH_VENV_BOOTSTRAP_SCRIPT"; then
            echo "  FAIL: sbatch script still bootstraps the venv with a bare python3"
            VENV_BOOTSTRAP_FAIL=1
        fi
        # Fresh per-job venv keyed on the Slurm job id -- no shared venv, no flock.
        if ! grep -Fq '.sflow_venv-${SLURM_JOB_ID' "$BATCH_VENV_BOOTSTRAP_SCRIPT"; then
            echo "  FAIL: sbatch script does not create a per-job venv keyed on SLURM_JOB_ID"
            VENV_BOOTSTRAP_FAIL=1
        fi
        # The word "flock" still appears in an explanatory comment, so match the
        # actual flock *command* to detect a real regression to shared-venv locking.
        if grep -Fq 'flock -w' "$BATCH_VENV_BOOTSTRAP_SCRIPT"; then
            echo "  FAIL: sbatch script still uses flock for shared-venv locking"
            VENV_BOOTSTRAP_FAIL=1
        fi
        # Well-known absolute location tried first, with a PATH fallback.
        if ! grep -Fq '/usr/bin/python3' "$BATCH_VENV_BOOTSTRAP_SCRIPT"; then
            echo "  FAIL: sbatch script missing well-known absolute python3 location"
            VENV_BOOTSTRAP_FAIL=1
        fi
        if ! grep -Fq 'command -v python3' "$BATCH_VENV_BOOTSTRAP_SCRIPT"; then
            echo "  FAIL: sbatch script missing PATH python3 fallback"
            VENV_BOOTSTRAP_FAIL=1
        fi
        if [ "$VENV_BOOTSTRAP_FAIL" -eq 0 ]; then
            echo "  PASS: sbatch venv bootstrap resolves system python3 and creates a fresh per-job venv (no flock)"
        else
            FAIL=$((FAIL + VENV_BOOTSTRAP_FAIL))
            TOTAL=$((TOTAL + VENV_BOOTSTRAP_FAIL))
            FAILED_LABELS="$FAILED_LABELS  - sbatch venv bootstrap resolution\n"
        fi
    else
        echo "  FAIL: sbatch venv bootstrap script not generated"
        FAIL=$((FAIL + 1))
        TOTAL=$((TOTAL + 1))
        FAILED_LABELS="$FAILED_LABELS  - sbatch venv bootstrap script missing\n"
    fi

    # -- Post-wait: verify ${{ variables.X.domain }} resolved correctly --
    DOMAIN_RESOLVED="$COMPOSE_DOMAIN_DIR/resolved.yaml"
    if [ -f "$DOMAIN_RESOLVED" ]; then
        DOMAIN_FAIL=0
        if grep -q '\[1, 4, 16\]' "$DOMAIN_RESOLVED"; then
            echo "  PASS: variables.CONCURRENCY.domain resolved to [1, 4, 16]"
        else
            echo "  FAIL: variables.CONCURRENCY.domain not resolved in compose output"
            DOMAIN_FAIL=1
        fi
        if grep -q "sglang.*vllm.*trtllm" "$DOMAIN_RESOLVED"; then
            echo "  PASS: variables.FRAMEWORK.domain resolved to framework list"
        else
            echo "  FAIL: variables.FRAMEWORK.domain not resolved in compose output"
            DOMAIN_FAIL=1
        fi
        if [ "$DOMAIN_FAIL" -gt 0 ]; then
            FAIL=$((FAIL + DOMAIN_FAIL))
            TOTAL=$((TOTAL + DOMAIN_FAIL))
            FAILED_LABELS="$FAILED_LABELS  - variables.X.domain resolution\n"
        fi
    fi

    # -- Post-wait: verify compose -r rewrites resolved vars inside deferred Jinja --
    COMPOSE_DEFERRED_RESOLVED="$PREFLIGHT_DIR/compose_deferred_jinja/resolved.yaml"
    COMPOSE_DEFERRED_FAIL=0
    if [ ! -f "$COMPOSE_DEFERRED_RESOLVED" ]; then
        echo "  FAIL: compose deferred-Jinja e2e output missing"
        COMPOSE_DEFERRED_FAIL=1
    else
        if grep -q 'variables.INFRA_NODE_INDEX' "$COMPOSE_DEFERRED_RESOLVED"; then
            echo "  FAIL: compose deferred-Jinja output still references variables.INFRA_NODE_INDEX"
            COMPOSE_DEFERRED_FAIL=1
        fi
        if grep -q 'if 0 == 0' "$COMPOSE_DEFERRED_RESOLVED"; then
            echo "  PASS: compose -r rewrote resolved vars inside deferred Jinja"
        else
            echo "  FAIL: compose deferred-Jinja output did not inline the resolved literal"
            COMPOSE_DEFERRED_FAIL=1
        fi
    fi
    if [ "$COMPOSE_DEFERRED_FAIL" -gt 0 ]; then
        FAIL=$((FAIL + COMPOSE_DEFERRED_FAIL))
        TOTAL=$((TOTAL + COMPOSE_DEFERRED_FAIL))
        FAILED_LABELS="$FAILED_LABELS  - compose deferred-Jinja resolution\n"
    fi

    # -- Post-wait: verify compose -r resolves nodes.indices expression strings to a YAML list value --
    COMPOSE_INDICES_RESOLVED="$PREFLIGHT_DIR/compose_indices_expression/resolved.yaml"
    COMPOSE_INDICES_FAIL=0
    if [ ! -f "$COMPOSE_INDICES_RESOLVED" ]; then
        echo "  FAIL: compose nodes.indices e2e output missing"
        COMPOSE_INDICES_FAIL=1
    else
        export COMPOSE_INDICES_RESOLVED
        if python - <<'PY'
import os
from pathlib import Path
import yaml

resolved_path = Path(os.environ["COMPOSE_INDICES_RESOLVED"])
data = yaml.safe_load(resolved_path.read_text())
tasks = {task["name"]: task for task in data["workflow"]["tasks"]}
indices = tasks["frontend_server"]["resources"]["nodes"]["indices"]
exclude = tasks["worker_server"]["resources"]["nodes"]["exclude"]
assert indices in ("[0, 1]", [0, 1]), indices
assert exclude in ("[0, 1]", [0, 1]), exclude
PY
        then
            echo "  PASS: compose -r resolves resources.nodes.indices/exclude expression strings to [0, 1]"
        else
            echo "  FAIL: compose nodes.indices/exclude output did not resolve to [0, 1]"
            COMPOSE_INDICES_FAIL=1
        fi
    fi
    if [ "$COMPOSE_INDICES_FAIL" -gt 0 ]; then
        FAIL=$((FAIL + COMPOSE_INDICES_FAIL))
        TOTAL=$((TOTAL + COMPOSE_INDICES_FAIL))
        FAILED_LABELS="$FAILED_LABELS  - compose nodes.indices/exclude expression resolution\n"
    fi

    # -- Post-wait: verify dry-run assigns nodes from indices+count in the configured order --
    COMPOSE_INDICES_DRYRUN_FAIL=0
    if [ ! -f "$COMPOSE_INDICES_DRYRUN_LOG" ]; then
        echo "  FAIL: dry-run nodes.indices/count log missing"
        COMPOSE_INDICES_DRYRUN_FAIL=1
    else
        export COMPOSE_INDICES_DRYRUN_LOG
        if python - <<'PY'
import os
import re
from pathlib import Path

text = Path(os.environ["COMPOSE_INDICES_DRYRUN_LOG"]).read_text()
expected = {
    "ordered_pool_0": "slurm_cluster-node3",
    "ordered_pool_1": "slurm_cluster-node0",
    "ordered_pool_2": "slurm_cluster-node1",
    "ordered_pool_3": "slurm_cluster-node2",
}
for task_name, node_name in expected.items():
    pattern = rf"\[\d+\]\s+{re.escape(task_name)}.*?nodelist:\s+\['{re.escape(node_name)}'\]"
    assert re.search(pattern, text, re.S), (task_name, node_name)
PY
        then
            echo "  PASS: dry-run assigns indices+count replicas in configured order"
        else
            echo "  FAIL: dry-run did not preserve indices+count ordering"
            COMPOSE_INDICES_DRYRUN_FAIL=1
        fi
    fi
    if [ "$COMPOSE_INDICES_DRYRUN_FAIL" -gt 0 ]; then
        FAIL=$((FAIL + COMPOSE_INDICES_DRYRUN_FAIL))
        TOTAL=$((TOTAL + COMPOSE_INDICES_DRYRUN_FAIL))
        FAILED_LABELS="$FAILED_LABELS  - dry-run nodes.indices+count ordering\n"
    fi

    # -- Post-wait: verify a ':' in an unquoted task script command remains a string --
    COLON_SCRIPT_PREFLIGHT_FAIL=0
    for colon_output in "$COLON_SCRIPT_COMPOSED" "$COLON_SCRIPT_BATCH_CONFIG"; do
        if [ ! -f "$colon_output" ]; then
            echo "  FAIL: colon task script output missing: $colon_output"
            COLON_SCRIPT_PREFLIGHT_FAIL=1
            continue
        fi
        export COLON_SCRIPT_OUTPUT="$colon_output"
        if python - <<'PY'
import os
from pathlib import Path
import yaml

config_path = Path(os.environ["COLON_SCRIPT_OUTPUT"])
data = yaml.safe_load(config_path.read_text())
tasks = {task["name"]: task for task in data["workflow"]["tasks"]}
assert tasks["worker"]["script"][0] == 'echo "My GPUs: $CUDA_VISIBLE_DEVICES"'
PY
        then
            echo "  PASS: colon in task script is preserved as a plain command string in $(basename "$colon_output")"
        else
            echo "  FAIL: colon in task script was not preserved as a command string in $(basename "$colon_output")"
            COLON_SCRIPT_PREFLIGHT_FAIL=1
        fi
    done
    if [ "$COLON_SCRIPT_PREFLIGHT_FAIL" -gt 0 ]; then
        FAIL=$((FAIL + COLON_SCRIPT_PREFLIGHT_FAIL))
        TOTAL=$((TOTAL + COLON_SCRIPT_PREFLIGHT_FAIL))
        FAILED_LABELS="$FAILED_LABELS  - colon in task script preservation\n"
    fi

    # -- Post-wait: verify readiness probe list appears as two readiness checks in dry-run plan --
    READINESS_AND_FAIL=0
    if [ ! -f "$READINESS_AND_DRYRUN_LOG" ]; then
        echo "  FAIL: readiness probe list dry-run log missing"
        READINESS_AND_FAIL=1
    else
        if ! grep -q 'readiness: log_watch (pattern=readiness one)' "$READINESS_AND_DRYRUN_LOG"; then
            echo "  FAIL: first readiness probe missing from dry-run plan"
            READINESS_AND_FAIL=1
        fi
        if ! grep -q 'readiness: log_watch (pattern=readiness two)' "$READINESS_AND_DRYRUN_LOG"; then
            echo "  FAIL: second readiness probe missing from dry-run plan"
            READINESS_AND_FAIL=1
        fi
        if [ "$READINESS_AND_FAIL" -eq 0 ]; then
            echo "  PASS: readiness probe list expands to multiple readiness checks"
        fi
    fi
    if [ "$READINESS_AND_FAIL" -gt 0 ]; then
        FAIL=$((FAIL + READINESS_AND_FAIL))
        TOTAL=$((TOTAL + READINESS_AND_FAIL))
        FAILED_LABELS="$FAILED_LABELS  - readiness probe list dry-run expansion\n"
    fi

    # -- Post-wait: verify old single readiness probe object still works --
    READINESS_SINGLE_FAIL=0
    if [ ! -f "$READINESS_SINGLE_DRYRUN_LOG" ]; then
        echo "  FAIL: single readiness probe dry-run log missing"
        READINESS_SINGLE_FAIL=1
    else
        if ! grep -q 'readiness: log_watch (pattern=single readiness)' "$READINESS_SINGLE_DRYRUN_LOG"; then
            echo "  FAIL: single readiness probe missing from dry-run plan"
            READINESS_SINGLE_FAIL=1
        fi
        if grep -q 'ValidationError\|Traceback' "$READINESS_SINGLE_DRYRUN_LOG"; then
            echo "  FAIL: single readiness probe dry-run emitted validation error"
            READINESS_SINGLE_FAIL=1
        fi
        if [ "$READINESS_SINGLE_FAIL" -eq 0 ]; then
            echo "  PASS: single readiness probe object remains compatible"
        fi
    fi
    if [ "$READINESS_SINGLE_FAIL" -gt 0 ]; then
        FAIL=$((FAIL + READINESS_SINGLE_FAIL))
        TOTAL=$((TOTAL + READINESS_SINGLE_FAIL))
        FAILED_LABELS="$FAILED_LABELS  - single readiness probe compatibility\n"
    fi

    # -- Post-wait: verify sflow sample copy flows --
    SAMPLE_SELF_OUT="$PREFLIGHT_DIR/sample_copy_self/local_hello_world.yaml"
    SAMPLE_MODULAR_OUT="$PREFLIGHT_DIR/sample_copy_modular/inference_x_v2"
    SAMPLE_COPY_FAIL=0
    if [ -s "$SAMPLE_SELF_OUT" ]; then
        echo "  PASS: sample copied self-contained workflow to custom output path"
    else
        echo "  FAIL: sample self-contained copy missing or empty"
        SAMPLE_COPY_FAIL=1
    fi
    if [ -d "$SAMPLE_MODULAR_OUT" ] && \
       [ -f "$SAMPLE_MODULAR_OUT/slurm_config.yaml" ] && \
       [ -f "$SAMPLE_MODULAR_OUT/bulk_input.csv" ]; then
        echo "  PASS: sample copied modular workflow folder with key files"
    else
        echo "  FAIL: sample modular copy missing expected files"
        SAMPLE_COPY_FAIL=1
    fi
    if [ "$SAMPLE_COPY_FAIL" -gt 0 ]; then
        FAIL=$((FAIL + SAMPLE_COPY_FAIL))
        TOTAL=$((TOTAL + SAMPLE_COPY_FAIL))
        FAILED_LABELS="$FAILED_LABELS  - sample copy flows\n"
    fi

    # -- Post-wait: verify CLI --set wins over CSV column in bulk-input --
    # batch: generated sbatch scripts must call `sflow run --set GPUS_PER_NODE=77`
    # and must NOT pass the CSV value (GPUS_PER_NODE=4) for that variable.
    if [ -d "$BATCH_BULK_SET_DIR" ]; then
        SET_FAIL=0
        scripts_found=0
        for sh_file in "$BATCH_BULK_SET_DIR"/bulk_input_*/*.sh; do
            [ -f "$sh_file" ] || continue
            scripts_found=$((scripts_found + 1))
            if ! grep -q -- '--set GPUS_PER_NODE=77' "$sh_file"; then
                echo "  FAIL: CLI --set GPUS_PER_NODE=77 missing in $(basename "$sh_file")"
                SET_FAIL=1
            fi
            if grep -q -- '--set GPUS_PER_NODE=4\b' "$sh_file"; then
                echo "  FAIL: CSV GPUS_PER_NODE=4 not overridden in $(basename "$sh_file")"
                SET_FAIL=1
            fi
        done
        if [ "$scripts_found" -eq 0 ]; then
            echo "  FAIL: no scripts generated in $BATCH_BULK_SET_DIR"
            SET_FAIL=1
        fi
        if [ -f "$BATCH_BULK_SET_OUT" ] && \
           ! grep -q "CLI --set value will take precedence" "$BATCH_BULK_SET_OUT"; then
            echo "  FAIL: expected 'CLI --set value will take precedence' warning (batch)"
            SET_FAIL=1
        fi
        if [ "$SET_FAIL" -eq 0 ]; then
            echo "  PASS: batch bulk-input --set overrides CSV column (CLI wins)"
        else
            FAIL=$((FAIL + SET_FAIL))
            TOTAL=$((TOTAL + SET_FAIL))
            FAILED_LABELS="$FAILED_LABELS  - batch bulk-input --set precedence\n"
        fi
    fi

    # compose: merged YAMLs must carry the CLI value for GPUS_PER_NODE (77), not 4.
    if [ -d "$COMPOSE_BULK_SET_DIR" ]; then
        SET_FAIL=0
        yamls_found=0
        for yaml_file in "$COMPOSE_BULK_SET_DIR"/compose_*/*.yaml; do
            [ -f "$yaml_file" ] || continue
            yamls_found=$((yamls_found + 1))
            # Extract GPUS_PER_NODE variable block: expect value '77' from CLI, not 4 from CSV.
            gpn_value=$(awk '
                /name: GPUS_PER_NODE/ {found=1; next}
                found && /value:/ {
                    sub(/.*value:[[:space:]]*/, "")
                    gsub(/["'\'']/, "")
                    print
                    exit
                }
            ' "$yaml_file")
            if [ "$gpn_value" != "77" ]; then
                echo "  FAIL: GPUS_PER_NODE expected 77 (CLI), got '$gpn_value' in $(basename "$yaml_file")"
                SET_FAIL=1
            fi
        done
        if [ "$yamls_found" -eq 0 ]; then
            echo "  FAIL: no yamls generated in $COMPOSE_BULK_SET_DIR"
            SET_FAIL=1
        fi
        if [ -f "$COMPOSE_BULK_SET_OUT" ] && \
           ! grep -q "CLI --set value will take precedence" "$COMPOSE_BULK_SET_OUT"; then
            echo "  FAIL: expected 'CLI --set value will take precedence' warning (compose)"
            SET_FAIL=1
        fi
        if [ "$SET_FAIL" -eq 0 ]; then
            echo "  PASS: compose bulk-input --set overrides CSV column (CLI wins)"
        else
            FAIL=$((FAIL + SET_FAIL))
            TOTAL=$((TOTAL + SET_FAIL))
            FAILED_LABELS="$FAILED_LABELS  - compose bulk-input --set precedence\n"
        fi
    fi

    # -- Post-wait: verify sflow_batch_dir column in results.csv --
    for mode in batch_bulk_submit batch_bulk_input; do
        csv_file=$(find "$PREFLIGHT_DIR/$mode" -name results.csv -print -quit 2>/dev/null)
        if [ -f "$csv_file" ]; then
            csv_file=$(abs_path "$csv_file")
            if head -1 "$csv_file" | grep -q "sflow_batch_dir"; then
                bulk_dir=$(basename "$(dirname "$csv_file")")
                if grep -q "$bulk_dir" "$csv_file"; then
                    echo "  PASS: sflow_batch_dir column present and correct in $csv_file"
                else
                    echo "  FAIL: sflow_batch_dir value mismatch in $csv_file"
                    FAIL=$((FAIL + 1))
                    TOTAL=$((TOTAL + 1))
                    FAILED_LABELS="$FAILED_LABELS  - sflow_batch_dir value mismatch ($mode)\n"
                fi
            else
                echo "  FAIL: sflow_batch_dir column missing from $csv_file"
                FAIL=$((FAIL + 1))
                TOTAL=$((TOTAL + 1))
                FAILED_LABELS="$FAILED_LABELS  - sflow_batch_dir column missing ($mode)\n"
            fi
        fi
    done

    fi

    echo ""
    echo "===== Preflight Summary: $PASS/$TOTAL passed, $FAIL failed ====="
    echo ""
    echo "===== Results Directory: $PREFLIGHT_DIR ====="
    file_count=$(find "$PREFLIGHT_DIR" -type f | wc -l)
    echo "  ($file_count file(s) total)"
    echo ""

    if [ -n "$PREFLIGHT_SKIP_NOTES" ]; then
        echo "Skipped checks:"
        echo -e "$PREFLIGHT_SKIP_NOTES"
    fi

    if [ "$FAIL" -gt 0 ]; then
        echo "Failed tests:"
        echo -e "$FAILED_LABELS"
        echo "Failure details:"
        echo -e "$FAILED_DETAILS"
        echo "ERROR: $FAIL preflight check(s) failed — aborting before job submission."
        exit 1
    fi

fi

# =========================================================================
# Real e2e tests (submit jobs to Slurm)
# =========================================================================
if [ -n "$SUBMIT" ] && [ -z "$PREFLIGHT_ONLY" ]; then
    echo ""
    echo "===== All preflight checks passed — proceeding to job submission ====="
    echo ""
    set -x
    cd "$SCRIPT_DIR/../tests/e2e_tests"
    E2E_PARTITION="${CLI_PARTITION:-my_partition}"
    E2E_ACCOUNT="${CLI_ACCOUNT:-user}"
    E2E_SBATCH_OUTPUT="$REPO_DIR/sflow_output/%j-sflow-submit.out"
    E2E_SBATCH_ERROR="$REPO_DIR/sflow_output/%j-sflow-submit.err"
    # Known-broken/flaky Slurm nodes to keep every e2e submission off of. CI pins
    # the live list via the $SLURM_E2E_EXCLUDE_NODES job variable (forwarded to the
    # remote by run_slurm_e2e_over_ssh.py); the default here is the fallback for
    # local/manual runs. Add a node here (or to the CI variable) to drain it from
    # e2e without touching any recipe.
    E2E_EXCLUDE_NODES="${SLURM_E2E_EXCLUDE_NODES:-gb-nvl-137-compute02,gb-nvl-137-compute14}"
    E2E_BATCH_EXTRA_ARGS=(
        "--sbatch-output" "$E2E_SBATCH_OUTPUT"
        "--sbatch-error" "$E2E_SBATCH_ERROR"
        # Exercise the hardware monitor on every submitted workflow (OOTB defaults:
        # all scopes + csv/svg report). The independent monitor-coverage check in
        # sample_test.sh then asserts each run produced sflow_monitor.log,
        # regardless of the workflow's own pass/fail.
        "--enable-workflow-monitor"
    )
    # Flag and value MUST be separate argv tokens ("-e" "--exclude=..."); a glued
    # single token folds a leading space into the value and the secondary-backend
    # salloc drops the exclude silently.
    if [ -n "$E2E_EXCLUDE_NODES" ]; then
        E2E_BATCH_EXTRA_ARGS+=("-e" "--exclude=$E2E_EXCLUDE_NODES")
    fi
    if [ -n "${COLON_SCRIPT_FIXTURE:-}" ] && [ -f "$COLON_SCRIPT_FIXTURE" ]; then
        SFLOW_COLON_SCRIPT_FIXTURE="$COLON_SCRIPT_FIXTURE" \
        SFLOW_COLON_SCRIPT_OUTPUT_DIR="$REPO_DIR/sflow_output/colon_in_task_script" \
            ./sample_test.sh -p "$E2E_PARTITION" -A "$E2E_ACCOUNT" -m "$MODEL_PATH" -t "$TEST_TYPE" --submit -- "${E2E_BATCH_EXTRA_ARGS[@]}" # 09 has some GPU issues
    else
        ./sample_test.sh -p "$E2E_PARTITION" -A "$E2E_ACCOUNT" -m "$MODEL_PATH" -t "$TEST_TYPE" --submit -- "${E2E_BATCH_EXTRA_ARGS[@]}" # 09 has some GPU issues
    fi

    set +x
elif [ -z "$SUBMIT" ]; then
    echo "Preflight only (no -S flag). To submit jobs, re-run with -S."
else
    echo "Preflight only (-P flag). Skipping job submission."
fi
