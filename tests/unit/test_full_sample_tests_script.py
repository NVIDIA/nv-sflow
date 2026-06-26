from pathlib import Path


def test_scripts_do_not_embed_plaintext_gitlab_details():
    scripts_dir = Path("scripts")
    forbidden = ("gitlab", "GITLAB", "gitlab-master", "oauth2")

    for script in scripts_dir.glob("*.sh"):
        text = script.read_text()
        for term in forbidden:
            assert term not in text, f"{script} embeds {term!r}"


def test_infmax_recipe_source_comes_from_env_and_can_skip():
    script = Path("scripts/full_sample_tests.sh").read_text()
    fetch_block = script.split("fetch_infmax_recipes() {", 1)[1].split(
        "set_infmax_suite_overrides() {", 1
    )[0]
    summary_block = script.split('echo "===== Preflight Summary:', 1)[1]

    assert 'INFMAX_RECIPE_REPO="${INFMAX_RECIPE_REPO:-}"' in script
    assert 'INFMAX_RECIPE_REF="${INFMAX_RECIPE_REF:-}"' in script
    assert 'INFMAX_RECIPE_SUBDIR="${INFMAX_RECIPE_SUBDIR:-}"' in script
    assert "INFMAX_RECIPE_SKIP_REASON=" in fetch_block
    assert "WARN: skipping Infmax recipe fetch" in fetch_block
    assert "return 1" in fetch_block
    assert "if fetch_infmax_recipes; then" in script
    assert "record_preflight_skip" in script
    assert "Skipped checks:" in summary_block


def test_full_sample_submit_mode_uses_sample_test_for_infmax_tracking():
    script = Path("scripts/full_sample_tests.sh").read_text()
    submit_block = script.split(
        'if [ -n "$SUBMIT" ] && [ -z "$PREFLIGHT_ONLY" ]; then', 1
    )[1]
    sample_test = Path("tests/e2e_tests/sample_test.sh").read_text()

    assert "infmax/batch_test.sh" not in script
    assert "INFMAX_SUBMIT_PID" not in submit_block
    assert "bash ./infmax/batch_test.sh" not in submit_block
    assert "./sample_test.sh" in submit_block
    assert 'E2E_SBATCH_OUTPUT="$REPO_DIR/sflow_output/%j-sflow-submit.out"' in submit_block
    assert 'E2E_SBATCH_ERROR="$REPO_DIR/sflow_output/%j-sflow-submit.err"' in submit_block
    assert '"--sbatch-output" "$E2E_SBATCH_OUTPUT"' in submit_block
    assert '"--sbatch-error" "$E2E_SBATCH_ERROR"' in submit_block
    assert "-e --exclude=gb-nvl-137-compute09,gb-nvl-137-compute16,gb-nvl-137-compute03" in submit_block


def test_full_sample_supports_infmax_only_flag():
    script = Path("scripts/full_sample_tests.sh").read_text()

    assert '-inf) TEST_TYPE="inf" ;;' in script
    assert 'echo "  -inf  infmax batch suites only"' in script
    assert 'if [ "$TEST_TYPE" = "inf" ]; then' in script
    assert '-t "$TEST_TYPE"' in script


def test_sample_test_submits_infmax_jobs_into_main_job_id_list():
    script = Path("tests/e2e_tests/sample_test.sh").read_text()

    assert "submit_infmax_batch_suites" in script
    assert 'JOB_IDS+=("$job_id")' in script.split("submit_infmax_batch_suites", 1)[1]
    assert "===== Part 3: Infmax multi-node batch suites =====" in script


def test_sample_test_uses_repo_output_dir_for_submitted_batches():
    script = Path("tests/e2e_tests/sample_test.sh").read_text()

    assert 'E2E_OUTPUT_DIR="${E2E_OUTPUT_DIR:-$REPO_DIR/sflow_output}"' in script
    assert 'search_dir="${E2E_OUTPUT_DIR:-sflow_output}"' in script
    assert '"${BATCH_OUTPUT_ARGS[@]}"' in script


def test_sample_test_supports_infmax_only_type():
    script = Path("tests/e2e_tests/sample_test.sh").read_text()
    part2 = script.split("Part 2: Modular examples", 1)[1].split(
        "Part 3: Infmax multi-node batch suites", 1
    )[0]
    part3 = script.split("Part 3: Infmax multi-node batch suites", 1)[1]

    assert "[-t s|m|inf|a|smoke|min]" in script
    assert '&& [ "$TEST_TYPE" != "inf" ]' in script
    assert '&& [ "$TEST_TYPE" != "smoke" ]' in script
    assert 'if [ "$TEST_TYPE" = "inf" ]' not in part2
    assert 'if [ "$TEST_TYPE" = "inf" ] || [ "$TEST_TYPE" = "m" ] || [ "$TEST_TYPE" = "a" ] || [ "$TEST_TYPE" = "smoke" ] || [ "$TEST_TYPE" = "min" ]; then' in part3


def test_full_sample_and_sample_test_support_min_submit_mode():
    full_sample = Path("scripts/full_sample_tests.sh").read_text()
    sample_test = Path("tests/e2e_tests/sample_test.sh").read_text()
    min_self_contained_block = sample_test.split("MIN_SELF_CONTAINED=(", 1)[1].split(
        ")",
        1,
    )[0]

    assert "  --min  minimal Slurm submit set" in full_sample
    assert '--min) TEST_TYPE="min" ;;' in full_sample
    assert "  --smoke  curated Slurm smoke subset with broad coverage" in full_sample
    assert '--smoke) TEST_TYPE="smoke" ;;' in full_sample
    assert '        -smoke) TEST_TYPE="smoke" ;;' not in full_sample
    assert 'echo "  -t min Minimal representative set' in sample_test
    assert 'if [ "$TEST_TYPE" = "min" ]; then' in sample_test
    assert 'run_check "run local_dag (live)"' in full_sample
    assert 'sflow run "$EXAMPLES_DIR/local_dag.yaml"' in full_sample
    assert 'run_check "run local_variable_domain (live, optional)"' in full_sample
    assert 'local_dag.yaml' not in min_self_contained_block
    assert 'local_variable_domain.yaml' not in min_self_contained_block
    assert 'slurm_auto_replica.yaml' in min_self_contained_block
    assert 'slurm_dynamo_trtllm_disagg.yaml' in min_self_contained_block
    assert 'slurm_resource_release_after.yaml' in min_self_contained_block
    assert 'slurm_trtllm_serve_disagg.yaml' in min_self_contained_block
    assert 'MODULAR_ROW_ARGS=(--row 2)' in sample_test
    assert "submit_min_infmax_suite" in sample_test
    assert "$INFMAX_DIR/dsr1-fp4-gb200-multi_node-trtllm" in sample_test


def test_infmax_kimi_suite_pins_runtime_overrides():
    full_sample = Path("scripts/full_sample_tests.sh").read_text()
    sample_test = Path("tests/e2e_tests/sample_test.sh").read_text()

    expected_base_override = 'INFMAX_SUITE_OVERRIDES=(-s "CONCURRENCY=[16,32]")'
    expected_kimi_override = 'INFMAX_SUITE_OVERRIDES+=(-s "DYNAMO_VERSION=1.0.1" -e "--container-remap-root")'
    assert expected_base_override in full_sample
    assert expected_base_override in sample_test
    assert expected_kimi_override in full_sample
    assert expected_kimi_override in sample_test
    assert full_sample.count('"${INFMAX_SUITE_OVERRIDES[@]}"') == 2
    assert sample_test.count('"${INFMAX_SUITE_OVERRIDES[@]}"') == 3


def test_e2e_submit_enables_workflow_monitor_and_checks_coverage():
    """Every submitted e2e workflow runs with --enable-workflow-monitor, and
    sample_test.sh asserts each produced sflow_monitor.log INDEPENDENTLY of the
    job's pass/fail (its own parseable coverage line)."""
    full_sample = Path("scripts/full_sample_tests.sh").read_text()
    sample_test = Path("tests/e2e_tests/sample_test.sh").read_text()
    submit_block = full_sample.split(
        'if [ -n "$SUBMIT" ] && [ -z "$PREFLIGHT_ONLY" ]; then', 1
    )[1]

    # Injected for ALL submitted jobs via the shared extra-args array.
    assert '"--enable-workflow-monitor"' in submit_block

    # Independent monitor-coverage check + the parseable summary line the CI
    # helper (run_slurm_e2e_over_ssh.parse_monitor_coverage) keys on.
    assert "===== Monitor Coverage =====" in sample_test
    assert (
        "workflows produced a monitor overview with metrics (sflow_monitor.log)"
        in sample_test
    )
    # The coverage loop is independent of the PASS/FAIL results loop.
    assert "MONITOR_PRESENT" in sample_test and "MONITOR_TOTAL" in sample_test
    assert "name 'sflow_monitor.log'" in sample_test
    # Existence is not enough: the overview must carry a populated Metric Summary
    # (positive sample count + real numeric metric rows), not just an empty file.
    assert "monitor_log_has_content" in sample_test
    assert "Metric Summary" in sample_test
    assert "(no numeric samples collected)" in sample_test
    assert "cpu_utilization_pct" in sample_test

    # Every submitted e2e workflow must be monitored, including the focused
    # colon-in-task-script job, which submits via its OWN `sflow batch` (it does
    # not receive E2E_BATCH_EXTRA_ARGS), so it must enable the monitor explicitly
    # or it would be the lone uncovered workflow tripping the coverage gate.
    colon_block = sample_test.split("submit_colon_task_script_e2e() {", 1)[1].split(
        "\n}\n", 1
    )[0]
    assert "--enable-workflow-monitor" in colon_block


def test_sample_test_checks_disagg_monitor_targeting():
    """The disagg recipes monitor the prefill/decode servers from the benchmark
    task (used_by_tasks); sample_test.sh must verify the generated cross reports
    sampled the SERVERS' resources, on its own parseable gate line, and the CI
    helper must parse + enforce it independently."""
    sample_test = Path("tests/e2e_tests/sample_test.sh").read_text()
    checker = Path("tests/e2e_tests/check_disagg_monitor.py")

    # Dedicated stdlib checker exists and is invoked per disagg workflow.
    assert checker.is_file()
    assert "check_disagg_monitor.py" in sample_test
    assert "===== Monitor Targeting (disagg used_by_tasks) =====" in sample_test
    assert "-name '*__monitored_by__benchmark'" in sample_test
    # Parseable summary line, mirrored by the CI parser's regex.
    assert (
        "disagg workflows monitored the correct (server) resources" in sample_test
    )


def test_disagg_recipes_monitor_servers_from_benchmark():
    """Every slurm self-contained disagg recipe must trigger a monitor from the
    benchmark task that samples the prefill/decode servers' resources."""
    import yaml

    disagg = sorted(Path("examples").glob("slurm_*disagg.yaml"))
    assert disagg, "expected slurm *_disagg.yaml recipes under examples/"
    for recipe in disagg:
        data = yaml.safe_load(recipe.read_text())
        tasks = {t["name"]: t for t in data["workflow"]["tasks"]}
        assert "benchmark" in tasks, f"{recipe}: no benchmark task"
        monitor = tasks["benchmark"].get("monitor")
        assert monitor is not None, f"{recipe}: benchmark task has no monitor"
        used = monitor.get("resources", {}).get("used_by_tasks")
        assert used == ["prefill_server", "decode_server"], (
            f"{recipe}: benchmark monitor must target the servers, got {used}"
        )
        assert monitor.get("report", {}).get("enabled") is True, (
            f"{recipe}: benchmark monitor report must be enabled"
        )
        # The packaged sample copy must match the example.
        sample = Path("src/sflow/samples") / recipe.name
        assert sample.read_text() == recipe.read_text(), (
            f"{sample} is out of sync with {recipe}"
        )


def test_self_contained_benchmark_recipes_monitor_servers():
    """Every self-contained slurm recipe whose benchmark drives server task(s)
    must trigger a monitor from the benchmark sampling those servers' resources
    (used_by_tasks) -- not only the disagg ones. Standalone benchmarks with no
    server (e.g. the aiperf template) are exempt. Packaged samples must match."""
    import yaml

    server_tasks = {"prefill_server", "decode_server", "agg_server", "sglang_server"}
    checked = 0
    for recipe in sorted(Path("examples").glob("slurm_*.yaml")):
        data = yaml.safe_load(recipe.read_text())
        tasks = {t["name"]: t for t in data.get("workflow", {}).get("tasks", [])}
        if "benchmark" not in tasks:
            continue
        servers = server_tasks & tasks.keys()
        if not servers:
            continue  # standalone benchmark (no server to monitor)
        checked += 1
        monitor = tasks["benchmark"].get("monitor")
        assert monitor is not None, f"{recipe}: benchmark task has no monitor"
        used = monitor.get("resources", {}).get("used_by_tasks") or []
        assert used, f"{recipe}: benchmark monitor has no used_by_tasks"
        # Targets the recipe's server task(s), and they exist in the recipe.
        assert set(used) <= server_tasks, f"{recipe}: unexpected used_by_tasks {used}"
        assert set(used) <= tasks.keys(), f"{recipe}: used_by_tasks refers to missing tasks"
        assert monitor.get("report", {}).get("enabled") is True, (
            f"{recipe}: benchmark monitor report must be enabled"
        )
        sample = Path("src/sflow/samples") / recipe.name
        assert sample.read_text() == recipe.read_text(), (
            f"{sample} is out of sync with {recipe}"
        )
    assert checked >= 9, f"expected >=9 benchmark+server recipes, checked {checked}"


def test_full_sample_covers_monitor_cli_injection():
    """Preflight exercises --enable-workflow-monitor / --enable-task-monitor:
    app-layer injection (run --dry-run), split_list_arg list forms, unknown-task
    rejection, and batch threading into the sbatch script + composed snapshot."""
    script = Path("scripts/full_sample_tests.sh").read_text()
    block = script.split("# -- sflow run/batch: monitor CLI injection", 1)[1].split(
        "# -- sflow batch -f (multi-file): modular examples --", 1
    )[0]

    # Workflow-level injection visible in the dry-run plan.
    assert "run --enable-workflow-monitor injects a workflow monitor" in block
    assert "--enable-workflow-monitor --dry-run" in block
    assert "Enabled default workflow monitor" in block
    assert "Planned monitors" in block
    # Task-level injection across comma / whitespace / repeated-flag forms.
    assert (
        "--enable-task-monitor 'prepare_data,preprocess' "
        "--enable-task-monitor 'train export_model'" in block
    )
    assert "Enabled default task monitor for 'train'" in block
    # Negative: an unknown task name is rejected.
    assert "rejects an unknown task" in block
    assert "refers to unknown task" in block
    # batch threads the flags into the generated script AND the YAML snapshot.
    assert "grep -q -- '--enable-workflow-monitor'" in block
    assert "grep -q -- '--enable-task-monitor'" in block
    assert "grep -q 'monitor:'" in block


def test_sample_test_monitor_content_check_requires_populated_summary(tmp_path, fp):
    """The e2e monitor-coverage gate must accept only a *populated* overview:
    a real sflow_monitor.log with a Metric Summary + positive sample count is
    OK, while a header-only / "(no numeric samples collected)" file is rejected
    (existence alone is not sufficient)."""
    import subprocess

    fp.allow_unregistered(True)  # this test drives a real `bash` subprocess

    script = Path("tests/e2e_tests/sample_test.sh").read_text()
    start = script.index("monitor_log_has_content() {")
    end = script.index("\n}\n", start) + len("\n}\n")
    func = script[start:end]

    populated = tmp_path / "good.log"
    populated.write_text(
        "Sflow Monitor\n=============\nWorkflow : wf\nSamples  : 1392\n\n"
        "Metric Summary\n--------------\n"
        "cpu       cpu_utilization_pct         %       48     9.82    16.16    28.19\n"
    )
    empty = tmp_path / "empty.log"
    empty.write_text(
        "Sflow Monitor\n=============\nWorkflow : wf\nSamples  : 0\n\n"
        "Metric Summary\n--------------\n(no numeric samples collected)\n"
    )

    harness = func + '\nif monitor_log_has_content "$1"; then echo MATCH; else echo NOMATCH; fi\n'
    good = subprocess.run(
        ["bash", "-c", harness, "_", str(populated)], capture_output=True, text=True
    )
    bad = subprocess.run(
        ["bash", "-c", harness, "_", str(empty)], capture_output=True, text=True
    )

    assert good.stdout.strip() == "MATCH", good.stdout + good.stderr
    assert bad.stdout.strip() == "NOMATCH", bad.stdout + bad.stderr


def test_infmax_suites_include_monitor_config():
    """Every Infmax `sflow batch` invocation must layer in monitor/monitor.yaml
    so the monitor tests are exercised alongside each suite.
    """
    full_sample = Path("scripts/full_sample_tests.sh").read_text()
    sample_test = Path("tests/e2e_tests/sample_test.sh").read_text()

    monitor_default = (
        'INFMAX_MONITOR_CONFIG="${INFMAX_MONITOR_CONFIG:-$INFMAX_DIR/monitor/monitor.yaml}"'
    )
    assert monitor_default in full_sample
    assert monitor_default in sample_test

    # fetch_infmax_recipes must include the monitor file in its required_path
    # probe so a stale cache (older recipe revision without monitor/) triggers
    # a re-fetch instead of silently running without monitor coverage.
    assert full_sample.count('"$INFMAX_MONITOR_CONFIG"') >= 3
    assert sample_test.count('"$INFMAX_MONITOR_CONFIG"') >= 3
    assert full_sample.count('-f "$INFMAX_MONITOR_CONFIG"') == 2
    assert sample_test.count('-f "$INFMAX_MONITOR_CONFIG"') == 3


def test_sample_scripts_force_under_dev_sflow_checkout():
    helper = Path("scripts/use_under_dev_sflow.sh").read_text()
    full_sample = Path("scripts/full_sample_tests.sh").read_text()
    sample_test = Path("tests/e2e_tests/sample_test.sh").read_text()
    full_sample_setup_block = full_sample.split('source "$SCRIPT_DIR/use_under_dev_sflow.sh"', 1)[
        1
    ].split("RESULTS_DIR=$(mktemp -d)", 1)[0]
    default_version_block = full_sample.split(
        "# -- sflow batch under-dev install: should be editable from the local checkout --",
        1,
    )[1].split("# -- sflow batch -e with variables.X.domain expression --", 1)[0]
    default_version_summary_block = full_sample.split(
        "# -- Post-wait: verify the under-dev batch install is editable from the local checkout --",
        1,
    )[1].split("# -- Post-wait: verify ${{ variables.X.domain }}", 1)[0]
    submit_block = full_sample.split(
        'if [ -n "$SUBMIT" ] && [ -z "$PREFLIGHT_ONLY" ]; then', 1
    )[1]

    assert "assert_under_dev_sflow_editable_install" in helper
    assert "source  : local editable dev" in helper
    assert "exec \"$SFLOW_TEST_PYTHON\" -m sflow \"$@\"" in helper
    # Under-dev batch runs install the local checkout editable via the injected
    # --sflow-source-path (mutually exclusive with --sflow-version).
    assert '--sflow-source-path "$SFLOW_UNDER_DEV_REPO"' in helper
    assert 'if [ -z "$IS_REAL_SUBMIT" ]; then' in full_sample_setup_block
    assert "setup_under_dev_sflow \"$REPO_DIR\" || exit 1" in full_sample_setup_block
    assert "Skipping local under-dev sflow setup for real Slurm submit" in full_sample_setup_block
    assert 'if [ -z "$IS_REAL_SUBMIT" ] && [ -f "$EXTRA_ARGS_EXAMPLE" ]; then' in default_version_block
    assert 'if [ -z "$IS_REAL_SUBMIT" ] && [ -f "$BATCH_DEFAULT_SCRIPT" ]; then' in default_version_summary_block
    assert "cleanup_under_dev_sflow" in full_sample
    assert "setup_under_dev_sflow \"$REPO_DIR\" || exit 1" in sample_test
    assert "cleanup_under_dev_sflow" in sample_test
    assert '--workspace-dir "$REPO_DIR"' in sample_test
    assert '--sflow-venv-path "$WORKSPACE_DIR"' not in sample_test
    assert 'SFLOW_COLON_SCRIPT_OUTPUT_DIR="$REPO_DIR/sflow_output/colon_in_task_script"' in submit_block


def test_full_sample_summary_check_matches_current_layout():
    script = Path("scripts/full_sample_tests.sh").read_text()
    summary_block = script.split(
        "# -- Post-wait: verify live run summary and command-only command logs --", 1
    )[1].split("# -- Post-wait: verify ${{ variables.X.domain }}", 1)[0]

    assert "'End Summary'" not in summary_block
    assert "'Counts       :'" in summary_block
    assert "'FAILED/CANCELLED Tasks :'" in summary_block
    assert "'Task Duration Chart'" in summary_block
    assert "'Timeline'" in summary_block


def test_full_sample_covers_backend_agnostic_examples():
    script = Path("scripts/full_sample_tests.sh").read_text()

    assert "dry-run docker_hello_world uses docker_run default operator" in script
    assert "dry-run docker_multi_node assigns remote Docker hosts and GPUs" in script
    assert "dry-run kubernetes_hello_world uses k8s operator" in script
    assert "operator: docker_run" in script
    assert "CUDA_VISIBLE_DEVICES: 0" in script
    assert "gpus: device=0" in script
    assert "id=kubernetes" in script


def test_full_sample_covers_dry_run_report_reformat():
    script = Path("scripts/full_sample_tests.sh").read_text()

    # Standardized layout + compact-by-default Tasks + --verbose detail.
    assert "dry-run report uses standardized sections + compact tasks by default" in script
    assert "dry-run --verbose expands full per-task detail" in script
    assert "── Plan " in script
    assert "── Tasks " in script
    assert "(use --verbose for full per-task details)" in script
    # Storage / uploads sections + replica auto-rename.
    assert "dry-run renders storage targets and planned uploads" in script
    assert "── Storage targets " in script
    assert "[results_bucket] S3StorageTarget" in script
    assert "auto-renamed per replica" in script
    # sbatch out/err surfaced in the Plan via `sflow batch`.
    assert "sflow batch dry-run Plan shows sbatch out/err paths" in script
    assert "sbatch out:" in script
    assert "sbatch err:" in script


def test_full_sample_covers_slurm_env_inheritance_and_sflow_aliases():
    script = Path("scripts/full_sample_tests.sh").read_text()

    assert "run fake slurm controller env inheritance and sflow aliases" in script
    assert "SLURM_CUSTOM_CONTROLLER_ENV=controller-kept" in script
    assert "SLURM_JOB_ID=777777" in script
    assert "SFLOW_BACKEND_JOB_ID=777777" in script
    assert "SFLOW_BACKEND_NODELIST=fake-node" in script
    assert "SFLOW_BACKEND_NUM_NODES=1" in script
    assert "SFLOW_BACKEND_STEP_ID=3" in script
    assert "SFLOW_TASK_NODE_NAME=fake-node" in script
    assert "SFLOW_TASK_PROCESS_ID=5" in script
    assert "SFLOW_TASK_LOCAL_PROCESS_ID=0" in script
    assert "SFLOW_TASK_NUM_PROCESSES=1" in script


def test_full_sample_covers_tui_live_run_with_local_dag():
    script = Path("scripts/full_sample_tests.sh").read_text()
    tui_block = script.split("# -- sflow run (live): verify TUI launch path --", 1)[1].split(
        "# -- sflow run (live): per-task log offload ON/OFF x --tui ON/OFF matrix --",
        1,
    )[0]

    assert 'run_check "run local_dag with TUI (live)"' in tui_block
    assert 'sflow run "$EXAMPLES_DIR/local_dag.yaml"' in tui_block
    assert "--tui" in tui_block
    assert "--dry-run" not in tui_block
    assert "--output-dir \"$TUI_RUN_DIR\"" in tui_block


def test_full_sample_covers_offload_tui_logging_matrix():
    """The offload x --tui matrix section must drive all four combinations and
    assert: every combo writes the per-task <task>.log, task output never leaks
    into sflow.log, no scattered <task>.orchestration.log sidecar is left behind
    (offload merges diagnostics into <task>.log), and on a TTY the output streams
    to the console (offload auto-falls back to streaming).
    """
    script = Path("scripts/full_sample_tests.sh").read_text()
    block = script.split(
        "# -- sflow run (live): per-task log offload ON/OFF x --tui ON/OFF matrix --",
        1,
    )[1].split(
        "# -- sflow run/batch: plain script commands containing ':' must stay strings --",
        1,
    )[0]

    # Fixture + helper are materialized and driven by a single run_check.
    assert 'OFFLOAD_TUI_FIXTURE="$OFFLOAD_TUI_DIR/offload_matrix.yaml"' in block
    assert 'OFFLOAD_TUI_CHECK="$OFFLOAD_TUI_DIR/offload_tui_matrix_check.sh"' in block
    assert (
        'run_check "run per-task log offload x TUI matrix (4 combos: file + console) (live)"'
        in block
    )
    assert 'bash "$OFFLOAD_TUI_CHECK" "$OFFLOAD_TUI_FIXTURE" "$OFFLOAD_TUI_DIR"' in block

    # The task prints a concatenated sentinel so it only appears as real task
    # stdout, never in the logged command text.
    assert "printf '%s%s\\n' OFFLOADMATRIX SENTINEL" in block
    assert 'marker="OFFLOADMATRIXSENTINEL"' in block

    # All four offload x tui combinations are exercised.
    assert "for offload in --no-offload-task-logs --offload-task-logs; do" in block
    assert 'for tui in "" --tui; do' in block

    # (a) per-task <task>.log is always asserted to contain the task output;
    assert "assert_task_log" in block
    assert "name printer.log" in block
    # (b) task output must NOT leak into sflow.log;
    assert "task output leaked into sflow.log" in block
    # (c) driver diagnostics are merged into <task>.log -> NO scattered sidecar;
    assert "*.orchestration.log" in block
    assert "unexpected scattered orchestration sidecar" in block
    # (d) on a TTY the output streams to the console (PTY-captured).
    assert "pty.spawn" in block
    assert "not streamed to the console on a TTY" in block


def test_full_sample_covers_result_parsing_sample_yaml():
    script = Path("scripts/full_sample_tests.sh").read_text()
    result_block = script.split(
        "# -- sflow run (live): verify consolidated result parsing from logs and files --",
        1,
    )[1].split(
        "# -- sflow run (live): verify per-task uploads and workflow.upload_all with",
        1,
    )[0]

    assert 'RESULT_PARSING_SAMPLE="$EXAMPLES_DIR/local_result_parsing.yaml"' in result_block
    assert 'run_check "run result parsing workflow (live)"' in result_block
    assert "RESULT_PARSING_SAMPLE_PASS" in result_block
    assert "cat >" not in result_block


def test_full_sample_resource_rehearsal_title_highlights_reuse():
    script = Path("scripts/full_sample_tests.sh").read_text()

    assert "Resource Occupancy" in script
    assert "Resource release rehearsal" not in script


def test_full_sample_covers_reserved_env_collision_warning():
    script = Path("scripts/full_sample_tests.sh").read_text()
    collision_block = script.split(
        "# -- sflow run --dry-run: user variables colliding with reserved envs warn --",
        1,
    )[1].split(
        "# -- sflow run --dry-run: self-contained slurm examples --",
        1,
    )[0]

    assert "reserved_env_collision.yaml" in collision_block
    assert "SFLOW_TASK_OUTPUT_DIR" in collision_block
    assert "CUDA_VISIBLE_DEVICES" in collision_block
    assert "Reserved env collisions" in collision_block
    assert "MODEL_PATH" in collision_block
    assert "⚠ MODEL_PATH" in collision_block


def test_full_sample_visualize_preflight_does_not_require_graphviz():
    script = Path("scripts/full_sample_tests.sh").read_text()
    visualize_block = script.split('# -- sflow visualize --', 1)[1].split(
        '# -- sflow sample --', 1
    )[0]

    assert '--format mermaid' in visualize_block
    assert 'visualize_vllm_disagg.mmd' in visualize_block
    assert 'visualize_vllm_disagg.png' not in visualize_block


def test_full_sample_covers_multi_backend_slurm():
    script = Path("scripts/full_sample_tests.sh").read_text()
    block = script.split(
        "# -- sflow run (fake Slurm): multi-backend binds each task to its own backend allocation --",
        1,
    )[1].split("# -- sflow run --dry-run: modular (multi-file) --", 1)[0]

    assert "run fake slurm multi-backend binds each task to its own backend" in block
    assert "slurm_multi_backend.yaml" in block
    assert "--set PARTITION_A=alpha" in block
    assert "--set PARTITION_B=beta" in block
    # Each backend allocates its own pool on its own partition.
    assert "--partition alpha" in block
    assert "--partition beta" in block
    # task_a -> cluster_a allocation (5001/alpha-node).
    assert "--job-name task_a" in block
    assert "--jobid 5001" in block
    assert "--nodelist alpha-node" in block
    # task_b -> cluster_b allocation (5002/beta-node).
    assert "--job-name task_b" in block
    assert "--jobid 5002" in block
    assert "--nodelist beta-node" in block
    # Each backend runs its own operator's container image.
    assert "--set IMAGE_A=nvcr.io/sflow-test/mb:alpha" in block
    assert "--set IMAGE_B=nvcr.io/sflow-test/mb:beta" in block
    assert "--container-image nvcr.io/sflow-test/mb:alpha" in block
    assert "--container-image nvcr.io/sflow-test/mb:beta" in block
    # Both owned allocations are released on completion.
    assert "scancel" in block


def test_full_sample_covers_multi_backend_driver():
    script = Path("scripts/full_sample_tests.sh").read_text()
    block = script.split(
        "# -- sflow run (inside fake `sflow batch` driver): leader reuses the driver",
        1,
    )[1].split("# -- sflow run --dry-run: modular (multi-file) --", 1)[0]

    assert "run fake slurm batch driver leader reuses alloc and others salloc" in block
    # Driver sbatch env: leader allocation + per-backend-salloc markers.
    assert "export SLURM_JOB_ID=6001" in block
    assert "export SFLOW_SLURM_MULTI_BACKEND_SALLOC=1" in block
    assert "export SFLOW_SLURM_WRAPPER_BACKEND=cluster_a" in block
    # Only the non-leader backend sallocs; the leader reuses the driver alloc.
    assert "--partition beta" in block
    # task_a (leader) reuses the driver job 6001; task_b sallocs its own (5002).
    assert "--jobid 6001" in block
    assert "--jobid 5002" in block
    # Only the non-leader's own salloc job is scancelled (leader alloc not owned).
    assert "scancel" in block


def test_sample_test_runs_real_multi_backend():
    script = Path("tests/e2e_tests/sample_test.sh").read_text()

    # Real multi-backend coverage is a `sflow batch` multi-backend driver job
    # (each backend runs its own salloc), NOT a single-allocation batch job
    # (which would collapse both backends into one pool).
    assert "run_multi_backend_real" in script
    assert 'sflow batch "$EXAMPLES_DIR/slurm_multi_backend.yaml"' in script
    assert "per-backend salloc via sflow batch" in script
    # Two DIFFERENT partitions (one per component), overridable via env and
    # defaulting to the CI e2e partitions.
    assert '--set "PARTITION_A=$part_a"' in script
    assert '--set "PARTITION_B=$part_b"' in script
    assert "MULTI_BACKEND_PARTITION_A:-genesisq" in script
    assert "MULTI_BACKEND_PARTITION_B:-gamoraq" in script
    # Each task echoes its backend + assigned nodes; distinct nodes prove each
    # operator bound to its own het component (job ids collapse to the leader).
    assert "task_a backend=cluster_a job=" in script
    assert "task_b backend=cluster_b job=" in script
    # Runs for both CI submit modes (min and smoke), plus s/a.
    assert "s|a|smoke|min)" in script
    # NOT bulk-submitted (single-partition) in smoke/all; the driver job is authoritative.
    assert "slurm_multi_backend.yaml)" in script
    # Merged into the shared submit -> wait(sacct) -> validate loop: the driver
    # job id joins JOB_IDS and the results loop validates distinct nodes.
    assert 'JOB_IDS+=("$mb_job_id")' in script
    assert 'MULTI_BACKEND_JOB_IDS+=("$mb_job_id")' in script
    assert "Submitted batch job" in script
    assert "is_multi_backend_result" in script
    assert "multi_backend_run_ok" in script
    assert "PASS (multi-backend;" in script
    assert "MULTI_BACKEND_STATUS" not in script
    # The run is kicked off after the async submissions, before the wait loop.
    assert script.index("\nrun_multi_backend_real\n") < script.index(
        "Waiting for jobs to complete"
    )


def test_sample_test_excuses_cuda_infra_failures_with_warning():
    """A job that fails only because the GPU driver/fabric was not ready on its
    node (a machine issue) must be EXCUSED from the pass/fail threshold, but
    loudly warned about -- never silently masked as a success."""
    script = Path("tests/e2e_tests/sample_test.sh").read_text()

    # Detection helper covering the GB200 'system not ready' + missing-runtime
    # signatures from both torch and cupy.
    assert "cuda_infra_failure()" in script
    assert "system not yet initialized" in script
    assert "cudaErrorSystemNotReady" in script
    assert "CUDA initialization: Unexpected error from cudaGetDeviceCount" in script
    assert "No CUDA runtime is found" in script

    # Would-be FAILs are reclassified as excused via the helper (both the
    # 'Successful requests: 0' and 'no success indicator' branches).
    assert "mark_cuda_excused" in script
    assert "CUDA-INFRA EXCUSED" in script
    assert "elif cuda_infra_failure " in script
    assert 'if cuda_infra_failure "$out_dir"; then' in script

    # The real pass count is preserved for the CI threshold parser, and the
    # excused count is emitted on its own parseable line + a loud warning.
    assert 'echo "$PASSED/$TOTAL jobs passed"' in script
    assert (
        '"$CUDA_INFRA/$TOTAL jobs excused due to CUDA/GPU infrastructure failures (not counted as failures)"'
        in script
    )
    assert "⚠ WARNING" in script


def test_sample_test_cuda_infra_detection_matches_real_symbols(tmp_path, fp):
    """Pin the detection against the exact symbols seen in CI: the cupy
    'cudaErrorSystemNotReady: system not yet initialized' infra failure is
    excused, while a genuine recipe error (pydantic validation) is not."""
    import subprocess

    fp.allow_unregistered(True)  # this test drives a real `bash` subprocess

    script = Path("tests/e2e_tests/sample_test.sh").read_text()
    start = script.index("cuda_infra_failure() {")
    end = script.index("\n}\n", start) + len("\n}\n")
    func = script[start:end]

    cuda_log = tmp_path / "cuda" / "prefill_server_0"
    cuda_log.mkdir(parents=True)
    (cuda_log / "prefill_server_0.log").write_text(
        "2026-06-16 06:32:06,371 - sflow.task.prefill_server_0 - INFO - 0: "
        "cupy_backends.cuda.api.runtime.CUDARuntimeError: cudaErrorSystemNotReady: "
        "system not yet initialized\n"
    )
    real_log = tmp_path / "real" / "prefill_server_0"
    real_log.mkdir(parents=True)
    (real_log / "prefill_server_0.log").write_text(
        "pydantic_core._pydantic_core.ValidationError: 1 validation error for "
        "VllmConfig\n  Value error, Number of experts in the model must be > 0\n"
    )

    harness = func + '\nif cuda_infra_failure "$1"; then echo MATCH; else echo NOMATCH; fi\n'
    cuda = subprocess.run(
        ["bash", "-c", harness, "_", str(tmp_path / "cuda")],
        capture_output=True,
        text=True,
    )
    real = subprocess.run(
        ["bash", "-c", harness, "_", str(tmp_path / "real")],
        capture_output=True,
        text=True,
    )

    assert cuda.stdout.strip() == "MATCH", cuda.stdout + cuda.stderr
    assert real.stdout.strip() == "NOMATCH", real.stdout + real.stderr
