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

    assert "infmax/batch_test.sh" not in script
    assert "INFMAX_SUBMIT_PID" not in submit_block
    assert "bash ./infmax/batch_test.sh" not in submit_block
    assert "./sample_test.sh" in submit_block


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


def test_sample_test_supports_infmax_only_type():
    script = Path("tests/e2e_tests/sample_test.sh").read_text()
    part2 = script.split("Part 2: Modular examples", 1)[1].split(
        "Part 3: Infmax multi-node batch suites", 1
    )[0]
    part3 = script.split("Part 3: Infmax multi-node batch suites", 1)[1]

    assert "[-t s|m|inf|a|smoke]" in script
    assert '&& [ "$TEST_TYPE" != "inf" ]' in script
    assert '&& [ "$TEST_TYPE" != "smoke" ]' in script
    assert 'if [ "$TEST_TYPE" = "inf" ]' not in part2
    assert 'if [ "$TEST_TYPE" = "inf" ] || [ "$TEST_TYPE" = "m" ] || [ "$TEST_TYPE" = "a" ] || [ "$TEST_TYPE" = "smoke" ]; then' in part3


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
    assert sample_test.count('"${INFMAX_SUITE_OVERRIDES[@]}"') == 2


def test_sample_scripts_force_under_dev_sflow_checkout():
    helper = Path("scripts/use_under_dev_sflow.sh").read_text()
    full_sample = Path("scripts/full_sample_tests.sh").read_text()
    sample_test = Path("tests/e2e_tests/sample_test.sh").read_text()

    assert "assert_under_dev_sflow_editable_install" in helper
    assert "source  : local editable dev" in helper
    assert "exec \"$SFLOW_TEST_PYTHON\" -m sflow \"$@\"" in helper
    assert "--sflow-version" in helper
    assert "setup_under_dev_sflow \"$REPO_DIR\"" in full_sample
    assert "cleanup_under_dev_sflow" in full_sample
    assert "setup_under_dev_sflow \"$REPO_DIR\"" in sample_test
    assert "cleanup_under_dev_sflow" in sample_test


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
