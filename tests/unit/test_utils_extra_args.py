# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared backend extra-args de-dup/merge helpers."""

from sflow.utils.extra_args import (
    dedup_merge_extra_args,
    extra_arg_key,
    normalize_extra_args,
)


class TestNormalizeExtraArgs:
    def test_none_and_empty(self):
        assert normalize_extra_args(None) == []
        assert normalize_extra_args([]) == []
        assert normalize_extra_args(["", "   "]) == []

    def test_leading_whitespace_stripped_into_clean_token(self):
        # The multi-backend regression: a CLI value that carried a leading space.
        assert normalize_extra_args([" --exclude=n1,n2"]) == ["--exclude=n1,n2"]

    def test_bundled_flags_split_into_separate_tokens(self):
        assert normalize_extra_args(["--a --b=1"]) == ["--a", "--b=1"]

    def test_wellformed_tokens_unchanged(self):
        assert normalize_extra_args(["--gres=gpu:4", "--exclusive"]) == [
            "--gres=gpu:4",
            "--exclusive",
        ]

    def test_quoted_value_with_space_preserved_as_single_token(self):
        assert normalize_extra_args(['--comment="my job"']) == ["--comment=my job"]

    def test_non_string_entries_coerced(self):
        assert normalize_extra_args([4, "--time=10"]) == ["4", "--time=10"]

    def test_unbalanced_quotes_fall_back_to_stripped_original(self):
        # shlex can't split this; never worse than the old verbatim passthrough.
        assert normalize_extra_args(['  --foo="bar ']) == ['--foo="bar']


class TestExtraArgKey:
    def test_bare_flag_keys_on_itself(self):
        assert extra_arg_key("--exclusive") == "--exclusive"

    def test_single_valued_flag_keys_on_option_name(self):
        # A later value overrides the earlier one (same key).
        assert extra_arg_key("--network=host") == "--network"
        assert extra_arg_key("--network=bridge") == "--network"

    def test_value_without_equals_keys_on_option_name(self):
        assert extra_arg_key("--gres=gpu:8") == "--gres"
        assert extra_arg_key("--gres=gpu:4") == "--gres"

    def test_repeatable_key_value_flags_key_on_option_plus_key(self):
        # Distinct keys coexist; the same key overrides.
        assert extra_arg_key("--env=FOO=1") == "--env=FOO"
        assert extra_arg_key("--env=BAR=2") == "--env=BAR"
        assert extra_arg_key("--env=FOO=9") == "--env=FOO"

    def test_space_separated_flag_keys_on_first_token(self):
        assert extra_arg_key("--gpus-per-node 4") == "--gpus-per-node"

    def test_blank_arg_returned_as_is(self):
        assert extra_arg_key("") == ""
        assert extra_arg_key("   ") == "   "


class TestDedupMergeExtraArgs:
    def test_empty_inputs(self):
        assert dedup_merge_extra_args([], []) == []

    def test_base_only_and_override_only(self):
        assert dedup_merge_extra_args(["--a"], []) == ["--a"]
        assert dedup_merge_extra_args([], ["--b"]) == ["--b"]

    def test_override_wins_on_conflicting_option(self):
        # CLI (override) --gres=gpu:4 replaces recipe (base) --gres=gpu:8.
        assert dedup_merge_extra_args(["--gres=gpu:8"], ["--gres=gpu:4"]) == [
            "--gres=gpu:4"
        ]

    def test_base_option_keeps_position_value_overridden(self):
        merged = dedup_merge_extra_args(
            ["--exclusive", "--gres=gpu:8", "--time=01:00:00"],
            ["--gres=gpu:4", "--constraint=gpu"],
        )
        assert merged == [
            "--exclusive",
            "--gres=gpu:4",
            "--time=01:00:00",
            "--constraint=gpu",
        ]

    def test_repeatable_key_value_flags_preserved(self):
        merged = dedup_merge_extra_args(
            ["--env=FOO=1", "--env=BAR=2"],
            ["--env=FOO=9", "--env=BAZ=3"],
        )
        assert merged == ["--env=FOO=9", "--env=BAR=2", "--env=BAZ=3"]

    def test_later_entry_wins_within_a_single_list(self):
        assert dedup_merge_extra_args(["--gres=gpu:1", "--gres=gpu:2"], []) == [
            "--gres=gpu:2"
        ]

    def test_bare_flag_deduped(self):
        assert dedup_merge_extra_args(["--exclusive"], ["--exclusive"]) == [
            "--exclusive"
        ]
