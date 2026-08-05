# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared install-spec helpers used by both ``sflow batch`` and ``sflow upgrade``."""

from sflow.utils.install import (
    DEFAULT_SFLOW_GIT_BRANCH,
    DEFAULT_SFLOW_GIT_URL,
    sflow_git_install_url,
    sflow_git_spec,
    sflow_index_url_error,
    sflow_pypi_requirement,
    sflow_version_error,
)


class TestGitInstallUrl:
    def test_bare_ref_uses_the_public_oss_repo(self):
        assert (
            sflow_git_install_url("main")
            == f"git+{DEFAULT_SFLOW_GIT_URL}@{DEFAULT_SFLOW_GIT_BRANCH}"
        )
        assert sflow_git_install_url("v0.1.0") == f"git+{DEFAULT_SFLOW_GIT_URL}@v0.1.0"

    def test_none_defaults_to_main(self):
        assert sflow_git_install_url(None) == f"git+{DEFAULT_SFLOW_GIT_URL}@main"

    def test_repo_url_with_ref_is_prefixed_once(self):
        assert (
            sflow_git_install_url("https://git.example.com/x/sflow.git@develop")
            == "git+https://git.example.com/x/sflow.git@develop"
        )

    def test_already_git_prefixed_url_is_left_alone(self):
        url = "git+https://git.example.com/x/sflow.git@develop"
        assert sflow_git_install_url(url) == url


class TestGitSpec:
    def test_defaults_to_main_on_the_public_repo(self):
        assert sflow_git_spec(None, None) == "main"

    def test_branch_only_stays_a_bare_ref(self):
        # A bare ref means "default repo", which sflow_git_install_url expands.
        assert sflow_git_spec(None, "develop") == "develop"

    def test_custom_repo_produces_repo_at_ref(self):
        assert (
            sflow_git_spec("https://git.example.com/x/sflow.git", None)
            == "https://git.example.com/x/sflow.git@main"
        )
        assert (
            sflow_git_spec("https://git.example.com/x/sflow.git", "topic")
            == "https://git.example.com/x/sflow.git@topic"
        )

    def test_explicitly_passing_the_default_repo_is_still_a_bare_ref(self):
        assert sflow_git_spec(DEFAULT_SFLOW_GIT_URL, "v2") == "v2"

    def test_blank_values_fall_back_to_defaults(self):
        assert sflow_git_spec("  ", "  ") == "main"


class TestPypiRequirement:
    def test_empty_installs_latest(self):
        assert sflow_pypi_requirement(None) == "sflow"
        assert sflow_pypi_requirement("   ") == "sflow"

    def test_bare_version_is_pinned(self):
        assert sflow_pypi_requirement("0.2.1") == "sflow==0.2.1"

    def test_operator_spec_is_passed_through(self):
        assert sflow_pypi_requirement(">=0.2,<0.3") == "sflow>=0.2,<0.3"
        assert sflow_pypi_requirement("~=0.2.0") == "sflow~=0.2.0"


class TestIndexUrlError:
    def test_plain_url_is_accepted(self):
        assert sflow_index_url_error("https://host/artifactory/api/pypi/r/simple") is None

    def test_none_is_accepted(self):
        assert sflow_index_url_error(None) is None

    def test_embedded_credentials_are_rejected(self):
        msg = sflow_index_url_error("https://user:pw@host/simple")
        assert msg and "embedded credentials" in msg

    def test_query_and_fragment_are_rejected(self):
        assert "query parameters" in (sflow_index_url_error("https://h/s?token=x") or "")
        assert "query parameters" in (sflow_index_url_error("https://h/s#frag") or "")

    def test_hint_is_customizable(self):
        msg = sflow_index_url_error("https://u:p@h/s", hint="use a keyring.")
        assert msg and msg.endswith("use a keyring.")


class TestVersionError:
    def test_empty_is_always_accepted(self):
        assert sflow_version_error(None, registry=False) is None
        assert sflow_version_error("", registry=True) is None

    def test_git_route_accepts_refs_and_urls(self):
        assert sflow_version_error("main", registry=False) is None
        assert (
            sflow_version_error("https://h/x.git@develop", registry=False) is None
        )

    def test_git_route_rejects_whitespace(self):
        msg = sflow_version_error(">=0.2, <0.3", registry=False)
        assert msg and "not a valid git ref" in msg

    def test_pypi_route_accepts_versions_and_specifiers(self):
        assert sflow_version_error("0.2.1", registry=True) is None
        assert sflow_version_error(">=0.2,<0.3", registry=True) is None

    def test_pypi_route_rejects_urls_and_direct_references(self):
        assert sflow_version_error("https://h/x.git", registry=True) is not None
        assert sflow_version_error("sflow @ git+https://h/x.git", registry=True) is not None

    def test_option_name_appears_in_the_message(self):
        msg = sflow_version_error("a b", registry=False, option="--branch")
        assert msg and msg.startswith("--branch ")
