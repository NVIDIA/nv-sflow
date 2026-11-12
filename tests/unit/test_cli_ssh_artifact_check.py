from unittest.mock import Mock
import json
import subprocess

import pytest
import typer

from sflow.ssh import delegate as ssh_delegate
from sflow.ssh.invocation import argv_to_invocation


@pytest.fixture
def remote_artifact_recipe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recipe = tmp_path / "flow.yaml"
    recipe.write_text(
        "version: '0.1'\n"
        "artifacts:\n  - name: MODEL\n    uri: fs:///remote/missing-model\n"
        "workflow:\n  name: artifact-check\n  tasks:\n"
        "    - name: hello\n      script: [echo hello]\n"
    )
    monkeypatch.setattr(
        ssh_delegate,
        "_resolve_remote_environment",
        lambda *a, **k: ("/remote/cache", "test-platform"),
    )
    # Stop after preflight without uploading or executing anything remotely.
    monkeypatch.setattr(
        ssh_delegate, "_copy_input", Mock(side_effect=RuntimeError("preflight passed"))
    )
    return recipe


@pytest.mark.parametrize("command", ["run", "batch"])
def test_ssh_skip_artifact_check_bypasses_remote_preflight(
    remote_artifact_recipe, monkeypatch, command
):
    invocation = argv_to_invocation(
        [
            command,
            "-f",
            str(remote_artifact_recipe),
            "--ssh",
            "login",
            "--skip-artifact-check",
        ]
    )
    check = Mock(side_effect=AssertionError("remote artifact check must be skipped"))
    monkeypatch.setattr(ssh_delegate, "_check_remote_artifacts", check)

    with pytest.raises(RuntimeError, match="preflight passed"):
        ssh_delegate.delegate(invocation, transport=Mock())

    check.assert_not_called()
    assert "--skip-artifact-check" in invocation.raw_args


def test_ssh_missing_artifact_still_fails_with_override_hint(
    remote_artifact_recipe, capsys
):
    invocation = argv_to_invocation(
        [
            "run",
            "-f",
            str(remote_artifact_recipe),
            "--ssh",
            "login",
        ]
    )
    transport = Mock()
    transport.capture.return_value = subprocess.CompletedProcess(
        [], 0, stdout=json.dumps(["/remote/missing-model"]), stderr=""
    )

    with pytest.raises(typer.Exit) as exc:
        ssh_delegate.delegate(invocation, transport=transport)

    assert exc.value.exit_code == 1
    error = capsys.readouterr().err
    assert "/remote/missing-model" in error
    assert "--artifact NAME=URI" in error
    assert "--skip-artifact-check" in error


def test_ssh_artifact_override_is_checked_instead_of_original(remote_artifact_recipe):
    invocation = argv_to_invocation(
        [
            "run",
            "-f",
            str(remote_artifact_recipe),
            "--ssh",
            "login",
            "--artifact",
            "MODEL=fs:///remote/present-model",
        ]
    )
    transport = Mock()
    transport.capture.return_value = subprocess.CompletedProcess(
        [], 0, stdout="[]", stderr=""
    )

    with pytest.raises(RuntimeError, match="preflight passed"):
        ssh_delegate.delegate(invocation, transport=transport)

    command = transport.capture.call_args.args[1]
    assert "/remote/present-model" in command
    assert "/remote/missing-model" not in command
