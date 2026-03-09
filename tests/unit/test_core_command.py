# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shlex

import pytest

from sflow.core.command import Command, format_command


def test_command_requires_exec():
    with pytest.raises(ValueError, match="Executable is required"):
        Command(exec="")


def test_as_list_includes_exec_opts_and_args_in_order():
    cmd = Command(exec="srun", args=["echo", "hello"])
    cmd.add_opt("--account", "test_account")
    cmd.add_opt("--nodelist", "eos0001")
    cmd.add_opt("--jobid", "1111111")

    assert cmd.as_list() == [
        "srun",
        "--account",
        "test_account",
        "--nodelist",
        "eos0001",
        "--jobid",
        "1111111",
        "echo",
        "hello",
    ]
    assert list(cmd) == cmd.as_list()


def test_add_opt_flag_adds_option_without_value():
    cmd = Command(exec="srun", args=["echo", "hello"])
    cmd.add_opt("--verbose")
    assert cmd.as_list() == ["srun", "--verbose", "echo", "hello"]


def test_add_opt_overwrites_by_default():
    cmd = Command(exec="prog")
    cmd.add_opt("--x", 1)
    cmd.add_opt("--x", 2)  # overwrite
    assert cmd.as_list() == ["prog", "--x", "2"]


def test_add_opt_append_keeps_duplicates():
    cmd = Command(exec="prog")
    cmd.add_opt("--x", 1, append=True)
    cmd.add_opt("--x", 2, append=True)
    assert cmd.as_list() == ["prog", "--x", "1", "--x", "2"]


def test_add_arg_appends_positional_arguments():
    cmd = Command(exec="prog")
    cmd.add_arg("a").add_arg(2)
    assert cmd.as_list() == ["prog", "a", "2"]


def test_as_str_shell_quotes_like_shlex_join():
    cmd = Command(exec="echo", args=["hello, world"])
    assert cmd.as_str() == shlex.join(["echo", "hello, world"])


def test_str_dunder_matches_as_str():
    cmd = Command(exec="echo", args=["hello"])
    assert str(cmd) == cmd.as_str()


def test_format_command_for_command_str_and_list():
    cmd = Command(exec="srun", args=["echo", "hello, world"])
    cmd.add_opt("--account", "test_account")
    cmd.add_opt("--nodelist", "eos0001")
    cmd.add_opt("--jobid", "1111111")

    expected = "srun --account test_account --nodelist eos0001 --jobid 1111111 echo 'hello, world'"
    assert format_command(cmd) == expected
    assert format_command(cmd.as_str()) == cmd.as_str()
    assert format_command(cmd.as_list()) == expected
