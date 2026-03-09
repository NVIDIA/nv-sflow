# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from sflow.utils.parser import LinesParser, ParseLogHandler


def test_lines_parser_first_matching_pattern_wins_and_preserves_types():
    # "7" matches both patterns, but the first one should win (int via :d)
    parser = LinesParser(["Value: {x:d}", "Value: {x}"])
    parser.add_line("Value: 7")
    assert parser.parsed_dict() == {"x": 7}


def test_lines_parser_accumulates_multiple_values_and_flattens_singletons():
    parser = LinesParser(["A {a:d}", "B {b}"])
    parser.add_lines(["A 1", "A 2", "B hi"])
    assert parser.parsed_dict() == {"a": [1, 2], "b": "hi"}


def test_lines_parser_w_star_allows_empty_word_field():
    # partition uses {partition:w*} which should accept empty string between pipes
    pattern = (
        "{job_id}|{job_name}|{partition:w*}|{account}|{alloc_cpus:d}|{state}|"
        "{exit_code:d}:{exit_signal:d}"
    )
    parser = LinesParser([pattern])
    parser.add_line("123|job||acct|4|RUNNING|0:0")
    assert parser.parsed_dict()["partition"] == ""


def test_parsing_log_handler_collects_from_log_records():
    logger = logging.getLogger("sflow.tests.parser")
    logger.setLevel(logging.INFO)

    handler = ParseLogHandler(["X={x:d}", "Y={y}"])
    logger.addHandler(handler)
    try:
        logger.info("X=3")
        logger.info("Y=hello")
        assert handler.get_parsed_dict() == {"x": 3, "y": "hello"}
    finally:
        logger.removeHandler(handler)
