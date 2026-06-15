# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from parse import compile as parse_compile, with_pattern

_logger = logging.getLogger(__name__)


@with_pattern(r"\w*")
def parse_nullable_w(text):
    return text


class LinesParser:
    """A parser that extracts structured data from lines of text using parse patterns.

    This class allows you to define multiple parse patterns and apply them to lines of text
    to extract named values. It accumulates results across multiple lines and provides
    a convenient way to retrieve the parsed data.
    """

    def __init__(self, parse_patterns: list[str]):
        """Initialize the LinesParser with a list of parse patterns.

        Args:
            parse_patterns: A list of parse pattern strings that define how to extract
                named values from text lines. Patterns use the 'parse' library
                syntax (e.g., "Value: {value:d}" to extract an integer named 'value').
        """
        self._parse_patterns = parse_patterns
        # Compile each pattern once. ``parse.parse()`` rebuilds the regex on every
        # call, which is wasteful when lines are fed one-by-one from a live task
        # stream; precompiling keeps the per-line cost to a single regex match.
        self._compiled = []
        for pattern in parse_patterns:
            try:
                self._compiled.append(
                    parse_compile(
                        pattern,
                        extra_types={"w*": parse_nullable_w},
                        case_sensitive=True,
                    )
                )
            except Exception as e:
                # Best-effort: skip an unparseable pattern rather than crash the
                # workflow (task output parsing has always been best-effort).
                _logger.warning(f"Ignoring invalid parse pattern {pattern!r}: {e}")
        self._parsed_dict = {}

    def add_line(self, line: str):
        """Parse a single line of text using the configured patterns.

        Attempts to match the line against each parse pattern in order. When a pattern
        matches, the named values are extracted and stored, then processing stops for
        that line.

        Args:
            line: The text line to parse.
        """
        for parser in self._compiled:
            result = parser.parse(line)
            if result:
                for key, value in result.named.items():
                    if key not in self._parsed_dict:
                        self._parsed_dict[key] = []
                    self._parsed_dict[key].append(value)
                break

    def add_lines(self, lines: list[str]):
        """Parse a list of lines of text using the configured patterns.

        Args:
            lines: The list of text lines to parse.
        """
        for line in lines:
            self.add_line(line)

    def parsed_dict(self):
        # If only one value was parsed for each key, return the value directly
        # Otherwise, return the list of values
        result = {}
        for key, values in self._parsed_dict.items():
            if len(values) == 1:
                result[key] = values[0]
            else:
                result[key] = values
        return result


class ParseLogHandler(logging.Handler):
    def __init__(self, patterns: list[str]):
        super().__init__()
        self._parse_patterns = patterns
        self._parser = LinesParser(patterns)

    def emit(self, record: logging.LogRecord):
        self._parser.add_line(record.getMessage())

    def get_parsed_dict(self):
        return self._parser.parsed_dict()
