# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from sflow.utils.logging import temporary_handler


class _CollectingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class TestTemporaryHandler:
    def test_adds_and_removes_handler(self):
        logger = logging.getLogger("sflow.tests.temp_handler.basic")
        logger.handlers = []
        logger.propagate = False
        logger.setLevel(logging.DEBUG)

        handler = _CollectingHandler()

        assert handler not in logger.handlers
        with temporary_handler(logger, handler):
            assert handler in logger.handlers
            logger.info("hello")
        assert handler not in logger.handlers

        # The handler saw the log while attached.
        assert handler.messages == ["hello"]

    def test_removes_handler_on_exception(self):
        logger = logging.getLogger("sflow.tests.temp_handler.exception")
        logger.handlers = []
        logger.propagate = False

        handler = _CollectingHandler()

        with pytest.raises(RuntimeError):
            with temporary_handler(logger, handler):
                assert handler in logger.handlers
                raise RuntimeError("boom")

        assert handler not in logger.handlers

    def test_does_not_remove_if_already_attached(self):
        logger = logging.getLogger("sflow.tests.temp_handler.already")
        logger.handlers = []
        logger.propagate = False

        handler = _CollectingHandler()
        logger.addHandler(handler)

        with temporary_handler(logger, handler):
            assert handler in logger.handlers

        # Should still be attached, since it was there before the context.
        assert handler in logger.handlers
