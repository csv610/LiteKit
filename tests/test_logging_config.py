"""Tests for logging configuration module."""

import logging
import os
import tempfile

import pytest

from litekit.logging_config import configure_logging


class TestLoggingConfig:
    def test_configure_logging_defaults(self):
        root_logger = configure_logging()
        assert root_logger.level == logging.INFO

    def test_custom_level(self):
        root_logger = configure_logging(level=logging.DEBUG)
        assert root_logger.level == logging.DEBUG

    def test_verbosity_mapping(self):
        mappings = {
            0: logging.CRITICAL,
            1: logging.ERROR,
            2: logging.WARNING,
            3: logging.INFO,
            4: logging.DEBUG,
        }
        for verbosity, expected_level in mappings.items():
            root_logger = configure_logging(verbosity=verbosity)
            assert root_logger.level == expected_level, f"Failed for verbosity={verbosity}"

    def test_invalid_verbosity_defaults_to_warning(self):
        root_logger = configure_logging(verbosity=99)
        assert root_logger.level == logging.WARNING

    def test_console_handler_added_when_enabled(self):
        root_logger = configure_logging(enable_console=True)
        assert any(
            isinstance(h, logging.StreamHandler)
            for h in root_logger.handlers
        )
