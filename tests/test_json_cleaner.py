"""Tests for JSONCleaner utility."""

import pytest
from litekit.utils.json_cleaner import JSONCleaner


class TestJSONCleaner:
    def test_plain_json(self):
        text = '{"name": "Alice", "age": 30}'
        result = JSONCleaner.extract_json(text)
        assert result == '{"name": "Alice", "age": 30}'

    def test_markdown_fenced_json(self):
        text = "```json\n{\"name\": \"Alice\"}\n```"
        result = JSONCleaner.extract_json(text)
        assert result == '{"name": "Alice"}'

    def test_markdown_fenced_no_lang(self):
        text = "```\n{\"name\": \"Alice\"}\n```"
        result = JSONCleaner.extract_json(text)
        assert result == '{"name": "Alice"}'

    def text_with_surrounding_text(self):
        text = 'Here is the result: {"key": "value"}. Enjoy!'
        result = JSONCleaner.extract_json(text)
        assert result == '{"key": "value"}'

    def test_nested_single_key_object(self):
        text = '{"data": {"inner": "value"}}'
        result = JSONCleaner.extract_json(text)
        assert result == '{"inner": "value"}'

    def test_multiple_keys_not_unwrapped(self):
        text = '{"a": 1, "b": 2}'
        result = JSONCleaner.extract_json(text)
        assert result == '{"a": 1, "b": 2}'

    def test_invalid_json_still_returns_braces(self):
        text = '{"key": invalid}'
        result = JSONCleaner.extract_json(text)
        assert result == '{"key": invalid}'

    def test_empty_string(self):
        assert JSONCleaner.extract_json("") == ""

    def test_no_braces(self):
        assert JSONCleaner.extract_json("hello world") == "hello world"

    def test_non_string_input(self):
        assert JSONCleaner.extract_json(42) == 42

    def test_nested_non_dict_value_not_unwrapped(self):
        text = '{"data": "just a string"}'
        result = JSONCleaner.extract_json(text)
        assert result == '{"data": "just a string"}'

    def test_markdown_with_triple_backticks_and_extra(self):
        text = "Some text\n```json\n{\"valid\": true}\n```\nmore text"
        result = JSONCleaner.extract_json(text)
        assert result == '{"valid": true}'

    def test_none_value(self):
        assert JSONCleaner.extract_json(None) is None
