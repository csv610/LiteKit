"""Tests for print_response utility."""

from io import StringIO
import sys
from pydantic import BaseModel
from litekit.utils.print_response import print_response, print_simple_result


def capture_print(func, *args, **kwargs):
    out = StringIO()
    original = sys.stdout
    sys.stdout = out
    try:
        func(*args, **kwargs)
        return out.getvalue()
    finally:
        sys.stdout = original


class SampleModel(BaseModel):
    name: str
    count: int


class TestPrintResponse:
    def test_print_none(self):
        output = capture_print(print_response, None, title="Test")
        assert "No result to display" in output

    def test_print_string(self):
        output = capture_print(print_response, "hello", title="Test")
        assert "hello" in output

    def test_print_dict(self):
        output = capture_print(
            print_response, {"name": "Alice", "count": 42}, title="Result"
        )
        assert "NAME" in output
        assert "Alice" in output
        assert "COUNT" in output
        assert "42" in output

    def test_print_pydantic(self):
        model = SampleModel(name="Bob", count=10)
        output = capture_print(print_response, model)
        assert "NAME" in output
        assert "Bob" in output
        assert "COUNT" in output
        assert "10" in output

    def test_print_nested_dict(self):
        data = {"metadata": {"version": 1, "tags": ["a", "b"]}}
        output = capture_print(print_response, data)
        assert "Version" in output
        assert "1" in output
        assert "a" in output

    def test_custom_title(self):
        output = capture_print(print_response, "data", title="Custom")
        assert "CUSTOM" in output


class TestPrintSimpleResult:
    def test_print_none(self):
        output = capture_print(print_simple_result, None)
        assert "No result to display" in output

    def test_print_string(self):
        output = capture_print(print_simple_result, "hello", title="Test")
        assert "hello" in output

    def test_print_flat_dict(self):
        output = capture_print(
            print_simple_result, {"x": 1, "y": 2}, title="Coords"
        )
        assert "x: 1" in output
        assert "y: 2" in output
