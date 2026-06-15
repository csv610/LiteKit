"""Tests for save_model_response utility."""

import json
import os
import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel

from litekit.utils.save_response import save_model_response


class SampleModel(BaseModel):
    name: str
    value: int


class TestSaveModelResponse:
    def test_save_pydantic_model_without_extension(self, tmp_path):
        path = tmp_path / "output"
        model = SampleModel(name="test", value=42)
        result = save_model_response(model, str(path))
        assert result == path.with_suffix(".json").resolve()
        assert result.exists()
        with open(result) as f:
            data = json.load(f)
        assert data == {"name": "test", "value": 42}

    def test_save_pydantic_model_with_extension(self, tmp_path):
        path = tmp_path / "output.json"
        model = SampleModel(name="test", value=42)
        result = save_model_response(model, str(path))
        assert result == path.resolve()
        assert result.exists()
        with open(result) as f:
            data = json.load(f)
        assert data == {"name": "test", "value": 42}

    def test_save_string_without_extension(self, tmp_path):
        path = tmp_path / "note"
        result = save_model_response("# Hello", str(path))
        assert result == path.with_suffix(".md").resolve()
        assert result.exists()
        with open(result) as f:
            assert f.read() == "# Hello"

    def test_save_string_with_extension(self, tmp_path):
        path = tmp_path / "note.md"
        result = save_model_response("# Hello", str(path))
        assert result == path.resolve()
        assert result.exists()
        with open(result) as f:
            assert f.read() == "# Hello"

    def test_save_string_with_wrong_extension(self, tmp_path):
        path = tmp_path / "note.txt"
        result = save_model_response("# Hello", str(path))
        assert result == path.resolve()
        assert result.exists()
        with open(result) as f:
            assert f.read() == "# Hello"

    def test_exclude_none_in_pydantic(self, tmp_path):
        class ModelWithNone(BaseModel):
            name: str
            optional: str | None = None

        path = tmp_path / "output.json"
        model = ModelWithNone(name="test")
        save_model_response(model, str(path))
        with open(path) as f:
            data = json.load(f)
        assert "optional" not in data

    def test_invalid_type_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported model type"):
            save_model_response(42, str(tmp_path / "out"))

    def test_creates_parent_directories(self, tmp_path):
        nested = tmp_path / "sub" / "dir" / "output.json"
        model = SampleModel(name="test", value=1)
        result = save_model_response(model, str(nested))
        assert result.exists()

    def test_pathlib_path_accepted(self, tmp_path):
        model = SampleModel(name="test", value=1)
        result = save_model_response(model, tmp_path / "model")
        assert isinstance(result, Path)
        assert result.exists()
