"""Tests for LiteMCQClient."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from litekit.lite_mcq_client import (
    LiteMCQClient,
    MCQInput,
    CorrectOption,
    MultipleChoiceAnswer,
    MultipleChoiceSolverResponse,
    _dict_to_mcq_input,
)


class TestFormatOptions:
    def test_dict_options(self):
        client = LiteMCQClient.__new__(LiteMCQClient)
        options = {"A": "Paris", "B": "London"}
        result = client._format_options(options)
        assert "A: Paris" in result
        assert "B: London" in result

    def test_list_options(self):
        client = LiteMCQClient.__new__(LiteMCQClient)
        options = ["Paris", "London", "Berlin"]
        result = client._format_options(options)
        assert "A: Paris" in result
        assert "B: London" in result
        assert "C: Berlin" in result

    def test_empty_list(self):
        client = LiteMCQClient.__new__(LiteMCQClient)
        result = client._format_options([])
        assert result == ""


class TestCreatePrompt:
    def test_basic(self):
        client = LiteMCQClient.__new__(LiteMCQClient)
        inp = MCQInput(question="What is 2+2?", options=["1", "2", "3", "4"])
        prompt = client._create_prompt(inp)
        assert "What is 2+2?" in prompt
        assert "A: 1" in prompt
        assert "B: 2" in prompt
        assert "C: 3" in prompt
        assert "D: 4" in prompt

    def test_with_context(self):
        client = LiteMCQClient.__new__(LiteMCQClient)
        inp = MCQInput(
            question="Q?", options=["A", "B"], context="Some background"
        )
        prompt = client._create_prompt(inp)
        assert "Some background" in prompt
        assert "Context" in prompt

    def test_with_images(self):
        client = LiteMCQClient.__new__(LiteMCQClient)
        inp = MCQInput(
            question="Q?", options=["A", "B"], image_paths=["img1.jpg", "img2.jpg"]
        )
        prompt = client._create_prompt(inp)
        assert "2 image(s) are provided" in prompt

    def test_with_context_and_images(self):
        client = LiteMCQClient.__new__(LiteMCQClient)
        inp = MCQInput(
            question="Q?",
            options=["A", "B"],
            context="Ctx",
            image_paths=["img.jpg"],
        )
        prompt = client._create_prompt(inp)
        assert "Ctx" in prompt
        assert "1 image(s)" in prompt


class TestLiteMCQClientInit:
    @patch("litekit.lite_mcq_client.LiteClient")
    def test_successful_init(self, mock_lite_client):
        client = LiteMCQClient(model="gpt-4", temperature=0.3)
        assert client.client is not None

    @patch("litekit.lite_mcq_client.ModelConfig")
    def test_init_failure_raises(self, mock_config):
        mock_config.side_effect = Exception("bad config")
        with pytest.raises(RuntimeError, match="Failed to initialize"):
            LiteMCQClient(model="bad-model")


class TestSolve:
    @patch("litekit.lite_mcq_client.LiteClient")
    def test_solve_returns_pydantic_directly(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        answer = MultipleChoiceAnswer(
            question="Q?",
            correct_options=[CorrectOption(key="B", value="4")],
            reasoning="Math",
            confidence=1.0,
        )
        mock_client.generate_text.return_value = MultipleChoiceSolverResponse(
            answer=answer
        )
        client = LiteMCQClient(model="gpt-4")
        inp = MCQInput(question="What is 2+2?", options=["1", "2", "3", "4"])
        result = client.solve(inp)
        assert result is not None
        assert result.correct_options[0].key == "B"
        assert result.confidence == 1.0

    @patch("litekit.lite_mcq_client.LiteClient")
    def test_solve_returns_json_string(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.generate_text.return_value = (
            '{"question": "Q?", "correct_options": [{"key": "A", "value": "Paris"}], '
            '"reasoning": "France capital", "confidence": 0.9}'
        )
        client = LiteMCQClient(model="gpt-4")
        inp = MCQInput(question="Capital of France?", options=["Paris", "London"])
        result = client.solve(inp)
        assert result is not None
        assert result.correct_options[0].key == "A"

    @patch("litekit.lite_mcq_client.LiteClient")
    def test_solve_with_override_config(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.generate_text.return_value = MultipleChoiceSolverResponse(
            answer=MultipleChoiceAnswer(
                question="Q?",
                correct_options=[CorrectOption(key="A", value="Yes")],
                reasoning="R",
                confidence=0.5,
            )
        )
        client = LiteMCQClient(model="gpt-4")
        from litekit.config import ModelConfig
        override = ModelConfig(model="gpt-3.5-turbo")
        inp = MCQInput(question="Test?", options=["Yes", "No"])
        result = client.solve(inp, model_config=override)
        assert result is not None

    @patch("litekit.lite_mcq_client.LiteClient")
    def test_solve_with_images(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        answer = MultipleChoiceAnswer(
            question="Q?",
            correct_options=[CorrectOption(key="A", value="Cat")],
            reasoning="Image shows cat",
            confidence=0.8,
        )
        mock_client.generate_text.return_value = MultipleChoiceSolverResponse(
            answer=answer
        )
        client = LiteMCQClient(model="gpt-4")
        inp = MCQInput(
            question="What animal?",
            options=["Cat", "Dog"],
            image_paths=["cat.jpg"],
        )
        result = client.solve(inp)
        assert result is not None
        args, kwargs = mock_client.generate_text.call_args
        assert kwargs["model_input"].image_paths == ["cat.jpg"]


class TestDictToMCQInput:
    def test_full_dict(self):
        data = {
            "question": "Q?",
            "options": ["A", "B"],
            "context": "ctx",
            "image_paths": ["img.jpg"],
        }
        inp = _dict_to_mcq_input(data)
        assert inp.question == "Q?"
        assert inp.options == ["A", "B"]
        assert inp.context == "ctx"
        assert inp.image_paths == ["img.jpg"]

    def test_minimal_dict(self):
        data = {"question": "Q?", "options": ["A"]}
        inp = _dict_to_mcq_input(data)
        assert inp.question == "Q?"
        assert inp.context is None
        assert inp.image_paths is None
