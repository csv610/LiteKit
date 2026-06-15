"""Tests for LiteResponseJudge."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from litekit.lite_response_judge import (
    ResponseJudge,
    PromptBuilder,
    UserInput,
    ModelConfig,
    EvaluationModel,
    CriteriaScores,
)


class TestPromptBuilder:
    def test_system_prompt(self):
        prompt = PromptBuilder.create_system_prompt()
        assert isinstance(prompt, str)
        assert "EVALUATION PRINCIPLES" in prompt
        assert "Accuracy" in prompt

    def test_build_user_prompt_with_all_fields(self):
        inp = UserInput(
            model_response="42",
            user_prompt="What is 2+2?",
            ground_truth="4",
            context="Math",
        )
        result = PromptBuilder.build_user_prompt(inp)
        assert "### User Prompt:" in result
        assert "What is 2+2?" in result
        assert "### Model Response:" in result
        assert "42" in result
        assert "### Ground Truth:" in result
        assert "4" in result
        assert "### Context:" in result
        assert "Math" in result

    def test_build_user_prompt_minimal(self):
        inp = UserInput(model_response="Some response")
        result = PromptBuilder.build_user_prompt(inp)
        assert "### User Prompt:" not in result
        assert "### Model Response:" in result
        assert "Some response" in result
        assert "### Ground Truth:" not in result
        assert "### Context:" not in result

    def test_build_user_prompt_without_user_prompt(self):
        inp = UserInput(
            model_response="Answer",
            ground_truth="Correct answer",
        )
        result = PromptBuilder.build_user_prompt(inp)
        assert "### User Prompt:" not in result
        assert "### Ground Truth:" in result


class TestResponseJudgeInit:
    def test_valid_init(self):
        config = ModelConfig(model="gpt-4")
        judge = ResponseJudge(model_config=config)
        assert judge.model_config is config
        assert judge.client is not None

    def test_invalid_config_type(self):
        with pytest.raises(TypeError, match="ResponseJudge requires a ModelConfig"):
            ResponseJudge(model_config="not-a-config")  # type: ignore

    def test_none_config_raises(self):
        with pytest.raises(TypeError, match="ResponseJudge requires a ModelConfig"):
            ResponseJudge(model_config=None)  # type: ignore


class TestResponseJudgeEvaluate:
    @patch("litekit.lite_response_judge.LiteClient")
    def test_evaluate_basic(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.generate_text.return_value = EvaluationModel(
            criteria=CriteriaScores(
                accuracy=0.9,
                completeness=0.8,
                relevance=0.95,
                clarity=0.85,
            ),
            overall_score=0.875,
            is_correct=True,
            reasoning="Good response",
            feedback="Could be more detailed",
        )
        config = ModelConfig(model="gpt-4")
        judge = ResponseJudge(model_config=config)
        inp = UserInput(
            model_response="The capital of France is Paris.",
            user_prompt="What is the capital of France?",
        )
        result = judge.evaluate(inp)
        assert isinstance(result, EvaluationModel)
        assert result.overall_score == 0.875
        assert result.is_correct is True
        assert result.criteria.accuracy == 0.9

    @patch("litekit.lite_response_judge.LiteClient")
    def test_evaluate_invalid_input_type(self, mock_client_class):
        config = ModelConfig(model="gpt-4")
        judge = ResponseJudge(model_config=config)
        with pytest.raises(TypeError, match="evaluate"):
            judge.evaluate("not-a-user-input")  # type: ignore

    @patch("litekit.lite_response_judge.LiteClient")
    def test_evaluate_empty_response_raises(self, mock_client_class):
        config = ModelConfig(model="gpt-4")
        judge = ResponseJudge(model_config=config)
        with pytest.raises(ValueError, match="model_response must not be empty"):
            judge.evaluate(UserInput(model_response=""))

    @patch("litekit.lite_response_judge.LiteClient")
    def test_evaluate_whitespace_response_raises(self, mock_client_class):
        config = ModelConfig(model="gpt-4")
        judge = ResponseJudge(model_config=config)
        with pytest.raises(ValueError, match="model_response must not be empty"):
            judge.evaluate(UserInput(model_response="   "))
