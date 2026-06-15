"""Tests for the config module (dataclasses and validation)."""

import pytest
from pydantic import BaseModel
from litekit.config import (
    ModelConfig,
    ModelInput,
    MCQInput,
    UserInput,
    ChatConfig,
    ModelOutput,
    DEFAULT_PROMPT,
)


class TestModelConfig:
    def test_valid_config(self):
        config = ModelConfig(model="gpt-4", temperature=0.5)
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.timeout is None

    def test_empty_model_name(self):
        with pytest.raises(ValueError, match="model name cannot be empty"):
            ModelConfig(model="")

    def test_whitespace_model_name(self):
        with pytest.raises(ValueError, match="model name cannot be empty"):
            ModelConfig(model="   ")

    def test_temperature_too_low(self):
        with pytest.raises(ValueError, match="temperature must be between"):
            ModelConfig(model="gpt-4", temperature=-0.1)

    def test_temperature_too_high(self):
        with pytest.raises(ValueError, match="temperature must be between"):
            ModelConfig(model="gpt-4", temperature=2.1)

    def test_temperature_boundary_valid(self):
        c1 = ModelConfig(model="gpt-4", temperature=0.0)
        c2 = ModelConfig(model="gpt-4", temperature=2.0)
        assert c1.temperature == 0.0
        assert c2.temperature == 2.0

    def test_negative_timeout(self):
        with pytest.raises(ValueError, match="timeout must be positive"):
            ModelConfig(model="gpt-4", timeout=-1)

    def test_zero_timeout(self):
        with pytest.raises(ValueError, match="timeout must be positive"):
            ModelConfig(model="gpt-4", timeout=0)

    def test_positive_timeout(self):
        config = ModelConfig(model="gpt-4", timeout=30)
        assert config.timeout == 30

    def test_default_temperature(self):
        config = ModelConfig(model="gpt-4")
        assert config.temperature == 0.2


class TestModelInput:
    def test_basic_input(self):
        inp = ModelInput(user_prompt="Hello")
        assert inp.user_prompt == "Hello"
        assert inp.image_path is None
        assert inp.image_paths is None
        assert inp.system_prompt is None
        assert inp.response_format is None

    def test_empty_prompt_without_image_raises(self):
        with pytest.raises(ValueError, match="user_prompt cannot be empty"):
            ModelInput(user_prompt="")

    def test_whitespace_prompt_without_image_raises(self):
        with pytest.raises(ValueError, match="user_prompt cannot be empty"):
            ModelInput(user_prompt="   ")

    def test_empty_prompt_with_image_gets_default(self):
        inp = ModelInput(user_prompt="", image_path="/path/to/img.jpg")
        assert inp.user_prompt == DEFAULT_PROMPT

    def test_empty_prompt_with_image_paths_gets_default(self):
        inp = ModelInput(
            user_prompt="", image_paths=["/path/to/img.jpg"]
        )
        assert inp.user_prompt == DEFAULT_PROMPT

    def test_system_prompt_normalized_to_none(self):
        inp = ModelInput(
            user_prompt="Hello", system_prompt=""
        )
        assert inp.system_prompt is None
        inp2 = ModelInput(
            user_prompt="Hello", system_prompt="   "
        )
        assert inp2.system_prompt is None
        inp3 = ModelInput(
            user_prompt="Hello", system_prompt="Be helpful"
        )
        assert inp3.system_prompt == "Be helpful"

    def test_empty_response_format_string_normalized(self):
        inp = ModelInput(user_prompt="Hello", response_format="")
        assert inp.response_format is None

    def test_response_format_pydantic_class(self):
        class MyModel(BaseModel):
            name: str

        inp = ModelInput(
            user_prompt="Hello", response_format=MyModel
        )
        assert inp.response_format is MyModel

    def test_default_prompt_constant(self):
        assert DEFAULT_PROMPT == "Describe this image in detail"


class TestMCQInput:
    def test_with_list_options(self):
        inp = MCQInput(
            question="What is 2+2?",
            options=["1", "2", "3", "4"],
        )
        assert inp.question == "What is 2+2?"
        assert inp.options == ["1", "2", "3", "4"]
        assert inp.context is None
        assert inp.image_paths is None

    def test_with_dict_options(self):
        inp = MCQInput(
            question="What is 2+2?",
            options={"A": "1", "B": "2", "C": "3", "D": "4"},
        )
        assert inp.options["A"] == "1"

    def test_with_context_and_images(self):
        inp = MCQInput(
            question="Q?",
            options=["A", "B"],
            context="Some context",
            image_paths=["img1.jpg"],
        )
        assert inp.context == "Some context"
        assert inp.image_paths == ["img1.jpg"]


class TestUserInput:
    def test_basic(self):
        inp = UserInput(model_response="42")
        assert inp.model_response == "42"
        assert inp.user_prompt is None
        assert inp.ground_truth is None
        assert inp.context is None

    def test_full(self):
        inp = UserInput(
            model_response="42",
            user_prompt="What is 2+2?",
            ground_truth="4",
            context="Math",
        )
        assert inp.model_response == "42"
        assert inp.user_prompt == "What is 2+2?"
        assert inp.ground_truth == "4"
        assert inp.context == "Math"


class TestChatConfig:
    def test_defaults(self):
        cfg = ChatConfig()
        assert cfg.max_history == 10
        assert cfg.auto_save is False
        assert cfg.save_dir == "."

    def test_custom(self):
        cfg = ChatConfig(max_history=20, auto_save=True, save_dir="/tmp")
        assert cfg.max_history == 20
        assert cfg.auto_save is True
        assert cfg.save_dir == "/tmp"


class TestModelOutput:
    def test_defaults(self):
        out = ModelOutput()
        assert out.data is None
        assert out.markdown is None
        assert out.metadata == {}

    def test_with_data(self):
        out = ModelOutput(data={"key": "value"}, markdown="# Hello")
        assert out.data == {"key": "value"}
        assert out.markdown == "# Hello"

    def test_serialization(self):
        out = ModelOutput(data=[1, 2, 3])
        dumped = out.model_dump()
        assert dumped["data"] == [1, 2, 3]
        assert dumped["metadata"] == {}
