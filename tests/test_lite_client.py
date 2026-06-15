"""Tests for LiteClient (with mocked litellm)."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from litekit.lite_client import LiteClient, _is_pydantic_model
from litekit.config import ModelConfig, ModelInput


class TestIsPydanticModel:
    def test_is_pydantic_class(self):
        class MyModel(BaseModel):
            pass

        assert _is_pydantic_model(MyModel) is True

    def test_is_not_pydantic_class(self):
        assert _is_pydantic_model(str) is False
        assert _is_pydantic_model(dict) is False
        assert _is_pydantic_model(None) is False
        assert _is_pydantic_model(42) is False

    def test_instance_is_not_pydantic_class(self):
        class MyModel(BaseModel):
            pass

        assert _is_pydantic_model(MyModel()) is False


class MockMessage:
    def __init__(self, content=None, parsed=None):
        self.content = content
        self.parsed = parsed


class MockChoice:
    def __init__(self, message):
        self.message = message


class MockResponse:
    def __init__(self, choices):
        self.choices = choices


class TestCreateMessage:
    def test_basic_text_message(self):
        inp = ModelInput(user_prompt="Hello")
        messages = LiteClient.create_message(inp)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"][0]["type"] == "text"
        assert messages[0]["content"][0]["text"] == "Hello"

    def test_with_system_prompt(self):
        inp = ModelInput(user_prompt="Hi", system_prompt="Be concise.")
        messages = LiteClient.create_message(inp)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be concise."
        assert messages[1]["role"] == "user"

    def test_with_pydantic_response_format_adds_schema(self):
        class MyModel(BaseModel):
            name: str

        inp = ModelInput(
            user_prompt="Extract name",
            response_format=MyModel,
        )
        messages = LiteClient.create_message(inp)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "JSON" in messages[0]["content"]
        assert "name" in messages[0]["content"]

    def test_with_image_path(self):
        with patch("litekit.lite_client.ImageUtils.encode_to_base64") as mock_encode:
            mock_encode.return_value = "data:image/jpeg;base64,abc123"
            inp = ModelInput(
                user_prompt="Describe this",
                image_path="/path/img.jpg",
            )
            messages = LiteClient.create_message(inp)
            assert len(messages) == 1
            content = messages[0]["content"]
            assert len(content) == 2
            assert content[0]["type"] == "text"
            assert content[1]["type"] == "image_url"
            assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,abc123"

    def test_with_multiple_image_paths(self):
        with patch("litekit.lite_client.ImageUtils.encode_to_base64") as mock_encode:
            mock_encode.return_value = "data:image/jpeg;base64,img"
            inp = ModelInput(
                user_prompt="Compare",
                image_paths=["img1.jpg", "img2.jpg"],
            )
            messages = LiteClient.create_message(inp)
            content = messages[0]["content"]
            assert len(content) == 3

    def test_duplicate_image_paths_deduped(self):
        with patch("litekit.lite_client.ImageUtils.encode_to_base64") as mock_encode:
            mock_encode.return_value = "data:image/jpeg;base64,img"
            inp = ModelInput(
                user_prompt="Dupes",
                image_path="/same.jpg",
                image_paths=["/same.jpg", "/other.jpg"],
            )
            messages = LiteClient.create_message(inp)
            image_urls = [
                c["image_url"]["url"]
                for c in messages[0]["content"]
                if c.get("type") == "image_url"
            ]
            assert len(image_urls) == 2


class TestGenerateText:
    def test_no_config_raises(self):
        client = LiteClient()
        with pytest.raises(ValueError, match="ModelConfig must be provided"):
            client.generate_text(ModelInput(user_prompt="Hi"))

    @patch("litekit.lite_client.completion")
    def test_basic_text_response(self, mock_completion):
        mock_completion.return_value = MockResponse(
            choices=[MockChoice(message=MockMessage(content="Hello back!"))]
        )
        config = ModelConfig(model="gpt-4", temperature=0.5)
        client = LiteClient(model_config=config)
        result = client.generate_text(ModelInput(user_prompt="Hi"))
        assert result == "Hello back!"

    @patch("litekit.lite_client.completion")
    def test_retry_on_failure(self, mock_completion):
        mock_completion.side_effect = [
            Exception("API error"),
            MockResponse(
                choices=[MockChoice(message=MockMessage(content="Success on retry"))]
            ),
        ]
        config = ModelConfig(model="gpt-4")
        client = LiteClient(model_config=config)
        result = client.generate_text(ModelInput(user_prompt="Hi"), retries=2)
        assert result == "Success on retry"

    @patch("litekit.lite_client.completion")
    def test_all_retries_exhausted(self, mock_completion):
        mock_completion.side_effect = Exception("Persistent error")
        config = ModelConfig(model="gpt-4")
        client = LiteClient(model_config=config)
        with pytest.raises(Exception, match="Persistent error"):
            client.generate_text(ModelInput(user_prompt="Hi"), retries=1)

    @patch("litekit.lite_client.completion")
    def test_pydantic_response_via_litellm_parsed(self, mock_completion):
        class MyModel(BaseModel):
            name: str
            age: int

        mock_completion.return_value = MockResponse(
            choices=[MockChoice(message=MockMessage(parsed=MyModel(name="Alice", age=30)))]
        )
        config = ModelConfig(model="gpt-4")
        client = LiteClient(model_config=config)
        result = client.generate_text(
            ModelInput(user_prompt="Extract", response_format=MyModel)
        )
        assert isinstance(result, MyModel)
        assert result.name == "Alice"
        assert result.age == 30

    @patch("litekit.lite_client.completion")
    @patch("litekit.lite_client.JSONCleaner.extract_json")
    def test_pydantic_response_manual_fallback(self, mock_cleaner, mock_completion):
        class MyModel(BaseModel):
            name: str
            age: int

        mock_completion.return_value = MockResponse(
            choices=[MockChoice(message=MockMessage(content='{"name": "Bob", "age": 25}'))]
        )
        mock_cleaner.return_value = '{"name": "Bob", "age": 25}'
        config = ModelConfig(model="gpt-4")
        client = LiteClient(model_config=config)
        result = client.generate_text(
            ModelInput(user_prompt="Extract", response_format=MyModel)
        )
        assert isinstance(result, MyModel)
        assert result.name == "Bob"
        assert result.age == 25

    @patch("litekit.lite_client.completion")
    def test_override_model_config(self, mock_completion):
        mock_completion.return_value = MockResponse(
            choices=[MockChoice(message=MockMessage(content="Result"))]
        )
        config = ModelConfig(model="gpt-4")
        override = ModelConfig(model="gpt-3.5-turbo")
        client = LiteClient(model_config=config)
        result = client.generate_text(
            ModelInput(user_prompt="Hi"), model_config=override
        )
        assert result == "Result"
        assert mock_completion.call_args[1]["model"] == "gpt-3.5-turbo"

    @patch("litekit.lite_client.completion")
    def test_timeout_passed_to_completion(self, mock_completion):
        mock_completion.return_value = MockResponse(
            choices=[MockChoice(message=MockMessage(content="Ok"))]
        )
        config = ModelConfig(model="gpt-4", timeout=60)
        client = LiteClient(model_config=config)
        client.generate_text(ModelInput(user_prompt="Hi"))
        assert mock_completion.call_args[1]["timeout"] == 60

    def test_file_not_found_error(self):
        config = ModelConfig(model="gpt-4")
        client = LiteClient(model_config=config)
        with pytest.raises(FileNotFoundError):
            client.generate_text(
                ModelInput(user_prompt="Describe", image_path="/nonexistent.jpg"), retries=0
            )
