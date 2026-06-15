"""Tests for LiteChat (with mocked litellm)."""

from unittest.mock import MagicMock, patch

import pytest

from litekit.lite_chat import LiteChat
from litekit.config import ModelConfig, ChatConfig, ModelInput


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


class TestLiteChatInit:
    def test_default_config(self):
        chat = LiteChat()
        assert chat.model_config is None
        assert chat.max_history == 10
        assert chat.auto_save is False
        assert chat.save_dir == "."
        assert len(chat.conversation_history) == 0
        assert chat.current_image_path is None

    def test_with_model_config(self):
        model_cfg = ModelConfig(model="gpt-4")
        chat = LiteChat(model_config=model_cfg)
        assert chat.model_config is model_cfg

    def test_with_chat_config(self):
        chat_cfg = ChatConfig(max_history=6, auto_save=True, save_dir="/tmp")
        chat = LiteChat(chat_config=chat_cfg)
        assert chat.max_history == 6
        assert chat.auto_save is True
        assert chat.save_dir == "/tmp"

    def test_odd_max_history_is_decremented(self):
        chat_cfg = ChatConfig(max_history=7)
        chat = LiteChat(chat_config=chat_cfg)
        assert chat.max_history == 6


class TestAddMessageToHistory:
    def test_add_message(self):
        chat = LiteChat()
        chat.add_message_to_history("user", "Hello")
        assert len(chat.conversation_history) == 1
        assert chat.conversation_history[0]["role"] == "user"
        assert chat.conversation_history[0]["content"] == "Hello"

    def test_trim_history_to_max(self):
        chat_cfg = ChatConfig(max_history=2)
        chat = LiteChat(chat_config=chat_cfg)
        chat.add_message_to_history("user", "A")
        chat.add_message_to_history("assistant", "B")
        chat.add_message_to_history("user", "C")
        assert len(chat.conversation_history) == 1
        assert chat.conversation_history[0]["content"] == "C"

    def test_trim_with_single_message_edge_case(self):
        chat_cfg = ChatConfig(max_history=0)
        chat = LiteChat(chat_config=chat_cfg)
        chat.add_message_to_history("user", "A")
        assert len(chat.conversation_history) == 0

    def test_max_history_zero(self):
        chat_cfg = ChatConfig(max_history=0)
        chat = LiteChat(chat_config=chat_cfg)
        assert chat.max_history == 0
        chat.add_message_to_history("user", "A")
        assert len(chat.conversation_history) == 0


class TestCreateMessage:
    def test_basic(self):
        with patch("litekit.lite_chat.ImageUtils.encode_to_base64") as mock_encode:
            mock_encode.return_value = "data:image/jpeg;base64,abc"
            chat = LiteChat()
            inp = ModelInput(user_prompt="Hello")
            messages = chat.create_message(inp)
            assert len(messages) == 1
            assert messages[0]["role"] == "user"

    def test_with_image(self):
        with patch("litekit.lite_chat.ImageUtils.encode_to_base64") as mock_encode:
            mock_encode.return_value = "data:image/jpeg;base64,abc"
            chat = LiteChat()
            inp = ModelInput(user_prompt="Describe", image_path="/path/img.jpg")
            messages = chat.create_message(inp)
            assert len(messages) == 1
            content = messages[0]["content"]
            assert isinstance(content, list)
            assert content[0]["type"] == "text"
            assert content[1]["type"] == "image_url"

    def test_with_image_no_prompt_gets_default(self):
        chat = LiteChat()
        with patch("litekit.lite_chat.ImageUtils.encode_to_base64") as mock_encode:
            mock_encode.return_value = "data:image/jpeg;base64,abc"
            inp = ModelInput(user_prompt="", image_path="/img.jpg")
            assert inp.user_prompt != ""
            messages = chat.create_message(inp)
            assert len(messages) == 1

    def test_conversation_history_included(self):
        chat = LiteChat()
        chat.add_message_to_history("user", "Previous question")
        chat.add_message_to_history("assistant", "Previous answer")
        inp = ModelInput(user_prompt="New question")
        messages = chat.create_message(inp)
        assert len(messages) == 3
        assert messages[0]["content"] == "Previous question"
        assert messages[1]["content"] == "Previous answer"
        assert messages[2]["content"] == "New question"


class TestGenerateText:
    @patch("litekit.lite_chat.completion")
    def test_basic_generation(self, mock_completion):
        mock_completion.return_value = MockResponse(
            choices=[MockChoice(message=MockMessage(content="Response"))]
        )
        config = ModelConfig(model="gpt-4")
        chat = LiteChat(model_config=config)
        result = chat.generate_text(ModelInput(user_prompt="Hi"))
        assert result == "Response"
        assert len(chat.conversation_history) == 2

    @patch("litekit.lite_chat.completion")
    def test_no_config_raises(self, mock_completion):
        chat = LiteChat()
        with pytest.raises(ValueError, match="ModelConfig must be provided"):
            chat.generate_text(ModelInput(user_prompt="Hi"))

    @patch("litekit.lite_chat.completion")
    def test_override_model_config(self, mock_completion):
        mock_completion.return_value = MockResponse(
            choices=[MockChoice(message=MockMessage(content="Resp"))]
        )
        config = ModelConfig(model="gpt-4")
        override = ModelConfig(model="gpt-3.5-turbo")
        chat = LiteChat(model_config=config)
        chat.generate_text(ModelInput(user_prompt="Hi"), model_config=override)
        assert mock_completion.call_args[1]["model"] == "gpt-3.5-turbo"

    @patch("litekit.lite_chat.completion")
    def test_error_raises_exception(self, mock_completion):
        mock_completion.side_effect = Exception("API failed")
        config = ModelConfig(model="gpt-4")
        chat = LiteChat(model_config=config)
        with pytest.raises(Exception, match="API failed"):
            chat.generate_text(ModelInput(user_prompt="Hi"))


class TestConversationManagement:
    def test_reset_conversation(self):
        chat = LiteChat()
        chat.add_message_to_history("user", "Hi")
        chat.add_message_to_history("assistant", "Hello")
        assert len(chat.conversation_history) == 2
        chat.reset_conversation()
        assert chat.conversation_history == []
        assert chat.current_image_path is None

    def test_get_conversation_history(self):
        chat = LiteChat()
        chat.add_message_to_history("user", "Q")
        history = chat.get_conversation_history()
        assert len(history) == 1
        history.append({"role": "assistant", "content": "A"})
        assert len(chat.conversation_history) == 1

    @patch("litekit.lite_chat.completion")
    def test_history_built_from_multiple_turns(self, mock_completion):
        mock_completion.return_value = MockResponse(
            choices=[MockChoice(message=MockMessage(content="Answer"))]
        )
        config = ModelConfig(model="gpt-4")
        chat = LiteChat(model_config=config)
        chat.generate_text(ModelInput(user_prompt="Q1"))
        chat.generate_text(ModelInput(user_prompt="Q2"))
        history = chat.get_conversation_history()
        assert len(history) == 4
        assert history[0]["content"] == "Q1"
        assert history[1]["content"] == "Answer"
        assert history[2]["content"] == "Q2"
        assert history[3]["content"] == "Answer"


class TestFormatContent:
    def test_format_string(self):
        assert LiteChat._format_content("hello") == "hello"

    def test_format_list(self):
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ]
        result = LiteChat._format_content(content)
        assert result == "Hello"

    def test_format_empty_list(self):
        result = LiteChat._format_content([{"type": "image_url", "image_url": {"url": "data:"}}])
        assert result == str([{"type": "image_url", "image_url": {"url": "data:"}}])

    def test_format_other(self):
        assert LiteChat._format_content(42) == "42"
