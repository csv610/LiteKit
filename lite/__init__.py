"""LiteLLM - Unified interface for multiple LLM providers.

This package provides a unified client for interacting with text and vision models
from multiple providers (OpenAI, Ollama, Google Gemini, Perplexity).
"""

from .lite_client import LiteClient
from .config import ModelConfig

__version__ = "0.1.0"
__all__ = ["LiteClient", "ModelConfig"]
