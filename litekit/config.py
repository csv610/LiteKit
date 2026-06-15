"""Configuration for available models and vision processing parameters."""

from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any, Type
from pydantic import BaseModel, Field

# Vision model defaults
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 2000
DEFAULT_PROMPT = "Describe this image in detail"

# Image processing
SUPPORTED_IMAGE_TYPES = ("jpg", "jpeg", "png", "gif", "webp")
IMAGE_MIME_TYPE = "image/jpeg"


@dataclass
class ModelConfig:
    """Configuration for model interactions."""

    model: str
    temperature: float = DEFAULT_TEMPERATURE
    timeout: Optional[int] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.model or not self.model.strip():
            raise ValueError("model name cannot be empty")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(
                f"temperature must be between 0.0 and 2.0, got {self.temperature}"
            )
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError(
                f"timeout must be positive, got {self.timeout}"
            )


@dataclass
class ChatConfig:
    """Configuration for chat session management."""

    max_history: int = 10
    auto_save: bool = False
    save_dir: str = "."


@dataclass
class ModelInput:
    """Input parameters for model interactions."""

    user_prompt: str = ""
    image_path: Optional[str] = None
    image_paths: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    response_format: Optional[Union[Type[BaseModel], str]] = None

    def __post_init__(self):
        """Validate input after initialization."""
        # Use default prompt if empty and images are provided
        if (not self.user_prompt or not self.user_prompt.strip()) and (
            self.image_path or self.image_paths
        ):
            self.user_prompt = DEFAULT_PROMPT

        if not self.user_prompt or not self.user_prompt.strip():
            if not self.image_path and not self.image_paths:
                raise ValueError(
                    "user_prompt cannot be empty unless an image_path(s) is provided"
                )

        # Normalize empty system_prompt to None
        if self.system_prompt is not None and not self.system_prompt.strip():
            self.system_prompt = None

        # Normalize empty response_format to None if it's a string
        if isinstance(self.response_format, str) and not self.response_format.strip():
            self.response_format = None


@dataclass
class MCQInput:
    """Input parameters for multiple-choice question solving."""

    question: str
    options: Union[List[str], Dict[str, str]]
    context: Optional[str] = None
    image_paths: Optional[List[str]] = None


@dataclass
class UserInput:
    """Standardized input for model response evaluation."""

    model_response: str
    user_prompt: Optional[str] = None
    ground_truth: Optional[str] = None
    context: Optional[str] = None


class ModelOutput(BaseModel):
    """Standardized output structure for all medical providers."""

    data: Optional[Any] = None
    markdown: Optional[str] = None
    metadata: Optional[dict] = Field(default_factory=dict)
