"""Core LiteLLM library."""

__version__ = "0.1.0"

from .lite_client import LiteClient
from .config import ModelConfig, ModelOutput, ModelInput, MCQInput, UserInput, ChatConfig
from .image_utils import ImageUtils
from .logging_config import configure_logging
from .utils import save_model_response
from .lite_response_judge import ResponseJudge, EvaluationModel
from .lite_mcq_client import LiteMCQClient
from .lite_chat import LiteChat

__all__ = [
    "LiteClient",
    "LiteChat",
    "LiteMCQClient",
    "ModelConfig",
    "ModelInput",
    "MCQInput",
    "UserInput",
    "ChatConfig",
    "ModelOutput",
    "ImageUtils",
    "configure_logging",
    "save_model_response",
    "ResponseJudge",
    "EvaluationModel",
]
