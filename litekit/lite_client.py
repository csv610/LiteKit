"""Unified LiteClient for text and vision model interactions."""

import json
import logging
from typing import Any, Dict, List, Optional, Union

import litellm
from litellm import completion
from pydantic import BaseModel

from .config import ModelConfig, ModelInput
from .image_utils import ImageUtils
from .utils.json_cleaner import JSONCleaner

logger = logging.getLogger(__name__)


def _is_pydantic_model(obj: Any) -> bool:
    """Check if an object is a Pydantic BaseModel class (not an instance)."""
    return isinstance(obj, type) and issubclass(obj, BaseModel)


class LiteClient:
    """Unified client for interacting with both text and vision models."""

    def __init__(self, model_config: Optional[ModelConfig] = None):
        self.model_config = model_config

    @staticmethod
    def create_message(model_input: ModelInput) -> List[Dict[str, Any]]:
        """Build a message list for the completion API from a ModelInput."""
        messages = []
        system_content = model_input.system_prompt or ""

        if _is_pydantic_model(model_input.response_format):
            schema_json = model_input.response_format.model_json_schema()
            json_instruction = (
                "IMPORTANT: You MUST return a valid JSON object "
                f"matching this schema:\n{json.dumps(schema_json)}"
            )
            system_content = (
                f"{system_content}\n\n{json_instruction}"
                if system_content
                else json_instruction
            )

        if system_content:
            messages.append({"role": "system", "content": system_content})

        body = [{"type": "text", "text": model_input.user_prompt}]

        unique_paths = {}
        if model_input.image_path:
            unique_paths[model_input.image_path] = None
        for path in (model_input.image_paths or []):
            unique_paths[path] = None
        for image_path in unique_paths:
            base64_url = ImageUtils.encode_to_base64(image_path)
            body.append({"type": "image_url", "image_url": {"url": base64_url}})

        messages.append({"role": "user", "content": body})
        return messages

    def generate_text(
        self,
        model_input: ModelInput,
        model_config: Optional[ModelConfig] = None,
        retries: int = 2,
    ) -> Union[str, BaseModel]:
        """Generate text from a prompt or analyze an image with a prompt.

        Args:
            model_input: Input with prompt and optional image params
            model_config: Optional override config
            retries: Number of retry attempts on failure (default 2)

        Returns:
            Generated text string or a parsed Pydantic model instance

        Raises:
            ValueError: If no ModelConfig is available
            Exception: Last failure if all retries are exhausted
        """
        config = model_config or self.model_config
        if not config:
            raise ValueError("ModelConfig must be provided")

        for attempt in range(retries + 1):
            try:
                logger.info(
                    "Generating completion (attempt %d) with model: %s",
                    attempt + 1,
                    config.model,
                )
                messages = self.create_message(model_input)

                completion_params: Dict[str, Any] = dict(
                    model=config.model,
                    messages=messages,
                    temperature=config.temperature,
                    response_format=model_input.response_format,
                )
                if config.timeout is not None:
                    completion_params["timeout"] = config.timeout

                response = completion(**completion_params)

                # litellm passes Pydantic response_format as JSON schema to the
                # provider. Some providers return the parsed model in
                # response.choices[0].message.parsed.  For the rest, we parse
                # the JSON string manually.
                if _is_pydantic_model(model_input.response_format):
                    parsed = getattr(
                        response.choices[0].message, "parsed", None
                    )
                    if parsed is not None:
                        logger.info(
                            "Parsed response as %s via litellm",
                            model_input.response_format.__name__,
                        )
                        return parsed

                    response_content = response.choices[0].message.content
                    cleaned_json = JSONCleaner.extract_json(response_content)
                    try:
                        parsed_response = (
                            model_input.response_format.model_validate_json(
                                cleaned_json
                            )
                        )
                        logger.info(
                            "Parsed response as %s (manual fallback)",
                            model_input.response_format.__name__,
                        )
                        return parsed_response
                    except Exception as parse_err:
                        logger.warning(
                            "Failed to parse %s on attempt %d: %s",
                            model_input.response_format.__name__,
                            attempt + 1,
                            parse_err,
                        )
                        if attempt == retries:
                            raise
                        continue

                return response.choices[0].message.content

            except (KeyboardInterrupt, SystemExit):
                raise
            except FileNotFoundError as e:
                logger.error("File not found: %s", e)
                raise
            except litellm.exceptions.APIError as e:
                logger.error("Attempt %d failed (API error): %s", attempt + 1, e)
                if attempt < retries:
                    continue
                raise
            except Exception as e:
                logger.error("Attempt %d failed: %s", attempt + 1, e)
                if attempt < retries:
                    continue
                raise
