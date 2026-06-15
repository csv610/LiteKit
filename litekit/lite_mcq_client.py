"""Multiple-choice question solver using LiteLLM."""

import json
import argparse
import string
from typing import Optional, Union, List, Dict

from pydantic import BaseModel, Field
from litekit.lite_client import LiteClient
from litekit.config import ModelConfig, ModelInput, MCQInput


class CorrectOption(BaseModel):
    key: str = Field(..., description="The key or index of the correct option (e.g., 'A', 'B', '0', '1')")
    value: str = Field(..., description="The text of the correct option")


class MultipleChoiceAnswer(BaseModel):
    question: str = Field(..., description="The multiple-choice question")
    correct_options: List[CorrectOption] = Field(
        ...,
        description="List of correct option(s). Can be single or multiple."
    )
    reasoning: str = Field(
        ...,
        description="Detailed explanation of why these are the correct answers"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence level in the answer (0.0 to 1.0)"
    )


class MultipleChoiceSolverResponse(BaseModel):
    answer: MultipleChoiceAnswer


class LiteMCQClient:
    """Solves multiple-choice questions using LiteLLM."""

    SYSTEM_PROMPT = """You are an expert at solving multiple-choice questions accurately.
Your task is to:
1. Carefully read the question and all options
2. Identify which option(s) are correct
3. Provide clear reasoning for your answer
4. Be confident in your answer when you have strong justification

Always respond in the requested JSON format."""

    def __init__(self, model: str = "gemini/gemini-2.5-flash", temperature: float = 0.2):
        """Initialize the MCQ Client with a model and temperature."""
        try:
            model_config = ModelConfig(model=model, temperature=temperature)
            self.client = LiteClient(model_config=model_config)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LiteMCQClient: {e}") from e

    def solve(self, question: MCQInput, model_config: Optional[ModelConfig] = None) -> Optional[MultipleChoiceAnswer]:
        """
        Solve a multiple-choice question.

        Args:
            question: MCQInput object containing the question, options, and optional context/images
            model_config: Optional ModelConfig to override the client's model configuration

        Returns:
            MultipleChoiceAnswer with correct option(s) and reasoning
        """
        try:
            # Use provided model_config or the client's default
            if model_config:
                client = LiteClient(model_config=model_config)
            else:
                client = self.client

            # Create prompt
            prompt = self._create_prompt(question)

            # Build model input
            model_input = ModelInput(
                user_prompt=prompt,
                response_format=MultipleChoiceSolverResponse,
                system_prompt=self.SYSTEM_PROMPT
            )

            # Add images if provided
            if question.image_paths:
                model_input.image_paths = question.image_paths

            # Get response
            response = client.generate_text(model_input=model_input)

            # Check if response is the expected Pydantic model
            if isinstance(response, MultipleChoiceSolverResponse):
                return response.answer
            if isinstance(response, str):
                try:
                    parsed = MultipleChoiceSolverResponse.model_validate_json(response)
                    return parsed.answer
                except Exception:
                    try:
                        return MultipleChoiceAnswer.model_validate_json(response)
                    except Exception:
                        return None
            
            return None

        except Exception as e:
            raise RuntimeError(f"Error solving question: {e}") from e

    def _format_options(self, options: Union[List[str], Dict[str, str]]) -> str:
        """Format options as a string for the prompt."""
        if isinstance(options, dict):
            formatted = "\n".join([f"{key}: {value}" for key, value in options.items()])
        else:
            # Convert list to dict with A, B, C, etc. keys
            keys = list(string.ascii_uppercase[:len(options)])
            formatted = "\n".join([f"{key}: {value}" for key, value in zip(keys, options)])
        return formatted

    def _create_prompt(self, question: MCQInput) -> str:
        """Create a prompt for the model."""
        formatted_options = self._format_options(question.options)

        prompt = f"""Question: {question.question}

Options:
{formatted_options}

Please analyze this question carefully and determine which option(s) is/are correct.
Provide your answer in the specified JSON format."""

        if question.context:
            prompt = f"""Context:
{question.context}

{prompt}"""

        if question.image_paths:
            num_images = len(question.image_paths)
            prompt += f"\n\nNote: {num_images} image(s) are provided. Please analyze them along with the question."

        return prompt


def _dict_to_mcq_input(data: dict) -> MCQInput:
    """Convert a dictionary to MCQInput object."""
    return MCQInput(
        question=data.get("question", ""),
        options=data.get("options", []),
        context=data.get("context"),
        image_paths=data.get("image_paths")
    )


def main():
    """Command-line interface for the MCQ solver."""
    parser = argparse.ArgumentParser(
        description="Solve multiple-choice questions using LiteLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Simple question with options
  python -m litekit.lite_mcq_client -q "What is the capital of France?" \\
    -o "A: London" "B: Paris" "C: Berlin"

  # Question with context and multiple images
  python -m litekit.lite_mcq_client -q "Which image contains a tiger?" \\
    -c "You have 4 animal images" \\
    -i image1.jpg image2.jpg image3.jpg image4.jpg \\
    -o "Image 1" "Image 2" "Image 3" "Image 4"

  # Load from JSON file
  python -m litekit.lite_mcq_client -f questions.json

  # Load from JSON file with custom model
  python -m litekit.lite_mcq_client -f questions.json -m gpt-4
        """
    )

    parser.add_argument(
        "-q", "--question",
        type=str,
        help="The multiple-choice question"
    )
    parser.add_argument(
        "-o", "--options",
        nargs="+",
        help="Options as space-separated strings"
    )
    parser.add_argument(
        "-c", "--context",
        type=str,
        default=None,
        help="Context/background information (optional)"
    )
    parser.add_argument(
        "-i", "--images",
        nargs="+",
        default=None,
        help="Paths to image files (multiple images supported)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="gemini/gemini-2.5-flash",
        help="Model name (default: gemini/gemini-2.5-flash)"
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Path to JSON file with question"
    )

    args = parser.parse_args()

    client = LiteMCQClient(model=args.model, temperature=0.2)

    if args.file:
        # Load question from file
        with open(args.file, "r") as f:
            data = json.load(f)
        question = _dict_to_mcq_input(data)
        answer = client.solve(question)
        print_answer(answer)
    elif args.question and args.options:
        # Single question
        question = MCQInput(
            question=args.question,
            options=args.options,
            context=args.context,
            image_paths=args.images
        )
        answer = client.solve(question)
        print_answer(answer)
    else:
        parser.print_help()


def print_answer(answer: Optional[MultipleChoiceAnswer]) -> None:
    """Pretty print an answer."""
    if answer is None:
        print("Failed to solve question")
        return

    print("\n" + "="*60)
    print(f"question: {answer.question}")
    print("-"*60)

    answers = [opt.key for opt in answer.correct_options]
    print(f"answers: {answers}")
    print(f"confidence: {answer.confidence:.1%}")
    print("-"*60)
    print(f"Reasoning:\n{answer.reasoning}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
