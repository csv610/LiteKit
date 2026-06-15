import logging
from typing import Optional
from pydantic import BaseModel, Field

from .lite_client import LiteClient
from .config import ModelConfig, ModelInput, UserInput

logger = logging.getLogger(__name__)


class CriteriaScores(BaseModel):
    """Per-dimension evaluation scores (0.0–1.0)."""

    accuracy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Factual correctness and alignment with Ground Truth (if provided).",
    )

    completeness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Degree to which all parts of the User Prompt are addressed.",
    )

    relevance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Absence of irrelevant, fabricated, or unnecessary information.",
    )

    clarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Logical structure, coherence, and readability of the response.",
    )


class EvaluationModel(BaseModel):
    """Structured schema returned by the judge model."""

    criteria: CriteriaScores = Field(
        ...,
        description="Per-dimension evaluation scores across core criteria.",
    )

    overall_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Overall normalized score between 0.0 and 1.0. "
            "Should reflect the combined assessment of all criteria."
        ),
    )

    is_correct: bool = Field(
        ...,
        description=(
            "True only if the Model Response is fundamentally correct and free of "
            "major factual errors."
        ),
    )

    reasoning: str = Field(
        ...,
        description=(
            "Concise explanation referencing key strengths, weaknesses, and "
            "notable issues in the evaluation."
        ),
    )

    feedback: Optional[str] = Field(
        None,
        description="Specific, actionable suggestions for improving the response.",
    )


class PromptBuilder:
    """Constructs prompts for LLM-based evaluation."""

    SYSTEM_PROMPT = (
        "You are an impartial, expert evaluator assessing the quality and correctness "
        "of a Model Response. You must evaluate ONLY using the provided sections: "
        "User Prompt (if present), Model Response, Ground Truth (if present), and Context (if present). "
        "Do not introduce external knowledge. If Ground Truth is provided, treat it as authoritative.\n\n"
        "EVALUATION PRINCIPLES:\n"
        "1. Accuracy (highest priority): Penalize factual errors, contradictions, and hallucinations.\n"
        "2. Completeness: Ensure all parts of the User Prompt are addressed.\n"
        "3. Relevance: Irrelevant or fabricated details reduce score.\n"
        "4. Clarity: Structure matters, but correctness outweighs style.\n\n"
        "SCORING CALIBRATION (0.0–1.0):\n"
        "1.0 = Fully correct and complete.\n"
        "0.8–0.9 = Minor omissions but fundamentally correct.\n"
        "0.5–0.7 = Partially correct with meaningful gaps.\n"
        "0.2–0.4 = Major errors or missing key elements.\n"
        "0.0–0.1 = Fundamentally incorrect or fabricated.\n"
        "Use partial credit appropriately. Do not inflate scores for fluent but incorrect answers.\n\n"
        "Semantic equivalence is acceptable if meaning matches Ground Truth. "
        "Additional correct information is allowed unless it introduces errors.\n\n"
        "Return ONLY a JSON object with: score (0.0–1.0), is_correct (boolean), "
        "reasoning (concise explanation), and optional feedback (actionable improvements)."
    )

    @staticmethod
    def create_system_prompt() -> str:
        """Return the system-level instruction for the judge model."""
        return PromptBuilder.SYSTEM_PROMPT

    @staticmethod
    def build_user_prompt(user_input: UserInput) -> str:
        sections = []

        if user_input.user_prompt:
            sections.append(f"### User Prompt:\n{user_input.user_prompt}")

        sections.append(f"### Model Response:\n{user_input.model_response}")

        if user_input.ground_truth:
            sections.append(f"### Ground Truth:\n{user_input.ground_truth}")

        if user_input.context:
            sections.append(f"### Context:\n{user_input.context}")

        sections.append("Evaluate the Model Response using all provided sections.")

        return "\n\n".join(sections)


class ResponseJudge:
    """LLM-as-a-judge evaluation engine."""

    def __init__(self, model_config: ModelConfig):
        if not isinstance(model_config, ModelConfig):
            raise TypeError("ResponseJudge requires a ModelConfig instance.")

        self.model_config = model_config
        self.client = LiteClient(model_config=self.model_config)

    def evaluate(self, user_input: UserInput) -> EvaluationModel:
        if not isinstance(user_input, UserInput):
            raise TypeError("evaluate() requires a UserInput instance.")

        if not user_input.model_response or not user_input.model_response.strip():
            raise ValueError("UserInput.model_response must not be empty.")

        constructed_prompt = PromptBuilder.build_user_prompt(user_input)

        model_input = ModelInput(
            user_prompt=constructed_prompt,
            system_prompt=PromptBuilder.SYSTEM_PROMPT,
            response_format=EvaluationModel,
        )

        logger.info(
            "Running LLM judge evaluation",
            extra={
                "model": self.model_config.model,
                "has_ground_truth": bool(user_input.ground_truth),
            },
        )

        result = self.client.generate_text(model_input=model_input)

        # Assume LiteClient enforces schema via response_format
        return result


def main():
    """CLI wrapper for ResponseJudge."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Evaluate a model response.")
    parser.add_argument("-p", "--prompt", help="Original user prompt")
    parser.add_argument("-r", "--response", required=True, help="Model response to evaluate")
    parser.add_argument("-g", "--ground-truth", help="Expected ground truth answer")
    parser.add_argument("-m", "--model", default="gemini/gemini-2.5-flash", help="Judge model")

    args = parser.parse_args()

    model_config = ModelConfig(model=args.model)
    judge = ResponseJudge(model_config=model_config)

    user_input = UserInput(
        model_response=args.response,
        user_prompt=args.prompt,
        ground_truth=args.ground_truth,
    )

    try:
        result = judge.evaluate(user_input=user_input)

        print("\n" + "=" * 60)
        print("EVALUATION RESULT")
        print("=" * 60)
        print(f"Overall Score: {result.overall_score:.2f}")
        print(f"Is Correct:    {result.is_correct}")
        print(f"Reasoning:     {result.reasoning}")
        if result.feedback:
            print(f"Feedback:      {result.feedback}")
        
        print("\nCriteria Scores:")
        print(f"  Accuracy:     {result.criteria.accuracy:.2f}")
        print(f"  Completeness: {result.criteria.completeness:.2f}")
        print(f"  Relevance:    {result.criteria.relevance:.2f}")
        print(f"  Clarity:      {result.criteria.clarity:.2f}")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

