import argparse
import logging
import json
import sys
import os
import re
from datetime import datetime
from typing import Optional
from litellm import completion
from pydantic import BaseModel, Field, ValidationError

# Configure logging to file only
def setup_logging(log_file: str = "nobel_prize_info.log") -> logging.Logger:
    """
    Configure logging to write to file only.

    Args:
        log_file: Path to log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # File handler with restricted permissions
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)

    # Add handlers to logger
    logger.addHandler(file_handler)

    # Set restrictive file permissions (owner read/write only)
    try:
        os.chmod(log_file, 0o600)
    except OSError:
        pass  # File may not exist yet, permissions will be set on first write

    return logger

logger = setup_logging()

# ==============================================================================
# Pydantic Models
# ==============================================================================

class FAQItem(BaseModel):
    """Frequently asked question about the discovery."""
    question: str = Field(..., description="The frequently asked question")
    answer: str = Field(..., description="The answer to the question")


class GlossaryItem(BaseModel):
    """A glossary term with its definition."""
    term: str = Field(..., description="The technical term or concept")
    definition: str = Field(..., description="Clear, concise definition of the term")


class PrizeWinner(BaseModel):
    """Represents a Nobel Prize winner and their contribution."""

    name: str = Field(..., description="Full name of the prize winner(s)")
    year: int = Field(..., description="Year the prize was awarded", ge=1901, le=2025)
    category: str = Field(..., description="Nobel Prize category (Physics, Chemistry, Medicine, etc.)")
    contribution: str = Field(
        ...,
        description="Objective, concise description of the scientific work that won the prize. Focus on what was discovered/invented, not superlatives."
    )

    # Historical context - objective facts
    history: str = Field(
        ...,
        description="Chronological history of the discovery with specific dates, names, and events. Include key experimental methods and findings. Avoid subjective language. Focus on facts and progression of research.",
        min_length=50
    )

    # Scientific impact - measurable effects
    impact: str = Field(
        ...,
        description="Measurable scientific impact: how the discovery changed our understanding of natural phenomena. Include specific examples of what became possible because of this work. Avoid phrases like 'profound,' 'revolutionary,' or 'transformed.' Use objective language.",
        min_length=50
    )

    # Cross-disciplinary influence
    foundation: str = Field(
        ...,
        description="Specific ways this discovery influenced or enabled research in other scientific fields. Give concrete examples of which fields were affected and how. Avoid vague statements.",
        min_length=50
    )

    # Practical applications - real world use cases
    applications: list[str] = Field(
        ...,
        description="List of specific, verified real-world applications and implementations. Only include applications that are actually used today or in development. Be concrete, not speculative.",
        min_length=1
    )

    # Current relevance - factual assessment
    relevancy: str = Field(
        ...,
        description="How the idea is still valid and relevant today. Explain current research directions, active fields using this work, ongoing investigations, and practical uses in modern science and industry. Include specific examples of how the discovery continues to influence contemporary work. Avoid subjective claims about importance.",
        min_length=50
    )

    # Subsequent research - building on the work
    advancements: list[str] = Field(
        ...,
        description="List of specific advancements, new discoveries, and technologies developed based on this work. Include years where known. Focus on factual extensions of the original discovery.",
        min_length=1
    )

    # Methodological improvements
    refinements: list[str] = Field(
        ...,
        description="List of major improvements to the experimental methods, theoretical models, or practical techniques developed after the original work. Be specific about what was improved and how.",
        min_length=1
    )

    # Open questions and limitations
    gaps: list[str] = Field(
        ...,
        description="List of known limitations, open questions, unsolved problems, or areas not yet understood. Be specific about what remains unknown or incompletely explained.",
        min_length=1
    )

    # Important keywords
    keywords: list[str] = Field(
        ...,
        description="List of important keywords and technical terms related to the discovery. Include core concepts, methods, substances, phenomena, and fields that are central to understanding this work.",
        min_length=1
    )

    # Learning objectives
    learning_objectives: list[str] = Field(
        ...,
        description="List of learning objectives: what a student can learn from this discovery. Include conceptual understanding, methodological approaches, problem-solving techniques, and insights into the scientific process.",
        min_length=1
    )

    # Frequently asked questions about the discovery
    faq: list[FAQItem] = Field(
        ...,
        description="List of frequently asked questions about the discovery. Include common misconceptions, practical questions, and educational queries.",
        min_length=1
    )

    # Glossary of terms
    glossary: list[GlossaryItem] = Field(
        ...,
        description="List of key terms and concepts related to the discovery with clear, concise definitions. Include specialized vocabulary, technical terminology, and important concepts needed to understand the work.",
        min_length=1
    )


class PrizeResponse(BaseModel):
    """Response containing a list of Nobel Prize winners."""
    winners: list[PrizeWinner]


# ==============================================================================
# Core Functions
# ==============================================================================

def create_prompt(category: str, year: str) -> str:
    """
    Create the prompt for fetching Nobel Prize information.

    Args:
        category: Nobel Prize category
        year: Year of the prize

    Returns:
        Formatted prompt string for the LLM
    """
    return f"""Provide detailed, objective information about Nobel Prize winners in {category} for {year}.

IMPORTANT: Focus on factual, educational content. Avoid subjective language and superlatives.

For each winner, provide:
1. Contribution: What they discovered or invented (facts only)
2. History: Chronological facts with dates, names, methods used
3. Impact: Measurable changes in scientific understanding - what became possible?
4. Foundation: Specific cross-disciplinary influence with concrete examples
5. Applications: Real-world uses, not speculative
6. Relevancy: How the idea is still valid and relevant today - explain current research, active fields using this work, and practical uses in modern science/industry
7. Advancements: Specific improvements and extensions with dates
8. Refinements: Methodological and theoretical improvements
9. Gaps: Unknown questions, limitations, and unsolved problems
10. Keywords: Important keywords and technical terms related to the discovery (core concepts, methods, substances, phenomena, and fields)
11. Learning Objectives: What a student can learn from this discovery (conceptual understanding, methodological approaches, problem-solving techniques, insights into the scientific process)
12. FAQ: Frequently asked questions with answers (include common misconceptions, practical questions, and educational queries)
13. Glossary: Dictionary of key terms and concepts with clear, concise definitions (specialized vocabulary, technical terminology, important concepts needed to understand the work)

Use objective language. Avoid words like "revolutionary," "profound," "amazing," "transformed."
Instead, describe what specifically changed and how we know it changed."""


def _extract_winners(response) -> list[PrizeWinner]:
    """
    Extract winners from API response.

    Args:
        response: API response object with Pydantic validated data

    Returns:
        List of PrizeWinner instances

    Raises:
        ValueError: If no winners in response
    """
    if not response.choices or len(response.choices) == 0:
        raise ValueError("API returned empty response")

    if not hasattr(response.choices[0], 'message'):
        raise ValueError("API response missing message field")

    # LiteLLM returns content as JSON string, need to parse back to Pydantic model
    content_str = response.choices[0].message.content

    if isinstance(content_str, str):
        prize_response = PrizeResponse.model_validate_json(content_str)
    else:
        prize_response = content_str  # In case it's already parsed

    if not prize_response.winners or len(prize_response.winners) == 0:
        raise ValueError("No winners returned in response")

    return prize_response.winners


def _handle_api_error(error: Exception) -> None:
    """
    Handle and translate API errors to meaningful exceptions.

    Args:
        error: Exception raised during API call

    Raises:
        RuntimeError: With appropriate error message
    """
    error_str = str(error).lower()

    if "401" in str(error) or "authentication" in error_str:
        logger.error("Authentication failed: Check your API credentials")
        raise RuntimeError("API authentication failed. Check LITELLM_API_KEY or model-specific credentials.")
    elif "429" in str(error):
        logger.error("Rate limit exceeded")
        raise RuntimeError("API rate limit exceeded. Please try again later.")
    elif "404" in str(error):
        logger.error("Model not found")
        raise RuntimeError("Model not found or not available.")
    else:
        logger.error(f"Unexpected error: {error}")
        raise RuntimeError(f"Failed to fetch Nobel Prize information: {error}")


def fetch_nobel_winners(
    category: str,
    year: str,
    model: Optional[str] = None
) -> list[PrizeWinner]:
    """
    Fetch Nobel Prize winners for a specific field and year.

    Args:
        category: Nobel Prize category (Physics, Chemistry, Medicine, Literature, Peace, Economics)
        year: Year of the prize
        model: LLM model to use (defaults to environment variable or gemini/gemini-2.5-flash)

    Returns:
        List of PrizeWinner instances

    Raises:
        ValueError: If API response is invalid or model response doesn't match schema
        RuntimeError: If API call fails or required credentials are missing
    """
    if model is None:
        model = os.getenv("NOBEL_PRIZE_MODEL", "gemini/gemini-2.5-flash")

    if not re.match(r'^[a-zA-Z0-9\-\./_]+$', model):
        raise ValueError(f"Invalid model name: {model}. Only alphanumeric characters, hyphens, slashes, dots, and underscores are allowed.")

    logger.info(f"Fetching Nobel Prize information for {category} in {year} using model: {model}")

    messages = [
        {
            "role": "user",
            "content": create_prompt(category, year)
        }
    ]

    try:
        response = completion(
            model=model,
            messages=messages,
            response_format=PrizeResponse
        )

        winners = _extract_winners(response)
        logger.info(f"Successfully fetched {len(winners)} winner(s)")
        return winners

    except Exception as e:
        _handle_api_error(e)


# ==============================================================================
# Validation Functions
# ==============================================================================

def validate_year(year_str: str) -> int:
    """
    Validate and convert year string to integer.

    Args:
        year_str: Year as string

    Returns:
        Year as integer

    Raises:
        argparse.ArgumentTypeError: If year is invalid
    """
    try:
        year_int = int(year_str)
        current_year = datetime.now().year
        if year_int < 1901:
            raise argparse.ArgumentTypeError(
                f"Nobel Prizes began in 1901, got {year_int}"
            )
        if year_int > current_year:
            raise argparse.ArgumentTypeError(
                f"Year cannot be in the future (current year: {current_year})"
            )
        return year_int
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Year must be a valid number, got '{year_str}'"
        )


def validate_category(category_str: str) -> str:
    """
    Validate and normalize Nobel Prize category.

    Args:
        category_str: Category name (case-insensitive)

    Returns:
        Normalized category name

    Raises:
        argparse.ArgumentTypeError: If category is invalid
    """
    valid_fields = ["Physics", "Chemistry", "Medicine", "Literature", "Peace", "Economics"]
    valid_fields_lower = [f.lower() for f in valid_fields]

    if category_str.lower() not in valid_fields_lower:
        raise argparse.ArgumentTypeError(
            f"'{category_str}' is not a valid Nobel Prize category. "
            f"Valid categories are: {', '.join(valid_fields)}"
        )

    # Return normalized (proper case) field
    return valid_fields[valid_fields_lower.index(category_str.lower())]


# ==============================================================================
# CLI Functions
# ==============================================================================

def arguments_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Fetch detailed information about Nobel Prize winners",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nobel_prize_info.py -c Physics -y 2023
  python nobel_prize_info.py --category Chemistry --year 2022
  python nobel_prize_info.py -c Medicine -y 2021 -m claude-3-opus
  NOBEL_PRIZE_MODEL=gpt-4 python nobel_prize_info.py -c Literature -y 2020
        """
    )

    parser.add_argument(
        "-c", "--category",
        required=True,
        type=validate_category,
        help="Nobel Prize category (Physics, Chemistry, Medicine, Literature, Peace, Economics)"
    )

    parser.add_argument(
        "-y", "--year",
        required=True,
        type=validate_year,
        help="Year of the Nobel Prize (1901 onwards)"
    )

    parser.add_argument(
        "-m", "--model",
        default=None,
        help="LLM model to use (default: $NOBEL_PRIZE_MODEL or gemini/gemini-2.5-flash)"
    )

    return parser



def main() -> int:
    """
    Main entry point for the Nobel Prize information CLI.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = arguments_parser()
    args = parser.parse_args()

    try:
        logger.info(f"Fetching Nobel Prize information for {args.category} in {args.year}...")

        winners = fetch_nobel_winners(args.category, str(args.year), model=args.model)

        if not winners:
            logger.error("No winners returned from API")
            print("Error: No winners found for the specified category and year.")
            return 1

        # Save to JSON file with format: {category}_{year}.json
        output_filename = os.path.basename(f"{args.category.lower()}_{args.year}.json")

        data_to_save = {
            "category": args.category,
            "year": args.year,
            "winners_count": len(winners),
            "winners": [winner.model_dump() for winner in winners]
        }

        with open(output_filename, 'w') as f:
            json.dump(data_to_save, f, indent=4)

        os.chmod(output_filename, 0o600)

        logger.info(f"Successfully saved {len(winners)} winner(s) to {output_filename}")
        print(f"Results saved to {output_filename}")
        print(f"Winners retrieved: {len(winners)}")
        for i, winner in enumerate(winners, 1):
            print(f"  {i}. {winner.name} ({winner.year})")

        return 0

    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 1
    except IOError as e:
        logger.error(f"File I/O error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
