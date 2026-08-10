from litellm import completion
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import sys
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Image:
    """Represents an image in the response."""
    url: Optional[str] = None
    source_url: Optional[str] = None
    title: str = "No title"
    width: str = "Unknown width"
    height: str = "Unknown height"

    def __str__(self) -> str:
        return (f"Title: {self.title}\n"
                f"   Image URL: {self.url}\n"
                f"   Source URL: {self.source_url}\n"
                f"   Dimensions: {self.width}x{self.height}")


@dataclass
class ResponseData:
    """Represents structured response data."""
    content: str
    citations: List[str]
    images: List[Image]
    related_questions: List[str]


class PerplexityResponseParser:
    """Handles parsing and extraction of response data."""

    @staticmethod
    def extract_content(response: Any) -> str:
        """Extract main content from response."""
        try:
            return response.choices[0].message.content
        except (AttributeError, IndexError) as e:
            logger.error(f"Error extracting content: {e}")
            return ""

    @staticmethod
    def extract_citations(response: Any) -> List[str]:
        """Extract citations from response."""
        citations = getattr(response, "citations", None) or response.get("citations", None)
        return citations if citations else []

    @staticmethod
    def extract_images(response: Any) -> List[Image]:
        """Extract and parse images from response."""
        images_data = getattr(response, "images", None) or response.get("images", [])
        images = []

        for img_data in images_data:
            image = Image(
                url=img_data.get("image_url"),
                source_url=img_data.get("origin_url"),
                title=img_data.get("title", "No title"),
                width=img_data.get("width", "Unknown width"),
                height=img_data.get("height", "Unknown height")
            )
            images.append(image)

        return images

    @staticmethod
    def extract_related_questions(response: Any) -> List[str]:
        """Extract related questions from response."""
        # Try top-level attribute first
        related_questions = getattr(response, "related_questions", None)

        # Try nested in choices
        if related_questions is None:
            try:
                related_questions = getattr(response.choices[0], "related_questions", None)
            except (AttributeError, IndexError):
                related_questions = None

        return related_questions if related_questions else []

    def parse(self, response: Any) -> ResponseData:
        """Parse complete response into structured data."""
        return ResponseData(
            content=self.extract_content(response),
            citations=self.extract_citations(response),
            images=self.extract_images(response),
            related_questions=self.extract_related_questions(response)
        )


class PerplexityResponseFormatter:
    """Handles formatting of response data."""

    @staticmethod
    def format_content(content: str) -> str:
        """Format main content."""
        return f"\n{content}\n"

    @staticmethod
    def format_citations(citations: List[str]) -> str:
        """Format citations."""
        if not citations:
            return "\nNo citations found."

        formatted = "\nCitations:\n"
        for i, citation in enumerate(citations, start=1):
            formatted += f"{i}. {citation}\n"
        return formatted

    @staticmethod
    def format_images(images: List[Image]) -> str:
        """Format images."""
        if not images:
            return "No images returned in response."

        formatted = "Images returned:\n"
        for i, image in enumerate(images, 1):
            formatted += f"{i}. {image}\n"
        return formatted

    @staticmethod
    def format_related_questions(questions: List[str]) -> str:
        """Format related questions."""
        if not questions:
            return "No follow-up questions found."

        formatted = "Follow-up Questions:\n"
        for i, question in enumerate(questions, 1):
            formatted += f"{i}. {question}\n"
        return formatted

    def format_all(self, data: ResponseData) -> str:
        """Format all response data into a single string."""
        output = ""
        output += self.format_content(data.content)
        output += self.format_citations(data.citations)
        output += self.format_images(data.images)
        output += self.format_related_questions(data.related_questions)
        return output

    def save_to_file(self, data: ResponseData, filepath: str) -> None:
        """Save all response data to a file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.format_all(data))
            logger.info(f"Response saved to {filepath}")
        except IOError as e:
            logger.error(f"Error writing to file {filepath}: {e}")
            raise


class PerplexityChat:
    """Main class to interact with Perplexity API."""

    def __init__(self, model: str = "perplexity/sonar", output_dir: str = "responses", **kwargs):
        """
        Initialize PerplexityChat.

        Args:
            model: Model to use for API calls
            output_dir: Directory to save responses
            **kwargs: Additional arguments to pass to completion API
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.api_kwargs = {
            "return_images": True,
            "return_citations": True,
            "return_related_questions": True,
            **kwargs
        }
        self.parser = PerplexityResponseParser()
        self.formatter = PerplexityResponseFormatter()

    def _generate_filename(self, question: str) -> str:
        """Generate a filename based on timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"response_{timestamp}.txt"

    def generate_content(self, question: str) -> tuple[ResponseData, str]:
        """
        Ask a question and automatically save the response to a file.

        Args:
            question: The question to ask

        Returns:
            Tuple of (ResponseData object, output filepath)
        """
        logger.info(f"Querying Perplexity with: {question}")

        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": question}],
            **self.api_kwargs
        )

        data = self.parser.parse(response)

        # Always save to file
        output_file = self.output_dir / self._generate_filename(question)
        self.formatter.save_to_file(data, str(output_file))

        return data, str(output_file)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python perplx_chat.py <question> [output_dir]")
        sys.exit(1)

    question = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "responses"

    client = PerplexityChat(output_dir=output_dir)
    data, filepath = client.generate_content(question)
    print(f"Response saved to {filepath}")


if __name__ == "__main__":
    main()
