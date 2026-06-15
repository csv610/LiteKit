import re
import logging
import json

logger = logging.getLogger(__name__)


class JSONCleaner:
    """Utility to clean and extract JSON from model responses."""

    @staticmethod
    def extract_json(response_text: str) -> str:
        """Extracts the raw JSON string from a potentially markdown-wrapped or nested response."""
        if not isinstance(response_text, str):
            return response_text

        response_text = response_text.strip()

        # 1. First, try to extract JSON from markdown code blocks
        # Look for ```json or ``` at the start and ``` at the end
        if response_text.startswith("```"):
            # Find the end of the opening marker
            end_of_marker = response_text.find("\n", 3)
            if end_of_marker != -1:
                # Find the closing ``` after the opening marker
                start_of_closing = response_text.find("```", end_of_marker + 1)
                if start_of_closing != -1:
                    # Extract the content between the markers
                    response_text = response_text[
                        end_of_marker + 1 : start_of_closing
                    ].strip()

        # 2. Primary method: find the first { and last }
        start_index = response_text.find("{")
        end_index = response_text.rfind("}")
        if start_index != -1 and end_index != -1 and end_index > start_index:
            cleaned = response_text[start_index : end_index + 1].strip()

            # 3. Handle nested objects if the model wrapped the response in a field like "researchBrief" or "data"
            try:
                data = json.loads(cleaned)
                if isinstance(data, dict) and len(data) == 1:
                    key = list(data.keys())[0]
                    if isinstance(data[key], dict):
                        return json.dumps(data[key])
                return cleaned
            except (json.JSONDecodeError, TypeError, ValueError):
                return cleaned

        return response_text.strip()
