"""Example: Structured (Pydantic) output with LiteClient."""

from pydantic import BaseModel

from litekit import LiteClient, ModelConfig, ModelInput


class MovieReview(BaseModel):
    title: str
    rating: int
    summary: str
    genres: list[str]


config = ModelConfig(model="gpt-4o-mini", temperature=0.2)
client = LiteClient(model_config=config)

result = client.generate_text(
    ModelInput(
        user_prompt='Review "Inception" in a structured format.',
        response_format=MovieReview,
    )
)

if isinstance(result, MovieReview):
    print(f"Title: {result.title}")
    print(f"Rating: {result.rating}/10")
    print(f"Summary: {result.summary}")
    print(f"Genres: {', '.join(result.genres)}")
else:
    print("Raw response:", result)
