"""Example: Vision analysis with LiteClient."""

from litekit import LiteClient, ModelConfig, ModelInput

config = ModelConfig(model="gpt-4o", temperature=0.2)
client = LiteClient(model_config=config)

result = client.generate_text(
    ModelInput(
        user_prompt="What objects do you see in this image?",
        image_path="path/to/photo.jpg",
    )
)
print(result)
