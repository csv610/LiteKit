"""Example: Basic text generation with LiteClient."""

from litekit import LiteClient, ModelConfig, ModelInput

config = ModelConfig(model="gpt-4o-mini", temperature=0.3)
client = LiteClient(model_config=config)

result = client.generate_text(
    ModelInput(user_prompt="Explain what an LLM is in one sentence.")
)
print(result)
