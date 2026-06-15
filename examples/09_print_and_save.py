"""Example: Print responses and save to disk."""

from pydantic import BaseModel

from litekit import ModelOutput
from litekit.utils import print_response, print_simple_result, save_model_response

output = ModelOutput(
    data={"role": "assistant", "content": "Hello!"},
    markdown="# Assistant\n\nHello!",
    metadata={"model": "gpt-4o", "tokens": 42},
)

print_response(output, title="Model Output")
print_simple_result("All done!", title="Status")

path = save_model_response(output, "outputs/response.json")
print(f"Saved to: {path}")

class MyModel(BaseModel):
    name: str
    score: float

m = MyModel(name="test", score=9.5)
save_model_response(m, "outputs/result.json")
print_response(m)
