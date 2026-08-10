import os
from litellm import completion
from pydantic import BaseModel, Field
import json
import sys

class MedicineInfo(BaseModel):
    name: str = Field(..., description="The generic name of the medicine.")
    brand_name: str = Field(..., description="The common brand name for the medicine.")
    description: str = Field(..., description="A brief description of how the medicine works and its primary uses.")
    history: str = Field(..., description="A brief history of the medicine including the approval date from medical uses.")
    active_ingredient: str = Field(..., description="The main chemical component of the medicine.")
    uses: list[str] = Field(..., description="A list of medical conditions the medicine is used to treat.")
    side_effects: list[str] = Field(..., description="A list of common side effects associated with the medicine.")

class MedicineResponse(BaseModel):
    medicine: MedicineInfo

def cli(medicine):
    model = "gemini/gemini-2.5-flash"

    # Update the user message to request information about a specific medicine
    messages = [{"role": "user", "content": f"Provide detailed information about the medicine {medicine}."}]

    response = completion(
        model=model,
        messages=messages,
        response_format=MedicineResponse
    )

    # Parse and print the formatted JSON output
    json_string = response.choices[0].message.content
    data = json.loads(json_string)

    print(json.dumps(data, indent=4))

medicine = sys.argv[1]
cli(medicine)
