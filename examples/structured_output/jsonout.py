import os
from litellm import completion 
from pydantic import BaseModel
import json

class CalendarEvent(BaseModel):
  name: str
  date: str
  participants: list[str]

class EventsList(BaseModel):
    events: list[CalendarEvent]

model = "gemini/gemini-2.5-flash" 

messages = [{"role": "user", "content": "List 5 important events in the XIX century"}]

response = completion(
    model= model,
    messages=messages,
    response_format=EventsList
)

json_str = response.choices[0].message.content
# Parse the JSON string into a Python dictionary
data = json.loads(json_str)

# Use json.dumps to print the data in a human-readable format
print(json.dumps(data, indent=4))
