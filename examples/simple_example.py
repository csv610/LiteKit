from litellm import completion
import os
import sys

query = sys.argv[1]

model = "claude-sonnet-4-5-20250929"
model = "ollama/llama3.2"
model = "perplexity/sonar-pro" 
model = "gemini/gemini-2.5-flash"

messages=[{"role": "user", "content": query}]

response = completion(model=model, messages=messages)
print(response)
