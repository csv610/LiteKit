from litellm import completion
import os
import sys

question = sys.argv[1]
model="gemini/gemini-2.5-flash"

response = completion(
  model=model,
  messages = [{ "role": "user", "content": question}],
  reasoning_efforts = 'low'
)
print( response)

print( response.choices[0].message.content)
