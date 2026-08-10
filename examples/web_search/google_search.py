import sys
from litellm import completion

tools = [{"googleSearch": {}}] # 👈 ADD GOOGLE SEARCH

query = sys.argv[1]

response = completion(
    model="gemini/gemini-2.5-flash",
    messages=[{"role": "user", "content": query}],
    tools=tools,
)

print(response)
