import sys
from litellm import completion

# 👇 ADD URL CONTEXT
tools = [{"urlContext": {}}]

url  = sys.argv[1]

response = completion(
    model="gemini/gemini-2.5-flash",
    messages=[{"role": "user", "content": f"Summarize this document: {url}"}],
    tools=tools,
)

print(response)

