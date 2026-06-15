"""Example: Extract clean JSON from LLM responses."""

from litekit.utils.json_cleaner import JSONCleaner

raw = '```json\n{"name": "Alice", "age": 30}\n```'
cleaned = JSONCleaner.extract_json(raw)
print("Cleaned:", cleaned)

raw2 = '{"result": {"answer": 42}}'
cleaned2 = JSONCleaner.extract_json(raw2)
print("Unwrapped:", cleaned2)

raw3 = "invalid text"
cleaned3 = JSONCleaner.extract_json(raw3)
print("Fallback:", cleaned3)
