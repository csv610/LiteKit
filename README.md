# LiteKit

An unofficial, opinionated Python toolkit built on top of [LiteLLM (BerriAI)](https://github.com/BerriAI/litellm) that provides a unified client for text generation, vision analysis, multi-turn chat, MCQ solving, LLM-as-a-judge evaluation, image processing, and LMDB-backed storage.

## Installation

```bash
git clone https://github.com/csv610/LiteKit.git
cd LiteKit
pip install -e .
```

Or with dev dependencies:

```bash
pip install -e ".[dev]"
```

### Quick Setup

```bash
# Copy and fill in API keys
cp .env.example .env

# Or export directly
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

For local models, ensure [Ollama](https://ollama.ai) is running:

```bash
ollama pull gemma3
```

## Core Components

### LiteClient

Unified text and vision client with retry logic and Pydantic structured output. The primary interface for all LLM interactions.

```python
from litekit import LiteClient, ModelConfig, ModelInput

client = LiteClient()

# Text generation
result = client.generate_text(
    ModelInput(user_prompt="Explain quantum computing in simple terms")
)

# Vision analysis
result = client.generate_text(
    ModelInput(
        user_prompt="Describe this image",
        image_path="/path/to/image.jpg"
    )
)

# Structured output with Pydantic
from pydantic import BaseModel

class Recipe(BaseModel):
    name: str
    ingredients: list[str]
    steps: list[str]

result = client.generate_text(
    ModelInput(
        user_prompt="Recipe for chocolate cake",
        response_format=Recipe
    )
)
# result is a validated Recipe instance
```

### LiteChat

Multi-turn conversational client with history management and auto-save.

```python
from litekit import LiteChat, ModelConfig, ChatConfig

chat = LiteChat(
    chat_config=ChatConfig(max_history=10, auto_save=True)
)

response = chat.generate_text(ModelInput(user_prompt="Hello!"))
response = chat.generate_text(ModelInput(user_prompt="What did I just say?"))
# Maintains conversation context
```

Interactive CLI:

```bash
python -m litekit.lite_chat -m "gemini/gemini-2.0-flash"
```

### LiteMCQClient

Multiple-choice question solver with structured output.

```python
from litekit import LiteMCQClient, MCQInput

solver = LiteMCQClient()
answer = solver.solve(
    MCQInput(
        question="What is the capital of France?",
        options=["London", "Paris", "Berlin", "Madrid"],
        context="European geography"
    )
)
print(answer.correct_options)  # [CorrectOption(key='B', value='Paris')]
```

CLI:

```bash
python -m litekit.lite_mcq_client -q "What is 2+2?" -o "3,4,5,6"
```

### ResponseJudge

LLM-as-a-judge evaluation engine. Scores responses across four criteria (accuracy, completeness, relevance, clarity) with an overall score and pass/fail判定.

```python
from litekit import ResponseJudge, UserInput, ModelConfig

judge = ResponseJudge(ModelConfig(model="gpt-4"))
result = judge.evaluate(
    UserInput(
        model_response="Paris is the capital of France.",
        user_prompt="What is the capital of France?",
        ground_truth="Paris"
    )
)
print(result.overall_score)  # 0.0 - 1.0
print(result.is_correct)     # True / False
```

CLI:

```bash
python -m litekit.lite_response_judge -p "What is 2+2?" -r "4"
```

### Image Utilities

Comprehensive image validation, I/O, processing, and collection.

```python
from litekit.vision import (
    encode_to_base64, get_image_info, save_image,
    resize_to_max_size, square_image, convert_format,
    collect_images, auto_orient, crop
)

# Encode image for LLM vision
b64 = encode_to_base64("/path/to/image.jpg")

# Get metadata
info = get_image_info("/path/to/image.jpg")

# Resize to fit size limit
resize_to_max_size("large.jpg", max_size=5, size_unit="MB")

# Collect images from directory
images = collect_images("./photos", recursive=True, formats=["jpg", "png"])
```

### LMDB Storage

Key-value store with automatic compression and JSON import/export.

```python
from litekit import LMDBStorage

with LMDBStorage(db_path="/tmp/mydb.lmdb") as db:
    db.put("key1", "value1")
    print(db.get("key1"))   # "value1"
    print(db.num_keys())    # 1
    db.export_to_json("backup.json")
```

### Model Evaluation

Benchmark models across text and vision tasks:

```bash
python eval_models.py --mode all --runs 3
```

## Project Structure

```
LiteKit/
├── litekit/                     # Core package
│   ├── __init__.py              # Public API facade
│   ├── config.py                # ModelConfig, ModelInput, MCQInput, etc.
│   ├── lite_client.py           # Core LLM client (text + vision)
│   ├── lite_chat.py             # Multi-turn chat with history
│   ├── lite_mcq_client.py       # MCQ solver
│   ├── lite_response_judge.py   # LLM-as-a-judge evaluation
│   ├── image_utils.py           # Deprecated compat shim → vision/
│   ├── logging_config.py        # Centralized logging setup
│   ├── lmdb_storage.py          # LMDB key-value store
│   ├── vision/                  # Image validation, I/O, processing, collection
│   │   ├── validation.py
│   │   ├── io.py
│   │   ├── processing.py
│   │   └── collection.py
│   ├── utils/                   # JSON cleaning, pretty-print, save helpers
│   │   ├── json_cleaner.py
│   │   ├── print_response.py
│   │   └── save_response.py
│   └── storage/                 # Storage configuration
│       └── storage_config.py
├── tests/                       # pytest test suite (17 files)
├── eval_models.py               # Model evaluation benchmark
├── pyproject.toml               # Modern Python packaging
├── setup.py                     # Legacy setup (backward compat)
├── Makefile                     # Automation
└── .env.example                 # Environment template
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
make test

# Or directly
python -m pytest tests/ -v

# Lint and format
pylint litekit/ tests/
black litekit/ tests/
ruff check litekit/ tests/
```

## Requirements

- Python 3.8+
- An API key for at least one provider (OpenAI, Google Gemini, Anthropic) or a local Ollama instance

## License

MIT
