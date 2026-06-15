# LiteKit User Guide

> **Version:** 0.1.0  
> **License:** MIT  
> **Repository:** https://github.com/csv610/LiteKit

LiteKit is an unofficial, opinionated Python toolkit built on top of [LiteLLM](https://github.com/BerriAI/litellm) that provides a unified client for text generation, vision analysis, multi-turn chat, MCQ solving, LLM-as-a-judge evaluation, image processing, and LMDB-backed storage.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [LiteClient — Core LLM Client](#liteclient--core-llm-client)
- [LiteChat — Multi‑Turn Conversation](#litechat--multi-turn-conversation)
- [LiteMCQClient — Multiple‑Choice Solver](#litemcqclient--multiple-choice-solver)
- [ResponseJudge — LLM Evaluation](#responsejudge--llm-evaluation)
- [Vision Utilities](#vision-utilities)
- [LMDB Storage](#lmdb-storage)
- [Logging](#logging)
- [Utilities](#utilities)
- [Model Evaluation](#model-evaluation)
- [CLI Reference](#cli-reference)
- [Configuration Reference](#configuration-reference)
- [Full API Reference](#full-api-reference)
- [Troubleshooting](#troubleshooting)

---

## Installation

### From Source

```bash
git clone https://github.com/csv610/LiteKit.git
cd LiteKit
pip install -e .
```

With development dependencies:

```bash
pip install -e ".[dev]"
```

### Environment Setup

```bash
cp .env.example .env
# Edit .env with your API keys
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

For local models (via Ollama):

```bash
ollama pull gemma3
```

---

## Quick Start

```python
from litekit import LiteClient, ModelConfig, ModelInput

client = LiteClient()

# Basic text generation
result = client.generate_text(
    ModelInput(user_prompt="What is the capital of Japan?")
)
print(result)  # Tokyo

# Vision analysis
result = client.generate_text(
    ModelInput(
        user_prompt="Describe this image in detail.",
        image_path="photo.jpg",
    )
)

# Structured output with Pydantic
from pydantic import BaseModel

class Movie(BaseModel):
    title: str
    rating: int
    genres: list[str]

result = client.generate_text(
    ModelInput(
        user_prompt="Review Inception.",
        response_format=Movie,
    )
)
print(result.title, result.rating)
```

---

## LiteClient — Core LLM Client

`LiteClient` is the primary interface for all LLM interactions. It wraps `litellm.completion` with retry logic, vision support, and structured output parsing.

### Constructor

```python
LiteClient(model_config: Optional[ModelConfig] = None)
```

### Methods

#### `generate_text(model_input, model_config=None, retries=2) -> str | BaseModel`

```python
client = LiteClient(ModelConfig(model="gpt-4o-mini", temperature=0.3))

# Text only
result = client.generate_text(ModelInput(user_prompt="Hello!"))

# With system prompt
result = client.generate_text(ModelInput(
    user_prompt="Summarize this article.",
    system_prompt="You are a helpful assistant.",
))

# With images
result = client.generate_text(ModelInput(
    user_prompt="What do you see?",
    image_path="photo.jpg",
))

# Multiple images
result = client.generate_text(ModelInput(
    user_prompt="Compare these images.",
    image_paths=["photo1.jpg", "photo2.jpg"],
))

# Structured output
class Recipe(BaseModel):
    name: str
    ingredients: list[str]
    steps: list[str]

result = client.generate_text(ModelInput(
    user_prompt="Recipe for chocolate cake.",
    response_format=Recipe,
))
```

Returns a validated `Recipe` instance when `response_format` is a Pydantic class.

**Retry behavior:** On failure, retries up to `retries` times (default 2). Returns an error string if all attempts fail.

#### `create_message(model_input) -> list[dict]` (static)

Builds the message list for `litellm.completion` without executing it. Useful for debugging or when using the LiteLLM API directly.

```python
messages = LiteClient.create_message(ModelInput(
    user_prompt="Hello!",
    system_prompt="Be concise.",
))
# [{'role': 'system', 'content': 'Be concise.'},
#  {'role': 'user', 'content': [{'type': 'text', 'text': 'Hello!'}]}]
```

---

## LiteChat — Multi‑Turn Conversation

`LiteChat` extends `LiteClient` with conversation history, trimming, and optional auto-save to markdown files.

### Constructor

```python
LiteChat(
    model_config: Optional[ModelConfig] = None,
    chat_config: Optional[ChatConfig] = None,
)
```

`ChatConfig` parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_history` | `10` | Max messages kept; auto-decremented to even number |
| `auto_save` | `False` | Save each turn to a markdown file |
| `save_dir` | `"."` | Directory for saved conversations |

### Methods

#### `generate_text(model_input, model_config=None) -> str`

Maintains conversation context across calls:

```python
chat = LiteChat(
    model_config=ModelConfig(model="gpt-4o-mini"),
    chat_config=ChatConfig(max_history=6, auto_save=True),
)

chat.generate_text(ModelInput(user_prompt="My name is Alice."))
chat.generate_text(ModelInput(user_prompt="What is my name?"))
# Assistant: Your name is Alice.
```

#### `reset_conversation()`

Clears all history and the current image.

#### `get_conversation_history() -> list[dict]`

Returns a copy of the raw message history.

#### `save_conversation()`

Appends the latest exchange to a timestamped markdown file. Called automatically when `auto_save=True`.

---

## LiteMCQClient — Multiple‑Choice Solver

`LiteMCQClient` solves multiple-choice questions using a structured Pydantic output schema.

### Constructor

```python
LiteMCQClient(
    model: str = "gemini/gemini-2.5-flash",
    temperature: float = 0.2,
)
```

### Method

#### `solve(question: MCQInput, model_config=None) -> Optional[MultipleChoiceAnswer]`

```python
solver = LiteMCQClient(model="gpt-4o-mini")

# With list options
answer = solver.solve(MCQInput(
    question="What is 2+2?",
    options=["3", "4", "5", "6"],
    context="Basic arithmetic",
))

# With dict options
answer = solver.solve(MCQInput(
    question="Which planet is known as the Red Planet?",
    options={
        "A": "Venus",
        "B": "Mars",
        "C": "Jupiter",
        "D": "Saturn",
    },
))

if answer:
    print(f"Correct options: {[o.key for o in answer.correct_options]}")
    print(f"Reasoning: {answer.reasoning}")
    print(f"Confidence: {answer.confidence:.2f}")
```

Output is a `MultipleChoiceAnswer` with `question`, `correct_options`, `reasoning`, and `confidence`.

---

## ResponseJudge — LLM Evaluation

`ResponseJudge` implements LLM-as-a-judge — it evaluates a model response against a reference answer across four criteria.

### Constructor

```python
ResponseJudge(model_config: ModelConfig)
```

### Method

#### `evaluate(user_input: UserInput) -> EvaluationModel`

```python
judge = ResponseJudge(ModelConfig(model="gpt-4o-mini", temperature=0.0))

result = judge.evaluate(UserInput(
    user_prompt="What is the capital of France?",
    model_response="Paris is the capital of France.",
    ground_truth="Paris",
))

print(f"Score: {result.overall_score:.2f}")
print(f"Correct: {result.is_correct}")
print(f"Accuracy: {result.criteria.accuracy:.2f}")
print(f"Completeness: {result.criteria.completeness:.2f}")
print(f"Relevance: {result.criteria.relevance:.2f}")
print(f"Clarity: {result.criteria.clarity:.2f}")
print(f"Reasoning: {result.reasoning}")
if result.feedback:
    print(f"Feedback: {result.feedback}")
```

### Evaluation Criteria

| Criterion | Range | Description |
|-----------|-------|-------------|
| `accuracy` | 0–1 | Factual correctness against ground truth |
| `completeness` | 0–1 | All parts of the prompt addressed |
| `relevance` | 0–1 | No irrelevant or fabricated information |
| `clarity` | 0–1 | Logical structure and readability |

`overall_score` is the combined score (0–1). `is_correct` is `True` only when there are no major factual errors.

---

## Vision Utilities

The `litekit.vision` package provides comprehensive image handling without requiring any LLM calls.

### Constants

```python
from litekit.vision import (
    MAX_IMAGE_SIZE_MB,       # 50
    MAX_IMAGE_SIZE_BYTES,    # 50 * 1024 * 1024
    MIN_IMAGE_DIMENSION,     # 32
    IMAGE_MIME_TYPE,         # "image/jpeg"
)
```

### Validation

```python
from litekit.vision import is_valid_image, is_valid_size, is_valid_dimensions

is_valid_image("photo.jpg")         # Checks extension (jpg/png/gif/webp)
is_valid_size("photo.jpg")          # Checks file size <= 50 MB
is_valid_dimensions("photo.jpg")    # Checks width/height >= 32 px
```

### I/O

```python
from litekit.vision import (
    encode_to_base64,
    b64_to_pil,
    pil_to_b64,
    cv2_to_pil,
    pil_to_cv2,
    get_image_info,
    save_image,
    save_images_batch,
)

# Encode for LLM vision APIs
b64 = encode_to_base64("photo.jpg")
# "data:image/jpeg;base64,/9j/4AAQ..."

# PIL conversion
pil_img = b64_to_pil(b64)
b64_back = pil_to_b64(pil_img, image_format="PNG")

# OpenCV interop
cv_img = cv2_to_pil(pil_img)
pil_img = pil_to_cv2(cv_img)

# Metadata
info = get_image_info("photo.jpg")
# {'width': 1920, 'height': 1080, 'format': 'JPEG',
#  'color_mode': 'RGB', 'file_size_bytes': 524288, ...}

# Save
save_image(pil_img, "output.png", image_format="PNG")
save_images_batch([pil_img1, pil_img2], "output_dir/")
```

### Processing

```python
from litekit.vision import (
    create_blank_image,
    create_random_image,
    create_gradient_image,
    square_image,
    resize_to_dimensions,
    resize_to_max_size,
    resize_images_to_fit,
    convert_format,
    crop,
    auto_orient,
    remove_exif,
)

# Create images
img = create_blank_image(800, 600, color=(255, 0, 0))
img = create_random_image(100, 100)
img = create_gradient_image(400, 300, (0,0,255), (0,255,0))

# Resize
img = square_image("photo.jpg", max_size=1024)
img = resize_to_dimensions("photo.jpg", width=800, height=600)
img = resize_to_max_size("photo.jpg", max_size=5, size_unit="MB")

# Batch resize (total payload < 50 MB)
resized_paths = resize_images_to_fit(["photo1.jpg", "photo2.jpg"])

# Format conversion
webp_data = convert_format("photo.jpg", target_format="WEBP", quality=80)

# Crop and orient
img = crop("photo.jpg", left=100, top=100, right=500, bottom=500)
img = auto_orient("photo.jpg")
img = remove_exif("photo.jpg")
```

### Collection

```python
from litekit.vision import collect_images, collect_images_with_info

# Collect paths
images = collect_images(
    "./photos",
    recursive=True,
    formats=["jpg", "png"],
    validate=True,
    sort_by="name",  # "name" or "size"
)

# Collect with metadata
images_info = collect_images_with_info(
    "./photos",
    recursive=False,
    sort_by="name",
)
```

---

## LMDB Storage

`LMDBStorage` provides a key-value store using LMDB with automatic compression and JSON import/export.

### Constructor Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `db_path` | `"storage.lmdb"` | Path to database file |
| `capacity_mb` | `100` | Max database size in MB |
| `enable_logging` | `True` | Enable per-instance file logging |
| `compression_threshold` | `100` | Bytes above which values are gzip-compressed |
| `config` | `None` | `LMDBConfig` dataclass (overrides individual params) |

### Usage

```python
from litekit.lmdb_storage import LMDBStorage, LMDBConfig

# Context manager (recommended)
with LMDBStorage("/tmp/mydb.lmdb", capacity_mb=50) as db:
    db.put("key1", "value1")
    db.put("long_value", "x" * 1000)  # Auto-compressed

    print(db.get("key1"))         # "value1"
    print(db.exists("key1"))      # True
    print(db.num_keys())          # 2

    db.delete("key1")
    print(db.get("key1"))         # None

    # Export / import
    db.export_to_json("backup.json")
    db.import_from_json("backup.json")

    # Keys
    keys = db.get_keys()                     # list
    keys_gen = db.get_keys(as_generator=True) # generator

    # Stats
    print(db.get_stats())

# Manual lifecycle
db = LMDBStorage(config=LMDBConfig(db_path="/tmp/mydb.lmdb"))
db.put("manual", "value")
db.close()
```

### Key Features

- **Automatic compression:** Values larger than `compression_threshold` bytes are gzip-compressed transparently
- **Context manager:** Use `with` for automatic cleanup
- **JSON import/export:** Portable backup and restore
- **Logging:** Per-instance file logging with optional disable
- **Memory-efficient iteration:** Generator mode for large databases

---

## Logging

### `configure_logging(log_file, level, enable_console, verbosity)`

```python
from litekit import configure_logging
import logging

configure_logging(
    log_file="myapp.log",
    level=logging.INFO,
    enable_console=False,
    verbosity=None,  # 0=CRITICAL, 1=ERROR, 2=WARNING, 3=INFO, 4=DEBUG
)
```

- Directs all logs to `logs/` directory by default
- Removes existing console handlers, preserves file handlers
- Configures both root and `litellm` loggers
- `verbosity` overrides `level` if provided

---

## Utilities

### `print_response(result, title="Result")`

Pretty-prints Pydantic models, dicts, or any object recursively:

```python
from litekit.utils import print_response, print_simple_result
from litekit import ModelOutput

output = ModelOutput(
    data={"role": "assistant", "content": "Hello!"},
    markdown="# Hello",
    metadata={"model": "gpt-4o", "tokens": 42},
)
print_response(output, title="Model Output")

print_simple_result("Done!", title="Status")
```

### `save_model_response(model, output_path) -> Path`

Saves a Pydantic model as JSON or a string as markdown:

```python
from litekit.utils import save_model_response
from pydantic import BaseModel

class Result(BaseModel):
    name: str
    score: float

save_model_response(Result(name="test", score=9.5), "outputs/result.json")
save_model_response("Plain text response", "outputs/response.md")
```

### `JSONCleaner.extract_json(text) -> str`

Extracts clean JSON from LLM responses that may include markdown fences or extra wrapping:

```python
from litekit.utils.json_cleaner import JSONCleaner

JSONCleaner.extract_json('```json\n{"key": "value"}\n```')
# '{"key": "value"}'

JSONCleaner.extract_json('{"result": {"deep": true}}')
# '{"deep": true}'  (unwraps single-key wrappers)
```

---

## Model Evaluation

The `eval_models.py` script benchmarks models across text and vision tasks, reporting success rate, execution time, and token usage.

```bash
# Full evaluation (all models, all tests)
python eval_models.py

# Text-only
python eval_models.py --mode text

# Vision-only (requires images in current directory)
python eval_models.py --mode vision

# Specific models
python eval_models.py --models gemma4:e2b gemma4:e4b

# Multiple runs for statistical significance
python eval_models.py --runs 3
```

Test cases cover: factual questions (capital cities), explanations (quantum entanglement), creative writing (haiku), structured output via Pydantic schemas (research briefs, MCQs), and image description for vision models.

---

## CLI Reference

### LiteChat Interactive CLI

```bash
python -m litekit.lite_chat
python -m litekit.lite_chat -m "gpt-4o-mini"
python -m litekit.lite_chat -m "gemini/gemini-2.5-flash" -i photo.jpg --auto-save
```

| Flag | Default | Description |
|------|---------|-------------|
| `-m`, `--model` | `gemini/gemini-2.5-flash` | Model identifier |
| `-t`, `--temperature` | `0.2` | Sampling temperature |
| `-i`, `--image-path` | `None` | Image for vision on first turn |
| `--max-history` | `10` | Max messages in history |
| `--auto-save` | `False` | Save conversation to markdown |
| `--save-dir` | `.` | Save directory |
| `-v`, `--version` | — | Show version |

**Interactive commands:** `exit`, `history`, `clear`

### LiteMCQClient CLI

```bash
python -m litekit.lite_mcq_client \
    -q "What is the capital of France?" \
    -o "A: London" "B: Paris" "C: Berlin" "D: Madrid"

python -m litekit.lite_mcq_client \
    -f questions.json \
    -m "gpt-4o"
```

| Flag | Description |
|------|-------------|
| `-q`, `--question` | The question text |
| `-o`, `--options` | Options as space-separated strings |
| `-c`, `--context` | Optional context |
| `-i`, `--images` | Image paths (multiple) |
| `-m`, `--model` | Model name |
| `-f`, `--file` | Load question from JSON file |

### ResponseJudge CLI

```bash
python -m litekit.lite_response_judge \
    -p "What is 2+2?" \
    -r "4" \
    -g "4" \
    -m "gpt-4o-mini"
```

| Flag | Required | Description |
|------|----------|-------------|
| `-p`, `--prompt` | No | Original user prompt |
| `-r`, `--response` | **Yes** | Model response to evaluate |
| `-g`, `--ground-truth` | No | Expected answer |
| `-m`, `--model` | No | Judge model (default: `gemini/gemini-2.5-flash`) |

---

## Configuration Reference

### `ModelConfig`

| Field | Type | Default | Validation |
|-------|------|---------|------------|
| `model` | `str` | — | Required, non-empty |
| `temperature` | `float` | `0.2` | 0.0–2.0 |
| `timeout` | `Optional[int]` | `None` | Must be positive if set |

### `ChatConfig`

| Field | Type | Default |
|-------|------|---------|
| `max_history` | `int` | `10` |
| `auto_save` | `bool` | `False` |
| `save_dir` | `str` | `"."` |

### `ModelInput`

| Field | Type | Default |
|-------|------|---------|
| `user_prompt` | `str` | `""` |
| `image_path` | `Optional[str]` | `None` |
| `image_paths` | `Optional[list[str]]` | `None` |
| `system_prompt` | `Optional[str]` | `None` |
| `response_format` | `Optional[Type[BaseModel] \| str]` | `None` |

### `MCQInput`

| Field | Type | Default |
|-------|------|---------|
| `question` | `str` | — |
| `options` | `list[str] \| dict[str, str]` | — |
| `context` | `Optional[str]` | `None` |
| `image_paths` | `Optional[list[str]]` | `None` |

### `UserInput`

| Field | Type | Default |
|-------|------|---------|
| `model_response` | `str` | — |
| `user_prompt` | `Optional[str]` | `None` |
| `ground_truth` | `Optional[str]` | `None` |
| `context` | `Optional[str]` | `None` |

### `LMDBConfig`

| Field | Type | Default | Validation |
|-------|------|---------|------------|
| `db_path` | `str` | `"storage.lmdb"` | Non-empty |
| `capacity_mb` | `int` | `100` | > 0 |
| `enable_logging` | `bool` | `True` | — |
| `compression_threshold` | `int` | `100` | >= 0 |
| `max_key_size` | `int` | `511` | > 0 |

---

## Full API Reference

### `litekit` Package

```python
from litekit import (
    LiteClient,       # Core LLM client
    LiteChat,         # Multi-turn chat
    LiteMCQClient,    # MCQ solver
    ResponseJudge,    # LLM-as-a-judge
    EvaluationModel,  # Judge output schema
    ModelConfig,      # Model configuration
    ModelInput,       # Model input parameters
    MCQInput,         # MCQ input parameters
    UserInput,        # Judge input parameters
    ChatConfig,       # Chat configuration
    ModelOutput,      # Standardized output
    ImageUtils,       # Deprecated vision compat shim
    configure_logging,  # Logging setup
    save_model_response,  # Save output to file
)
```

### `litekit.vision` Package

```python
from litekit.vision import (
    # Constants
    IMAGE_MIME_TYPE, MAX_IMAGE_SIZE_MB, MAX_IMAGE_SIZE_BYTES,
    MAX_TOTAL_IMAGE_PAYLOAD_MB, MAX_TOTAL_IMAGE_PAYLOAD_BYTES,
    MIN_IMAGE_DIMENSION,

    # Validation
    is_valid_image, is_valid_size, is_valid_dimensions,

    # I/O
    encode_to_base64, b64_to_pil, pil_to_b64, cv2_to_pil, pil_to_cv2,
    get_image_info, save_image, save_images_batch,

    # Processing
    create_blank_image, create_random_image, create_gradient_image,
    resize_images_to_fit, square_image, resize_to_dimensions,
    convert_format, crop, auto_orient, remove_exif, resize_to_max_size,

    # Collection
    collect_images, collect_images_with_info,
)
```

### `litekit.lmdb_storage` Module

```python
from litekit.lmdb_storage import LMDBStorage, LMDBConfig
```

### `litekit.utils` Module

```python
from litekit.utils import print_response, print_simple_result, save_model_response
from litekit.utils.json_cleaner import JSONCleaner
```

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---------|-------------|----------|
| `AuthenticationError` | Missing or invalid API key | Check `.env` file or environment variables |
| `FileNotFoundError` on image | Path does not exist | Use absolute path or verify relative path |
| Structured output returns string, not Pydantic | Model does not support structured output | Use a model with JSON mode support (GPT-4o, Gemini 2.5) |
| LiteChat history not working | Odd `max_history` | LiteKit auto-decrements odd values to maintain pairs |
| `ModuleNotFoundError: litellm` | Dependencies not installed | Run `pip install -e .` |
| Empty response from `LiteMCQClient` | Model returned JSON not matching schema | Try a different model or lower temperature |
| LMDB `MapFullError` | Database capacity exceeded | Increase `capacity_mb` when creating `LMDBStorage` |
| `eval_models.py` finds no images | No images in current directory | Place `.jpg`/`.png` files or specify a different directory |
| Rate limiting errors | Too many requests in short time | Add delays between calls or switch to a model with higher rate limits |

---

## Directory Structure

```
LiteKit/
├── litekit/                     # Core package
│   ├── __init__.py              # Public API exports
│   ├── config.py                # ModelConfig, ModelInput, MCQInput, UserInput, ChatConfig, ModelOutput
│   ├── lite_client.py           # LiteClient — text + vision + structured output
│   ├── lite_chat.py             # LiteChat — multi-turn conversation
│   ├── lite_mcq_client.py       # LiteMCQClient — MCQ solver
│   ├── lite_response_judge.py   # ResponseJudge — LLM-as-a-judge
│   ├── lmdb_storage.py          # LMDBStorage — key-value store
│   ├── logging_config.py        # configure_logging()
│   ├── image_utils.py           # Deprecated compat shim for vision/
│   ├── vision/                  # Image validation, I/O, processing, collection
│   │   ├── __init__.py
│   │   ├── core.py
│   │   ├── validation.py
│   │   ├── io.py
│   │   ├── processing.py
│   │   └── collection.py
│   ├── utils/                   # Pretty-print, save, JSON cleaning
│   │   ├── __init__.py
│   │   ├── print_response.py
│   │   ├── save_response.py
│   │   └── json_cleaner.py
│   └── storage/                 # Storage configuration
│       └── storage_config.py
├── examples/                    # Example scripts
├── tests/                       # pytest test suite (247+ tests)
├── eval_models.py               # Model evaluation benchmark
├── pyproject.toml
├── setup.py
└── Makefile
```

---

## Examples

See the `examples/` directory for runnable scripts covering every major feature:

| Example | Description |
|---------|-------------|
| `01_basic_text_generation.py` | Text generation with LiteClient |
| `02_vision_analysis.py` | Vision analysis with image input |
| `03_multi_turn_chat.py` | Conversation with LiteChat |
| `04_mcq_solver.py` | Multiple-choice question solving |
| `05_response_judge.py` | LLM-as-a-judge evaluation |
| `06_lmdb_storage.py` | Key-value storage with LMDB |
| `07_vision_utilities.py` | Image encoding, resizing, format conversion |
| `08_json_cleaner.py` | JSON extraction from LLM responses |
| `09_print_and_save.py` | Pretty-print and save to disk |
| `10_structured_output.py` | Pydantic structured output |

---

## Makefile Commands

```bash
make venv           # Create virtual environment
make install        # Install dependencies
make test           # Run tests
make lint           # Run pylint
make format         # Run black
make run-cli-text   # Run CLI text mode
make run-cli-vision # Run CLI vision mode
make clean          # Remove cache and build files
```
